"""
Test per due fix nella generazione segnali di analyzer.py (audit 2026-09-05):

1. candlestick_strategy(): teneva l'ULTIMO pattern controllato, non quello
   con lo score più alto — un pattern debole controllato dopo uno forte
   poteva sovrascriverlo anche in direzione opposta, solo per l'ordine del
   codice.
2. ml_alpha_strategy(): un tentativo di "regressione logistica reale" si
   allenava su feature diverse da quelle usate in previsione (mismatch
   train/predict) - rimosso, resta solo lo score composito rule-based.
"""

import contextlib
import os
import sys
import unittest
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer
from analyzer import candlestick_strategy, ml_alpha_strategy, _stat_arb_score_from_means, statistical_arbitrage_strategy, smc_v3_strategy, detect_swing_points, calibrate_probability_for_display, smc_generic_zone_signal, _sniper_analyze, full_analyze, SNIPER_TIMEFRAME, SNIPER_FIXED_PROB, SNIPER_CONFIGS
from unittest.mock import patch


def _candle_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


class TestCandlestickStrategyHighestScoreWins(unittest.TestCase):
    def test_strong_pattern_not_overridden_by_weaker_conflicting_one(self):
        """candela0 e' costruita per matchare SIA Pinbar Bullish (score 7,
        BUY, controllato per primo) SIA Harami Bearish (score 5, SELL,
        controllato dopo) - stesso identico scenario che prima del fix
        avrebbe fatto vincere Harami Bearish solo perche' controllato dopo,
        nonostante il punteggio piu' basso."""
        df = _candle_df([
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 100, "High": 111, "Low": 99, "Close": 110},   # candela1
            {"Open": 105, "High": 107, "Low": 85, "Close": 103},   # candela0
        ])
        result = candlestick_strategy(df)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["score"], 7)
        self.assertIn("Pinbar Bullish", result["pattern"])
        # Il pattern piu' debole va comunque elencato nella motivazione,
        # solo non deve determinare il segnale finale.
        self.assertIn("Harami Bearish", result["reason"])

    def test_single_matching_pattern_still_works(self):
        # Engulfing Bullish pulito, nessun altro pattern in conflitto.
        df = _candle_df([
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Open": 110, "High": 111, "Low": 95, "Close": 96},    # candela1 bearish
            {"Open": 95,  "High": 121, "Low": 94, "Close": 120},   # candela0 engulfing bullish
        ])
        result = candlestick_strategy(df)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["score"], 8)
        self.assertIn("Engulfing Bullish", result["pattern"])

    def test_no_pattern_returns_neutral(self):
        df = _candle_df([
            {"Open": 100, "High": 100.5, "Low": 99.5, "Close": 100.1}
            for _ in range(5)
        ])
        result = candlestick_strategy(df)
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_too_few_candles_returns_neutral(self):
        df = _candle_df([{"Open": 100, "High": 101, "Low": 99, "Close": 100}] * 3)
        result = candlestick_strategy(df)
        self.assertEqual(result["signal"], "NEUTRAL")


def _ml_alpha_df(n=60, trend: str = "up") -> pd.DataFrame:
    rows = []
    for i in range(n):
        base = 100 + (i * 0.5 if trend == "up" else -i * 0.5 if trend == "down" else 0)
        rows.append({
            "Close": base, "ema9": base - (1 if trend == "up" else -1 if trend == "down" else 0),
            "ema20": base - (2 if trend == "up" else -2 if trend == "down" else 0),
            "ema50": base - (3 if trend == "up" else -3 if trend == "down" else 0),
            "ema200": base - (4 if trend == "up" else -4 if trend == "down" else 0),
            "rsi": 65 if trend == "up" else 35 if trend == "down" else 50,
            "rsi_fast": 70 if trend == "up" else 30 if trend == "down" else 50,
            "macd_hist": 1.0 if trend == "up" else -1.0 if trend == "down" else 0.0,
            "atr": 5.0,
        })
    return pd.DataFrame(rows)


class TestMlAlphaStrategyRuleBasedOnly(unittest.TestCase):
    """Dopo la rimozione del tentativo ML confuso, la funzione deve restare
    puramente deterministica (nessun accesso a sklearn/DB) e coerente con
    le stesse feature tecniche in ingresso."""

    def test_strong_uptrend_signals_buy(self):
        df = _ml_alpha_df(trend="up")
        mtf = {"1h": "BUY", "4h": "BUY", "1day": "BUY", "15min": "BUY", "5min": "BUY", "1min": "BUY"}
        result = ml_alpha_strategy(df, mtf, {"structure": "BULLISH"})
        self.assertEqual(result["signal"], "BUY")
        self.assertIn("rule-based", result["reason"])

    def test_strong_downtrend_signals_sell(self):
        df = _ml_alpha_df(trend="down")
        mtf = {"1h": "SELL", "4h": "SELL", "1day": "SELL", "15min": "SELL", "5min": "SELL", "1min": "SELL"}
        result = ml_alpha_strategy(df, mtf, {"structure": "BEARISH"})
        self.assertEqual(result["signal"], "SELL")

    def test_flat_market_is_neutral(self):
        df = _ml_alpha_df(trend="flat")
        mtf = {"1h": "NEUTRAL", "4h": "NEUTRAL"}
        result = ml_alpha_strategy(df, mtf, {"structure": "NEUTRAL"})
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_too_few_candles_returns_neutral(self):
        df = _ml_alpha_df(n=10, trend="up")
        result = ml_alpha_strategy(df, {}, {"structure": "NEUTRAL"})
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_no_sklearn_or_db_dependency(self):
        """Non deve piu' importare sklearn ne' toccare il DB - verificato
        indirettamente: nessuna eccezione anche senza DB_PATH/BOT_DIR
        validi nell'ambiente."""
        df = _ml_alpha_df(trend="up")
        os.environ.pop("DB_PATH", None)
        result = ml_alpha_strategy(df, {"1h": "BUY"}, {"structure": "BULLISH"})
        self.assertIn(result["signal"], ("BUY", "SELL", "NEUTRAL"))


class TestStatArbScoreFromMeans(unittest.TestCase):
    """_stat_arb_score_from_means() (2026-09-05): logica pura estratta da
    statistical_arbitrage_strategy() così che backtest.py possa riusarla
    con medie storiche gratuite invece di duplicarne le soglie."""

    def test_dxy_above_ma_signals_sell(self):
        result = _stat_arb_score_from_means(dxy=102.0, us10y=90.0, dxy_ma=100.0, tlt_ma=None)
        self.assertEqual(result["signal"], "SELL")

    def test_dxy_below_ma_signals_buy(self):
        result = _stat_arb_score_from_means(dxy=98.0, us10y=90.0, dxy_ma=100.0, tlt_ma=None)
        self.assertEqual(result["signal"], "BUY")

    def test_tlt_above_ma_signals_buy(self):
        # TLT alto = yields bassi = positivo per l'oro
        result = _stat_arb_score_from_means(dxy=100.0, us10y=92.0, dxy_ma=None, tlt_ma=90.0)
        self.assertEqual(result["signal"], "BUY")

    def test_tlt_below_ma_signals_sell(self):
        result = _stat_arb_score_from_means(dxy=100.0, us10y=88.0, dxy_ma=None, tlt_ma=90.0)
        self.assertEqual(result["signal"], "SELL")

    def test_conflicting_signals_cancel_to_neutral(self):
        # DXY dice SELL (+2), TLT dice BUY (+2): pareggio -> nessuna direzione vince
        result = _stat_arb_score_from_means(dxy=102.0, us10y=92.0, dxy_ma=100.0, tlt_ma=90.0)
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_small_deviation_is_neutral(self):
        result = _stat_arb_score_from_means(dxy=100.3, us10y=90.0, dxy_ma=100.0, tlt_ma=None)
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_missing_means_is_neutral(self):
        result = _stat_arb_score_from_means(dxy=100.0, us10y=90.0, dxy_ma=None, tlt_ma=None)
        self.assertEqual(result["signal"], "NEUTRAL")
        self.assertEqual(result["score"], 0)


class TestStatisticalArbitrageStrategyWrapper(unittest.TestCase):
    """Regressione: statistical_arbitrage_strategy() deve comportarsi
    esattamente come prima del refactor (delega a
    _stat_arb_score_from_means)."""

    def test_zero_dxy_or_us10y_is_neutral(self):
        self.assertEqual(statistical_arbitrage_strategy(4000.0, 0.0, 90.0)["signal"], "NEUTRAL")
        self.assertEqual(statistical_arbitrage_strategy(4000.0, 100.0, 0.0)["signal"], "NEUTRAL")

    def test_delegates_to_pure_scoring_with_fetched_means(self):
        import pandas as pd
        dxy_hist = pd.DataFrame({"close": [100.0] * 15})
        tlt_hist = pd.DataFrame({"close": [90.0] * 15})
        with patch("analyzer.get_dxy_history", return_value=dxy_hist), \
             patch("analyzer.get_tlt_history", return_value=tlt_hist):
            result = statistical_arbitrage_strategy(4000.0, 103.0, 90.0)
        self.assertEqual(result["signal"], "SELL")


class TestSmcV3StrategyUsesEvaluationTime(unittest.TestCase):
    """FIX (2026-09-06): smc_v3_strategy() usava datetime.now(TIMEZONE) —
    l'ora REALE del computer — per il filtro di sessione 14-19 IT, invece
    dell'orario della barra simulata. Risultato: un backtest che la
    richiama con dati storici otteneva sempre NEUTRAL a meno di girare lo
    script fra le 14 e le 19 reali (verificato: 39295 chiamate, 0 segnali,
    indipendentemente dai dati). Aggiunto un parametro opzionale `now` che
    di default preserva il comportamento live (datetime.now(TIMEZONE)) ma
    permette a un backtest di passare l'orario storico simulato."""

    def _df(self, n=40):
        idx = pd.date_range("2026-01-01", periods=n, freq="15min")
        return pd.DataFrame({
            "Open": [100.0] * n, "High": [100.5] * n, "Low": [99.5] * n,
            "Close": [100.0] * n, "Volume": [100] * n,
        }, index=idx)

    def test_explicit_now_outside_session_blocks_before_evaluating_setups(self):
        outside_hours = analyzer.TIMEZONE.localize(datetime(2026, 3, 4, 21, 45))
        with patch("analyzer.detect_bos_choch") as mock_choch:
            result = smc_v3_strategy(self._df(), self._df(), {}, {}, {}, now=outside_hours)
        self.assertEqual(result, {"signal": "NEUTRAL", "setup": None, "score": 0})
        mock_choch.assert_not_called()

    def test_explicit_now_inside_session_proceeds_to_evaluate_setups(self):
        """Prima del fix, questa chiamata sarebbe stata bloccata ogni volta
        che lo script gira fuori dalle 14-19 reali, indipendentemente
        dall'orario storico simulato passato: col fix, e' SOLO `now` a
        decidere se procedere."""
        inside_hours = analyzer.TIMEZONE.localize(datetime(2026, 3, 4, 15, 30))
        neutral_smc = {"choch": None, "bos": None, "structure": "NEUTRAL"}
        with patch("analyzer.detect_bos_choch", return_value=neutral_smc) as mock_choch:
            smc_v3_strategy(self._df(), self._df(), {}, {}, {}, now=inside_hours)
        mock_choch.assert_called()

    def test_omitting_now_preserves_live_behaviour_using_real_clock(self):
        """Comportamento live invariato: senza passare `now`, la funzione
        deve continuare a usare l'orologio reale (datetime.now)."""
        fake_real_now = analyzer.TIMEZONE.localize(datetime(2026, 3, 4, 3, 0))
        with patch("analyzer.datetime") as mock_datetime_cls:
            mock_datetime_cls.now.return_value = fake_real_now
            result = smc_v3_strategy(self._df(), self._df(), {}, {}, {})
        mock_datetime_cls.now.assert_called_once()
        self.assertEqual(result["signal"], "NEUTRAL")


class TestSmcV3StrategyOnlySetup1Remains(unittest.TestCase):
    """FIX (2026-09-07): primo backtest reale su 5 anni completi (1154
    trade) - PF 1.0 esatto nell'aggregato, ma solo perche' il Setup 1
    (CHoCH+OB) genuinamente positivo (n=161, +106.58R, confermato su
    ENTRAMBE le meta' cronologiche indipendenti 2021-2024 e 2024-2026)
    veniva esattamente compensato dagli altri 3 setup (Liquidity Sweep,
    OB+FVG Confluence, Premium/Discount - n=989, -110.12R). Rimossi.
    Questi test verificano che i setup rimossi non possano più generare un
    segnale (non solo che "in questo caso" non lo fanno) mockando i
    detector in modo che le loro condizioni sarebbero soddisfatte, e che i
    detector stessi non vengano nemmeno più chiamati."""

    def _df(self, n=40):
        idx = pd.date_range("2026-01-01", periods=n, freq="15min")
        return pd.DataFrame({
            "Open": [100.0] * n, "High": [100.5] * n, "Low": [99.5] * n,
            "Close": [100.0] * n, "Volume": [100] * n,
        }, index=idx)

    def test_setup1_bullish_still_fires(self):
        inside_hours = analyzer.TIMEZONE.localize(datetime(2026, 3, 4, 15, 30))
        smc_result = {"choch": "CHOCH_BULLISH", "bos": None, "structure": "BULLISH"}
        ob_result = {"bullish_ob": {"low": 99.0, "high": 100.5}}
        with patch("analyzer.detect_bos_choch", return_value=smc_result), \
             patch("analyzer.detect_order_blocks", return_value=ob_result):
            result = smc_v3_strategy(self._df(), self._df(), {}, {}, {}, now=inside_hours)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["setup"], "Setup 1: CHoCH + OB Bullish")

    def test_setups_2_through_5_never_fire_even_when_their_conditions_hold(self):
        inside_hours = analyzer.TIMEZONE.localize(datetime(2026, 3, 4, 15, 30))
        # Nessun bullish/bearish OB -> Setup 1 non puo' scattare, cosi'
        # isoliamo il comportamento degli altri setup.
        smc_result = {"choch": None, "bos": "BOS_BULLISH", "structure": "BULLISH"}
        with patch("analyzer.detect_bos_choch", return_value=smc_result), \
             patch("analyzer.detect_order_blocks", return_value={}), \
             patch("analyzer.detect_fvg", return_value={"bullish_fvg": {"bottom": 99.0, "top": 100.5}}) as mock_fvg, \
             patch("analyzer.detect_liquidity", return_value={"eqh": 100.0, "eql": 99.5}) as mock_liq, \
             patch("analyzer.detect_premium_discount", return_value="DISCOUNT") as mock_pd:
            result = smc_v3_strategy(self._df(), self._df(), {}, {}, {}, now=inside_hours)
        self.assertEqual(result, {"signal": "NEUTRAL", "setup": None, "score": 0})
        mock_fvg.assert_not_called()
        mock_liq.assert_not_called()
        mock_pd.assert_not_called()


class TestDetectSwingPointsVectorized(unittest.TestCase):
    """PERF (2026-09-06): il loop Python puro riga-per-riga rendeva
    impraticabile processare 1min su 5 anni pieni (1.77M barre). Sostituito
    con un rolling centrato, verificato dare risultati identici (~1300x più
    veloce). Qui si fissa il contratto della funzione, indipendentemente
    dall'implementazione."""

    def _df(self, highs, lows):
        n = len(highs)
        return pd.DataFrame({
            "Open": highs, "High": highs, "Low": lows, "Close": highs,
            "Volume": [100] * n,
        })

    def test_local_peak_is_flagged_swing_high(self):
        # lookback=5: indice 5 e' un picco isolato circondato da valori piu' bassi.
        highs = [100, 100, 100, 100, 100, 110, 100, 100, 100, 100, 100]
        lows  = [99]  * len(highs)
        df = detect_swing_points(self._df(highs, lows), lookback=5)
        self.assertTrue(df["swing_high"].iloc[5])
        self.assertFalse(df["swing_high"].iloc[4])
        self.assertFalse(df["swing_high"].iloc[6])

    def test_local_trough_is_flagged_swing_low(self):
        highs = [101] * 11
        lows  = [100, 100, 100, 100, 100, 90, 100, 100, 100, 100, 100]
        df = detect_swing_points(self._df(highs, lows), lookback=5)
        self.assertTrue(df["swing_low"].iloc[5])
        self.assertFalse(df["swing_low"].iloc[4])
        self.assertFalse(df["swing_low"].iloc[6])

    def test_edges_within_lookback_are_never_flagged(self):
        # Anche se il valore piu' alto in assoluto e' al bordo (indice 0),
        # non puo' avere una finestra completa -> mai marcato.
        highs = [200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        lows  = [99] * len(highs)
        df = detect_swing_points(self._df(highs, lows), lookback=5)
        self.assertFalse(df["swing_high"].iloc[0])

    def test_plateau_ties_all_flagged_not_just_one(self):
        """Se piu' barre nella finestra condividono lo stesso massimo, TUTTE
        risultano swing_high (confronto indipendente per barra, non un
        singolo vincitore) - stesso comportamento del loop originale."""
        highs = [100, 100, 100, 100, 100, 110, 110, 100, 100, 100, 100, 100]
        lows  = [99] * len(highs)
        df = detect_swing_points(self._df(highs, lows), lookback=5)
        self.assertTrue(df["swing_high"].iloc[5])
        self.assertTrue(df["swing_high"].iloc[6])


class TestCalibrateProbabilityForDisplay(unittest.TestCase):
    """calibrate_probability_for_display() - solo per il testo mostrato
    all'utente, misurata su calibration_report() su 17.596 trade reali
    (2026-09-08). Non deve mai essere usata per MIN_PROB o altre decisioni
    di trading - vedi commento sopra estimate_probability() in analyzer.py."""

    def test_below_lowest_anchor_clamps_flat(self):
        self.assertEqual(calibrate_probability_for_display(40), 33)
        self.assertEqual(calibrate_probability_for_display(55), 33)

    def test_above_highest_anchor_clamps_flat(self):
        self.assertEqual(calibrate_probability_for_display(97), 30)
        self.assertEqual(calibrate_probability_for_display(91), 30)

    def test_interpolates_between_anchors(self):
        # 63.6 e' a meta' strada tra i due primi ancoraggi (58.8 -> 68.4);
        # il valore atteso interpola tra 33.0 e 32.8.
        mid = calibrate_probability_for_display(63)
        self.assertIn(mid, (32, 33))

    def test_exact_anchor_returns_its_own_value(self):
        self.assertEqual(calibrate_probability_for_display(68), 33)

    def test_output_stays_close_to_observed_range(self):
        # Su tutto il dominio [40,97] il risultato deve restare nella
        # fascia realmente osservata (30-33%), mai un numero fuori scala.
        for raw in range(40, 98):
            out = calibrate_probability_for_display(raw)
            self.assertTrue(29 <= out <= 34, f"raw={raw} -> {out}")


def _sniper_price_df(n=35, close=100.0, atr=2.0) -> pd.DataFrame:
    rows = [
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
         "atr": atr, "rsi": 50.0, "adx": 20.0}
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


class TestSmcGenericZoneSignal(unittest.TestCase):
    """smc_generic_zone_signal() — la stessa identica logica validata nel
    backtest 5 anni del 2026-09-08 (smc_generic_strategy in
    strategy_blocks_test.py): CHoCH + prezzo dentro la zona dell'Order
    Block (con margine di mezzo ATR). detect_bos_choch/detect_order_blocks
    sono mockate: la loro correttezza è già coperta altrove, qui si
    verifica solo la logica di zona/margine di QUESTA funzione."""

    def test_too_few_rows_returns_neutral(self):
        df = _sniper_price_df(n=10)
        result = smc_generic_zone_signal(df)
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_bullish_choch_with_price_in_ob_zone_returns_buy(self):
        df = _sniper_price_df(close=100.0, atr=2.0)
        with patch("analyzer.detect_bos_choch", return_value={"choch": "CHOCH_BULLISH"}), \
             patch("analyzer.detect_order_blocks", return_value={"bullish_ob": {"low": 98.0, "high": 100.5}}):
            result = smc_generic_zone_signal(df)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["score"], 8)

    def test_bullish_choch_with_price_outside_ob_zone_plus_margin_returns_neutral(self):
        df = _sniper_price_df(close=110.0, atr=2.0)  # ben oltre high(100.5)+atr*0.5
        with patch("analyzer.detect_bos_choch", return_value={"choch": "CHOCH_BULLISH"}), \
             patch("analyzer.detect_order_blocks", return_value={"bullish_ob": {"low": 98.0, "high": 100.5}}):
            result = smc_generic_zone_signal(df)
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_bearish_choch_with_price_in_ob_zone_returns_sell(self):
        df = _sniper_price_df(close=100.0, atr=2.0)
        with patch("analyzer.detect_bos_choch", return_value={"choch": "CHOCH_BEARISH"}), \
             patch("analyzer.detect_order_blocks", return_value={"bearish_ob": {"low": 99.5, "high": 102.0}}):
            result = smc_generic_zone_signal(df)
        self.assertEqual(result["signal"], "SELL")

    def test_no_choch_returns_neutral_even_with_ob_present(self):
        df = _sniper_price_df(close=100.0, atr=2.0)
        with patch("analyzer.detect_bos_choch", return_value={"choch": None}), \
             patch("analyzer.detect_order_blocks", return_value={"bullish_ob": {"low": 98.0, "high": 100.5}}):
            result = smc_generic_zone_signal(df)
        self.assertEqual(result["signal"], "NEUTRAL")


class TestSniperAnalyze(unittest.TestCase):
    """_sniper_analyze(timeframe_key) (2026-09-09, generalizzato per
    ospitare piu' cecchini — vedi SNIPER_CONFIGS) — qui testato sul primo
    cecchino, SMC+Stat Arb su "5min_sniper". La confluenza è stretta: serve
    che SMC e il secondo segnale concordino sulla STESSA direzione,
    esattamente come validato nel backtest (mode="agree" in strategy_
    blocks_test.py)."""

    def _run_sniper(self, smc_signal, second_signal, order_type="BUY", entry=100.0, timeframe_key=SNIPER_TIMEFRAME):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("analyzer.get_data", return_value=_sniper_price_df()))
            stack.enter_context(patch("analyzer.compute_indicators", side_effect=lambda df: df))
            stack.enter_context(patch("analyzer.detect_swing_points", side_effect=lambda df: df))
            stack.enter_context(patch("analyzer.get_dxy_price", return_value=104.0))
            stack.enter_context(patch("analyzer.get_us10y_price", return_value=4.2))
            stack.enter_context(patch("analyzer.detect_bos_choch", return_value={
                "structure": "BULLISH", "choch": None, "bos": None, "last_high": None, "last_low": None,
            }))
            stack.enter_context(patch("analyzer.detect_order_blocks", return_value={}))
            stack.enter_context(patch("analyzer.detect_fvg", return_value={}))
            stack.enter_context(patch("analyzer.detect_premium_discount", return_value="EQUILIBRIUM"))
            stack.enter_context(patch("analyzer.get_support_resistance", return_value={
                "support": 90, "resistance": 110, "s_near": 95, "r_near": 105,
            }))
            stack.enter_context(patch("analyzer.detect_market_regime", return_value={"regime": "NORMAL"}))
            stack.enter_context(patch("analyzer.smc_generic_zone_signal", return_value=smc_signal))
            stack.enter_context(patch("analyzer.statistical_arbitrage_strategy", return_value=second_signal))
            stack.enter_context(patch("analyzer.candlestick_strategy", return_value=second_signal))
            stack.enter_context(patch("analyzer.determine_order_type", return_value=(order_type, entry)))
            stack.enter_context(patch("analyzer.calculate_risk_levels", return_value={
                "sl": 95.0, "tp1": 105.0, "tp2": 110.0, "tp3": 115.0, "be": 100.0,
                "rr1": 1.0, "rr2": 2.0, "rr3": 3.0,
            }))
            return _sniper_analyze(timeframe_key)

    def test_both_agree_buy_produces_a_real_trade(self):
        result = self._run_sniper(
            smc_signal={"signal": "BUY", "score": 8},
            second_signal={"signal": "BUY", "score": 5},
        )
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["timeframe"], SNIPER_TIMEFRAME)
        self.assertEqual(result["prob"], SNIPER_FIXED_PROB)
        self.assertEqual(result["sl"], 95.0)
        self.assertEqual(result["tp2"], 110.0)

    def test_disagreement_returns_neutral_with_no_trade_levels(self):
        result = self._run_sniper(
            smc_signal={"signal": "BUY", "score": 8},
            second_signal={"signal": "SELL", "score": 5},
        )
        self.assertEqual(result["signal"], "NEUTRAL")
        self.assertEqual(result["entry"], 0.0)
        self.assertEqual(result["prob"], 0)

    def test_smc_neutral_returns_neutral_regardless_of_second_signal(self):
        result = self._run_sniper(
            smc_signal={"signal": "NEUTRAL", "score": 0},
            second_signal={"signal": "BUY", "score": 5},
        )
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_prob_display_is_the_real_backtested_win_rate_not_the_generic_calibration(self):
        """prob_display qui NON deve passare per calibrate_probability_for_
        display() (calibrata sull'aggregato a 8 strategie, ~30-33% ovunque)
        — mostrerebbe un numero fuorviante per una strategia con un win
        rate storico reale molto più alto e diverso."""
        result = self._run_sniper(
            smc_signal={"signal": "BUY", "score": 8},
            second_signal={"signal": "BUY", "score": 5},
        )
        self.assertEqual(result["prob_display"], round(SNIPER_CONFIGS[SNIPER_TIMEFRAME]["backtest_wr"]))


class TestSniperConfigsRegistry(unittest.TestCase):
    """I due cecchini H1 (2026-09-09, richiesta esplicita: SMC+Candlestick
    e SMC+Stat Arb operano SEPARATI sullo stesso H1, mai uniti in un solo
    blocco) devono usare il vero secondo segnale della loro config, e mai
    scambiarsi tra loro o coi timeframe reali."""

    def _run(self, timeframe_key, smc_signal, second_signal):
        with contextlib.ExitStack() as stack:
            get_data_mock = stack.enter_context(patch("analyzer.get_data", return_value=_sniper_price_df()))
            stack.enter_context(patch("analyzer.compute_indicators", side_effect=lambda df: df))
            stack.enter_context(patch("analyzer.detect_swing_points", side_effect=lambda df: df))
            stack.enter_context(patch("analyzer.get_dxy_price", return_value=104.0))
            stack.enter_context(patch("analyzer.get_us10y_price", return_value=4.2))
            stack.enter_context(patch("analyzer.detect_bos_choch", return_value={
                "structure": "BULLISH", "choch": None, "bos": None, "last_high": None, "last_low": None,
            }))
            stack.enter_context(patch("analyzer.detect_order_blocks", return_value={}))
            stack.enter_context(patch("analyzer.detect_fvg", return_value={}))
            stack.enter_context(patch("analyzer.detect_premium_discount", return_value="EQUILIBRIUM"))
            stack.enter_context(patch("analyzer.get_support_resistance", return_value={
                "support": 90, "resistance": 110, "s_near": 95, "r_near": 105,
            }))
            stack.enter_context(patch("analyzer.detect_market_regime", return_value={"regime": "NORMAL"}))
            stack.enter_context(patch("analyzer.smc_generic_zone_signal", return_value=smc_signal))
            stack.enter_context(patch("analyzer.statistical_arbitrage_strategy", return_value=second_signal))
            stack.enter_context(patch("analyzer.candlestick_strategy", return_value=second_signal))
            stack.enter_context(patch("analyzer.determine_order_type", return_value=("BUY", 100.0)))
            stack.enter_context(patch("analyzer.calculate_risk_levels", return_value={
                "sl": 95.0, "tp1": 105.0, "tp2": 110.0, "tp3": 115.0, "be": 100.0,
                "rr1": 1.0, "rr2": 2.0, "rr3": 3.0,
            }))
            return _sniper_analyze(timeframe_key), get_data_mock

    def test_h1_candlestick_and_h1_stat_arb_have_independent_real_win_rates(self):
        result_cs, _ = self._run(
            "1h_sniper_candlestick",
            smc_signal={"signal": "BUY", "score": 8}, second_signal={"signal": "BUY", "score": 5},
        )
        result_sa, _ = self._run(
            "1h_sniper_stat_arb",
            smc_signal={"signal": "BUY", "score": 8}, second_signal={"signal": "BUY", "score": 5},
        )
        self.assertEqual(result_cs["timeframe"], "1h_sniper_candlestick")
        self.assertEqual(result_sa["timeframe"], "1h_sniper_stat_arb")
        # Win rate storici diversi (43.5% vs 42.2%, backtest 2026-09-08) -
        # se fossero uguali vorrebbe dire che uno dei due sta leggendo la
        # config dell'altro.
        self.assertNotEqual(result_cs["prob_display"], result_sa["prob_display"])
        self.assertEqual(result_cs["prob_display"], round(SNIPER_CONFIGS["1h_sniper_candlestick"]["backtest_wr"]))
        self.assertEqual(result_sa["prob_display"], round(SNIPER_CONFIGS["1h_sniper_stat_arb"]["backtest_wr"]))

    def test_h1_snipers_fetch_1h_candles_not_5min(self):
        _, get_data_mock = self._run(
            "1h_sniper_stat_arb",
            smc_signal={"signal": "BUY", "score": 8}, second_signal={"signal": "BUY", "score": 5},
        )
        self.assertEqual(get_data_mock.call_args.kwargs.get("interval"), "1h")


class TestFullAnalyzeSniperDispatch(unittest.TestCase):
    def test_sniper_timeframe_dispatches_to_sniper_analyze(self):
        sentinel = {"signal": "BUY", "timeframe": SNIPER_TIMEFRAME}
        with patch("analyzer._sniper_analyze", return_value=sentinel) as mock_sniper:
            result = full_analyze(timeframe_focus=SNIPER_TIMEFRAME)
        mock_sniper.assert_called_once()
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
