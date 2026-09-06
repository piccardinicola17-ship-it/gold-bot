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

import os
import sys
import unittest
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer
from analyzer import candlestick_strategy, ml_alpha_strategy, _stat_arb_score_from_means, statistical_arbitrage_strategy, smc_v3_strategy
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


if __name__ == "__main__":
    unittest.main()
