"""
Test per le funzioni pure di backtest.py — la logica che decide, candela
per candela, l'esito di un trade simulato. In particolare _check_trade_bar,
che ha già un bug reale documentato e corretto in passato (la protezione
break-even scattava a soli $10 di movimento invece che solo dopo un vero
TP1, producendo esiti diversi da quelli che il bot live avrebbe prodotto).
"""

import contextlib
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest
from backtest import (
    ENTRY_COST, EXIT_COST,
    _is_pending, _order_touched, _execution_entry, _execution_exit,
    _check_trade_bar, _r_result, _safe, _make_setup,
    monte_carlo_drawdown, parameter_sensitivity_backtest,
    _stat_arb_for_date, _get_cross_asset_series_and_ma,
    calibration_report, format_calibration_report,
)


def _bar(high: float, low: float) -> dict:
    return {"High": high, "Low": low}


def _buy_trade(entry=100.0, sl=90.0, tp1=110.0, tp2=120.0, tp3=130.0, order_type="BUY",
                activated=True, **overrides) -> dict:
    trade = {
        "signal": "BUY", "order_type": order_type, "raw_entry": entry, "entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "activated": activated,
        "initial_risk": entry - sl,
    }
    trade.update(overrides)
    return trade


def _sell_trade(entry=100.0, sl=110.0, tp1=90.0, tp2=80.0, tp3=70.0, order_type="SELL",
                 activated=True, **overrides) -> dict:
    trade = {
        "signal": "SELL", "order_type": order_type, "raw_entry": entry, "entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "activated": activated,
        "initial_risk": sl - entry,
    }
    trade.update(overrides)
    return trade


class TestIsPending(unittest.TestCase):
    def test_limit_and_stop_are_pending(self):
        self.assertTrue(_is_pending("BUY LIMIT"))
        self.assertTrue(_is_pending("SELL STOP"))

    def test_market_is_not_pending(self):
        self.assertFalse(_is_pending("BUY"))
        self.assertFalse(_is_pending("SELL"))


class TestOrderTouched(unittest.TestCase):
    def test_buy_limit_touched_when_low_reaches_entry(self):
        trade = {"signal": "BUY", "order_type": "BUY LIMIT", "raw_entry": 100.0}
        self.assertTrue(_order_touched(trade, high=105, low=99))
        self.assertFalse(_order_touched(trade, high=105, low=101))

    def test_sell_stop_touched_when_low_reaches_entry(self):
        trade = {"signal": "SELL", "order_type": "SELL STOP", "raw_entry": 100.0}
        self.assertTrue(_order_touched(trade, high=101, low=99))
        self.assertFalse(_order_touched(trade, high=101, low=100.5))

    def test_market_order_always_touched(self):
        trade = {"signal": "BUY", "order_type": "BUY", "raw_entry": 100.0}
        self.assertTrue(_order_touched(trade, high=50, low=1))


class TestExecutionCosts(unittest.TestCase):
    def test_buy_pays_cost_on_entry_and_exit(self):
        self.assertAlmostEqual(_execution_entry("BUY", 100.0), 100.0 + ENTRY_COST)
        self.assertAlmostEqual(_execution_exit("BUY", 100.0), 100.0 - EXIT_COST)

    def test_sell_pays_cost_in_opposite_direction(self):
        self.assertAlmostEqual(_execution_entry("SELL", 100.0), 100.0 - ENTRY_COST)
        self.assertAlmostEqual(_execution_exit("SELL", 100.0), 100.0 + EXIT_COST)


class TestCheckTradeBar(unittest.TestCase):
    def test_sl_checked_before_targets_on_ambiguous_candle(self):
        """Se SL e TP1 sono toccati nella stessa candela, prevale sempre
        l'esito peggiore (SL) — non si inventa una sequenza intrabar favorevole."""
        trade = _buy_trade(entry=100, sl=90, tp1=110)
        result = _check_trade_bar(trade, _bar(high=115, low=85))
        self.assertEqual(result.outcome, "LOSS")

    def test_buy_tp3_directly(self):
        trade = _buy_trade()
        result = _check_trade_bar(trade, _bar(high=135, low=105))
        self.assertEqual(result.outcome, "WIN_TP3")

    def test_buy_tp1_then_reverts_to_be_next_bar(self):
        """TP1 raggiunto in una candela non chiude nulla (solo segna
        tp1_hit); un ritorno all'entry chiude a WIN_BE SOLO dalla candela
        successiva — mai nella stessa candela di TP1."""
        trade = _buy_trade()
        first = _check_trade_bar(trade, _bar(high=112, low=108))  # tocca TP1
        self.assertIsNone(first.outcome)
        self.assertTrue(trade["tp1_hit"])

        # Solo dopo che il chiamante propaga tp1_hit_before_bar (come fa
        # run_backtest tra una candela e l'altra) il ritorno a BE chiude.
        trade["tp1_hit_before_bar"] = True
        second = _check_trade_bar(trade, _bar(high=105, low=99))  # torna sotto l'entry
        self.assertEqual(second.outcome, "WIN_BE")

    def test_buy_tp1_hit_same_bar_as_be_return_does_not_close(self):
        """Bug storico corretto: la protezione BE non deve scattare nella
        STESSA candela in cui TP1 viene raggiunto, solo da quella dopo."""
        trade = _buy_trade()
        result = _check_trade_bar(trade, _bar(high=112, low=99))  # tocca TP1 e torna sotto l'entry, stessa candela
        self.assertIsNone(result.outcome, "non deve chiudere a BE nella stessa candela di TP1")
        self.assertTrue(trade["tp1_hit"])

    def test_sell_direction_mirrors_buy(self):
        trade = _sell_trade()
        result = _check_trade_bar(trade, _bar(high=95, low=65))
        self.assertEqual(result.outcome, "WIN_TP3")

    def test_pending_not_yet_activated_returns_no_outcome(self):
        trade = _buy_trade(order_type="BUY LIMIT", activated=False, entry=100)
        result = _check_trade_bar(trade, _bar(high=200, low=150))  # non tocca mai l'entry
        self.assertIsNone(result.outcome)
        self.assertFalse(trade["activated"])

    def test_pending_activates_and_can_close_same_bar(self):
        trade = _buy_trade(order_type="BUY LIMIT", activated=False, entry=100, sl=90)
        result = _check_trade_bar(trade, _bar(high=100.5, low=85))  # tocca l'entry E lo SL nella stessa barra
        self.assertTrue(result.activated_now)
        self.assertEqual(result.outcome, "LOSS")


class TestRResult(unittest.TestCase):
    def test_win_be_and_no_outcome_are_zero(self):
        trade = {"signal": "BUY", "entry": 100.0, "initial_risk": 10.0}
        self.assertEqual(_r_result(trade, "WIN_BE", 100.0), 0.0)
        self.assertEqual(_r_result(trade, "NEVER_TRIGGERED", None), 0.0)

    def test_buy_win_is_positive_r(self):
        trade = {"signal": "BUY", "entry": 100.0, "initial_risk": 10.0}
        self.assertAlmostEqual(_r_result(trade, "WIN_TP1", 110.0), 1.0)

    def test_sell_win_is_positive_r(self):
        trade = {"signal": "SELL", "entry": 100.0, "initial_risk": 10.0}
        self.assertAlmostEqual(_r_result(trade, "WIN_TP1", 90.0), 1.0)

    def test_loss_is_negative_r(self):
        trade = {"signal": "BUY", "entry": 100.0, "initial_risk": 10.0}
        self.assertAlmostEqual(_r_result(trade, "LOSS", 90.0), -1.0)


class TestSafe(unittest.TestCase):
    def test_strips_markdown_special_chars(self):
        self.assertEqual(_safe("TRENDING_UP *bold* `code` [link]"), "TRENDING UP bold code link")

    def test_plain_text_unchanged(self):
        self.assertEqual(_safe("plain text 123"), "plain text 123")


def _flat_window(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": [4400.0] * rows, "High": [4405.0] * rows, "Low": [4395.0] * rows,
        "Close": [4400.0] * rows, "atr": [5.0] * rows, "rsi": [50.0] * rows,
        "adx": [20.0] * rows,
    })


@contextlib.contextmanager
def _mock_pipeline(regime: str, signal: str, prob: int = 80):
    """Mocka l'intera catena di analyzer chiamata da _make_setup, isolando
    solo i due punti che questi test vogliono controllare: il regime
    (detect_market_regime) e il segnale aggregato (aggregate_strategies).
    Il resto ritorna stub innocui — non è il loro comportamento sotto test."""
    neutral = {"signal": "NEUTRAL", "score": 0}

    def _risk(sig, entry, atr, regime_):
        d = 1 if sig == "BUY" else -1
        return {
            "sl": entry - d * 10, "tp1": entry + d * 10,
            "tp2": entry + d * 20, "tp3": entry + d * 30, "be": entry,
        }

    with patch.object(backtest, "detect_swing_points", lambda w, lookback=5: w), \
         patch.object(backtest, "detect_bos_choch", lambda w: {"structure": "NEUTRAL"}), \
         patch.object(backtest, "detect_order_blocks", lambda w: {}), \
         patch.object(backtest, "detect_fvg", lambda w: {}), \
         patch.object(backtest, "detect_liquidity", lambda w: {}), \
         patch.object(backtest, "detect_premium_discount", lambda w, smc: "EQUILIBRIUM"), \
         patch.object(backtest, "get_support_resistance", lambda w: {}), \
         patch.object(backtest, "detect_market_regime", lambda w: {"regime": regime}), \
         patch.object(backtest, "trend_following_strategy", lambda w: dict(neutral)), \
         patch.object(backtest, "momentum_strategy", lambda w, mtf: dict(neutral)), \
         patch.object(backtest, "candlestick_strategy", lambda w: dict(neutral)), \
         patch.object(backtest, "order_flow_strategy", lambda w: dict(neutral)), \
         patch.object(backtest, "ml_alpha_strategy", lambda w, mtf, smc: dict(neutral)), \
         patch.object(backtest, "aggregate_strategies", lambda strategies, regime_data, timeframe: {
             "signal": signal, "total_score": 5, "buy_count": 1, "sell_count": 0,
         }), \
         patch.object(backtest, "determine_order_type", lambda *a, **k: ("MARKET", None)), \
         patch.object(backtest, "calculate_risk_levels", _risk), \
         patch.object(backtest, "estimate_probability", lambda *a, **k: prob):
        yield


class TestMakeSetupRegimeDirectionBlocks(unittest.TestCase):
    """Regression (audit 2026-09-04): _make_setup() non applicava i filtri
    regime/direzione bloccati che agent_orchestrator.py applica live — ogni
    backtest fin qui includeva trade che il bot live rifiuta categoricamente.
    _BLOCKED_REGIMES_BY_TF/_BLOCKED_REGIME_DIRECTION_BY_TF in backtest.py
    devono restare un mirror esatto di quelli in agent_orchestrator.py."""

    def test_sell_blocked_in_normal_regime_on_1day(self):
        with _mock_pipeline(regime="NORMAL", signal="SELL"):
            setup = _make_setup(_flat_window(), "1day", min_prob=50)
        self.assertIsNone(setup)

    def test_buy_not_blocked_in_normal_regime_on_1day(self):
        with _mock_pipeline(regime="NORMAL", signal="BUY"):
            setup = _make_setup(_flat_window(), "1day", min_prob=50)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["signal"], "BUY")

    def test_sell_not_blocked_in_normal_regime_on_other_tf(self):
        with _mock_pipeline(regime="NORMAL", signal="SELL"):
            setup = _make_setup(_flat_window(), "4h", min_prob=50)
        self.assertIsNotNone(setup)

    def test_whole_regime_blocked_on_1h_ranging(self):
        with _mock_pipeline(regime="RANGING", signal="BUY"):
            setup = _make_setup(_flat_window(), "1h", min_prob=50)
        self.assertIsNone(setup)

    def test_whole_regime_blocked_on_4h_ranging(self):
        with _mock_pipeline(regime="RANGING", signal="SELL"):
            setup = _make_setup(_flat_window(), "4h", min_prob=50)
        self.assertIsNone(setup)

    def test_trending_down_blocked_on_1day(self):
        with _mock_pipeline(regime="TRENDING_DOWN", signal="BUY"):
            setup = _make_setup(_flat_window(), "1day", min_prob=50)
        self.assertIsNone(setup)

    def test_normal_regime_not_blocked_on_1h(self):
        with _mock_pipeline(regime="NORMAL", signal="BUY"):
            setup = _make_setup(_flat_window(), "1h", min_prob=50)
        self.assertIsNotNone(setup)


class TestMonteCarloDrawdown(unittest.TestCase):
    """Fase A (2026-09-04): distribuzione dei drawdown possibili sulla stessa
    sequenza di trade, non solo l'ordine storico osservato."""

    def test_empty_input_returns_zero_sims(self):
        result = monte_carlo_drawdown([], n_sims=100)
        self.assertEqual(result["n_sims"], 0)

    def test_observed_drawdown_matches_manual_calculation(self):
        # equity = [1, 0, -1, 0], picco corrente = [1,1,1,1], dd = [0,-1,-2,-1]
        result = monte_carlo_drawdown([1.0, -1.0, -1.0, 1.0], n_sims=50, seed=1)
        self.assertAlmostEqual(result["observed_max_drawdown_r"], -2.0, places=6)

    def test_deterministic_with_seed(self):
        r1 = monte_carlo_drawdown([1.0, -1.0, 0.5, -0.5, 2.0], n_sims=500, seed=42)
        r2 = monte_carlo_drawdown([1.0, -1.0, 0.5, -0.5, 2.0], n_sims=500, seed=42)
        self.assertEqual(r1, r2)

    def test_percentiles_are_ordered(self):
        result = monte_carlo_drawdown([1.0, -1.0, -1.0, 2.0, -0.5, 1.5], n_sims=1000, seed=7)
        self.assertLessEqual(result["worst_simulated_dd_r"], result["p5_worst_case_dd_r"])
        self.assertLessEqual(result["p5_worst_case_dd_r"], result["p25_dd_r"])
        self.assertLessEqual(result["p25_dd_r"], result["median_dd_r"])
        self.assertLessEqual(result["median_dd_r"], result["p75_dd_r"])
        self.assertLessEqual(result["p75_dd_r"], result["p95_best_case_dd_r"])

    def test_all_positive_results_have_zero_drawdown(self):
        result = monte_carlo_drawdown([1.0, 2.0, 1.5], n_sims=200, seed=3)
        self.assertEqual(result["median_dd_r"], 0.0)


class TestParameterSensitivityBacktest(unittest.TestCase):
    """Fase A (2026-09-04): piccole variazioni di min_prob non dovrebbero
    cambiare drasticamente il risultato — altrimenti è overfitting sulla
    soglia esatta in uso."""

    def _stats_for(self, win_rate_by_prob: dict):
        def _fake_run_backtest(interval, bars, min_prob):
            return {
                "concluded": 20, "win_rate": win_rate_by_prob.get(min_prob, 40),
                "profit_factor": 1.2, "total_r": 3.0,
            }
        return _fake_run_backtest

    def test_stable_when_win_rate_barely_moves(self):
        fake = self._stats_for({45: 42, 50: 43, 55: 45, 60: 44, 65: 46})
        with patch("backtest.run_backtest", side_effect=fake):
            result = parameter_sensitivity_backtest(base_min_prob=55, deltas=(-10, -5, 0, 5, 10))
        self.assertTrue(result["stable"])
        self.assertLessEqual(result["win_rate_range"], 15)

    def test_unstable_when_win_rate_swings(self):
        fake = self._stats_for({45: 20, 50: 25, 55: 45, 60: 70, 65: 75})
        with patch("backtest.run_backtest", side_effect=fake):
            result = parameter_sensitivity_backtest(base_min_prob=55, deltas=(-10, -5, 0, 5, 10))
        self.assertFalse(result["stable"])
        self.assertGreater(result["win_rate_range"], 15)

    def test_excludes_thresholds_with_too_few_trades(self):
        def _fake_run_backtest(interval, bars, min_prob):
            if min_prob == 45:
                return {"concluded": 1, "win_rate": 100, "profit_factor": 0, "total_r": 0}
            return {"concluded": 20, "win_rate": 50, "profit_factor": 1.1, "total_r": 2.0}
        with patch("backtest.run_backtest", side_effect=_fake_run_backtest):
            result = parameter_sensitivity_backtest(base_min_prob=55, deltas=(-10, -5, 0, 5, 10))
        self.assertEqual(result["win_rate_range"], 0.0)
        self.assertTrue(result["stable"])

    def test_clamps_min_prob_to_valid_range(self):
        with patch("backtest.run_backtest", return_value={
            "concluded": 20, "win_rate": 50, "profit_factor": 1.0, "total_r": 1.0,
        }) as mock_run:
            parameter_sensitivity_backtest(base_min_prob=5, deltas=(-10, 0))
        called_probs = [c.kwargs["min_prob"] for c in mock_run.call_args_list]
        self.assertIn(1, called_probs)  # 5 + (-10) clampato a 1, non negativo


class TestStatArbForDate(unittest.TestCase):
    """stat_arb nel backtest (2026-09-05): DXY/TLT storici gratuiti
    (yfinance) al posto del NEUTRO fisso - lookup causale (.asof), mai il
    futuro."""

    def _fake_series(self, dates, values):
        return pd.Series(values, index=pd.DatetimeIndex(dates))

    def test_causal_lookup_uses_last_known_value_not_future(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        dxy = self._fake_series(dates, [100.0, 100.0, 103.0, 100.0, 100.0])
        dxy_ma = self._fake_series(dates, [100.0] * 5)
        tlt = self._fake_series(dates, [90.0] * 5)
        tlt_ma = self._fake_series(dates, [90.0] * 5)
        with patch("backtest._get_cross_asset_series_and_ma", side_effect=[
            (dxy, dxy_ma), (tlt, tlt_ma),
        ]):
            # Query su una data TRA il giorno 3 (DXY=103, SELL) e il 4esimo
            # (torna a 100): .asof() deve prendere l'ultimo valore noto
            # PRIMA della query, non guardare avanti.
            result = _stat_arb_for_date(pd.Timestamp("2026-01-03 12:00:00"))
        self.assertEqual(result["signal"], "SELL")

    def test_empty_history_returns_neutral(self):
        with patch("backtest._get_cross_asset_series_and_ma", return_value=(pd.Series(dtype=float), pd.Series(dtype=float))):
            result = _stat_arb_for_date(pd.Timestamp("2026-01-03"))
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_date_before_any_history_returns_neutral(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        dxy = self._fake_series(dates, [100.0] * 5)
        dxy_ma = self._fake_series(dates, [100.0] * 5)
        with patch("backtest._get_cross_asset_series_and_ma", side_effect=[
            (dxy, dxy_ma), (dxy, dxy_ma),
        ]):
            result = _stat_arb_for_date(pd.Timestamp("2025-01-01"))
        self.assertEqual(result["signal"], "NEUTRAL")


class TestGetCrossAssetSeriesAndMa(unittest.TestCase):
    def setUp(self):
        import backtest as bt
        bt._cross_asset_ma_cache.clear()

    def test_caches_after_first_fetch(self):
        import backtest as bt
        fake_df = pd.DataFrame(
            {"Close": [100.0 + i for i in range(15)]},
            index=pd.date_range("2026-01-01", periods=15, freq="D"),
        )
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = fake_df
            s1, ma1 = _get_cross_asset_series_and_ma("DX-Y.NYB")
            s2, ma2 = _get_cross_asset_series_and_ma("DX-Y.NYB")
        self.assertEqual(mock_ticker.call_count, 1)  # secondo giro dalla cache
        self.assertEqual(len(s1), 15)
        self.assertTrue(s1.equals(s2))

    def test_fetch_failure_returns_empty_series_not_raise(self):
        import backtest as bt
        bt._cross_asset_ma_cache.pop("BROKEN", None)
        with patch("yfinance.Ticker", side_effect=Exception("rete giu'")):
            s, ma = _get_cross_asset_series_and_ma("BROKEN")
        self.assertTrue(s.empty)
        self.assertTrue(ma.empty)


class TestCalibrationReport(unittest.TestCase):
    """calibration_report() (2026-09-05): estimate_probability() e'
    dichiaratamente uno score euristico, non una probabilita' statistica
    reale - questo verifica solo che ordini correttamente (fasce di prob
    piu' alte vincono piu' spesso), non che il numero esatto sia calibrato."""

    def test_empty_input(self):
        result = calibration_report([])
        self.assertEqual(result["n_total"], 0)

    def test_buckets_win_rate_computed_correctly(self):
        pairs = [(60, True)] * 15 + [(60, False)] * 5  # fascia 55-64%, WR 75%
        result = calibration_report(pairs)
        bucket = next(b for b in result["buckets"] if b["label"] == "55-64%")
        self.assertEqual(bucket["n"], 20)
        self.assertEqual(bucket["actual_win_rate"], 75.0)
        self.assertTrue(bucket["reliable"])

    def test_bucket_below_min_n_flagged_unreliable(self):
        pairs = [(60, True)] * 5
        result = calibration_report(pairs)
        bucket = next(b for b in result["buckets"] if b["label"] == "55-64%")
        self.assertFalse(bucket["reliable"])

    def test_monotonic_when_higher_prob_wins_more(self):
        pairs = (
            [(55, True)] * 12 + [(55, False)] * 13 +   # 55-64%: WR 48%
            [(80, True)] * 18 + [(80, False)] * 3       # 75-84%: WR 86%
        )
        result = calibration_report(pairs)
        self.assertTrue(result["monotonic"])

    def test_not_monotonic_when_higher_prob_wins_less(self):
        pairs = (
            [(80, True)] * 5 + [(80, False)] * 20 +    # 75-84%: WR 20%
            [(55, True)] * 20 + [(55, False)] * 5       # 55-64%: WR 80%
        )
        result = calibration_report(pairs)
        self.assertFalse(result["monotonic"])

    def test_insufficient_reliable_buckets_returns_none(self):
        pairs = [(60, True)] * 3
        result = calibration_report(pairs)
        self.assertIsNone(result["monotonic"])

    def test_format_does_not_raise_on_empty(self):
        msg = format_calibration_report(calibration_report([]), "1day")
        self.assertIn("Nessun trade concluso", msg)

    def test_format_includes_bucket_rows(self):
        pairs = [(60, True)] * 15 + [(60, False)] * 10
        msg = format_calibration_report(calibration_report(pairs), "1day")
        self.assertIn("55-64%", msg)
        self.assertIn("60.0%", msg)


if __name__ == "__main__":
    unittest.main()
