"""
Property-based test (Hypothesis) sui moduli numerici puri — Fase A del
2026-09-04. A differenza dei test esistenti (casi specifici scelti a
mano), qui Hypothesis genera migliaia di input casuali e verifica
un'invariante che deve valere SEMPRE, non solo per gli esempi che abbiamo
pensato di scrivere.

Unico file della suite con una dipendenza esterna (hypothesis, vedi
requirements-dev.txt) — deliberatamente separato da requirements.txt
perché il bot live non ne ha bisogno.
"""

import os
import sys
import unittest

from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_manager as tm
from backtest import _r_result, monte_carlo_drawdown

_price = st.floats(min_value=100, max_value=10000, allow_nan=False, allow_infinity=False)
_small = st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False)


class TestCalculateTradePipsProperties(unittest.TestCase):
    @given(entry=_price, exit_price=_price)
    @settings(max_examples=200)
    def test_buy_and_sell_are_mirror_images(self, entry, exit_price):
        buy = tm.calculate_trade_pips("BUY", entry, exit_price)
        sell = tm.calculate_trade_pips("SELL", entry, exit_price)
        self.assertAlmostEqual(buy, -sell, places=6)

    @given(entry=_price)
    @settings(max_examples=100)
    def test_zero_pips_when_entry_equals_exit(self, entry):
        self.assertEqual(tm.calculate_trade_pips("BUY", entry, entry), 0.0)
        self.assertEqual(tm.calculate_trade_pips("SELL", entry, entry), 0.0)

    @given(entry=_price, delta=st.floats(min_value=0.1, max_value=500, allow_nan=False))
    @settings(max_examples=200)
    def test_buy_profits_when_price_rises(self, entry, delta):
        self.assertGreater(tm.calculate_trade_pips("BUY", entry, entry + delta), 0)
        self.assertLess(tm.calculate_trade_pips("SELL", entry, entry + delta), 0)


class TestRResultProperties(unittest.TestCase):
    def _trade(self, signal="BUY", entry=100.0, initial_risk=10.0):
        return {"signal": signal, "entry": entry, "initial_risk": initial_risk}

    @given(entry=_price, initial_risk=st.floats(min_value=0.5, max_value=200, allow_nan=False),
           exit_price=_price)
    @settings(max_examples=200)
    def test_win_be_and_no_outcome_always_zero_regardless_of_price(self, entry, initial_risk, exit_price):
        trade = self._trade(entry=entry, initial_risk=initial_risk)
        for outcome in ("WIN_BE", "NEVER_TRIGGERED", "NO_OUTCOME"):
            self.assertEqual(_r_result(trade, outcome, exit_price), 0.0)

    @given(entry=_price, initial_risk=st.floats(min_value=0.5, max_value=200, allow_nan=False))
    @settings(max_examples=100)
    def test_none_exit_price_always_zero(self, entry, initial_risk):
        trade = self._trade(entry=entry, initial_risk=initial_risk)
        self.assertEqual(_r_result(trade, "LOSS", None), 0.0)

    @given(entry=_price, initial_risk=st.floats(min_value=0.5, max_value=200, allow_nan=False),
           k=st.floats(min_value=-5, max_value=5, allow_nan=False))
    @settings(max_examples=300)
    def test_buy_r_result_scales_linearly_with_risk_multiples(self, entry, initial_risk, k):
        # Se il prezzo si muove esattamente k * initial_risk a favore, il
        # risultato in R deve essere k (a meno di arrotondamento a 3 decimali).
        exit_price = entry + k * initial_risk
        trade = self._trade(signal="BUY", entry=entry, initial_risk=initial_risk)
        self.assertAlmostEqual(_r_result(trade, "WIN_TP1", exit_price), k, places=2)

    @given(entry=_price, initial_risk=st.floats(min_value=0.5, max_value=200, allow_nan=False),
           k=st.floats(min_value=-5, max_value=5, allow_nan=False))
    @settings(max_examples=300)
    def test_sell_r_result_is_negated_buy(self, entry, initial_risk, k):
        exit_price = entry + k * initial_risk
        buy_trade = self._trade(signal="BUY", entry=entry, initial_risk=initial_risk)
        sell_trade = self._trade(signal="SELL", entry=entry, initial_risk=initial_risk)
        buy_r = _r_result(buy_trade, "WIN_TP1", exit_price)
        sell_r = _r_result(sell_trade, "WIN_TP1", exit_price)
        self.assertAlmostEqual(buy_r, -sell_r, places=6)


class TestMonteCarloDrawdownProperties(unittest.TestCase):
    @given(r_results=st.lists(_small, min_size=1, max_size=60))
    @settings(max_examples=100)
    def test_observed_drawdown_never_positive(self, r_results):
        result = monte_carlo_drawdown(r_results, n_sims=50, seed=0)
        self.assertLessEqual(result["observed_max_drawdown_r"], 1e-9)

    @given(r_results=st.lists(_small, min_size=2, max_size=60))
    @settings(max_examples=50)
    def test_percentiles_always_ordered(self, r_results):
        result = monte_carlo_drawdown(r_results, n_sims=200, seed=0)
        self.assertLessEqual(result["worst_simulated_dd_r"], result["p5_worst_case_dd_r"] + 1e-9)
        self.assertLessEqual(result["p5_worst_case_dd_r"], result["p25_dd_r"] + 1e-9)
        self.assertLessEqual(result["p25_dd_r"], result["median_dd_r"] + 1e-9)
        self.assertLessEqual(result["median_dd_r"], result["p75_dd_r"] + 1e-9)
        self.assertLessEqual(result["p75_dd_r"], result["p95_best_case_dd_r"] + 1e-9)

    @given(r_results=st.lists(st.floats(min_value=0.01, max_value=500, allow_nan=False), min_size=1, max_size=30))
    @settings(max_examples=50)
    def test_all_positive_results_never_drawdown(self, r_results):
        result = monte_carlo_drawdown(r_results, n_sims=50, seed=0)
        self.assertEqual(result["observed_max_drawdown_r"], 0.0)
        self.assertEqual(result["median_dd_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
