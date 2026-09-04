"""
Test per analyzer._detect_market_regime_fallback — il classificatore
(ADX/Bollinger-width/ROC) che guida davvero le decisioni di trading live,
diverso da quello di regime_detector.py usato solo dal comando /regime
(vedi la nota aggiunta in regime_detector.format_regime_message il
2026-09-03). Verifica i confini delle 5 soglie che decidono il regime.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import _detect_market_regime_fallback


def _frame(adx=15.0, bb_width=0.02, atr=1.0, avg_atr=1.0, ema20=100.0, ema50=100.0, roc=0.0,
           n_history=20) -> pd.DataFrame:
    """Costruisce un DataFrame minimo con `n_history` righe di ATR costante
    (=avg_atr) e una riga finale con gli indicatori passati — riproduce
    esattamente le colonne lette da _detect_market_regime_fallback."""
    rows = [{"adx": adx, "bb_width": bb_width, "atr": avg_atr,
              "ema20": ema20, "ema50": ema50, "roc": roc} for _ in range(n_history)]
    rows.append({"adx": adx, "bb_width": bb_width, "atr": atr,
                  "ema20": ema20, "ema50": ema50, "roc": roc})
    return pd.DataFrame(rows)


class TestRegimeFallback(unittest.TestCase):
    def test_trending_up(self):
        df = _frame(adx=30, ema20=105, ema50=100, roc=0.5)
        self.assertEqual(_detect_market_regime_fallback(df)["regime"], "TRENDING_UP")

    def test_trending_down(self):
        df = _frame(adx=30, ema20=95, ema50=100, roc=-0.5)
        self.assertEqual(_detect_market_regime_fallback(df)["regime"], "TRENDING_DOWN")

    def test_adx_high_but_ema_flat_is_not_trending(self):
        """ADX alto da solo non basta: senza un ordine chiaro EMA20/EMA50 e
        un ROC coerente, non deve essere classificato come trend."""
        df = _frame(adx=30, ema20=100, ema50=100, roc=0.0, bb_width=0.02)
        self.assertNotIn(_detect_market_regime_fallback(df)["regime"], ("TRENDING_UP", "TRENDING_DOWN"))

    def test_ranging(self):
        df = _frame(adx=10, bb_width=0.005)
        self.assertEqual(_detect_market_regime_fallback(df)["regime"], "RANGING")

    def test_volatile(self):
        # atr molto più alto della media recente, ma non qualifica per trend/ranging
        df = _frame(adx=15, bb_width=0.02, atr=5.0, avg_atr=1.0)
        self.assertEqual(_detect_market_regime_fallback(df)["regime"], "VOLATILE")

    def test_normal_fallback(self):
        df = _frame(adx=20, bb_width=0.02, atr=1.0, avg_atr=1.0, ema20=100, ema50=100, roc=0.0)
        self.assertEqual(_detect_market_regime_fallback(df)["regime"], "NORMAL")

    def test_nan_indicators_treated_as_zero_not_crash(self):
        df = _frame()
        df.loc[df.index[-1], "adx"] = float("nan")
        df.loc[df.index[-1], "bb_width"] = float("nan")
        df.loc[df.index[-1], "ema20"] = float("nan")
        result = _detect_market_regime_fallback(df)  # non deve sollevare
        self.assertIn(result["regime"], ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "NORMAL"))


if __name__ == "__main__":
    unittest.main()
