"""
Test per analyzer._detect_market_regime_fallback — il classificatore
(ADX/Bollinger-width/ROC) che guida davvero le decisioni di trading live.
Fino al 2026-09-05 esisteva anche regime_detector.py, un secondo
classificatore con soglie diverse usato solo dal comando /regime (poteva
mostrare un regime diverso da quello su cui il bot decideva davvero nello
stesso istante) — rimosso, /regime ora usa questa stessa funzione. Verifica
i confini delle 5 soglie che decidono il regime.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import _detect_market_regime_fallback, format_live_regime_message


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


class TestFormatLiveRegimeMessage(unittest.TestCase):
    """/regime (2026-09-05) usa lo stesso format_live_regime_message() per
    presentare l'output di detect_market_regime() — non più un secondo
    classificatore separato (regime_detector.py, rimosso)."""

    def test_known_regime_shows_mapped_label(self):
        msg = format_live_regime_message({"regime": "TRENDING_UP", "adx": 30.5, "atr": 4.2})
        self.assertIn("Trending Up", msg)
        self.assertIn("30.5", msg)
        self.assertIn("4.2", msg)

    def test_unknown_regime_falls_back_to_raw_value(self):
        msg = format_live_regime_message({"regime": "SOMETHING_NEW", "adx": 1, "atr": 1})
        self.assertIn("SOMETHING_NEW", msg)

    def test_missing_fields_do_not_crash(self):
        msg = format_live_regime_message({})
        self.assertIn("UNKNOWN", msg)


if __name__ == "__main__":
    unittest.main()
