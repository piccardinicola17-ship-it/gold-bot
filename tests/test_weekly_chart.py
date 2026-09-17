"""
Test per weekly_chart.py — generalizzato il 2026-09-17 (richiesta
esplicita dell'utente: "la foto per l'analisi giornaliera e l'analisi
giornaliera, come facciamo per quella settimanale") per servire sia la
vista settimanale (4h, invariata) sia quella giornaliera (1h, nuova) con
la STESSA identica logica di zone/grafico — solo lookback e titolo
diversi, per non duplicare ~150 righe quasi identiche in due file.

Non testa compute_smc_context/detect_bos_choch ecc. (già validate
altrove in analyzer.py) — solo compute_weekly_zones e
render_weekly_outlook_chart, che sono la parte toccata da questa sessione.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weekly_chart as wc


def _make_df(n=60, start=4300.0, step=1.0) -> pd.DataFrame:
    """Serie sintetica con un trend leggero crescente — sufficiente per
    compute_weekly_zones/render_weekly_outlook_chart, che lavorano solo
    su OHLC grezzo (nessun indicatore con warmup lungo)."""
    index = pd.date_range("2026-09-01", periods=n, freq="1h")
    closes = start + np.arange(n) * step
    return pd.DataFrame({
        "Open": closes - 0.5, "High": closes + 1.0,
        "Low": closes - 1.0, "Close": closes,
    }, index=index)


class TestComputeWeeklyZones(unittest.TestCase):
    def test_keys_are_generic_period_not_week(self):
        zones = wc.compute_weekly_zones(_make_df(60))
        self.assertIn("period_high", zones)
        self.assertIn("period_low", zones)
        self.assertNotIn("week_high", zones)
        self.assertNotIn("week_low", zones)

    def test_default_lookback_matches_old_42_bar_window(self):
        df = _make_df(60)
        zones_default = wc.compute_weekly_zones(df)
        zones_explicit = wc.compute_weekly_zones(df, lookback_bars=42)
        self.assertEqual(zones_default, zones_explicit)

    def test_custom_lookback_narrows_the_high_low_window(self):
        """Con un trend monotono crescente, una finestra più stretta
        (es. 24 barre per una vista "giornaliera" su candele 1h) vede un
        massimo/minimo più vicini al prezzo attuale di una larga (42)."""
        df = _make_df(60)
        zones_wide   = wc.compute_weekly_zones(df, lookback_bars=42)
        zones_narrow = wc.compute_weekly_zones(df, lookback_bars=5)
        self.assertLess(
            zones_narrow["period_high"] - zones_narrow["period_low"],
            zones_wide["period_high"] - zones_wide["period_low"],
        )

    def test_current_price_is_last_close(self):
        df = _make_df(10)
        zones = wc.compute_weekly_zones(df)
        self.assertEqual(zones["current_price"], round(float(df["Close"].iloc[-1]), 2))


class TestRenderOutlookChart(unittest.TestCase):
    def test_returns_existing_png_with_custom_title(self):
        df = _make_df(60)
        zones = wc.compute_weekly_zones(df, lookback_bars=24)
        path = wc.render_weekly_outlook_chart(df, zones, "BUY", title="GOLD DAILY OUTLOOK — XAU/USD (1H)")
        try:
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.remove(path)

    def test_default_title_used_when_not_specified(self):
        df = _make_df(60)
        zones = wc.compute_weekly_zones(df)
        path = wc.render_weekly_outlook_chart(df, zones, "SELL")
        try:
            self.assertTrue(os.path.exists(path))
        finally:
            os.remove(path)

    def test_works_with_a_narrow_lookback_used_for_daily_view(self):
        """La combinazione lookback_bars=24 (giornaliera) + render non
        deve mai sollevare — stesso identico percorso di codice della
        vista settimanale, solo con zone più strette."""
        df = _make_df(96)
        zones = wc.compute_weekly_zones(df, lookback_bars=24)
        path = wc.render_weekly_outlook_chart(df, zones, "NEUTRAL")
        try:
            self.assertTrue(os.path.exists(path))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
