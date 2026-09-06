"""
Test per macro_predictor.py — prima volta testato (nessun test esisteva).

Copre il fix del 2026-09-06: FRED_SERIES distingue ora "mom_pct" (il
valore è una variazione % mese su mese, es. Core CPI m/m — la serie FRED
sottostante è un indice-livello da trasformare) da "level" (il valore è
già un conteggio grezzo comparabile direttamente al forecast, es.
Unemployment Claims — trattarlo come mom_pct calcolerebbe una sorpresa
completamente diversa da quella su cui il modello è stato allenato).
"""

import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import macro_predictor as mp


def _series(values: dict) -> pd.Series:
    s = pd.Series(values)
    s.index = pd.to_datetime(list(values.keys()))
    return s.sort_index()


class TestNewReleaseLevel(unittest.TestCase):
    """_new_release_level: nessuna trasformazione, solo l'ultimo valore."""

    def test_new_value_detected(self):
        s = _series({"2026-08-22": 204000.0, "2026-08-29": 206000.0})
        with mock.patch("macro_predictor._fetch_fred_series", return_value=s):
            result = mp._new_release_level("ICSA", last_seen={})
        self.assertIsNotNone(result)
        date_iso, value = result
        self.assertEqual(value, 206000.0)

    def test_already_seen_returns_none(self):
        s = _series({"2026-08-22": 204000.0, "2026-08-29": 206000.0})
        with mock.patch("macro_predictor._fetch_fred_series", return_value=s):
            result = mp._new_release_level("ICSA", last_seen={"ICSA": "2026-08-29"})
        self.assertIsNone(result)

    def test_empty_series_returns_none(self):
        with mock.patch("macro_predictor._fetch_fred_series", return_value=pd.Series(dtype=float)):
            result = mp._new_release_level("ICSA", last_seen={})
        self.assertIsNone(result)


class TestPredictReactionLevelSeries(unittest.TestCase):
    """predict_reaction per una serie 'level' (Unemployment Claims) — la
    sorpresa deve essere (livello attuale - livello forecast), MAI una
    variazione percentuale."""

    def setUp(self):
        self.model = {
            "event_name": "Unemployment Claims", "horizon": "reaction_1m",
            "slope": 0.001, "intercept": 0.0,
            "surprise_mean": 0.0, "surprise_std": 10000.0,
            "surprise_zscore_clip": 4.0, "n": 130,
        }

    def test_surprise_is_raw_level_difference_not_percent_change(self):
        s = _series({"2026-08-22": 204000.0, "2026-08-29": 219000.0})
        with mock.patch("macro_predictor._load_model", return_value=self.model), \
             mock.patch("macro_predictor._fetch_fred_series", return_value=s), \
             mock.patch("macro_predictor.load_fred_last_seen", return_value={}), \
             mock.patch("macro_predictor.save_fred_last_seen"):
            pred = mp.predict_reaction("Unemployment Claims", "214K")
        self.assertIsNotNone(pred)
        # actual 219000 (ultimo valore FRED) - forecast 214000 (parsed da "214K") = 5000
        self.assertAlmostEqual(pred["surprise_raw"], 5000.0, places=1)
        self.assertEqual(pred["value_type"], "level")
        self.assertEqual(pred["actual_value"], 219000.0)
        self.assertEqual(pred["forecast_value"], 214000.0)

    def test_non_numeric_forecast_returns_none_without_marking_progress_lost(self):
        s = _series({"2026-08-29": 219000.0})
        with mock.patch("macro_predictor._load_model", return_value=self.model), \
             mock.patch("macro_predictor._fetch_fred_series", return_value=s), \
             mock.patch("macro_predictor.load_fred_last_seen", return_value={}), \
             mock.patch("macro_predictor.save_fred_last_seen") as mock_save:
            pred = mp.predict_reaction("Unemployment Claims", "N/A")
        self.assertIsNone(pred)
        # il rilascio va comunque segnato come visto, anche se la previsione
        # non si completa per un altro motivo (stesso principio di
        # Core CPI m/m) — altrimenti si ritenterebbe lo stesso dato per sempre.
        mock_save.assert_called_once()


class TestPredictReactionMomPctSeriesUnaffected(unittest.TestCase):
    """Regressione: Core CPI m/m (value_type='mom_pct') deve comportarsi
    esattamente come prima del refactor per supportare 'level'."""

    def setUp(self):
        self.model = {
            "event_name": "Core CPI m/m", "horizon": "reaction_30m",
            "slope": -2.41, "intercept": 0.79,
            "surprise_mean": 0.0, "surprise_std": 0.13,
            "surprise_zscore_clip": 4.0, "n": 83,
        }

    def test_surprise_is_percent_change_not_raw_level(self):
        s = _series({"2026-06-01": 300.0, "2026-07-01": 300.9})  # +0.3% m/m
        with mock.patch("macro_predictor._load_model", return_value=self.model), \
             mock.patch("macro_predictor._fetch_fred_series", return_value=s), \
             mock.patch("macro_predictor.load_fred_last_seen", return_value={}), \
             mock.patch("macro_predictor.save_fred_last_seen"):
            pred = mp.predict_reaction("Core CPI m/m", "0.2")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["value_type"], "mom_pct")
        self.assertAlmostEqual(pred["actual_value"], 0.3, places=1)
        self.assertAlmostEqual(pred["surprise_raw"], 0.1, places=1)


class TestFormatPrediction(unittest.TestCase):
    def test_level_type_formats_as_raw_count_not_percent(self):
        pred = {
            "value_type": "level", "actual_value": 219000.0, "forecast_value": 214000.0,
            "surprise_zscore": 0.5, "predicted_reaction_usd": -1.2,
            "horizon": "reaction_1m", "n_historical": 130,
        }
        text = mp.format_prediction(pred)
        self.assertIn("219,000", text)
        self.assertIn("214,000", text)
        self.assertNotIn("219000.00%", text)

    def test_mom_pct_type_formats_as_percent(self):
        pred = {
            "value_type": "mom_pct", "actual_value": 0.3, "forecast_value": 0.2,
            "surprise_zscore": 0.8, "predicted_reaction_usd": -0.9,
            "horizon": "reaction_30m", "n_historical": 83,
        }
        text = mp.format_prediction(pred)
        self.assertIn("+0.30%", text)
        self.assertIn("+0.20%", text)


if __name__ == "__main__":
    unittest.main()
