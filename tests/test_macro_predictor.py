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


class TestNewReleaseMomDiff(unittest.TestCase):
    """_new_release_mom_diff (2026-09-08, ADP Non-Farm Employment Change):
    differenza ASSOLUTA dal mese precedente, arrotondata al migliaio come
    il comunicato ("155K") — mai una percentuale, mai il livello grezzo."""

    def test_computes_absolute_thousands_difference(self):
        s = _series({"2026-07-01": 132650000.0, "2026-08-01": 132803000.0})
        with mock.patch("macro_predictor._fetch_fred_series", return_value=s):
            result = mp._new_release_mom_diff("ADPMNUSNERSA", last_seen={})
        self.assertIsNotNone(result)
        _, value = result
        self.assertEqual(value, 153000.0)

    def test_already_seen_returns_none(self):
        s = _series({"2026-07-01": 132650000.0, "2026-08-01": 132803000.0})
        with mock.patch("macro_predictor._fetch_fred_series", return_value=s):
            result = mp._new_release_mom_diff("ADPMNUSNERSA", last_seen={"ADPMNUSNERSA": "2026-08-01"})
        self.assertIsNone(result)


class TestPredictReactionMomDiffSeries(unittest.TestCase):
    def setUp(self):
        self.model = {
            "event_name": "ADP Non-Farm Employment Change", "horizon": "reaction_1m",
            "slope": -1.77, "intercept": -0.09,
            "surprise_mean": 32880.0, "surprise_std": 473367.0,
            "surprise_zscore_clip": 4.0, "n": 184,
        }

    def test_surprise_is_thousands_not_percent(self):
        s = _series({"2026-07-01": 132650000.0, "2026-08-01": 132803000.0})
        with mock.patch("macro_predictor._load_model", return_value=self.model), \
             mock.patch("macro_predictor._fetch_fred_series", return_value=s), \
             mock.patch("macro_predictor.load_fred_last_seen", return_value={}), \
             mock.patch("macro_predictor.save_fred_last_seen"):
            pred = mp.predict_reaction("ADP Non-Farm Employment Change", "118K")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["value_type"], "mom_diff")
        self.assertEqual(pred["source_tier"], "fred")
        self.assertEqual(pred["actual_value"], 153000.0)
        self.assertEqual(pred["forecast_value"], 118000.0)
        self.assertAlmostEqual(pred["surprise_raw"], 35000.0, places=1)


def _rss_item(title: str, summary: str = "", pub_date: str = "Tue, 01 Sep 2026 14:00:00 GMT", link: str = "https://example.com/x") -> dict:
    return {"title": title, "summary": summary, "pub_date": pub_date, "link": link}


class TestNewReleaseRss(unittest.TestCase):
    """_new_release_rss (2026-09-08): ISM Manufacturing/Services PMI e CB
    Consumer Confidence non hanno una serie FRED nativa (verificato con
    ricerca diretta) - lette dai comunicati PR Newswire via RSS. Titoli
    presi verbatim da comunicati reali osservati su 9 mesi consecutivi
    2026, non inventati."""

    def test_ism_manufacturing_extracted_from_title(self):
        items = [_rss_item("Manufacturing PMI® at 54.6%; August 2026 ISM® Manufacturing PMI® Report")]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"], last_seen={})
        self.assertIsNotNone(result)
        _, value = result
        self.assertEqual(value, 54.6)

    def test_ism_services_extracted_from_title(self):
        items = [_rss_item("Services PMI® at 55.4%; August 2026 ISM® Services PMI® Report")]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("ISM Services PMI", mp.RSS_SERIES["ISM Services PMI"], last_seen={})
        self.assertIsNotNone(result)
        _, value = result
        self.assertEqual(value, 55.4)

    def test_manufacturing_pattern_does_not_match_services_title(self):
        """Regressione: i due pattern non devono incrociarsi - un titolo
        Services non deve mai far scattare il modello Manufacturing."""
        items = [_rss_item("Services PMI® at 55.4%; August 2026 ISM® Services PMI® Report")]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"], last_seen={})
        self.assertIsNone(result)

    def test_unrelated_press_release_returns_none(self):
        items = [_rss_item("ACME Corp Announces New Manufacturing Facility in Ohio")]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"], last_seen={})
        self.assertIsNone(result)

    def test_cb_consumer_confidence_extracted_from_body_not_title(self):
        """Il titolo reale ('US Consumer Confidence Edged Down Slightly in
        August') non contiene il numero - deve leggerlo dal riassunto."""
        items = [_rss_item(
            title="US Consumer Confidence Edged Down Slightly in August",
            summary="The Conference Board Consumer Confidence Index® decreased by 0.8 points to 89.4 (1985=100) in August, down from 90.2 in July.",
        )]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("CB Consumer Confidence", mp.RSS_SERIES["CB Consumer Confidence"], last_seen={})
        self.assertIsNotNone(result)
        _, value = result
        self.assertEqual(value, 89.4)

    def test_cb_consumer_confidence_short_summary_returns_none_not_a_guess(self):
        """Se il riassunto RSS e' troppo corto per contenere la frase col
        numero, mai indovinare - stesso principio di un ritardo FRED."""
        items = [_rss_item(
            title="US Consumer Confidence Edged Down Slightly in August",
            summary="Consumer confidence moderated slightly in August for a second consecutive month.",
        )]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("CB Consumer Confidence", mp.RSS_SERIES["CB Consumer Confidence"], last_seen={})
        self.assertIsNone(result)

    def test_already_seen_pub_date_returns_none(self):
        items = [_rss_item(
            "Manufacturing PMI® at 54.6%; August 2026 ISM® Manufacturing PMI® Report",
            pub_date="Tue, 01 Sep 2026 14:00:00 GMT",
        )]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss(
                "ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"],
                last_seen={"ISM Manufacturing PMI": "2026-09-01T14:00:00+00:00"},
            )
        self.assertIsNone(result)

    def test_picks_the_most_recent_of_multiple_matches(self):
        items = [
            _rss_item("Manufacturing PMI® at 52.4%; July 2026 ISM® Manufacturing PMI® Report", pub_date="Tue, 01 Jul 2026 14:00:00 GMT"),
            _rss_item("Manufacturing PMI® at 54.6%; August 2026 ISM® Manufacturing PMI® Report", pub_date="Tue, 01 Sep 2026 14:00:00 GMT"),
        ]
        with mock.patch("macro_predictor._fetch_rss", return_value=items):
            result = mp._new_release_rss("ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"], last_seen={})
        self.assertIsNotNone(result)
        _, value = result
        self.assertEqual(value, 54.6)

    def test_feed_unreachable_does_not_raise(self):
        with mock.patch("macro_predictor._fetch_rss", side_effect=Exception("timeout")):
            result = mp._new_release_rss("ISM Manufacturing PMI", mp.RSS_SERIES["ISM Manufacturing PMI"], last_seen={})
        self.assertIsNone(result)


class TestPredictReactionRssSeries(unittest.TestCase):
    def setUp(self):
        self.model = {
            "event_name": "ISM Manufacturing PMI", "horizon": "reaction_30m",
            "slope": -1.59, "intercept": -0.05,
            "surprise_mean": 0.15, "surprise_std": 1.71,
            "surprise_zscore_clip": 4.0, "n": 196,
        }

    def test_end_to_end_via_rss_source(self):
        items = [_rss_item("Manufacturing PMI® at 54.6%; August 2026 ISM® Manufacturing PMI® Report")]
        with mock.patch("macro_predictor._load_model", return_value=self.model), \
             mock.patch("macro_predictor._fetch_rss", return_value=items), \
             mock.patch("macro_predictor.load_rss_macro_last_seen", return_value={}), \
             mock.patch("macro_predictor.save_rss_macro_last_seen"):
            pred = mp.predict_reaction("ISM Manufacturing PMI", "55.2")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["value_type"], "index")
        self.assertEqual(pred["source_tier"], "rss")
        self.assertEqual(pred["actual_value"], 54.6)
        self.assertAlmostEqual(pred["surprise_raw"], -0.6, places=1)


class TestFormatPredictionNewTypes(unittest.TestCase):
    def test_mom_diff_type_formats_with_sign_and_thousands_separator(self):
        pred = {
            "value_type": "mom_diff", "source_tier": "fred",
            "actual_value": 153000.0, "forecast_value": 118000.0,
            "surprise_zscore": 0.4, "predicted_reaction_usd": 1.1,
            "horizon": "reaction_1m", "n_historical": 184,
        }
        text = mp.format_prediction(pred)
        self.assertIn("+153,000", text)
        self.assertIn("+118,000", text)

    def test_index_type_formats_with_one_decimal_no_percent(self):
        pred = {
            "value_type": "index", "source_tier": "rss",
            "actual_value": 54.6, "forecast_value": 55.2,
            "surprise_zscore": -0.4, "predicted_reaction_usd": 0.6,
            "horizon": "reaction_30m", "n_historical": 196,
        }
        text = mp.format_prediction(pred)
        self.assertIn("54.6", text)
        self.assertIn("55.2", text)
        self.assertNotIn("54.6%", text)

    def test_rss_source_tier_gets_the_less_reliable_disclaimer(self):
        pred = {
            "value_type": "index", "source_tier": "rss",
            "actual_value": 54.6, "forecast_value": 55.2,
            "surprise_zscore": -0.4, "predicted_reaction_usd": 0.6,
            "horizon": "reaction_30m", "n_historical": 196,
        }
        text = mp.format_prediction(pred)
        self.assertIn("comunicato stampa", text)

    def test_fred_source_tier_does_not_get_the_rss_disclaimer(self):
        pred = {
            "value_type": "level", "source_tier": "fred",
            "actual_value": 219000.0, "forecast_value": 214000.0,
            "surprise_zscore": 0.5, "predicted_reaction_usd": -1.2,
            "horizon": "reaction_1m", "n_historical": 130,
        }
        text = mp.format_prediction(pred)
        self.assertNotIn("comunicato stampa", text)


if __name__ == "__main__":
    unittest.main()
