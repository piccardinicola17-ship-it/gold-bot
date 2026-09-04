import unittest
from unittest.mock import patch

import breaking_news as bn


class TestClassifyFedText(unittest.TestCase):
    def test_hawkish_phrase_detected(self):
        r = bn.classify_fed_text("The committee sees upside risks to inflation and favors a restrictive stance.")
        self.assertEqual(r["label"], "HAWKISH")
        self.assertEqual(r["xau_bias"], "SELL")

    def test_dovish_phrase_detected(self):
        r = bn.classify_fed_text("It is now appropriate to reduce the target range given inflation has eased.")
        self.assertEqual(r["label"], "DOVISH")
        self.assertEqual(r["xau_bias"], "BUY")

    def test_neutral_text(self):
        r = bn.classify_fed_text("The board approved routine administrative matters.")
        self.assertEqual(r["label"], "NEUTRO")
        self.assertEqual(r["xau_bias"], "N/D")


class TestClassifyFiscalText(unittest.TestCase):
    def test_detects_debt_ceiling(self):
        r = bn.classify_fiscal_text("Congress debates the debt ceiling ahead of the deadline.")
        self.assertTrue(r["shock_detected"])
        self.assertIn("debt ceiling", r["matched"])

    def test_no_shock_on_unrelated_text(self):
        r = bn.classify_fiscal_text("The Fed chair gave a routine speech on economic outlook.")
        self.assertFalse(r["shock_detected"])


class TestClassifyGeopolitical(unittest.TestCase):
    def test_critical_risk_off(self):
        r = bn.classify_geopolitical_text("Reports of a military strike near the strait of hormuz.")
        self.assertTrue(r["risk_off"])
        self.assertEqual(r["xau_bias"], "BUY")

    def test_deescalation(self):
        r = bn.classify_geopolitical_text("A ceasefire agreed between the two sides.")
        self.assertFalse(r["risk_off"])
        self.assertEqual(r["xau_bias"], "SELL")


class TestCheckBreakingNewsFiscalWiring(unittest.TestCase):
    """Regression: classify_fiscal_text() era definito ma mai chiamato da
    check_breaking_news() (bug trovato in audit il 2026-09-04) — nessun
    alert su debt ceiling/shutdown/downgrade poteva mai scattare."""

    def _fake_items(self, title):
        return [{"title": title, "link": "https://example.com/1", "pub_date": "", "summary": ""}]

    def test_fiscal_shock_surfaces_in_alert_classification(self):
        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Treasury warns on government shutdown risk"),
            [],
        ]):
            alerts, seen = bn.check_breaking_news(set())
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0]["classification"].get("shock_detected"))
        self.assertIn("government shutdown", alerts[0]["classification"]["matched"])

    def test_no_fiscal_keywords_no_shock_flag(self):
        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Fed chair speaks on labor market conditions"),
            [],
        ]):
            alerts, seen = bn.check_breaking_news(set())
        self.assertEqual(len(alerts), 1)
        self.assertNotIn("shock_detected", alerts[0]["classification"])

    def test_seen_ids_dedup_across_calls(self):
        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Same headline twice"),
            [],
        ]):
            alerts1, seen1 = bn.check_breaking_news(set())
        self.assertEqual(len(alerts1), 1)
        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Same headline twice"),
            [],
        ]):
            alerts2, seen2 = bn.check_breaking_news(seen1)
        self.assertEqual(len(alerts2), 0)


class TestFormatBreakingAlert(unittest.TestCase):
    def test_shock_detected_line_included(self):
        alert = {
            "source": "fed_press",
            "title": "Statement on debt ceiling",
            "summary": "",
            "link": "",
            "classification": {"label": "NEUTRO", "xau_bias": "N/D", "matched": [], "shock_detected": True},
            "geopolitical": None,
        }
        msg = bn.format_breaking_alert(alert)
        self.assertIn("shock fiscale", msg)


if __name__ == "__main__":
    unittest.main()
