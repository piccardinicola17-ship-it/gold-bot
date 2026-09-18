import unittest
from unittest.mock import patch

import breaking_news as bn


class TestClassifyMonetaryPolicyText(unittest.TestCase):
    def test_hawkish_phrase_detected(self):
        r = bn.classify_monetary_policy_text("The committee sees upside risks to inflation and favors a restrictive stance.")
        self.assertEqual(r["label"], "HAWKISH")
        self.assertEqual(r["xau_bias"], "SELL")

    def test_dovish_phrase_detected(self):
        r = bn.classify_monetary_policy_text("It is now appropriate to reduce the target range given inflation has eased.")
        self.assertEqual(r["label"], "DOVISH")
        self.assertEqual(r["xau_bias"], "BUY")

    def test_neutral_text(self):
        r = bn.classify_monetary_policy_text("The board approved routine administrative matters.")
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
    alert su debt ceiling/shutdown/downgrade poteva mai scattare.

    seen_ids parte già con "__baseline_fed_press__" (la prima fonte
    iterata): simula una fonte già stabilita da tempo, così questi test
    restano concentrati sul wiring fiscale/dedup e non sul comportamento di
    baseline al primo incontro di una fonte (coperto a parte sotto, in
    TestNewSourceBaselineDoesNotSpam)."""

    _ESTABLISHED_SOURCE = {"__baseline_fed_press__": True}

    def _fake_items(self, title):
        return [{"title": title, "link": "https://example.com/1", "pub_date": "", "summary": ""}]

    def _first_source_only(self, title):
        """side_effect robusto al numero di fonti in check_breaking_news():
        il primo fetch (qualunque fonte sia) ritorna l'item finto, tutti i
        successivi ritornano vuoto — evita che aggiungere/rimuovere fonti
        (es. BCE/BoJ/BoE il 2026-09-18) rompa questi test per StopIteration."""
        import itertools
        return itertools.chain([self._fake_items(title)], itertools.repeat([]))

    def test_fiscal_shock_surfaces_in_alert_classification(self):
        with patch.object(bn, "_fetch_rss", side_effect=self._first_source_only("Treasury warns on government shutdown risk")):
            alerts, seen = bn.check_breaking_news(dict(self._ESTABLISHED_SOURCE))
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0]["classification"].get("shock_detected"))
        self.assertIn("government shutdown", alerts[0]["classification"]["matched"])

    def test_no_fiscal_keywords_no_shock_flag(self):
        with patch.object(bn, "_fetch_rss", side_effect=self._first_source_only("Fed chair speaks on labor market conditions")):
            alerts, seen = bn.check_breaking_news(dict(self._ESTABLISHED_SOURCE))
        self.assertEqual(len(alerts), 1)
        self.assertNotIn("shock_detected", alerts[0]["classification"])

    def test_seen_ids_dedup_across_calls(self):
        with patch.object(bn, "_fetch_rss", side_effect=self._first_source_only("Same headline twice")):
            alerts1, seen1 = bn.check_breaking_news(dict(self._ESTABLISHED_SOURCE))
        self.assertEqual(len(alerts1), 1)
        with patch.object(bn, "_fetch_rss", side_effect=self._first_source_only("Same headline twice")):
            alerts2, seen2 = bn.check_breaking_news(seen1)
        self.assertEqual(len(alerts2), 0)


class TestNewSourceBaselineDoesNotSpam(unittest.TestCase):
    """BUG REALE (2026-09-18): al deploy che ha aggiunto BCE/BoJ/BoE,
    is_first_run in gold_bot.py era già False (Fed girava da mesi) —
    is_first_run è un flag GLOBALE, non per-fonte, quindi non protegge una
    fonte aggiunta dopo che le altre hanno già storico: lo storico recente
    delle 3 fonti nuove (fino a 20 item ciascuna) è stato inviato come
    "breaking", 20 messaggi Telegram in sequenza al primo giro dopo il
    deploy. Fix: baseline esplicita per singola fonte dentro seen_ids."""

    def _fake_items(self, *titles):
        return [{"title": t, "link": f"https://example.com/{i}", "pub_date": "", "summary": ""}
                for i, t in enumerate(titles)]

    def test_first_encounter_of_a_source_produces_no_alerts(self):
        with patch.object(bn, "_fetch_rss", return_value=self._fake_items("A", "B", "C")):
            alerts, seen = bn.check_breaking_news({})
        self.assertEqual(len(alerts), 0, "una fonte mai vista prima deve solo stabilire la baseline, non allertare")

    def test_baseline_marker_is_saved_so_next_run_alerts_normally(self):
        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Old backlog item"), [], [], [], [], [],
        ]):
            _, seen_after_baseline = bn.check_breaking_news({})
        self.assertIn("__baseline_fed_press__", seen_after_baseline)

        with patch.object(bn, "_fetch_rss", side_effect=[
            self._fake_items("Old backlog item", "Genuinely new item"), [], [], [], [], [],
        ]):
            alerts, _ = bn.check_breaking_news(seen_after_baseline)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["title"], "Genuinely new item")

    def test_already_established_source_alerts_normally(self):
        seen = {"__baseline_fed_press__": True}
        with patch.object(bn, "_fetch_rss", side_effect=[self._fake_items("Fresh alert"), [], [], [], [], []]):
            alerts, _ = bn.check_breaking_news(seen)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["title"], "Fresh alert")


class TestFetchArticleText(unittest.TestCase):
    """2026-09-18, richiesta esplicita dell'utente: "Di cosa parla: non
    disponibile" non va bene quando il link porta a un testo leggibile.
    fetch_article_text() recupera il comunicato/discorso completo (HTML o
    PDF) quando l'RSS non include un estratto — vedi gold_bot.
    check_breaking_news_job, che lo usa prima di chiamare l'AI."""

    def test_empty_url_returns_empty_string(self):
        self.assertEqual(bn.fetch_article_text(""), "")

    def test_html_content_is_stripped_to_plain_text(self):
        class FakeResponse:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = ("<html><head><style>.x{color:red}</style></head>"
                     "<body><p>Real content here.</p></body></html>")
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/page")
        self.assertIn("Real content here.", text)
        self.assertNotIn("color:red", text)

    def test_main_tag_isolates_article_from_navigation(self):
        """Bug reale verificato dal vivo su ecb.europa.eu il 2026-09-18:
        senza isolare <main>, uno stripping grezzo su tutta la pagina
        restituiva migliaia di caratteri di menu di navigazione PRIMA del
        vero comunicato, che finiva tagliato fuori da max_chars."""
        class FakeResponse:
            headers = {"Content-Type": "text/html"}
            text = (
                "<html><header><nav><ul><li>Menu item one</li>"
                "<li>Menu item two</li></ul></nav></header>"
                "<main><p>The real press release content.</p></main>"
                "<footer>Footer links</footer></html>"
            )
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/page", max_chars=50)
        self.assertIn("The real press release content.", text)
        self.assertNotIn("Menu item", text)

    def test_fed_article_div_isolates_article_from_navigation(self):
        """Il sito della Fed non usa <main> ma un <div id="article"> fisso
        (verificato dal vivo su comunicati e discorsi reali)."""
        class FakeResponse:
            headers = {"Content-Type": "text/html"}
            text = (
                "<html><nav>Lots of Fed navigation links here</nav>"
                '<div id="content"><div id="article">'
                "<p>The real speech content.</p>"
                '</div><div id="lastUpdate">Last update</div></div></html>'
            )
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/page", max_chars=50)
        self.assertIn("The real speech content.", text)
        self.assertNotIn("navigation links", text)

    def test_unrecognized_layout_falls_back_to_whole_page(self):
        """Nessuno dei due marcatori conosciuti: meglio l'intera pagina
        (rumorosa) che stringa vuota — stesso principio del resto del
        modulo (mai bloccare per un sito non ancora visto)."""
        class FakeResponse:
            headers = {"Content-Type": "text/html"}
            text = "<html><body><p>Plain unrecognized page content.</p></body></html>"
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/page")
        self.assertIn("Plain unrecognized page content.", text)

    def test_pdf_content_is_extracted_via_pypdf(self):
        """BoJ pubblica le decisioni di politica monetaria SOLO come PDF
        (verificato: ogni link whatsnew.xml punta a un .pdf) — senza
        supporto PDF, ogni alert BoJ resterebbe perennemente "non
        disponibile"."""
        class FakePage:
            def extract_text(self):
                return "Policy statement text."

        class FakeReader:
            def __init__(self, *a, **k):
                self.pages = [FakePage()]

        class FakeResponse:
            headers = {"Content-Type": "application/pdf"}
            content = b"%PDF-fake"
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()), \
             patch.object(bn, "PdfReader", FakeReader):
            text = bn.fetch_article_text("https://example.com/doc.pdf")
        self.assertIn("Policy statement text.", text)

    def test_unsupported_content_type_returns_empty(self):
        class FakeResponse:
            headers = {"Content-Type": "application/json"}
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/data.json")
        self.assertEqual(text, "")

    def test_network_failure_returns_empty_string_not_exception(self):
        with patch.object(bn.requests, "get", side_effect=Exception("timeout")):
            text = bn.fetch_article_text("https://example.com/page")
        self.assertEqual(text, "")

    def test_truncates_to_max_chars(self):
        class FakeResponse:
            headers = {"Content-Type": "text/html"}
            text = "<p>" + ("x" * 5000) + "</p>"
            def raise_for_status(self): pass

        with patch.object(bn.requests, "get", return_value=FakeResponse()):
            text = bn.fetch_article_text("https://example.com/long", max_chars=100)
        self.assertEqual(len(text), 100)


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
