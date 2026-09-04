"""
Test per news_analyst.py — _escape_md (il fix per i 3 punti di Markdown
non escapato trovati il 2026-09-03), _find_macro_db_info, i rami "nessun
contenuto" delle funzioni di formattazione, e la regola di onestà
aggiunta ad analyze_breaking_news (non inventare un bias BUY/SELL dal
solo titolo quando l'RSS non porta un vero estratto del discorso —
verificato empiricamente lo stesso giorno sul discorso Waller).

_call_groq richiede GROQ_API_KEY (assente in questo ambiente locale) e
ritorna un placeholder senza fare rete — non testiamo la vera risposta
dell'LLM, ma il CONTESTO che le funzioni gli costruiscono, con un mock.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import news_analyst as na


class TestEscapeMd(unittest.TestCase):
    def test_escapes_all_special_chars(self):
        self.assertEqual(na._escape_md("a_b*c`d[e]f"), "a\\_b\\*c\\`d\\[e\\]f")

    def test_plain_text_unchanged(self):
        self.assertEqual(na._escape_md("Fed rate decision today"), "Fed rate decision today")

    def test_odd_underscore_count_no_longer_breaks_markdown(self):
        """Il bug reale: un titolo con un numero dispari di underscore
        (es. un nome di regime TRENDING_UP) rompeva il parsing Markdown
        dell'intero messaggio Telegram. Dopo l'escape non ci sono più
        underscore non protetti."""
        title = "TRENDING_UP breaks parsing"
        escaped = na._escape_md(title)
        self.assertNotIn("_", escaped.replace("\\_", ""))

    def test_empty_string(self):
        self.assertEqual(na._escape_md(""), "")


class TestFindMacroDbInfo(unittest.TestCase):
    def test_matches_known_event(self):
        info = na._find_macro_db_info("Core CPI m/m")
        self.assertTrue(info)
        self.assertEqual(info["impatto"], "MOLTO ALTO")

    def test_case_insensitive_substring_match(self):
        info = na._find_macro_db_info("us nfp report tonight")
        self.assertTrue(info)

    def test_unknown_event_returns_empty_dict(self):
        self.assertEqual(na._find_macro_db_info("Some Random Low-Impact Event"), {})


class TestFormatNewsMessageHeadlineSanitization(unittest.TestCase):
    """FIX: la sanificazione manuale delle headline in format_news_message
    rimuoveva solo * _ ` ma non [ ] — a differenza di _escape_md (usata
    altrove nello stesso file) che gestisce tutti e 5 i caratteri. Un
    titolo con parentesi quadre poteva rompere il parsing Markdown di
    Telegram. current_price=4400 (>100) per non innescare la chiamata di
    rete a fxratesapi dentro format_news_message."""

    def test_square_brackets_removed_from_headline(self):
        result = na.format_news_message(["Fed [Update]: rates unchanged"], current_price=4400)
        self.assertNotIn("[", result)
        self.assertNotIn("]", result)


class TestEmptyInputEarlyReturns(unittest.TestCase):
    def test_format_news_message_no_news(self):
        self.assertEqual(na.format_news_message([]), "Nessuna notizia disponibile al momento.")

    def test_get_macro_briefing_no_events(self):
        self.assertEqual(na.get_macro_briefing([]), "Nessun evento macro ad alto impatto oggi.")

    def test_get_bias_briefing_no_news(self):
        result = na.get_bias_briefing([])
        self.assertIn("NEUTRALE", result)


class TestAnalyzeBreakingNewsHonesty(unittest.TestCase):
    """Verifica che il contesto passato all'LLM dichiari esplicitamente
    l'assenza di un estratto reale (bug del 2026-09-03: senza questo
    segnale l'LLM inventava un bias plausibile dal solo titolo)."""

    def test_missing_summary_is_declared_explicitly_not_omitted(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=140):
            captured["system"] = system
            captured["user"] = user
            return "Cos'è: X\nDi cosa parla: Y\nPer l'oro: NEUTRO — Z"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_breaking_news("Discorso di un membro Fed", "Waller speaks", summary="")

        self.assertIn("NON DISPONIBILE", captured["user"])
        self.assertIn("Non indovinare il contenuto dal titolo", captured["user"])
        self.assertIn("Non inventare un bias plausibile dal solo titolo", captured["system"])

    def test_short_summary_below_threshold_also_treated_as_missing(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=140):
            captured["user"] = user
            return "ok"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_breaking_news("Comunicato Fed", "Title", summary="too short")

        self.assertIn("NON DISPONIBILE", captured["user"])

    def test_real_summary_is_passed_through_not_flagged_missing(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=140):
            captured["user"] = user
            return "ok"

        real_summary = "The FOMC decided today to maintain the target range for the federal funds rate."
        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_breaking_news("Comunicato Fed", "FOMC Statement", summary=real_summary)

        self.assertNotIn("NON DISPONIBILE", captured["user"])
        self.assertIn("Estratto:", captured["user"])


if __name__ == "__main__":
    unittest.main()
