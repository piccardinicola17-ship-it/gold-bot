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

    def test_no_currency_arg_keeps_old_behavior(self):
        """Retrocompatibilità: i chiamanti esistenti non passano currency,
        il match a substring resta invariato (comportamento pre-2026-09-16)."""
        info = na._find_macro_db_info("CPI y/y")
        self.assertTrue(info)

    def test_usd_currency_still_matches(self):
        info = na._find_macro_db_info("CPI y/y", currency="USD")
        self.assertTrue(info)

    def test_non_usd_currency_never_matches(self):
        """Bug reale trovato il 2026-09-16 aggiungendo il supporto per
        valute non-USD (GBP/EUR/JPY/...): MACRO_DB contiene solo logiche
        Fed-specifiche, ma il match era una semplice substring sul titolo
        senza controllare la valuta — "GBP CPI y/y" avrebbe agganciato la
        logica USD ("CPI alto -> Fed hawkish -> oro giù"), sbagliata (quella
        è la Bank of England, non la Fed). currency non-USD deve sempre
        tornare {} anche se il titolo contiene una chiave nota di MACRO_DB."""
        for currency in ("GBP", "EUR", "JPY", "gbp"):
            self.assertEqual(na._find_macro_db_info("CPI y/y", currency=currency), {})
            self.assertEqual(na._find_macro_db_info("NFP report tonight", currency=currency), {})


class TestFormatNewsMessageHeadlineSanitization(unittest.TestCase):
    """FIX: la sanificazione manuale delle headline in format_news_message
    rimuoveva solo * _ ` (cancellandoli, non escapandoli — perdendo pezzi
    reali del titolo) e ignorava del tutto [ ]. Ora usa _escape_md() come
    ovunque altrove nello stesso file: tutti e 5 i caratteri diventano
    sicuri per il parsing Markdown di Telegram MA restano nel testo (con
    backslash davanti), invece di sparire. current_price=4400 (>100) per
    non innescare la chiamata di rete a fxratesapi dentro
    format_news_message."""

    def test_square_brackets_escaped_not_removed_from_headline(self):
        result = na.format_news_message(["Fed [Update]: rates unchanged"], current_price=4400)
        self.assertIn("Fed \\[Update\\]: rates unchanged", result)


class TestFormatNewsMessageMultilineEntries(unittest.TestCase):
    """FIX (2026-09-13): get_extended_news() produce ogni voce su due righe
    ("fonte (data)\\ntitolo") — format_news_message prendeva solo la prima
    riga (raw.split("\\n")[0]), mostrando in produzione bullet come
    "Yahoo Entertainment (2026-09-10)" senza alcun titolo reale, scoperto
    da uno screenshot del bot live. Ora prende l'ultima riga non vuota."""

    def test_headline_shows_title_not_just_source_line(self):
        news = ["📰 *Yahoo Entertainment* (2026-09-10)\n_ECB raises rates, dollar strengthens_"]
        result = na.format_news_message(news, current_price=4400)
        self.assertIn("ECB raises rates, dollar strengthens", result)

    def test_single_line_entry_still_works(self):
        result = na.format_news_message(["Fed holds rates steady"], current_price=4400)
        self.assertIn("Fed holds rates steady", result)


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


class TestAnalyzeMacroEventCurrencyAware(unittest.TestCase):
    """2026-09-16: gli alert macro coprono anche eventi non-USD (GBP, EUR,
    JPY, ...) — l'LLM deve sapere esplicitamente di quale valuta si tratta,
    altrimenti assumerebbe implicitamente che sia sempre la Fed/USD (il
    titolo grezzo del calendario, es. "CPI y/y", non lo dice da solo)."""

    def test_usd_event_no_currency_note_needed(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_macro_event("CPI y/y", "0.3%", "0.2%", current_price=4400, currency="USD")

        self.assertNotIn("Valuta:", captured["user"])

    def test_non_usd_event_states_currency_explicitly(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_macro_event("CPI y/y", "0.3%", "0.2%", current_price=4400, currency="GBP")

        self.assertIn("Valuta: GBP", captured["user"])
        self.assertIn("non sulla Fed", captured["user"])

    def test_no_currency_arg_backward_compatible(self):
        """Chiamanti esistenti che non passano currency (default "") non
        devono rompersi né aggiungere una riga Valuta fuorviante."""
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_macro_event("Core CPI m/m", "0.3%", "0.2%", current_price=4400)

        self.assertNotIn("Valuta:", captured["user"])


class TestAnalyzeMacroEventRelatedContext(unittest.TestCase):
    """2026-09-16: quando un evento è la continuazione di una decisione
    macro già uscita poco prima (stessa valuta) — es. FOMC Press Conference
    dopo Federal Funds Rate della stessa riunione — il chiamante
    (gold_bot._find_related_macro_context) passa una riga di contesto che
    deve arrivare intatta nel prompt, con l'istruzione esplicita di non
    trattare l'evento come un esito ancora incerto."""

    def test_related_context_included_in_prompt(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["system"] = system
            captured["user"] = user
            return "Bias: SELL\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_macro_event(
                "FOMC Press Conference", "N/A", "N/A", current_price=4318.40,
                currency="USD",
                related_context="Contesto correlato: 30 minuti fa è già uscito "
                                 "\"Federal Funds Rate\" (bias SELL, prezzo oro -28.70$).",
            )

        self.assertIn("Contesto correlato", captured["user"])
        self.assertIn("Federal Funds Rate", captured["user"])
        self.assertIn("Contesto correlato", captured["system"])

    def test_no_related_context_by_default(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_macro_event("Core CPI m/m", "0.3%", "0.2%", current_price=4400)

        self.assertNotIn("Contesto correlato", captured["user"])

    def test_combined_event_forwards_related_context(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=90):
            captured["user"] = user
            return "Bias: SELL\nMotivo: test"

        events = [
            {"title": "FOMC Press Conference", "forecast": "N/A", "previous": "N/A", "currency": "USD"},
        ]
        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_combined_macro_event(
                events, current_price=4318.40,
                related_context="Contesto correlato: già uscito Federal Funds Rate.",
            )

        self.assertIn("Contesto correlato", captured["user"])


class TestAnalyzeCombinedMacroEventCurrency(unittest.TestCase):
    def test_single_event_forwards_currency(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=80):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_combined_macro_event(
                [{"title": "GDP q/q", "forecast": "0.5%", "previous": "0.4%", "currency": "NZD"}],
                current_price=4400,
            )

        self.assertIn("Valuta: NZD", captured["user"])

    def test_multi_event_group_states_shared_currency(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=90):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        events = [
            {"title": "Official Bank Rate", "forecast": "4.25%", "previous": "4.00%", "currency": "GBP"},
            {"title": "MPC Official Bank Rate Votes", "forecast": "N/A", "previous": "N/A", "currency": "GBP"},
        ]
        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_combined_macro_event(events, current_price=4400)

        self.assertIn("Valuta: GBP", captured["user"])

    def test_usd_group_has_no_currency_note(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=90):
            captured["user"] = user
            return "Bias: NEUTRO\nMotivo: test"

        events = [
            {"title": "Core Retail Sales m/m", "forecast": "0.3%", "previous": "0.2%", "currency": "USD"},
            {"title": "Retail Sales m/m", "forecast": "0.4%", "previous": "0.3%", "currency": "USD"},
        ]
        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.analyze_combined_macro_event(events, current_price=4400)

        self.assertNotIn("Valuta:", captured["user"])


class TestWeeklySmcNarrativeTimeframeLabel(unittest.TestCase):
    """2026-09-17: get_weekly_smc_narrative ora prende timeframe_label
    (default "4H", invariato per l'analisi weekend) per riusarla anche
    per l'analisi giornaliera del report mattutino su candele 1H — prima
    diceva sempre "4H" nel contesto a prescindere dai dati passati."""

    def _ctx(self):
        return {
            "structure": {"structure": "BULLISH", "bos": None, "choch": None,
                          "last_high": 4350.0, "last_low": 4300.0,
                          "prev_high": 4340.0, "prev_low": 4290.0},
            "order_blocks": {},
            "fvg": {},
            "liquidity": {},
            "premium_discount": "DISCOUNT",
            "mitigation": {},
            "regime": "TRENDING",
            "adx": 28,
        }

    def test_default_label_is_4h(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=260):
            captured["user"] = user
            return "narrativa di prova"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.get_weekly_smc_narrative(self._ctx(), current_price=4320.0)

        self.assertIn("Struttura di mercato (4H):", captured["user"])

    def test_custom_label_is_used_for_daily_analysis(self):
        captured = {}

        def fake_call_groq(system, user, max_tokens=260):
            captured["user"] = user
            return "narrativa di prova"

        with mock.patch.object(na, "_call_groq", fake_call_groq):
            na.get_weekly_smc_narrative(self._ctx(), current_price=4320.0, timeframe_label="1H")

        self.assertIn("Struttura di mercato (1H):", captured["user"])
        self.assertNotIn("Struttura di mercato (4H):", captured["user"])


if __name__ == "__main__":
    unittest.main()
