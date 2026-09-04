"""
Test per ai_assistant.py — copre il bug trovato in produzione il
2026-09-04: build_context_snapshot() non diceva mai all'LLM se il trade
fosse un ordine pending mai attivato o una posizione davvero aperta, né
quali target fossero stati realmente raggiunti. Senza quell'informazione
l'assistente ha inventato "ha già superato TP1 ed è vicino a TP2" per un
SELL LIMIT con entry_filled=0 (mai scattato), solo perché il prezzo
corrente era già sotto quei livelli — un'inferenza plausibile ma falsa.

full_analyze/get_news_sentiment/get_economic_events/get_extended_news/
get_upcoming_events sono mockate: qui testiamo solo il testo del contesto
costruito, non le vere chiamate di rete/LLM.
"""

import contextlib
import os
import sys
import unittest
from datetime import timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_assistant as aiasst


def _fake_full_analyze(**kwargs):
    return {
        "price": 4411.82, "regime": "TRENDING_DOWN", "signal": "NEUTRAL",
        "buy_count": 1, "sell_count": 2, "mtf_trends": {},
    }


@contextlib.contextmanager
def _mock_technical_levels():
    """build_context_snapshot() ora calcola livelli tecnici reali (1h/4h)
    per il blocco "LIVELLI TECNICI REALI" — senza questo mock ogni test
    farebbe vere chiamate di rete a get_data()."""
    with mock.patch("analyzer.get_data", return_value=mock.MagicMock()), \
         mock.patch("analyzer.compute_indicators", side_effect=lambda df: df), \
         mock.patch("analyzer.get_support_resistance", return_value={
             "support": 4300.0, "resistance": 4500.0, "s_near": 4400.0,
             "r_near": 4480.0, "pivot": 4450.0, "r1": 4470.0, "r2": 4520.0,
             "s1": 4380.0, "s2": 4350.0,
         }):
        yield


class TestBuildContextSnapshotTradeState(unittest.TestCase):
    def _run(self, trade: dict) -> str:
        with _mock_technical_levels(), \
             mock.patch("analyzer.full_analyze", side_effect=_fake_full_analyze), \
             mock.patch("analyzer.get_news_sentiment", return_value={"label": "bearish", "score": -3}), \
             mock.patch("analyzer.get_extended_news", return_value=[]), \
             mock.patch("analyzer.get_economic_events", return_value={"high_impact_today": False, "events": []}), \
             mock.patch("analyzer.get_upcoming_events", return_value=[]), \
             mock.patch("trade_manager.load_active_trade", return_value=trade):
            return aiasst.build_context_snapshot()

    def test_pending_order_never_activated_shows_waiting_not_progress(self):
        """Il caso reale del bug: SELL LIMIT mai attivato, prezzo attuale
        già sotto TP1/TP2 per puro caso di dove si trova il mercato ora —
        il contesto deve dire chiaramente che è ancora IN ATTESA, non
        lasciare che l'LLM deduca (sbagliando) un progresso verso i target."""
        trade = {
            "signal": "SELL", "order_type": "LIMIT", "entry": 4466.31,
            "timeframe": "1h", "sl": 4490.46, "tp1": 4442.16, "tp2": 4418.01, "tp3": 4389.04,
            "activated": 0, "tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0,
        }
        context = self._run(trade)
        self.assertIn("IN ATTESA", context)
        self.assertNotIn("target già raggiunti", context)
        # Non deve mai comparire una frase che suggerisca un TP raggiunto.
        self.assertNotIn("ATTIVO (posizione aperta)", context)

    def test_context_includes_real_technical_levels_not_just_bot_trade(self):
        """Il bug più importante della sessione: prima l'unico numero "a
        forma di target" nel contesto erano i TP1/TP2/TP3 del trade del
        BOT — a ogni "dove metto i TP" l'LLM li copiava, anche per un
        trade dell'utente di direzione opposta (TP discendenti proposti
        per un BUY, con TP3 sotto l'entry). Ora deve esserci una fonte di
        livelli separata, esplicitamente per questo scopo."""
        context = self._run({})
        self.assertIn("LIVELLI TECNICI REALI", context)
        self.assertIn("1H", context)
        self.assertIn("4H", context)
        self.assertIn("MAI i TP1/TP2/TP3 del campo 'Trade aperto dal BOT'", context)

    def test_activated_trade_with_tp1_hit_shows_real_progress(self):
        trade = {
            "signal": "SELL", "order_type": "LIMIT", "entry": 4466.31,
            "timeframe": "1h", "sl": 4490.46, "tp1": 4442.16, "tp2": 4418.01, "tp3": 4389.04,
            "activated": 1, "tp1_hit": 1, "tp2_hit": 0, "tp3_hit": 0,
        }
        context = self._run(trade)
        self.assertIn("ATTIVO (posizione aperta)", context)
        self.assertIn("target già raggiunti: TP1", context)
        self.assertNotIn("IN ATTESA", context)

    def test_activated_trade_no_target_hit_says_so_explicitly(self):
        trade = {
            "signal": "BUY", "order_type": "MARKET", "entry": 4400.0,
            "timeframe": "4h", "sl": 4380.0, "tp1": 4420.0, "tp2": 4440.0, "tp3": 4460.0,
            "activated": 1, "tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0,
        }
        context = self._run(trade)
        self.assertIn("nessuno ancora", context)

    def test_already_released_event_marked_as_such_not_upcoming(self):
        """Il secondo bug reale della stessa sessione: get_economic_events()
        ritornava sia gli eventi passati che futuri di oggi in "events",
        senza distinguerli — l'NFP delle 14:30, alle 16:00, veniva ancora
        presentato come "da tenere d'occhio". Ora deve comparire tra i
        "GIÀ USCITI", non tra quelli "ANCORA DA USCIRE"."""
        nfp_uscito = {
            "title": "Non-Farm Employment Change", "time": "14:30",
            "forecast": "55K", "previous": "-23K",
        }
        cal = {"high_impact_today": True, "events": [nfp_uscito], "upcoming": []}
        with _mock_technical_levels(), \
             mock.patch("analyzer.full_analyze", side_effect=_fake_full_analyze), \
             mock.patch("analyzer.get_news_sentiment", return_value={"label": "bearish", "score": -3}), \
             mock.patch("analyzer.get_extended_news", return_value=[]), \
             mock.patch("analyzer.get_economic_events", return_value=cal), \
             mock.patch("analyzer.get_upcoming_events", return_value=[]), \
             mock.patch("trade_manager.load_active_trade", return_value={}):
            context = aiasst.build_context_snapshot()
        self.assertIn("GIÀ USCITI", context)
        self.assertIn("Non-Farm Employment Change", context)
        self.assertNotIn("ANCORA DA USCIRE", context)

    def test_still_upcoming_event_marked_as_such(self):
        pending_event = {
            "title": "FOMC Statement", "time": "20:00",
            "forecast": "N/A", "previous": "N/A",
        }
        cal = {"high_impact_today": True, "events": [pending_event], "upcoming": [pending_event]}
        with _mock_technical_levels(), \
             mock.patch("analyzer.full_analyze", side_effect=_fake_full_analyze), \
             mock.patch("analyzer.get_news_sentiment", return_value={"label": "bearish", "score": -3}), \
             mock.patch("analyzer.get_extended_news", return_value=[]), \
             mock.patch("analyzer.get_economic_events", return_value=cal), \
             mock.patch("analyzer.get_upcoming_events", return_value=[]), \
             mock.patch("trade_manager.load_active_trade", return_value={}):
            context = aiasst.build_context_snapshot()
        self.assertIn("ANCORA DA USCIRE", context)
        self.assertIn("FOMC Statement", context)
        self.assertNotIn("GIÀ USCITI", context)

    def test_m5_signal_below_live_threshold_is_flagged(self):
        with _mock_technical_levels(), \
             mock.patch("analyzer.full_analyze", return_value={
            "price": 4411.82, "regime": "NORMAL", "signal": "SELL",
            "order_type": "SELL", "entry": 4411.0, "prob": 40,
        }), \
             mock.patch("analyzer.get_news_sentiment", return_value={"label": "bearish", "score": -3}), \
             mock.patch("analyzer.get_extended_news", return_value=[]), \
             mock.patch("analyzer.get_economic_events", return_value={"high_impact_today": False, "events": []}), \
             mock.patch("analyzer.get_upcoming_events", return_value=[]), \
             mock.patch("trade_manager.load_active_trade", return_value={}):
            context = aiasst.build_context_snapshot()
        self.assertIn("informativo, non ancora un trade", context)
        self.assertIn("sotto la soglia 65%", context)


class TestStatedPositionNote(unittest.TestCase):
    """_stated_position_note(): terzo bug della stessa sessione, il più
    insidioso. Una regola nel system prompt ("se l'utente descrive una sua
    posizione, ha priorità sul campo Trade") non bastava da sola —
    verificato dal vivo con Groq: con un trade SELL del bot in contesto e
    una domanda "ho un buy da 4394, dove metto i TP?", il modello ha
    comunque risposto con i TP del SELL del bot (target sotto al prezzo
    invece che sopra, coerenti con una SELL non con il BUY dichiarato).
    Serve un rinforzo strutturale agganciato alla domanda stessa, non solo
    un'istruzione generica nel prompt."""

    def test_detects_common_italian_phrasings(self):
        casi = [
            "Ho un buy da 4394, dove metto i TP?",
            "sono long a 4394, che ne pensi",
            "Ho un sell da 4470",
            "sono short da 4470, dove sl?",
            "ho comprato a 4394 ieri",
        ]
        for domanda in casi:
            with self.subTest(domanda=domanda):
                self.assertNotEqual(aiasst._stated_position_note(domanda), "")

    def test_generic_question_returns_empty_note(self):
        casi = [
            "Guardando il mercato ora fino a dove potrebbe arrivare?",
            "Cosa ne pensi dell'NFP di oggi?",
            "Conviene entrare adesso?",
        ]
        for domanda in casi:
            with self.subTest(domanda=domanda):
                self.assertEqual(aiasst._stated_position_note(domanda), "")

    def test_note_tells_model_to_ignore_bot_trade_field(self):
        note = aiasst._stated_position_note("Ho un buy da 4394, dove metto i TP?")
        self.assertIn("ignora del tutto SL/TP1/TP2/TP3 del campo 'Trade aperto dal BOT'", note)

    def test_note_fires_on_bare_number_without_verb_phrasing(self):
        """Il bug reale: 'ho un'operazione IN buy da 4394' non matchava il
        vecchio regex verbale (parole extra tra "un" e "buy"). Rilevare il
        prezzo invece del verbo copre qualunque formulazione."""
        note = aiasst._stated_position_note(
            "Io ho un operazione in buy da 4394, attualmente a break even, "
            "dammi i tp"
        )
        self.assertNotEqual(note, "")


class TestConversationMemory(unittest.TestCase):
    """Quarto bug della stessa sessione, e il più fondativo: ask_ai() era
    completamente stateless. L'utente ha descritto la sua posizione (BUY
    4394) in un messaggio, poi in quello successivo ha chiesto "il mio TP
    a 4460 verrà raggiunto?" senza ripetere i dettagli — assumendo
    ragionevolmente che il bot ricordasse. Senza storico l'LLM ha risposto
    sul trade sbagliato (il SELL LIMIT del bot) e ha persino detto di non
    avere il prezzo attuale, pur essendo sempre nel CONTESTO. Verificato
    dal vivo con Groq: con lo storico, la stessa domanda di follow-up
    ottiene una risposta coerente sulla posizione giusta."""

    def setUp(self):
        aiasst._conversation_history.clear()

    def tearDown(self):
        aiasst._conversation_history.clear()

    def test_turn_recorded_and_retrievable(self):
        aiasst._record_conversation_turn("domanda 1", "risposta 1")
        turns = aiasst._recent_conversation_turns()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["question"], "domanda 1")
        self.assertEqual(turns[0]["answer"], "risposta 1")

    def test_trims_to_max_turns(self):
        for i in range(aiasst._CONVERSATION_MAX_TURNS + 3):
            aiasst._record_conversation_turn(f"q{i}", f"a{i}")
        self.assertLessEqual(len(aiasst._conversation_history), aiasst._CONVERSATION_MAX_TURNS)
        # deve tenere le più recenti, non le più vecchie
        self.assertEqual(aiasst._conversation_history[-1]["question"], f"q{aiasst._CONVERSATION_MAX_TURNS + 2}")

    def test_stale_turns_pruned_by_ttl(self):
        aiasst._record_conversation_turn("vecchia", "risposta vecchia")
        aiasst._conversation_history[0]["ts"] -= timedelta(minutes=aiasst._CONVERSATION_TTL_MINUTES + 5)
        aiasst._record_conversation_turn("nuova", "risposta nuova")

        turns = aiasst._recent_conversation_turns()
        questions = [t["question"] for t in turns]
        self.assertNotIn("vecchia", questions)
        self.assertIn("nuova", questions)

    def test_ask_ai_includes_history_as_alternating_messages(self):
        aiasst._record_conversation_turn("domanda precedente", "risposta precedente")
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["messages"] = json["messages"]
            resp = mock.Mock()
            resp.raise_for_status = lambda: None
            resp.json = lambda: {"choices": [{"message": {"content": "nuova risposta"}}]}
            return resp

        with mock.patch.object(aiasst, "GROQ_API_KEY", "fake-key"), \
             mock.patch.object(aiasst, "requests") as mock_requests, \
             mock.patch.object(aiasst, "build_context_snapshot", return_value="(contesto finto)"):
            mock_requests.post = fake_post
            import asyncio
            asyncio.run(aiasst.ask_ai("nuova domanda"))

        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], {"role": "user", "content": "domanda precedente"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "risposta precedente"})
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("nuova domanda", messages[3]["content"])

    def test_ask_ai_records_the_new_turn_after_success(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            resp = mock.Mock()
            resp.raise_for_status = lambda: None
            resp.json = lambda: {"choices": [{"message": {"content": "risposta generata"}}]}
            return resp

        with mock.patch.object(aiasst, "GROQ_API_KEY", "fake-key"), \
             mock.patch.object(aiasst, "requests") as mock_requests, \
             mock.patch.object(aiasst, "build_context_snapshot", return_value="(contesto finto)"):
            mock_requests.post = fake_post
            import asyncio
            asyncio.run(aiasst.ask_ai("una domanda"))

        self.assertEqual(len(aiasst._conversation_history), 1)
        self.assertEqual(aiasst._conversation_history[0]["question"], "una domanda")
        self.assertEqual(aiasst._conversation_history[0]["answer"], "risposta generata")


if __name__ == "__main__":
    unittest.main()
