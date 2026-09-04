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

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_assistant as aiasst


def _fake_full_analyze(**kwargs):
    return {
        "price": 4411.82, "regime": "TRENDING_DOWN", "signal": "NEUTRAL",
        "buy_count": 1, "sell_count": 2, "mtf_trends": {},
    }


class TestBuildContextSnapshotTradeState(unittest.TestCase):
    def _run(self, trade: dict) -> str:
        with mock.patch("analyzer.full_analyze", side_effect=_fake_full_analyze), \
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

    def test_m5_signal_below_live_threshold_is_flagged(self):
        with mock.patch("analyzer.full_analyze", return_value={
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


if __name__ == "__main__":
    unittest.main()
