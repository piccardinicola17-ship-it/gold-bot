"""
Test per agent_orchestrator.py — copre agent_decision_maker (Regola 5 e
Regola 5bis), corrette il 2026-09-03: il check anti-slippage sui MARKET
order deve fallire in modo prudente (SKIP) quando il prezzo non è
disponibile o la sua scala (futures/spot) non è verificabile, invece di
saltare il controllo ed eseguire alla cieca; e un pending LIMIT/STOP non
deve più essere aperto se è già oltre la soglia (differenziata per
timeframe) che lo farebbe cancellare pochi secondi dopo dal monitor.

Sono chiamate dirette ad agent_decision_maker con un TradingState già
preparato per superare le Regole 1-4 (structure_ok/news_safe/risk_ok/rr),
così da isolare il comportamento della Regola 5/5bis.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_orchestrator import TradingState, agent_decision_maker


def _valid_state(**overrides) -> TradingState:
    """Uno stato che supera sempre le Regole 1-4 (setup valido, news
    sicure, risk ok, R:R sufficiente) — pronto per isolare la Regola 5/5bis."""
    state = TradingState(
        signal="BUY",
        order_type=overrides.pop("order_type", "BUY"),
        structure_ok=True,
        news_safe=True,
        risk_ok=True,
        rr=2.5,
        prob=60,
        regime="NORMAL",
        timeframe=overrides.pop("timeframe", "4h"),
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


class TestRegola5MarketOrders(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_price_unavailable(self):
        """Prima del fix: current_price=0 rendeva la condizione del check
        semplicemente falsa e si passava dritti a EXECUTE."""
        state = _valid_state(
            order_type="BUY", entry=4330.0, current_price=0.0, current_price_is_futures=False,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")
        self.assertIn("non disponibile", state.decision_reason)

    async def test_skips_when_price_scale_not_verified(self):
        """current_price da un fallback spot (is_futures=False): la scala
        rispetto a entry (futures) non è verificabile, non è sicuro
        confrontarli — deve fallire in modo prudente, non eseguire alla cieca."""
        state = _valid_state(
            order_type="BUY", entry=4330.0, current_price=4330.5, current_price_is_futures=False,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")
        self.assertIn("scala non verificabile", state.decision_reason)

    async def test_executes_when_price_close_and_scale_verified(self):
        state = _valid_state(
            order_type="BUY", entry=4330.0, current_price=4330.2, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "EXECUTE")

    async def test_skips_when_price_too_far_even_with_scale_verified(self):
        # 50 pip = $5.0 di soglia; $10 di distanza deve bloccare.
        state = _valid_state(
            order_type="BUY", entry=4330.0, current_price=4340.0, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")
        self.assertIn("troppo lontana", state.decision_reason)


class TestRegola5bisPendingOrders(unittest.IsolatedAsyncioTestCase):
    async def test_h4_buy_limit_within_threshold_executes(self):
        """Caso reale del 2026-09-03: BUY LIMIT H4 a 3.7x la distanza
        entry-SL (soglia H4 = 4.0x) deve poter aprirsi."""
        entry, sl = 4329.31, 4299.11
        sl_distance = entry - sl
        price = entry + 3.7 * sl_distance
        state = _valid_state(
            order_type="BUY LIMIT", timeframe="4h",
            entry=entry, sl=sl, current_price=price, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "EXECUTE")

    async def test_h4_buy_limit_beyond_threshold_skips(self):
        entry, sl = 4329.31, 4299.11
        sl_distance = entry - sl
        price = entry + 4.5 * sl_distance  # oltre la soglia H4 (4.0x)
        state = _valid_state(
            order_type="BUY LIMIT", timeframe="4h",
            entry=entry, sl=sl, current_price=price, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")
        self.assertIn("cancellata subito dal monitor", state.decision_reason)

    async def test_m15_buy_limit_same_distance_skips(self):
        """La stessa identica distanza che passa su H4 deve bloccarsi su
        M15 (soglia 1.5x, molto più stretta)."""
        entry, sl = 4329.31, 4299.11
        sl_distance = entry - sl
        price = entry + 3.7 * sl_distance
        state = _valid_state(
            order_type="BUY LIMIT", timeframe="15min",
            entry=entry, sl=sl, current_price=price, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")

    async def test_pending_order_not_blocked_by_unverified_scale(self):
        """Scelta di design deliberata (diversa dai MARKET order): se la
        scala del prezzo live non è verificabile, il pending non viene
        bloccato per questo — il controllo di distanza viene solo saltato,
        e lo stesso setup verrà comunque ricontrollato al giro successivo
        dal monitor (check_limit_invalidation, quello sì scale-aware)."""
        entry, sl = 4329.31, 4299.11
        price = entry + 100 * (entry - sl)  # distanza enorme, irrilevante qui
        state = _valid_state(
            order_type="BUY LIMIT", timeframe="4h",
            entry=entry, sl=sl, current_price=price, current_price_is_futures=False,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "EXECUTE")

    async def test_sell_limit_adverse_direction(self):
        """SELL LIMIT aspetta un rialzo: allontanarsi = prezzo scende
        sotto l'entry."""
        entry, sl = 4400.0, 4430.0  # SL sopra l'entry per una SELL
        sl_distance = sl - entry
        price = entry - 4.5 * sl_distance  # oltre la soglia H4 (4.0x), verso il basso
        state = _valid_state(
            order_type="SELL LIMIT", signal="SELL", timeframe="4h",
            entry=entry, sl=sl, current_price=price, current_price_is_futures=True,
        )
        await agent_decision_maker(state)
        self.assertEqual(state.final_decision, "SKIP")


if __name__ == "__main__":
    unittest.main()
