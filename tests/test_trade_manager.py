"""
Test per trade_manager.py — copre il ciclo di vita del trade (apertura,
chiusura, cancellazione, invalidazione pending) e in particolare le race
condition e i bug di correttezza trovati e corretti nella sessione del
2026-09-03: guardia status='OPEN' su _update_trade/close_trade, rowcount
non verificato su close_trade, soglia di invalidazione pending per
timeframe, dedup che ignora i CANCELLED.

Ogni test usa un DB SQLite temporaneo isolato (mai il goldbot.db reale) —
vedi setUp/tearDown.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_manager as tm


def _base_trade_data(**overrides) -> dict:
    data = {
        "signal": "BUY", "order_type": "BUY LIMIT", "entry": 4329.31, "sl": 4299.11,
        "tp1": 4359.51, "tp2": 4389.71, "tp3": 4425.95, "prob": 58, "regime": "NORMAL",
        "timeframe": "4h", "price": 4426.60, "risk_pct": 1.0, "strategies": {},
        "data_timestamp": "2026-09-03T11:00:00", "price_basis": 0.0, "early_be_level": 0,
    }
    data.update(overrides)
    data["setup_key"] = tm.build_setup_key(data)
    return data


class TradeManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        # ACTIVE_FILE è un path fisso calcolato una volta sola all'import
        # (str(BOT_DIR / "active_trades.json")), NON derivato da DB_PATH:
        # senza ripuntarlo anche lui, _sync_active_snapshot() (chiamata da
        # open_trade/close_trade) scriverebbe nel vero file locale del
        # progetto invece che in un file temporaneo — successo davvero
        # scrivendo questi test, corretto qui.
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)


class TestOpenCloseLifecycle(TradeManagerTestCase):
    def test_open_and_close_win(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        self.assertTrue(tm.close_trade(trade_id, "WIN_TP1", data["tp1"], "test"))

        # Rilettura diretta per non dipendere da funzioni di lettura extra
        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()
        self.assertEqual(row["status"], "CLOSED")
        self.assertEqual(row["result"], "WIN_TP1")
        self.assertEqual(row["tp1_hit"], 1)
        self.assertAlmostEqual(row["pnl_r"], 1.0)

    def test_close_trade_twice_second_call_is_noop(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        self.assertTrue(tm.close_trade(trade_id, "WIN_TP1", data["tp1"], "first"))
        # Un secondo close_trade sullo stesso trade_id non deve né sollevare
        # eccezioni né sovrascrivere il risultato già registrato.
        self.assertFalse(tm.close_trade(trade_id, "LOSS", data["sl"], "second"))

    def test_close_trade_returns_false_if_row_vanishes_between_select_and_update(self):
        """Simula la race trovata il 2026-09-03: la riga sparisce (es. un
        DELETE concorrente, tipo /api/reset) tra la SELECT e la UPDATE
        dentro close_trade. Prima del fix, close_trade tornava comunque
        True senza verificare se l'UPDATE avesse davvero toccato una riga.

        close_trade fa SELECT poi (tra le altre cose) chiama
        calculate_trade_pips() PRIMA della UPDATE, sulla stessa connessione:
        è il gancio giusto per iniettare, da una connessione separata, la
        'cancellazione concorrente' esattamente nella finestra reale."""
        data = _base_trade_data()
        trade_id = tm.open_trade(data)

        import sqlite3
        real_calculate_pips = tm.calculate_trade_pips

        def _calculate_pips_and_delete_concurrently(signal, entry, exit_price):
            side_conn = sqlite3.connect(self.tmpdb)
            side_conn.execute("DELETE FROM trades WHERE trade_id=?", (trade_id,))
            side_conn.commit()
            side_conn.close()
            return real_calculate_pips(signal, entry, exit_price)

        tm.calculate_trade_pips = _calculate_pips_and_delete_concurrently
        try:
            result = tm.close_trade(trade_id, "WIN_TP1", data["tp1"], "test")
        finally:
            tm.calculate_trade_pips = real_calculate_pips

        self.assertFalse(result, "close_trade deve tornare False se l'UPDATE non ha toccato nessuna riga")


class TestCloseTradeEarly(TradeManagerTestCase):
    """CLOSED_EARLY (aggiunto il 2026-09-04 per la chiusura protettiva
    pre-evento): a differenza di tutti gli altri risultati (livelli fissi
    TP1/TP2/TP3/SL/BE, R categorico) e di CANCELLED (0R fisso, nessun
    rischio reale), qui il trade era davvero attivo e va chiuso a un
    prezzo qualunque tra entry e sl — l'R deve riflettere quel prezzo
    reale, non essere forzato a 0 come nasconderebbe un guadagno o una
    perdita parziale veri."""

    def test_buy_closed_early_computes_proportional_negative_r(self):
        data = _base_trade_data(signal="BUY", order_type="BUY", entry=4329.31, sl=4299.11)
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        exit_price = 4315.0  # tra entry e sl: sfavorevole ma sl non toccato
        self.assertTrue(tm.close_trade(trade_id, "CLOSED_EARLY", exit_price, "test"))

        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()

        self.assertEqual(row["result"], "CLOSED_EARLY")
        expected_r = (exit_price - 4329.31) / (4329.31 - 4299.11)
        self.assertAlmostEqual(row["pnl_r"], expected_r, places=3)
        self.assertLess(row["pnl_r"], 0)
        self.assertGreater(row["pnl_r"], -1.0)  # peggiore di 0 ma non un LOSS pieno
        self.assertNotEqual(row["pips"], 0.0)   # pips reali, non azzerati come CANCELLED

    def test_sell_closed_early_computes_proportional_positive_r(self):
        data = _base_trade_data(signal="SELL", order_type="SELL", entry=4400.0, sl=4430.0)
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        exit_price = 4390.0  # favorevole per una SELL

        self.assertTrue(tm.close_trade(trade_id, "CLOSED_EARLY", exit_price, "test"))
        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()

        expected_r = -1 * (exit_price - 4400.0) / (4430.0 - 4400.0)
        self.assertAlmostEqual(row["pnl_r"], expected_r, places=3)
        self.assertGreater(row["pnl_r"], 0)

    def test_closed_early_does_not_mark_any_target_hit(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        self.assertTrue(tm.close_trade(trade_id, "CLOSED_EARLY", 4315.0, "test"))

        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()
        self.assertEqual(row["tp1_hit"], 0)
        self.assertEqual(row["tp2_hit"], 0)
        self.assertEqual(row["tp3_hit"], 0)
        self.assertEqual(row["be_hit"], 0)


class TestAmendClosedTrade(TradeManagerTestCase):
    """amend_closed_trade: correzione amministrativa di un trade GIA'
    chiuso (es. LOSS reale preso da uno SL durante un evento macro,
    corretto a posteriori a CLOSED_EARLY come se la chiusura protettiva
    pre-evento del 2026-09-04 fosse esistita già a quel momento) — a
    differenza di close_trade() (solo su status='OPEN'), richiede il
    trade già CLOSED e ricalcola anche i contatori di sessione."""

    def test_amends_result_price_pips_and_r(self):
        data = _base_trade_data(signal="BUY", order_type="BUY", entry=4481.19, sl=4447.11)
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        tm.close_trade(trade_id, "LOSS", 4447.11, "Stop loss raggiunto")

        self.assertTrue(tm.amend_closed_trade(trade_id, "CLOSED_EARLY", 4469.0, "Corretto a posteriori"))

        row = tm.get_trade_by_id(trade_id)
        self.assertEqual(row["result"], "CLOSED_EARLY")
        self.assertEqual(row["exit_price"], 4469.0)
        expected_r = (4469.0 - 4481.19) / (4481.19 - 4447.11)
        self.assertAlmostEqual(row["pnl_r"], expected_r, places=3)
        expected_pips = (4469.0 - 4481.19) / 0.10
        self.assertAlmostEqual(row["pips"], round(expected_pips, 1), places=1)

    def test_rejects_trade_still_open(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        # non chiuso: deve rifiutare, non correggere un trade ancora vivo
        self.assertFalse(tm.amend_closed_trade(trade_id, "CLOSED_EARLY", 4315.0, "test"))

    def test_rejects_unknown_trade_id(self):
        self.assertFalse(tm.amend_closed_trade("non-esiste", "CLOSED_EARLY", 4315.0, "test"))

    def test_session_loss_counter_corrected_when_loss_becomes_non_loss(self):
        data = _base_trade_data(signal="BUY", order_type="BUY", entry=4481.19, sl=4447.11)
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        tm.close_trade(trade_id, "LOSS", 4447.11, "Stop loss raggiunto")

        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        today = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(today["losses"], 1)
        self.assertEqual(today["consecutive_losses"], 1)
        self.assertAlmostEqual(today["pnl_r"], -1.0)
        conn.close()

        tm.amend_closed_trade(trade_id, "CLOSED_EARLY", 4469.0, "Corretto")

        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        after = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        # CLOSED_EARLY non e' una LOSS: il contatore va decrementato, non
        # lasciato a 1 come se il trade fosse ancora classificato perdente.
        self.assertEqual(after["losses"], 0)
        self.assertEqual(after["consecutive_losses"], 0)
        expected_r = (4469.0 - 4481.19) / (4481.19 - 4447.11)
        self.assertAlmostEqual(after["pnl_r"], expected_r, places=3)


class TestUpdateTradeOpenGuard(TradeManagerTestCase):
    def test_mark_tp_hit_is_noop_on_closed_trade(self):
        """_update_trade (usata da mark_tp1_hit/mark_tp2_hit/mark_tp3_hit)
        deve ignorare un trade già CLOSED, non riscriverne i campi."""
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        self.assertTrue(tm.close_trade(trade_id, "LOSS", data["sl"], "test"))

        tm.mark_tp3_hit(trade_id)  # non deve sollevare, e non deve avere effetto

        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()
        self.assertEqual(row["result"], "LOSS")
        self.assertEqual(
            row["tp3_hit"], 0,
            "mark_tp3_hit su un trade già chiuso come LOSS non deve settare tp3_hit=1 "
            "(altrimenti risultato incoerente: LOSS con TP3 raggiunto)",
        )

    def test_mark_tp_hit_works_on_open_trade(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        tm.mark_tp1_hit(trade_id)

        import sqlite3
        conn = sqlite3.connect(self.tmpdb)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
        conn.close()
        self.assertEqual(row["tp1_hit"], 1)


class TestSetupDedup(TradeManagerTestCase):
    def test_cancelled_setup_can_be_retried(self):
        """Fix del 2026-09-03: un pending CANCELLED (prezzo troppo lontano,
        mai partito) non deve bloccare un nuovo tentativo sullo stesso setup
        più tardi nella stessa candela — solo un WIN/LOSS/BE reale blocca."""
        data = _base_trade_data()
        self.assertFalse(tm.was_setup_seen(data["setup_key"]))

        trade_id_1 = tm.open_trade(data)
        tm.close_trade(trade_id_1, "CANCELLED", data["entry"], "prezzo troppo lontano")
        self.assertFalse(
            tm.was_setup_seen(data["setup_key"]),
            "un setup CANCELLED non deve risultare 'visto' — deve poter essere ritentato",
        )

        data2 = dict(data)
        data2["sl"] = 4297.79  # leggermente diverso, come nel caso reale osservato
        trade_id_2 = tm.open_trade(data2)  # non deve sollevare DuplicateSetupError
        self.assertNotEqual(trade_id_1, trade_id_2)

    def test_open_setup_blocks_duplicate(self):
        """Un trade ancora OPEN (o con esito reale WIN/LOSS/BE) DEVE
        continuare a bloccare un secondo tentativo identico."""
        data = _base_trade_data()
        tm.open_trade(data)
        self.assertTrue(tm.was_setup_seen(data["setup_key"]))
        with self.assertRaises(tm.DuplicateSetupError):
            tm.open_trade(dict(data))

    def test_win_setup_blocks_duplicate_forever(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        tm.close_trade(trade_id, "WIN_TP3", data["tp3"], "test")
        self.assertTrue(
            tm.was_setup_seen(data["setup_key"]),
            "un esito reale (WIN/LOSS/BE) deve restare bloccato per sempre, mai CANCELLED",
        )


class TestPendingInvalidation(TradeManagerTestCase):
    def _pending_trade(self, timeframe: str, signal: str = "BUY", order_type: str = "BUY LIMIT",
                        entry: float = 4329.31, sl: float = 4299.11, minutes_ago: float = 1.0) -> dict:
        ts = (datetime.now(tm.TIMEZONE) - timedelta(minutes=minutes_ago)).isoformat()
        return {
            "timestamp": ts, "timeframe": timeframe, "signal": signal,
            "order_type": order_type, "entry": entry, "sl": sl,
        }

    def test_h4_tolerates_further_adverse_move_than_m15(self):
        """Caso reale del 2026-09-03: un BUY LIMIT H4 a 3.7x la distanza
        entry-SL non deve invalidarsi (soglia H4 = 4.0x), mentre lo stesso
        identico scarto su M15 (soglia 1.5x) deve invalidarsi."""
        entry, sl = 4329.31, 4299.11
        sl_distance = entry - sl  # 30.20
        price = entry + 3.7 * sl_distance  # adverso di 3.7x

        h4_trade = self._pending_trade("4h", entry=entry, sl=sl)
        self.assertFalse(
            tm.check_limit_invalidation(h4_trade, price),
            "un BUY LIMIT H4 a 3.7x la distanza entry-SL non deve invalidarsi (soglia 4.0x)",
        )

        m15_trade = self._pending_trade("15min", entry=entry, sl=sl)
        self.assertTrue(
            tm.check_limit_invalidation(m15_trade, price),
            "lo stesso scarto su M15 (soglia 1.5x) deve invalidarsi",
        )

    def test_expires_by_time_regardless_of_price(self):
        trade = self._pending_trade("15min", minutes_ago=200)  # oltre i 90 min di TTL per 15min
        self.assertTrue(tm.check_limit_invalidation(trade, price=4329.31))  # prezzo = entry, nessuno scarto

    def test_adverse_distance_direction_buy_limit(self):
        # BUY LIMIT aspetta un ribasso: allontanarsi = prezzo sale sopra l'entry.
        trade = {"signal": "BUY", "order_type": "BUY LIMIT", "entry": 100.0}
        self.assertEqual(tm._pending_adverse_distance(trade, 105.0), 5.0)
        self.assertEqual(tm._pending_adverse_distance(trade, 95.0), 0.0)  # verso l'attivazione, non avverso

    def test_adverse_distance_direction_sell_limit(self):
        # SELL LIMIT aspetta un rialzo: allontanarsi = prezzo scende sotto l'entry.
        trade = {"signal": "SELL", "order_type": "SELL LIMIT", "entry": 100.0}
        self.assertEqual(tm._pending_adverse_distance(trade, 95.0), 5.0)
        self.assertEqual(tm._pending_adverse_distance(trade, 105.0), 0.0)


class TestPostTradeAnalysisForwardsTradeId(TradeManagerTestCase):
    """Regression (audit 2026-09-05): _post_trade_analysis() aveva trade_id
    disponibile in tutti e 3 i punti di chiamata (BE/SL/TP3) ma non lo
    passava mai ad analyze_last_trade() - cadeva sempre sul fallback
    "ultimo trade nel DB". Con due trade chiusi quasi in contemporanea
    (asyncio.create_task + sleep(1)) entrambe le analisi finivano per
    descrivere lo stesso trade."""

    async def _run(self, trade_id):
        bot = AsyncMock()
        with patch("self_learning.analyze_last_trade", return_value="ok") as mock_analyze, \
             patch("self_learning.format_learning_report"), \
             patch("self_learning.optimize_strategy_weights"), \
             patch("asyncio.sleep", new=AsyncMock()):
            await tm._post_trade_analysis(bot, "12345", trade_id)
        return mock_analyze

    def test_forwards_the_trade_id_that_just_closed(self):
        import asyncio
        mock_analyze = asyncio.run(self._run("real-trade-id-abc"))
        mock_analyze.assert_called_once_with("real-trade-id-abc")


class TestDecisionLog(TradeManagerTestCase):
    """Log strutturato interrogabile di ogni decisione EXECUTE/WAIT/SKIP
    (Fase A del 2026-09-04: osservabilità pura, nessun impatto sul trading)."""

    def test_log_and_retrieve_roundtrip(self):
        tm.log_decision("1h", "BUY", "NORMAL", 72, "EXECUTE", "Setup valido")
        rows = tm.get_recent_decisions(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timeframe"], "1h")
        self.assertEqual(rows[0]["signal"], "BUY")
        self.assertEqual(rows[0]["prob"], 72)
        self.assertEqual(rows[0]["decision"], "EXECUTE")
        self.assertEqual(rows[0]["reason"], "Setup valido")

    def test_most_recent_first(self):
        tm.log_decision("1h", "BUY", "NORMAL", 60, "SKIP", "primo")
        tm.log_decision("1h", "SELL", "NORMAL", 60, "SKIP", "secondo")
        rows = tm.get_recent_decisions(limit=10)
        self.assertEqual(rows[0]["reason"], "secondo")
        self.assertEqual(rows[1]["reason"], "primo")

    def test_filter_by_timeframe(self):
        tm.log_decision("1h", "BUY", "NORMAL", 60, "EXECUTE", "h1")
        tm.log_decision("4h", "BUY", "NORMAL", 60, "EXECUTE", "h4")
        rows = tm.get_recent_decisions(timeframe="4h", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "h4")

    def test_filter_by_decision(self):
        tm.log_decision("1h", "BUY", "NORMAL", 60, "EXECUTE", "eseguito")
        tm.log_decision("1h", "BUY", "RANGING", 60, "SKIP", "bloccato")
        rows = tm.get_recent_decisions(decision="SKIP", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "bloccato")

    def test_limit_is_respected(self):
        for i in range(5):
            tm.log_decision("1h", "BUY", "NORMAL", 60, "SKIP", f"n{i}")
        rows = tm.get_recent_decisions(limit=2)
        self.assertEqual(len(rows), 2)

    def test_does_not_raise_on_missing_prob(self):
        tm.log_decision("1h", "NEUTRAL", "", None, "SKIP", "nessun segnale")
        rows = tm.get_recent_decisions(limit=1)
        self.assertIsNone(rows[0]["prob"])


class TestStrategyVersionOnOpenTrade(TradeManagerTestCase):
    """open_trade() calcola strategy_version internamente se assente
    (Fase A, 2026-09-04) — nessuno dei due punti chiamanti in gold_bot.py
    deve doversene ricordare."""

    def test_computes_fingerprint_when_not_provided(self):
        import agent_orchestrator
        trade_id = tm.open_trade(_base_trade_data())
        row = tm.get_trade_by_id(trade_id)
        self.assertEqual(row["strategy_version"], agent_orchestrator.get_strategy_fingerprint())

    def test_respects_explicit_value_if_provided(self):
        trade_id = tm.open_trade(_base_trade_data(strategy_version="custom-abc123"))
        row = tm.get_trade_by_id(trade_id)
        self.assertEqual(row["strategy_version"], "custom-abc123")


class TestCalculateTradePips(unittest.TestCase):
    def test_buy_direction(self):
        self.assertAlmostEqual(tm.calculate_trade_pips("BUY", 4300.00, 4310.00), 100.0, places=1)
        self.assertAlmostEqual(tm.calculate_trade_pips("BUY", 4300.00, 4290.00), -100.0, places=1)

    def test_sell_direction(self):
        self.assertAlmostEqual(tm.calculate_trade_pips("SELL", 4300.00, 4290.00), 100.0, places=1)
        self.assertAlmostEqual(tm.calculate_trade_pips("SELL", 4300.00, 4310.00), -100.0, places=1)

    def test_case_insensitive_signal(self):
        self.assertAlmostEqual(
            tm.calculate_trade_pips("buy", 4300.00, 4310.00),
            tm.calculate_trade_pips("BUY", 4300.00, 4310.00),
        )


if __name__ == "__main__":
    unittest.main()
