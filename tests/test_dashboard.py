"""
Test per dashboard.py — copre compute_stats() (win rate/pips aggregati) e
la rimozione automatica dei trade CANCELLED dalla dashboard (2026-09-07,
close_trade() li elimina invece di marcarli — vedi anche
TestCancelledTradeIsDeleted in test_trade_manager.py).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashboard as db
import trade_manager as tm


def _trade(**overrides) -> dict:
    base = {
        "status": "CLOSED", "result": "WIN_TP1", "signal": "BUY",
        "entry": 4300.0, "exit_price": 4310.0, "pips": None,
        "pnl_r": 1.0, "tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0,
    }
    base.update(overrides)
    return base


class TestComputeStats(unittest.TestCase):
    def test_win_rate_counts_win_be_with_tp1_as_win(self):
        trades = [
            _trade(result="WIN_TP1", tp1_hit=1),
            _trade(result="WIN_BE", tp1_hit=1, pnl_r=0.0),   # TP1 raggiunto poi BE -> win
            _trade(result="WIN_BE", tp1_hit=0, pnl_r=0.0),   # BE puro -> neutro, non conta
            _trade(result="LOSS", pnl_r=-1.0),
        ]
        stats = db.compute_stats(trades)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["win_rate"], round(2 / 3 * 100, 1))

    def test_cancelled_excluded_from_stats(self):
        trades = [
            _trade(result="WIN_TP1", tp1_hit=1),
            _trade(result="CANCELLED", pnl_r=0.0, status="CLOSED"),
        ]
        stats = db.compute_stats(trades)
        self.assertEqual(stats["closed_total"], 1, "un CANCELLED non deve contare come trade chiuso")
        self.assertEqual(stats["win_rate"], 100.0)

    def test_win_rate_zero_when_no_decisive_trades(self):
        trades = [_trade(result="WIN_BE", tp1_hit=0, pnl_r=0.0)]
        stats = db.compute_stats(trades)
        self.assertEqual(stats["win_rate"], 0)


class TestSimulatedBudget(unittest.TestCase):
    """2026-09-16: budget simulato $ in dashboard (richiesta esplicita
    dell'utente) — un lotto FISSO di riferimento applicato ai prezzi
    entry/exit_price REALI già salvati per ogni trade, per vedere quanto
    avrebbe fruttato/perso ogni segnale senza bisogno di un conto demo
    MT5. Formula verificata contro risk_manager.calculate_lot_size:
    $ = (movimento_prezzo_a_favore - spread) × 100 once/lotto × lotto.
    spread_usd=0 esplicito nei test che isolano solo lotto/direzione, per
    non far dipendere quelle asserzioni dal valore di SIM_SPREAD_USD."""

    def test_pnl_usd_buy_winning(self):
        trade = _trade(signal="BUY", entry=4300.0, exit_price=4310.0)  # +$10
        # 10 * 100 * 0.02 = 20.0
        self.assertEqual(db._trade_pnl_usd(trade, lot_size=0.02, spread_usd=0.0), 20.0)

    def test_pnl_usd_sell_losing(self):
        """Prezzo sale ma il segnale è SELL -> perdita, non guadagno."""
        trade = _trade(signal="SELL", entry=4300.0, exit_price=4310.0)
        self.assertEqual(db._trade_pnl_usd(trade, lot_size=0.02, spread_usd=0.0), -20.0)

    def test_pnl_usd_matches_10_pips_per_001_lot_reference(self):
        """Il caso concreto usato per verificare la formula con l'utente:
        10 pip (= $1.00 di movimento, XAUUSD_PIP_SIZE=0.10) su 0.01 lotto
        deve fare $1.00 esatto, senza spread."""
        trade = _trade(signal="BUY", entry=4300.00, exit_price=4301.00)
        self.assertAlmostEqual(db._trade_pnl_usd(trade, lot_size=0.01, spread_usd=0.0), 1.00, places=6)

    def test_default_lot_size_and_spread_are_module_constants(self):
        trade = _trade(signal="BUY", entry=4300.0, exit_price=4310.0)
        expected = (10.0 - db.SIM_SPREAD_USD) * 100 * db.SIM_LOT_SIZE
        self.assertAlmostEqual(db._trade_pnl_usd(trade), expected, places=6)

    def test_spread_reduces_both_wins_and_losses(self):
        """Lo spread si paga sempre, indipendentemente dall'esito — un
        round-trip attraversa bid/ask una volta sola sia che chiuda in
        utile sia in perdita."""
        win = _trade(signal="BUY", entry=4300.0, exit_price=4310.0)
        loss = _trade(signal="BUY", entry=4300.0, exit_price=4290.0)
        pnl_win = db._trade_pnl_usd(win, lot_size=0.02, spread_usd=0.30)
        pnl_loss = db._trade_pnl_usd(loss, lot_size=0.02, spread_usd=0.30)
        # senza spread sarebbero +20.0 e -20.0; con spread (0.30*100*0.02=0.60)
        # diventano +19.40 e -20.60 — ridotto di 0.60 in ENTRAMBI i casi.
        self.assertAlmostEqual(pnl_win, 19.40, places=6)
        self.assertAlmostEqual(pnl_loss, -20.60, places=6)

    def test_spread_can_flip_a_marginal_win_into_a_loss(self):
        """Caso realistico: un movimento più piccolo dello spread stesso
        deve risultare in una perdita netta anche se la direzione era
        quella giusta — esattamente quello che succede in un conto reale."""
        trade = _trade(signal="BUY", entry=4300.0, exit_price=4300.20)  # +$0.20 lordo
        pnl = db._trade_pnl_usd(trade, lot_size=0.02, spread_usd=0.30)
        self.assertLess(pnl, 0)

    def test_compute_stats_exposes_sim_budget(self):
        trades = [
            _trade(signal="BUY", entry=4300.0, exit_price=4310.0),   # +$10 move
            _trade(signal="SELL", entry=4300.0, exit_price=4290.0),  # +$10 move (SELL wins)
        ]
        stats = db.compute_stats(trades)
        expected_total = 2 * (10.0 - db.SIM_SPREAD_USD) * 100 * db.SIM_LOT_SIZE
        self.assertAlmostEqual(stats["total_usd"], expected_total, places=6)
        self.assertAlmostEqual(
            stats["sim_budget_usd"], db.SIM_STARTING_BUDGET_USD + expected_total, places=6
        )
        self.assertEqual(stats["sim_starting_budget_usd"], db.SIM_STARTING_BUDGET_USD)
        self.assertEqual(stats["sim_lot_size"], db.SIM_LOT_SIZE)
        self.assertEqual(stats["sim_spread_usd"], db.SIM_SPREAD_USD)

    def test_cancelled_trade_contributes_zero_to_sim_budget(self):
        trades = [_trade(result="CANCELLED", entry=4300.0, exit_price=4400.0)]
        stats = db.compute_stats(trades)
        self.assertEqual(stats["total_usd"], 0.0)
        self.assertEqual(stats["sim_budget_usd"], db.SIM_STARTING_BUDGET_USD)


class TestCancelledTradeRemovedFromDashboard(unittest.TestCase):
    """FIX (2026-09-07): un CANCELLED (nessun rischio reale, 0R fisso) non
    deve più restare per sempre in dashboard/DB — close_trade() lo elimina
    subito invece di marcarlo. Prima di questo fix mostrava 0 pips invece
    della distanza ipotetica fino al prezzo di cancellazione (fix del
    2026-09-03); ora la riga non esiste affatto, quindi il problema non si
    pone più."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        # Vedi lo stesso commento in test_trade_manager.py: ACTIVE_FILE è un
        # path fisso, non derivato da DB_PATH — va ripuntato esplicitamente
        # per non scrivere nel vero file locale del progetto.
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        db.DB_PATH = self.tmpdb

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)

    def test_cancelled_trade_disappears_from_dashboard(self):
        data = {
            "signal": "BUY", "order_type": "BUY LIMIT", "entry": 4312.94, "sl": 4296.66,
            "tp1": 4329.22, "tp2": 4345.51, "tp3": 4365.05, "prob": 60, "regime": "NORMAL",
            "timeframe": "1h", "price": 4312.94, "risk_pct": 1.0, "strategies": {},
            "data_timestamp": "2026-09-03T11:00:00", "price_basis": 0.0, "early_be_level": 0,
        }
        data["setup_key"] = tm.build_setup_key(data)
        trade_id = tm.open_trade(data)
        tm.close_trade(trade_id, "CANCELLED", 4440.40, "prezzo troppo lontano")

        # init_db() migra sempre eventuali dati legacy da un file fisso
        # (data/active_trades.json), indipendentemente dal DB_PATH puntato —
        # quindi un DB "temporaneo" può contenere righe preesistenti. Non è
        # un bug da correggere qui: filtriamo sul trade_id creato da questo
        # test invece di assumere un DB vuoto.
        trades = [t for t in db._get_trades() if t.get("trade_id") == trade_id]
        self.assertEqual(
            len(trades), 0,
            "un trade CANCELLED deve sparire dalla dashboard, non solo mostrare 0 pips",
        )


class TestApiCorrectTrade(unittest.TestCase):
    """/api/correct-trade: correzione amministrativa di un trade già
    chiuso (es. LOSS reale preso da uno SL durante un evento macro,
    corretto a posteriori come se la chiusura protettiva pre-evento del
    2026-09-04 fosse già esistita a quel momento). Come /api/reset, deve
    richiedere il token anche da loopback (prima di questo endpoint solo
    /api/reset era escluso dall'esenzione loopback)."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        db.DB_PATH = self.tmpdb
        self._orig_token = db.DASHBOARD_TOKEN
        db.DASHBOARD_TOKEN = "test-token-123"
        self.client = db.app.test_client()

    def tearDown(self):
        db.DASHBOARD_TOKEN = self._orig_token
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)

    def _seed_loss_trade(self) -> str:
        data = {
            "signal": "BUY", "order_type": "BUY", "entry": 4481.19, "sl": 4447.11,
            "tp1": 4515.27, "tp2": 4549.35, "tp3": 4590.25, "prob": 85, "regime": "NORMAL",
            "timeframe": "4h", "price": 4481.19, "risk_pct": 1.0, "strategies": {},
            "data_timestamp": "2026-09-03T14:00:00", "price_basis": 0.0, "early_be_level": 0,
        }
        data["setup_key"] = tm.build_setup_key(data)
        trade_id = tm.open_trade(data)
        tm.activate_trade(trade_id)
        tm.close_trade(trade_id, "LOSS", 4447.11, "Stop loss raggiunto dal monitor virtuale")
        return trade_id

    def test_requires_token_even_from_loopback(self):
        trade_id = self._seed_loss_trade()
        resp = self.client.post("/api/correct-trade", json={
            "trade_id": trade_id, "result": "CLOSED_EARLY", "exit_price": 4469.0,
        })
        self.assertEqual(resp.status_code, 401)
        row = tm.get_trade_by_id(trade_id)
        self.assertEqual(row["result"], "LOSS", "senza token il trade non deve essere toccato")

    def test_corrects_trade_with_valid_token(self):
        trade_id = self._seed_loss_trade()
        resp = self.client.post(
            "/api/correct-trade?token=test-token-123",
            json={"trade_id": trade_id, "result": "CLOSED_EARLY", "exit_price": 4469.0,
                  "notes": "Corretto: la chiusura protettiva pre-evento non esisteva ancora"},
        )
        self.assertEqual(resp.status_code, 200)
        row = tm.get_trade_by_id(trade_id)
        self.assertEqual(row["result"], "CLOSED_EARLY")
        self.assertEqual(row["exit_price"], 4469.0)
        expected_r = (4469.0 - 4481.19) / (4481.19 - 4447.11)
        self.assertAlmostEqual(row["pnl_r"], expected_r, places=3)

    def test_rejects_invalid_result(self):
        trade_id = self._seed_loss_trade()
        resp = self.client.post(
            "/api/correct-trade?token=test-token-123",
            json={"trade_id": trade_id, "result": "NOT_A_REAL_RESULT", "exit_price": 4469.0},
        )
        self.assertEqual(resp.status_code, 400)

    def test_rejects_missing_exit_price(self):
        trade_id = self._seed_loss_trade()
        resp = self.client.post(
            "/api/correct-trade?token=test-token-123",
            json={"trade_id": trade_id, "result": "CLOSED_EARLY"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_unknown_trade_id_returns_404(self):
        resp = self.client.post(
            "/api/correct-trade?token=test-token-123",
            json={"trade_id": "non-esiste", "result": "CLOSED_EARLY", "exit_price": 4469.0},
        )
        self.assertEqual(resp.status_code, 404)


class TestApiEaAckAndFills(unittest.TestCase):
    """/api/ea/ack (con fill_price opzionale) e /api/ea/fills — tracking
    slippage aggiunto il 2026-09-15."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        db.DB_PATH = self.tmpdb
        self._orig_token = db.DASHBOARD_TOKEN
        db.DASHBOARD_TOKEN = "test-token-123"
        self._orig_bridge = tm.EA_BRIDGE_ENABLED
        tm.EA_BRIDGE_ENABLED = True
        self.client = db.app.test_client()

    def tearDown(self):
        db.DASHBOARD_TOKEN = self._orig_token
        tm.EA_BRIDGE_ENABLED = self._orig_bridge
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)

    def _open_pending_trade(self, entry=4300.0) -> str:
        data = {
            "signal": "BUY", "order_type": "BUY", "entry": entry, "sl": 4270.0,
            "tp1": 4340.0, "tp2": 4360.0, "tp3": 4390.0, "prob": 70, "regime": "NORMAL",
            "timeframe": "1h", "price": entry, "risk_pct": 1.0, "strategies": {},
            "data_timestamp": "2026-09-15T09:00:00", "price_basis": 0.0, "early_be_level": 0,
        }
        data["setup_key"] = tm.build_setup_key(data)
        return tm.open_trade(data)

    def test_ack_without_token_is_rejected(self):
        trade_id = self._open_pending_trade()
        resp = self.client.post("/api/ea/ack", json={"trade_id": trade_id})
        self.assertEqual(resp.status_code, 401)

    def test_ack_with_fill_price_is_visible_via_fills_endpoint(self):
        trade_id = self._open_pending_trade(entry=4300.0)
        resp = self.client.post(
            "/api/ea/ack?token=test-token-123",
            json={"trade_id": trade_id, "fill_price": 4300.4},
        )
        self.assertEqual(resp.status_code, 200)

        fills = self.client.get("/api/ea/fills").get_json()
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["trade_id"], trade_id)
        self.assertAlmostEqual(fills[0]["slippage"], 0.4, places=2)

    def test_ack_without_fill_price_leaves_fills_empty(self):
        trade_id = self._open_pending_trade()
        resp = self.client.post("/api/ea/ack?token=test-token-123", json={"trade_id": trade_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get("/api/ea/fills").get_json(), [])


class TestApiEaRequeue(unittest.TestCase):
    """/api/ea/requeue (2026-09-17): bug reale trovato in produzione — un
    bug nell'EA (GoldMindCopier.mq5) confermava /api/ea/ack anche quando
    l'apertura dell'ordine falliva (scadenza pending non supportata dal
    broker), quindi un BUY LIMIT spariva per sempre dalla coda senza mai
    arrivare sul conto MT5, pur restando regolarmente OPEN nella
    simulazione paper. Questo endpoint rimette a mano in coda un trade
    del genere prendendo i dati direttamente dal trade già registrato."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        db.DB_PATH = self.tmpdb
        self._orig_token = db.DASHBOARD_TOKEN
        db.DASHBOARD_TOKEN = "test-token-123"
        self._orig_bridge = tm.EA_BRIDGE_ENABLED
        tm.EA_BRIDGE_ENABLED = True
        self.client = db.app.test_client()

    def tearDown(self):
        db.DASHBOARD_TOKEN = self._orig_token
        tm.EA_BRIDGE_ENABLED = self._orig_bridge
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)

    def _open_pending_limit_trade(self, entry=4284.60) -> str:
        data = {
            "signal": "BUY", "order_type": "BUY LIMIT", "entry": entry, "sl": 4248.99,
            "tp1": 4320.21, "tp2": 4355.81, "tp3": 4398.54, "prob": 68, "regime": "NORMAL",
            "timeframe": "4h", "price": entry, "risk_pct": 1.0, "strategies": {},
            "data_timestamp": "2026-09-17T08:00:00", "price_basis": 0.0, "early_be_level": 0,
        }
        data["setup_key"] = tm.build_setup_key(data)
        return tm.open_trade(data)

    def test_requires_token_even_from_loopback(self):
        trade_id = self._open_pending_limit_trade()
        tm.ack_broker_order(trade_id)  # simula il bug reale: già tolto dalla coda
        resp = self.client.post("/api/ea/requeue", json={"trade_id": trade_id})
        self.assertEqual(resp.status_code, 401)
        self.assertNotIn(trade_id, tm.load_broker_orders_pending())

    def test_requeues_an_open_trade_with_its_real_data(self):
        trade_id = self._open_pending_limit_trade(entry=4284.60)
        tm.ack_broker_order(trade_id)  # simula il bug reale: già tolto dalla coda
        self.assertNotIn(trade_id, tm.load_broker_orders_pending())

        resp = self.client.post("/api/ea/requeue?token=test-token-123", json={"trade_id": trade_id})
        self.assertEqual(resp.status_code, 200)

        pending = tm.load_broker_orders_pending()
        self.assertIn(trade_id, pending)
        self.assertEqual(pending[trade_id]["order_type"], "BUY LIMIT")
        self.assertAlmostEqual(pending[trade_id]["entry"], 4284.60, places=2)

    def test_unknown_trade_id_returns_404(self):
        resp = self.client.post(
            "/api/ea/requeue?token=test-token-123", json={"trade_id": "non-esiste"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_missing_trade_id_returns_400(self):
        resp = self.client.post("/api/ea/requeue?token=test-token-123", json={})
        self.assertEqual(resp.status_code, 400)

    def test_closed_trade_cannot_be_requeued(self):
        trade_id = self._open_pending_limit_trade()
        tm.ack_broker_order(trade_id)
        tm.activate_trade(trade_id)
        tm.close_trade(trade_id, "WIN_TP1", 4320.21, "TP1 raggiunto")

        resp = self.client.post("/api/ea/requeue?token=test-token-123", json={"trade_id": trade_id})
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn(trade_id, tm.load_broker_orders_pending())


class TestMacroEventOutcomes(unittest.TestCase):
    """Tracciamento eventi macro in dashboard (2026-09-16, richiesta
    esplicita: "aggiungi nella dashboard gli eventi che si verificano a
    mercato e se il bot li prende... deve pure dare confermato o non
    confermato nel messaggio bias post evento come fa per quello bias
    evento") — /api/macro-event/manual per il caso non-automatico (bot
    riavviato durante la finestra), /api/data per l'esposizione in lettura."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        db.DB_PATH = self.tmpdb
        self._orig_token = db.DASHBOARD_TOKEN
        db.DASHBOARD_TOKEN = "test-token-123"
        self.client = db.app.test_client()

    def tearDown(self):
        db.DASHBOARD_TOKEN = self._orig_token
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)

    def test_requires_token_even_from_loopback(self):
        resp = self.client.post("/api/macro-event/manual", json={
            "group_key": "2026-09-16_20:00_USD", "title": "FOMC Statement",
            "event_time": "2026-09-16T20:00:00+02:00", "bias": "SELL",
            "price_pre_event": 4347.10,
        })
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(tm.get_macro_event_outcomes(), [])

    def test_manual_pre_only_seed_shows_as_pending(self):
        resp = self.client.post(
            "/api/macro-event/manual?token=test-token-123",
            json={
                "group_key": "2026-09-16_20:00_USD", "title": "FOMC Statement",
                "currency": "USD", "impact": "HIGH",
                "event_time": "2026-09-16T20:00:00+02:00", "bias": "SELL",
                "price_pre_event": 4347.10,
            },
        )
        self.assertEqual(resp.status_code, 200)
        rows = tm.get_macro_event_outcomes()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bias"], "SELL")
        self.assertIsNone(rows[0]["esito_post"])

    def test_manual_full_seed_with_confirmed_outcome(self):
        resp = self.client.post(
            "/api/macro-event/manual?token=test-token-123",
            json={
                "group_key": "2026-09-16_20:00_USD",
                "title": "Federal Funds Rate + FOMC Economic Projections + FOMC Statement",
                "currency": "USD", "impact": "HIGH",
                "event_time": "2026-09-16T20:00:00+02:00", "bias": "SELL",
                "price_pre_event": 4347.10,
                "price_immediate": 4318.40, "change_immediate": -28.70,
                "esito_immediate": "CONFERMATO",
                "price_post": 4318.40, "change_post": -28.70,
                "esito_post": "CONFERMATO", "minutes_post": 2,
            },
        )
        self.assertEqual(resp.status_code, 200)
        rows = tm.get_macro_event_outcomes()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["esito_immediate"], "CONFERMATO")
        self.assertEqual(row["esito_post"], "CONFERMATO")
        self.assertAlmostEqual(row["change_post"], -28.70, places=2)
        self.assertEqual(row["minutes_post"], 2)

    def test_missing_required_field_returns_400(self):
        resp = self.client.post(
            "/api/macro-event/manual?token=test-token-123",
            json={"group_key": "", "title": "FOMC Statement",
                  "event_time": "2026-09-16T20:00:00+02:00", "bias": "SELL",
                  "price_pre_event": 4347.10},
        )
        self.assertEqual(resp.status_code, 400)

    def test_api_data_exposes_macro_events(self):
        tm.save_macro_event_pre(
            "2026-09-16_20:00_USD", "FOMC Statement", "USD", "HIGH",
            "2026-09-16T20:00:00+02:00", 4347.10, "SELL",
        )
        resp = self.client.get("/api/data")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertIn("macro_events", payload)
        self.assertEqual(len(payload["macro_events"]), 1)
        self.assertEqual(payload["macro_events"][0]["title"], "FOMC Statement")


if __name__ == "__main__":
    unittest.main()
