"""
Test per dashboard.py — copre compute_stats() (win rate/pips aggregati) e
la correzione del 2026-09-03 per cui i trade CANCELLED mostrano 0 pips
invece della distanza ipotetica fino al prezzo di cancellazione.
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


class TestCancelledPipsDisplay(unittest.TestCase):
    """_get_trades() deve mostrare 0 pips per i CANCELLED, senza toccare
    il dato salvato in DB (solo visivo, vedi commento in dashboard.py)."""

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

    def test_cancelled_trade_shows_zero_pips(self):
        data = {
            "signal": "BUY", "order_type": "BUY LIMIT", "entry": 4312.94, "sl": 4296.66,
            "tp1": 4329.22, "tp2": 4345.51, "tp3": 4365.05, "prob": 60, "regime": "NORMAL",
            "timeframe": "1h", "price": 4312.94, "risk_pct": 1.0, "strategies": {},
            "data_timestamp": "2026-09-03T11:00:00", "price_basis": 0.0, "early_be_level": 0,
        }
        data["setup_key"] = tm.build_setup_key(data)
        trade_id = tm.open_trade(data)
        # Cancellato con un exit_price ben lontano dall'entry: senza il fix
        # calculate_trade_pips calcolerebbe un valore diverso da zero.
        tm.close_trade(trade_id, "CANCELLED", 4440.40, "prezzo troppo lontano")

        # init_db() migra sempre eventuali dati legacy da un file fisso
        # (data/active_trades.json), indipendentemente dal DB_PATH puntato —
        # quindi un DB "temporaneo" può contenere righe preesistenti. Non è
        # un bug da correggere qui: filtriamo sul trade_id creato da questo
        # test invece di assumere un DB vuoto.
        trades = [t for t in db._get_trades() if t.get("trade_id") == trade_id]
        self.assertEqual(len(trades), 1)
        self.assertEqual(
            trades[0]["pips"], 0.0,
            "un trade CANCELLED deve mostrare 0 pips in dashboard, non la distanza ipotetica",
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


if __name__ == "__main__":
    unittest.main()
