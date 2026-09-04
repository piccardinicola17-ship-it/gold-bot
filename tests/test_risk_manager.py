"""
Test per risk_manager.py — il file che decide se un trade può aprirsi
(soglie prob/R:R, cooldown dopo perdite consecutive, esposizione massima
per direzione, sizing). Mai testato prima. risk_manager._connect è lo
stesso oggetto funzione di trade_manager._connect (importato con `from
trade_manager import _connect`): punta a DB_PATH del modulo trade_manager
in cui è definito, quindi lo stesso pattern di isolamento (tm.DB_PATH +
tm.ACTIVE_FILE su un temporaneo) funziona anche qui.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_manager as tm
import risk_manager as rm


def _open_and_close(signal="BUY", timeframe="4h", result="WIN_TP1", entry=4300.0, sl=4270.0,
                     closed_at=None) -> str:
    """Apre e chiude un trade via trade_manager, per popolare il DB con
    dati realistici invece di INSERT manuali — riusa la logica già testata
    in test_trade_manager.py."""
    data = {
        "signal": signal, "order_type": signal, "entry": entry, "sl": sl,
        "tp1": entry + 30 if signal == "BUY" else entry - 30,
        "tp2": entry + 60 if signal == "BUY" else entry - 60,
        "tp3": entry + 90 if signal == "BUY" else entry - 90,
        "prob": 60, "regime": "NORMAL", "timeframe": timeframe,
        "price": entry, "risk_pct": 1.0, "strategies": {},
        "data_timestamp": datetime.now(tm.TIMEZONE).isoformat(), "price_basis": 0.0,
        "early_be_level": 0,
    }
    data["setup_key"] = tm.build_setup_key(data)
    trade_id = tm.open_trade(data)
    exit_price = data["tp1"] if result == "WIN_TP1" else sl
    tm.close_trade(trade_id, result, exit_price, "test")
    if closed_at is not None:
        with tm._connect() as conn:
            conn.execute(
                "UPDATE trades SET closed_at=? WHERE trade_id=?",
                (closed_at, trade_id),
            )
    return trade_id


class RiskManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
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


class TestCalculateLotSize(unittest.TestCase):
    def test_normal_case_is_tradable(self):
        result = rm.calculate_lot_size(account_balance=10000, risk_pct=1.0, entry=4300.0, sl=4290.0)
        self.assertTrue(result["tradable"])
        self.assertGreaterEqual(result["lot_size"], rm.MIN_LOT)
        self.assertLessEqual(result["actual_risk_pct"], 1.0 + 0.01)

    def test_invalid_inputs_not_tradable(self):
        for kwargs in (
            dict(account_balance=0, risk_pct=1.0, entry=4300.0, sl=4290.0),
            dict(account_balance=10000, risk_pct=0, entry=4300.0, sl=4290.0),
            dict(account_balance=10000, risk_pct=1.0, entry=4300.0, sl=4300.0),  # sl_distance=0
        ):
            with self.subTest(kwargs=kwargs):
                result = rm.calculate_lot_size(**kwargs)
                self.assertFalse(result["tradable"])
                self.assertIn("error", result)

    def test_tiny_balance_below_minimum_lot(self):
        # SL molto larga su un capitale piccolo: anche il lotto minimo
        # rischierebbe più del budget concesso.
        result = rm.calculate_lot_size(account_balance=50, risk_pct=1.0, entry=4300.0, sl=4000.0)
        self.assertFalse(result["tradable"])
        self.assertIn("minimum_lot_risk", result)

    def test_lot_size_capped_at_max_lot(self):
        result = rm.calculate_lot_size(account_balance=100_000_000, risk_pct=1.0, entry=4300.0, sl=4299.0)
        self.assertTrue(result["tradable"])
        self.assertLessEqual(result["lot_size"], rm.MAX_LOT)


class TestCooldownExpired(unittest.TestCase):
    def test_not_expired_when_recent(self):
        stopped_at = (datetime.now(rm.TIMEZONE) - timedelta(hours=1)).isoformat()
        self.assertFalse(rm._cooldown_expired({"session_stopped_at": stopped_at}))

    def test_expired_after_cooldown_hours(self):
        stopped_at = (datetime.now(rm.TIMEZONE) - timedelta(hours=rm.COOLDOWN_HOURS + 0.5)).isoformat()
        self.assertTrue(rm._cooldown_expired({"session_stopped_at": stopped_at}))

    def test_missing_timestamp_falls_back_to_todays_midnight(self):
        # Nessun session_stopped_at (DB vecchio): considera il blocco a
        # mezzanotte di oggi — comportamento "safe" esplicito nel codice.
        result = rm._cooldown_expired({"session_stopped_at": ""})
        expected = (datetime.now(rm.TIMEZONE).hour >= rm.COOLDOWN_HOURS)
        self.assertEqual(result, expected)


class TestIsNearNews(unittest.TestCase):
    def test_news_error_is_always_near(self):
        self.assertTrue(rm.is_near_news([], news_error=True))

    def test_no_events_is_not_near(self):
        self.assertFalse(rm.is_near_news([], news_error=False))

    def test_high_impact_event_within_buffer_is_near(self):
        soon = datetime.now(rm.TIMEZONE) + timedelta(minutes=rm.NEWS_BUFFER_MINUTES - 5)
        events = [{"impact": "HIGH", "datetime": soon.isoformat()}]
        self.assertTrue(rm.is_near_news(events))

    def test_high_impact_event_outside_buffer_is_not_near(self):
        far = datetime.now(rm.TIMEZONE) + timedelta(minutes=rm.NEWS_BUFFER_MINUTES + 30)
        events = [{"impact": "HIGH", "datetime": far.isoformat()}]
        self.assertFalse(rm.is_near_news(events))

    def test_low_impact_event_ignored_even_if_imminent(self):
        soon = datetime.now(rm.TIMEZONE) + timedelta(minutes=1)
        events = [{"impact": "LOW", "datetime": soon.isoformat()}]
        self.assertFalse(rm.is_near_news(events))


class TestConsecutiveLosses(RiskManagerTestCase):
    def test_zero_when_no_trades(self):
        self.assertEqual(rm.get_consecutive_losses(), 0)

    def test_counts_trailing_losses_only(self):
        now = datetime.now(tm.TIMEZONE)
        # Ordine cronologico: WIN, poi LOSS, LOSS, LOSS (le più recenti) —
        # deve contare solo le 3 LOSS finali, non quella prima del WIN.
        _open_and_close(result="WIN_TP1", closed_at=(now - timedelta(minutes=40)).isoformat())
        _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=30)).isoformat())
        _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=20)).isoformat())
        _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=10)).isoformat())
        self.assertEqual(rm.get_consecutive_losses(), 3)

    def test_resets_after_a_win(self):
        now = datetime.now(tm.TIMEZONE)
        _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=20)).isoformat())
        _open_and_close(result="WIN_TP2", closed_at=(now - timedelta(minutes=10)).isoformat())
        self.assertEqual(rm.get_consecutive_losses(), 0)


class TestDrawdownMultiplier(RiskManagerTestCase):
    def test_no_trades_no_drawdown_full_risk(self):
        self.assertEqual(rm.get_drawdown_multiplier(), 1.0)

    def test_multiplier_never_reaches_zero(self):
        """dd_multiplier riduce il rischio ma non lo azzera mai — solo 3 SL
        consecutivi bloccano la sessione, non il drawdown."""
        now = datetime.now(tm.TIMEZONE)
        for i in range(15):
            _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=15 - i)).isoformat())
        self.assertGreater(rm.get_drawdown_multiplier(), 0.0)
        self.assertEqual(rm.get_drawdown_multiplier(), 0.25)  # DD >= 10R -> minimo, mai zero


class TestCheckCanTrade(RiskManagerTestCase):
    def test_weekend_blocks(self):
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", is_weekend=True)
        self.assertFalse(result.allowed)

    def test_news_error_blocks(self):
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", news_error=True)
        self.assertFalse(result.allowed)

    def test_low_probability_blocks(self):
        result = rm.check_can_trade(prob=50, rr=2.5, signal="BUY", timeframe="4h")
        self.assertFalse(result.allowed)
        self.assertIn("sotto soglia", result.reason)

    def test_m5_requires_higher_probability_than_other_timeframes(self):
        # 60% basta su H4 (soglia 55%) ma non su M5 (soglia 65%, SL stretti).
        ok_h4 = rm.check_can_trade(prob=60, rr=2.5, signal="BUY", timeframe="4h")
        blocked_m5 = rm.check_can_trade(prob=60, rr=2.5, signal="BUY", timeframe="5min")
        self.assertTrue(ok_h4.allowed)
        self.assertFalse(blocked_m5.allowed)

    def test_low_rr_blocks(self):
        result = rm.check_can_trade(prob=70, rr=1.0, signal="BUY", timeframe="4h")
        self.assertFalse(result.allowed)

    def test_invalid_signal_blocks(self):
        result = rm.check_can_trade(prob=70, rr=2.5, signal="NEUTRAL", timeframe="4h")
        self.assertFalse(result.allowed)

    def test_near_news_blocks(self):
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", timeframe="4h", near_news=True)
        self.assertFalse(result.allowed)

    def test_valid_setup_is_allowed(self):
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", timeframe="4h")
        self.assertTrue(result.allowed)
        self.assertGreater(result.risk_pct, 0)

    def test_three_consecutive_losses_triggers_cooldown(self):
        now = datetime.now(tm.TIMEZONE)
        for i in range(3):
            _open_and_close(result="LOSS", closed_at=(now - timedelta(minutes=3 - i)).isoformat())
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", timeframe="4h")
        self.assertFalse(result.allowed)
        self.assertIn("cooldown", result.reason.lower())

    def test_max_same_direction_exposure_blocks(self):
        # Apre MAX_SAME_DIRECTION trade BUY ancora OPEN, poi un ulteriore
        # tentativo BUY deve essere bloccato per esposizione massima.
        for _ in range(rm.MAX_SAME_DIRECTION):
            data = {
                "signal": "BUY", "order_type": "BUY", "entry": 4300.0, "sl": 4270.0,
                "tp1": 4330.0, "tp2": 4360.0, "tp3": 4390.0, "prob": 60, "regime": "NORMAL",
                "timeframe": "4h", "price": 4300.0, "risk_pct": 1.0, "strategies": {},
                "data_timestamp": datetime.now(tm.TIMEZONE).isoformat(), "price_basis": 0.0,
                "early_be_level": 0,
            }
            data["setup_key"] = tm.build_setup_key(data)
            import time as _time
            _time.sleep(0.001)  # timestamp diverso -> setup_key diverso ad ogni giro
            tm.open_trade(data)
        result = rm.check_can_trade(prob=70, rr=2.5, signal="BUY", timeframe="4h")
        self.assertFalse(result.allowed)
        self.assertIn("Esposizione", result.reason)


if __name__ == "__main__":
    unittest.main()
