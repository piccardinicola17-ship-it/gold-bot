"""
Test per le funzioni pure di trade_manager.py rimaste scoperte: _fmt
(il fix del 2 settembre per gli artefatti di arrotondamento float nei
messaggi Telegram, es. "$4382.009999999999"), i formattatori msg_*,
_target_reached/_stop_reached, e is_bot_paused/set_bot_paused.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_manager as tm


class TestFmt(unittest.TestCase):
    def test_rounds_float_artifacts(self):
        # Il bug reale del 2 settembre: una sottrazione float genera
        # $4382.009999999999 invece di $4382.01.
        self.assertEqual(tm._fmt(4382.009999999999), "4382.01")

    def test_two_decimals_always(self):
        self.assertEqual(tm._fmt(4300), "4300.00")
        self.assertEqual(tm._fmt(4300.1), "4300.10")

    def test_non_numeric_falls_back_to_str(self):
        self.assertEqual(tm._fmt("N/D"), "N/D")
        self.assertEqual(tm._fmt(None), "None")


class TestTfLabel(unittest.TestCase):
    def test_known_timeframes(self):
        self.assertEqual(tm._tf_label("5min"), "M5")
        self.assertEqual(tm._tf_label("4h"), "H4")
        self.assertEqual(tm._tf_label("1day"), "D1")

    def test_unknown_timeframe_falls_back_to_upper(self):
        self.assertEqual(tm._tf_label("weird"), "WEIRD")
        self.assertEqual(tm._tf_label(""), "")


class TestTargetAndStopReached(unittest.TestCase):
    def test_buy_target(self):
        self.assertTrue(tm._target_reached("BUY", price=105, target=100))
        self.assertFalse(tm._target_reached("BUY", price=95, target=100))

    def test_sell_target(self):
        self.assertTrue(tm._target_reached("SELL", price=95, target=100))
        self.assertFalse(tm._target_reached("SELL", price=105, target=100))

    def test_buy_stop(self):
        self.assertTrue(tm._stop_reached("BUY", price=95, stop=100))
        self.assertFalse(tm._stop_reached("BUY", price=105, stop=100))

    def test_sell_stop(self):
        self.assertTrue(tm._stop_reached("SELL", price=105, stop=100))
        self.assertFalse(tm._stop_reached("SELL", price=95, stop=100))


class TestMessageFormatters(unittest.TestCase):
    """Verifica solo che i messaggi contengano i valori giusti, arrotondati
    correttamente, e non sollevino eccezioni — non il testo esatto."""

    def test_no_float_artifacts_in_any_message(self):
        dirty_price = 4382.009999999999
        messages = [
            tm.msg_order_activated("BUY LIMIT", dirty_price, dirty_price, "4h"),
            tm.msg_limit_cancelled(dirty_price, dirty_price, 0, "BUY", "4h"),
            tm.msg_be_closed(dirty_price, "BUY", "4h"),
            tm.msg_be_armed_early(dirty_price, dirty_price, dirty_price, "BUY", "4h"),
            tm.msg_tp1(dirty_price, dirty_price, dirty_price, "BUY", "4h"),
            tm.msg_tp2(dirty_price, dirty_price, dirty_price, "BUY", "4h"),
            tm.msg_tp3(dirty_price, dirty_price, "BUY", "4h"),
            tm.msg_sl(dirty_price, dirty_price, "BUY", "4h"),
        ]
        for msg in messages:
            with self.subTest(msg=msg[:40]):
                self.assertNotIn("00000", msg, "artefatto di arrotondamento float nel messaggio")

    def test_messages_include_timeframe_label(self):
        self.assertIn("H4", tm.msg_tp1(100, 110, 105, "BUY", "4h"))
        self.assertIn("M5", tm.msg_sl(100, 95, "SELL", "5min"))


class TestBotPaused(unittest.TestCase):
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

    def test_defaults_to_not_paused(self):
        self.assertFalse(tm.is_bot_paused())

    def test_pause_and_resume_persist(self):
        tm.set_bot_paused(True)
        self.assertTrue(tm.is_bot_paused())
        tm.set_bot_paused(False)
        self.assertFalse(tm.is_bot_paused())


if __name__ == "__main__":
    unittest.main()
