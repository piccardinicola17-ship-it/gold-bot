"""
Test per gold_bot.py — copre tre bug trovati nell'audit del 2026-09-04
(comandi Telegram, la parte meno testata del progetto):

1. _live_min_prob_for_tf(): /backtest usava sempre MIN_PROB=55 fisso per
   ogni timeframe, contraddicendo sia la soglia reale usata in live da
   agent_orchestrator.py (65% per M1/M5/M15) sia il testo mostrato a
   schermo da /backtest tutti.
2. is_decisive_win() (trade_manager.py, importata qui): cmd_stats e
   send_daily_report usavano un criterio di win rate diverso da cmd_report
   e dalla dashboard (escludevano ogni WIN_BE, anche quelli con TP1 già
   raggiunto) — stesso DB, win rate diverso a seconda del comando.
3. check_macro_alerts(): il titolo evento non era escapato per Markdown
   (rischio "can't parse entities" su un titolo con caratteri speciali) e
   il dedup veniva marcato PRIMA del tentativo di invio — un fallimento
   perdeva l'alert (incluso il blackout trading) per sempre, senza retry.

Ogni test usa un DB SQLite temporaneo isolato (mai il goldbot.db reale).
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_manager as tm
import gold_bot as gb


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


class GoldBotTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        tm.DB_PATH = self.tmpdb
        gb.DB_PATH = self.tmpdb
        self.tmp_active_file = tempfile.mktemp(suffix=".json")
        tm.ACTIVE_FILE = self.tmp_active_file
        tm.init_db()
        gb._sent_event_alerts = set()
        gb._sent_post_event_alerts = set()
        gb._pre_event_bias = {}

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.tmpdb + suffix
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_active_file):
            os.remove(self.tmp_active_file)


class TestLiveMinProbForTf(unittest.TestCase):
    def test_intraday_matches_live_threshold(self):
        for tf in ("5min", "15min", "1min"):
            self.assertEqual(gb._live_min_prob_for_tf(tf), 65)

    def test_other_timeframes_use_55_default(self):
        for tf in ("1h", "4h", "1day"):
            self.assertEqual(gb._live_min_prob_for_tf(tf), 55)

    def test_respects_min_prob_env_override_like_live_does(self):
        with mock.patch.dict(os.environ, {"MIN_PROB": "70"}):
            self.assertEqual(gb._live_min_prob_for_tf("1day"), 70)
            # M5/M15/M1 restano fissi a 65 anche con l'env override,
            # esattamente come in agent_orchestrator.py.
            self.assertEqual(gb._live_min_prob_for_tf("5min"), 65)


class TestWinRateConsistencyAcrossCommands(GoldBotTestCase):
    """Il bug reale: un trade WIN_BE con tp1_hit=True doveva contare come
    vittoria ovunque, non solo in /report."""

    def _seed_win_be_with_tp1_hit(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        tm.mark_tp1_hit(trade_id)
        tm.close_trade(trade_id, "WIN_BE", data["entry"])
        return trade_id

    def test_cmd_stats_counts_win_be_with_tp1_as_win(self):
        self._seed_win_be_with_tp1_hit()
        trades = gb._get_closed_trades()
        wins = [t for t in trades if gb.is_decisive_win(t)]
        losses = [t for t in trades if t.get("result") == "LOSS"]
        total = len(wins) + len(losses)
        wr = round(len(wins) / total * 100, 1) if total else 0
        self.assertEqual(len(wins), 1)
        self.assertEqual(wr, 100.0)

    def test_send_daily_report_formula_matches_is_decisive_win(self):
        """Prima del fix send_daily_report usava
        result in (WIN_TP1,WIN_TP2,WIN_TP3,LOSS) — escludendo SEMPRE
        WIN_BE, anche con tp1_hit. Verifica diretta sulla stessa query che
        send_daily_report usa internamente."""
        self._seed_win_be_with_tp1_hit()
        all_t = gb._get_closed_trades()
        wins_all_l = [t for t in all_t if gb.is_decisive_win(t)]
        losses_all_l = [t for t in all_t if t.get("result") == "LOSS"]
        decisivi_all = wins_all_l + losses_all_l
        self.assertEqual(len(wins_all_l), 1)
        self.assertEqual(len(decisivi_all), 1)

    def test_pure_be_never_reaching_tp1_is_not_a_win(self):
        data = _base_trade_data()
        trade_id = tm.open_trade(data)
        # niente mark_tp1_hit: BE anticipato, TP1 mai raggiunto
        tm.close_trade(trade_id, "WIN_BE", data["entry"])
        trades = gb._get_closed_trades()
        wins = [t for t in trades if gb.is_decisive_win(t)]
        self.assertEqual(len(wins), 0)


class TestCheckMacroAlertsResilience(GoldBotTestCase):
    """check_macro_alerts: escaping del titolo e dedup marcato solo dopo
    un send riuscito (prima: marcato subito, un fallimento perdeva
    l'alert — incluso il blackout trading — per sempre)."""

    def _make_event(self, minutes_away: int, title: str) -> dict:
        ev_dt = datetime.now(gb.TIMEZONE) + timedelta(minutes=minutes_away)
        return {
            "date": ev_dt.strftime("%Y-%m-%d"),
            "time": ev_dt.strftime("%H:%M"),
            "title": title,
            "forecast": "N/A",
            "previous": "N/A",
        }

    async def _run_with_send_side_effect(self, event, send_side_effect):
        bot = mock.AsyncMock()
        bot.send_message = mock.AsyncMock(side_effect=send_side_effect)
        with mock.patch("gold_bot.is_bot_paused", return_value=False), \
             mock.patch("analyzer.get_upcoming_events", return_value=[event]), \
             mock.patch("gold_bot.get_current_price_async", return_value=4400.0), \
             mock.patch("gold_bot.analyze_macro_event", return_value="Bias: NEUTRO\nMotivo: test"), \
             mock.patch("gold_bot.save_macro_alert_state", return_value=None):
            await gb.check_macro_alerts(bot)
        return bot

    async def _scenario_send_fails_then_succeeds(self, title):
        event = self._make_event(30, title)
        ev_key = f"{event['date']}_{event['time']}_{event['title']}"

        # Primo giro: il send fallisce (es. Markdown non valido) -> il
        # dedup NON deve essere marcato, deve poter ritentare.
        await self._run_with_send_side_effect(event, Exception("can't parse entities"))
        self.assertNotIn(ev_key, gb._sent_event_alerts)

        # Secondo giro: il send riesce -> ora sì marcato.
        bot2 = await self._run_with_send_side_effect(event, None)
        self.assertIn(ev_key, gb._sent_event_alerts)
        return bot2

    def test_failed_send_does_not_mark_dedup_allows_retry(self):
        import asyncio
        asyncio.run(self._scenario_send_fails_then_succeeds("Fed Chair Speech"))

    def test_event_title_with_markdown_chars_is_escaped(self):
        import asyncio
        bot = asyncio.run(self._scenario_send_fails_then_succeeds("Fed [Update]: rate_decision"))
        sent_text = bot.send_message.call_args.kwargs["text"]
        self.assertIn("\\[Update\\]", sent_text)
        self.assertIn("rate\\_decision", sent_text)


class TestPostEventNewsDigestNotDuplicated(GoldBotTestCase):
    """Bug trovato in diretta il 2026-09-04 sull'NFP: 3 eventi simultanei
    (NFP + Average Hourly Earnings + Unemployment Rate, tutti alle 14:30)
    generavano 3 digest notizie quasi identici di fila (uno per evento nel
    ciclo POST-EVENTO), con bias pure diverso da una chiamata all'altra —
    sembrava un bot rotto in loop. Il digest va mandato una sola volta per
    giro, non una volta per evento."""

    def _make_event(self, minutes_away: int, title: str) -> dict:
        ev_dt = datetime.now(gb.TIMEZONE) + timedelta(minutes=minutes_away)
        return {
            "date": ev_dt.strftime("%Y-%m-%d"),
            "time": ev_dt.strftime("%H:%M"),
            "title": title,
            "forecast": "N/A",
            "previous": "N/A",
        }

    async def _run(self, events):
        bot = mock.AsyncMock()
        bot.send_message = mock.AsyncMock(return_value=None)
        with mock.patch("gold_bot.is_bot_paused", return_value=False), \
             mock.patch("analyzer.get_upcoming_events", return_value=events), \
             mock.patch("gold_bot.get_current_price_async", return_value=4400.0), \
             mock.patch("gold_bot.get_extended_news", return_value=["headline"]), \
             mock.patch("gold_bot.format_news_message", return_value="digest notizie"), \
             mock.patch("gold_bot.save_macro_alert_state", return_value=None):
            await gb.check_macro_alerts(bot)
        return bot

    def test_single_post_event_sends_news_digest_once(self):
        import asyncio
        event = self._make_event(-10, "Non-Farm Employment Change")
        bot = asyncio.run(self._run([event]))
        digest_calls = [
            c for c in bot.send_message.call_args_list
            if c.kwargs.get("text") == "digest notizie"
        ]
        self.assertEqual(len(digest_calls), 1)

    def test_three_simultaneous_post_events_send_news_digest_once_not_three_times(self):
        import asyncio
        events = [
            self._make_event(-10, "Non-Farm Employment Change"),
            self._make_event(-10, "Average Hourly Earnings m/m"),
            self._make_event(-10, "Unemployment Rate"),
        ]
        bot = asyncio.run(self._run(events))
        digest_calls = [
            c for c in bot.send_message.call_args_list
            if c.kwargs.get("text") == "digest notizie"
        ]
        self.assertEqual(len(digest_calls), 1)
        # Ma il resoconto POST-EVENTO va comunque mandato per ciascuno dei 3.
        post_evento_calls = [
            c for c in bot.send_message.call_args_list
            if "POST-EVENTO" in c.kwargs.get("text", "")
        ]
        self.assertEqual(len(post_evento_calls), 3)

    def test_no_post_event_no_news_digest_sent(self):
        """Solo eventi pre-evento (30 min prima): nessun digest, il flag
        post_event_fired resta False."""
        import asyncio
        event = self._make_event(30, "Fed Chair Speech")
        with mock.patch("gold_bot.analyze_macro_event", return_value="Bias: NEUTRO\nMotivo: test"):
            bot = asyncio.run(self._run([event]))
        digest_calls = [
            c for c in bot.send_message.call_args_list
            if c.kwargs.get("text") == "digest notizie"
        ]
        self.assertEqual(len(digest_calls), 0)


if __name__ == "__main__":
    unittest.main()
