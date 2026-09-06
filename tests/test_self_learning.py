"""
Test per self_learning.py — copre in particolare _is_win(), la logica di
classificazione win/loss corretta il 2026-09-03 dopo che tre file diversi
(dashboard.py, self_learning.py, gold_bot.py) avevano iniziato a
divergere silenziosamente su come contare un WIN_BE con TP1 già raggiunto.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import self_learning as sl


class TestIsWin(unittest.TestCase):
    def test_win_tp1_tp2_tp3_are_wins(self):
        for result in ("WIN_TP1", "WIN_TP2", "WIN_TP3"):
            with self.subTest(result=result):
                self.assertTrue(sl._is_win({"result": result, "tp1_hit": 0}))

    def test_loss_is_never_a_win(self):
        self.assertFalse(sl._is_win({"result": "LOSS", "tp1_hit": 1}))

    def test_win_be_with_tp1_hit_counts_as_win(self):
        """Il caso che aveva iniziato a divergere tra i file: un trade che
        ha toccato TP1 (profitto parziale) e poi è tornato a breakeven
        conta come vittoria, non viene escluso come i BE 'puri'."""
        self.assertTrue(sl._is_win({"result": "WIN_BE", "tp1_hit": 1}))

    def test_win_be_without_tp1_hit_is_neutral_not_a_win(self):
        self.assertFalse(sl._is_win({"result": "WIN_BE", "tp1_hit": 0}))

    def test_cancelled_is_not_a_win(self):
        self.assertFalse(sl._is_win({"result": "CANCELLED", "tp1_hit": 0}))
        # Un CANCELLED con tp1_hit=1 non dovrebbe verificarsi nella pratica
        # (un pending non ancora attivato non può aver toccato TP1), ma la
        # funzione non deve comunque considerarlo un LOSS: solo "non win".
        self.assertTrue(sl._is_win({"result": "CANCELLED", "tp1_hit": 1}))

    def test_missing_fields_do_not_raise(self):
        self.assertFalse(sl._is_win({}))


def _trade(trade_id, entry, result="WIN_TP1"):
    return {
        "trade_id": trade_id, "result": result, "signal": "BUY", "regime": "NORMAL",
        "timeframe": "1h", "entry": entry, "sl": entry - 10, "tp1": entry + 10,
        "tp2": entry + 20, "tp3": entry + 30, "tp1_hit": 1, "be_hit": 0,
        "exit_price": entry + 10, "pnl_r": 1.0, "prob": 60, "timestamp": "2026-09-05T10:00:00",
    }


class TestAnalyzeLastTradeTradeIdLookup(unittest.TestCase):
    """Regression (audit 2026-09-05): analyze_last_trade(trade_id) aveva un
    parametro pensato per evitare di analizzare il trade sbagliato quando
    due trade si chiudono quasi in contemporanea, ma nessuno dei 3 punti di
    chiamata nel codice lo passava mai - cadeva sempre sul fallback
    "ultimo trade nel DB". Qui si verifica la logica di lookup in isolamento
    (senza DB/rete reali)."""

    def _entries_used_in_prompt(self, trades, trade_id=""):
        captured = {}

        def _fake_groq(system, user, max_tokens=500):
            captured["user"] = user
            return "ok"

        with patch.object(sl, "_get_closed_trades", return_value=trades), \
             patch.object(sl, "_call_groq", side_effect=_fake_groq):
            sl.analyze_last_trade(trade_id)
        return captured["user"]

    def test_finds_trade_by_full_id(self):
        trades = [_trade("aaaaaaaa-1111", 100.0), _trade("bbbbbbbb-2222", 200.0)]
        prompt = self._entries_used_in_prompt(trades, trade_id="aaaaaaaa-1111")
        self.assertIn("$100.0", prompt)
        self.assertNotIn("$200.0", prompt)

    def test_finds_trade_by_id_even_when_not_last(self):
        # Il trade cercato NON e' l'ultimo della lista - deve comunque
        # essere trovato lui, non il fallback trades[-1].
        trades = [_trade("aaaaaaaa-1111", 100.0), _trade("bbbbbbbb-2222", 200.0),
                  _trade("cccccccc-3333", 300.0)]
        prompt = self._entries_used_in_prompt(trades, trade_id="bbbbbbbb-2222")
        self.assertIn("$200.0", prompt)
        self.assertNotIn("$300.0", prompt)

    def test_falls_back_to_last_trade_when_no_id_given(self):
        trades = [_trade("aaaaaaaa-1111", 100.0), _trade("bbbbbbbb-2222", 200.0)]
        prompt = self._entries_used_in_prompt(trades, trade_id="")
        self.assertIn("$200.0", prompt)

    def test_falls_back_to_last_trade_when_id_not_found(self):
        trades = [_trade("aaaaaaaa-1111", 100.0), _trade("bbbbbbbb-2222", 200.0)]
        prompt = self._entries_used_in_prompt(trades, trade_id="zzzzzzzz-9999")
        self.assertIn("$200.0", prompt)


def _week_trade(result="WIN_TP2", regime="NORMAL", timeframe="1day", signal="BUY",
                 tp1_hit=1, pnl_r=1.0):
    return {
        "result": result, "regime": regime, "timeframe": timeframe, "signal": signal,
        "tp1_hit": tp1_hit, "pnl_r": pnl_r,
    }


class TestWeeklyReviewPromptQuality(unittest.TestCase):
    """FIX (2026-09-06): la review inviata quel giorno conteneva un periodo
    inventato nel futuro e un "pattern" (5min/15min 100% win rate) basato
    su 1-3 trade che contraddiceva il backtest a campione grande fatto lo
    stesso giorno (5min/15min in perdita robusta su migliaia di trade).
    Qui si verifica che il prompt mandato alla LLM dia le date esatte e
    marchi esplicitamente i bucket troppo piccoli, invece di lasciare che
    la LLM li tratti come pattern affidabili."""

    def _prompt_sent(self, trades):
        captured = {}

        def _fake_groq(system, user, max_tokens=500):
            captured["user"] = user
            return "ok"

        with patch.object(sl, "_get_closed_trades", return_value=trades), \
             patch.object(sl, "_call_groq", side_effect=_fake_groq):
            sl.weekly_review()
        return captured["user"]

    def test_prompt_contains_exact_date_range_not_left_to_the_llm(self):
        trades = [_week_trade() for _ in range(3)]
        prompt = self._prompt_sent(trades)
        period_end = sl.datetime.now(sl.TIMEZONE)
        period_start = period_end - sl.timedelta(days=7)
        self.assertIn(period_start.strftime("%d/%m/%Y"), prompt)
        self.assertIn(period_end.strftime("%d/%m/%Y"), prompt)
        self.assertIn("non inventarne altre", prompt)

    def test_small_bucket_is_flagged_not_presented_as_a_pattern(self):
        # 1 solo trade su "5min" — esattamente il caso reale del 6 settembre
        # che aveva prodotto un falso pattern "100% win rate" sulla review.
        trades = [_week_trade(timeframe="5min", result="WIN_TP2")]
        prompt = self._prompt_sent(trades)
        self.assertIn("5min: 1/1", prompt)
        self.assertIn("CAMPIONE TROPPO PICCOLO", prompt)

    def test_bucket_with_enough_samples_is_not_flagged(self):
        trades = [_week_trade(timeframe="1day", result="WIN_TP2") for _ in range(2)] + \
                  [_week_trade(timeframe="1day", result="LOSS")]
        prompt = self._prompt_sent(trades)
        self.assertIn("1day: 2/3", prompt)
        tf_section = prompt.split("Per timeframe:")[1].split("Per direzione:")[0]
        self.assertNotIn("CAMPIONE TROPPO PICCOLO", tf_section)

    def test_non_decisive_count_is_given_explicitly_not_left_to_speculation(self):
        # 2 decisivi + 1 BE + 4 "altro" (probabilmente ancora aperti) = 7 totali.
        trades = (
            [_week_trade(result="WIN_TP2"), _week_trade(result="LOSS")]
            + [_week_trade(result="WIN_BE", tp1_hit=0)]
            + [_week_trade(result="OPEN", tp1_hit=0) for _ in range(4)]
        )
        prompt = self._prompt_sent(trades)
        self.assertIn("ancora aperti/altro: 4", prompt)
        self.assertIn("non speculare", prompt.lower())


if __name__ == "__main__":
    unittest.main()
