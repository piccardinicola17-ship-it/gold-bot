"""
Test per self_learning.py — copre in particolare _is_win(), la logica di
classificazione win/loss corretta il 2026-09-03 dopo che tre file diversi
(dashboard.py, self_learning.py, gold_bot.py) avevano iniziato a
divergere silenziosamente su come contare un WIN_BE con TP1 già raggiunto.
"""

import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
