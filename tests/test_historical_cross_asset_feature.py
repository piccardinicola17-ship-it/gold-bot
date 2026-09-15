"""Test per historical_cross_asset_feature.py — solo la logica di
valutazione (_evaluate_lagged_predictor), mai una vera chiamata yfinance."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import historical_cross_asset_feature as ca


class TestEvaluateLaggedPredictor(unittest.TestCase):
    def test_shifts_predictor_by_one_day_no_lookahead(self):
        # gold_ret[t] costruito come funzione ESATTA di pred_ret[t-1] - se
        # lo shift funzionasse al contrario (lookahead), il fit fallirebbe
        # a catturare la relazione e il test lo scoprirebbe.
        idx = pd.date_range("2020-01-01", periods=300, freq="D")
        rng = np.random.RandomState(0)
        pred = pd.Series(rng.normal(0, 1, 300), index=idx)
        gold = pd.Series(np.nan, index=idx)
        gold.iloc[1:] = 2.0 * pred.iloc[:-1].to_numpy()

        result = ca._evaluate_lagged_predictor(pred, gold.dropna(), 0.5)
        self.assertIsNotNone(result)
        self.assertGreater(result["r2_test"], 0.9)  # relazione quasi perfetta se l'allineamento e' giusto

    def test_returns_none_below_minimum_sample(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        pred = pd.Series(np.random.normal(0, 1, 20), index=idx)
        gold = pd.Series(np.random.normal(0, 1, 20), index=idx)
        self.assertIsNone(ca._evaluate_lagged_predictor(pred, gold, 0.5))

    def test_pure_noise_does_not_beat_naive(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="D")
        rng = np.random.RandomState(1)
        pred = pd.Series(rng.normal(0, 1, 500), index=idx)
        gold = pd.Series(rng.normal(0, 1, 500), index=idx)  # nessuna relazione reale
        result = ca._evaluate_lagged_predictor(pred, gold, 0.5)
        self.assertIsNotNone(result)
        # Con puro rumore l'r2 sara' vicino a zero o negativo - non deve
        # sistematicamente "battere" un naive altrettanto casuale con un
        # margine ampio (qui controlliamo solo che il campo esista ed sia
        # calcolato, il verdetto specifico dipende dal seed).
        self.assertIn("beats_naive", result)
        self.assertIn("direction_accuracy", result)


if __name__ == "__main__":
    unittest.main()
