"""
Test per historical_cot_feature.py — copre il bug di rigore trovato il
2026-09-15 (stesso pattern già corretto in historical_fomc_text.py l'11/09):
_evaluate_interaction_split() marcava "conditioning_helps=True" anche
quando il modello condizionato aveva R² NEGATIVO, solo perché "meno
negativo" del baseline (anch'esso senza vera capacità predittiva).
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import historical_cot_feature as cf


def _cot_df(dates, zscores):
    return pd.DataFrame({
        "report_date": pd.to_datetime(dates),
        "release_date": pd.to_datetime(dates) + pd.Timedelta(days=cf.COT_RELEASE_LAG_DAYS),
        "net_long_pct": [0.0] * len(dates),
        "net_long_zscore": zscores,
    })


class TestCotAsof(unittest.TestCase):
    def test_never_returns_a_not_yet_released_report(self):
        cot_df = _cot_df(["2024-01-02", "2024-01-09"], [0.5, 2.0])
        # Evento il giorno stesso della release del secondo report (prima
        # che quel venerdi' sia passato) - deve vedere SOLO il primo.
        event_dates = pd.Series([pd.Timestamp("2024-01-09")])  # release_date del 2° report è 2024-01-12
        result = cf.cot_asof(event_dates, cot_df)
        self.assertEqual(result.iloc[0], 0.5)

    def test_picks_the_latest_already_released_report(self):
        cot_df = _cot_df(["2024-01-02", "2024-01-09", "2024-01-16"], [0.5, 2.0, -1.0])
        event_dates = pd.Series([pd.Timestamp("2024-01-20")])
        result = cf.cot_asof(event_dates, cot_df)
        self.assertEqual(result.iloc[0], -1.0)


class TestEvaluateInteractionSplit(unittest.TestCase):
    """Il bug centrale: senza "and r2_conditional > 0", un modello con R²
    negativo ma meno negativo del baseline passava come 'aiuta'."""

    def test_conditioning_does_not_help_when_r2_stays_negative(self):
        # Costruito apposta: sia il modello base sia quello condizionato
        # predicono peggio della semplice media (R² negativo per
        # entrambi) - un rumore quasi puro con pochissima struttura reale.
        rng = np.random.RandomState(0)
        n = 60
        df = pd.DataFrame({
            "surprise_zscore": rng.normal(0, 1, n),
            "cot_zscore": rng.normal(0, 1, n),
            "reaction_30m": rng.normal(0, 5, n),  # scala enorme, nessuna vera relazione con surprise/cot
        })
        result = cf._evaluate_interaction_split(df, "reaction_30m", 0.5)
        self.assertIsNotNone(result)
        if result["r2_conditional_on_cot"] <= 0:
            self.assertFalse(result["conditioning_helps"])

    def test_conditioning_helps_only_when_genuinely_positive_r2(self):
        # Relazione reale e forte SOLO nel gruppo "crowded" (|cot|>=mediana),
        # rumore puro nel gruppo "calm" - il modello condizionato deve
        # ottenere un R² positivo sul test, il baseline unico (una sola
        # pendenza per tutti) deve restare peggiore.
        rng = np.random.RandomState(1)
        n = 200
        cot = rng.normal(0, 1.5, n)
        surprise = rng.normal(0, 1, n)
        crowded_mask = np.abs(cot) >= np.median(np.abs(cot))
        reaction = np.where(crowded_mask, 3.0 * surprise, rng.normal(0, 3, n))
        reaction = reaction + rng.normal(0, 0.1, n)  # rumore piccolo, non annega il segnale
        df = pd.DataFrame({"surprise_zscore": surprise, "cot_zscore": cot, "reaction_30m": reaction})
        result = cf._evaluate_interaction_split(df, "reaction_30m", 0.5)
        self.assertIsNotNone(result)
        self.assertGreater(result["r2_conditional_on_cot"], 0)
        self.assertTrue(result["conditioning_helps"])

    def test_returns_none_below_minimum_sample(self):
        df = pd.DataFrame({
            "surprise_zscore": [0.1] * 10, "cot_zscore": [0.5] * 10, "reaction_30m": [1.0] * 10,
        })
        self.assertIsNone(cf._evaluate_interaction_split(df, "reaction_30m", 0.5))

    def test_includes_direction_accuracy(self):
        rng = np.random.RandomState(2)
        n = 100
        df = pd.DataFrame({
            "surprise_zscore": rng.normal(0, 1, n),
            "cot_zscore": rng.normal(0, 1, n),
            "reaction_30m": rng.normal(0, 1, n),
        })
        result = cf._evaluate_interaction_split(df, "reaction_30m", 0.5)
        self.assertIsNotNone(result)
        self.assertIn("direction_accuracy", result)
        self.assertGreaterEqual(result["direction_accuracy"], 0.0)
        self.assertLessEqual(result["direction_accuracy"], 1.0)


class TestValidateCotConditioning(unittest.TestCase):
    """Integrazione con un DB temporaneo (schema event_features reale) e
    una serie COT sintetica iniettata via mock — mai una vera chiamata di
    rete in questi test."""

    def setUp(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(self.tmpdb)
        conn.execute("""
            CREATE TABLE event_features (
                event_uid TEXT PRIMARY KEY, event_name TEXT, datetime_utc TEXT,
                surprise_zscore REAL, reaction_1m REAL, reaction_5m REAL,
                reaction_15m REAL, reaction_30m REAL, reaction_60m REAL
            )
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.tmpdb):
            os.remove(self.tmpdb)

    def test_unknown_event_returns_unavailable(self):
        result = cf.validate_cot_conditioning("Evento Mai Visto", db_path=self.tmpdb)
        self.assertFalse(result["available"])

    def test_insufficient_n_gives_explicit_verdict(self):
        conn = sqlite3.connect(self.tmpdb)
        for i in range(15):
            conn.execute(
                "INSERT INTO event_features VALUES (?,?,?,?,?,?,?,?,?)",
                (f"uid{i}", "Test Event", f"2024-01-{i+1:02d}T14:00:00+00:00",
                 0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
            )
        conn.commit()
        conn.close()

        cot_df = _cot_df(["2023-12-01"], [0.5])
        with mock.patch("historical_cot_feature.load_cot_series", return_value=cot_df):
            result = cf.validate_cot_conditioning("Test Event", db_path=self.tmpdb)
        self.assertLess(result["n"], 100)
        self.assertIn("DATI INSUFFICIENTI", result["verdict"])


class TestCheckCotConditionedEventsHealth(unittest.TestCase):
    """check_cot_conditioned_events_health() (2026-09-15, monitoraggio
    walk-forward locale) — sola lettura, mai rigenera cot_models.json."""

    def test_healthy_when_horizon_still_genuine(self):
        fake_result = {"n": 200, "genuine_horizons": ["reaction_30m", "reaction_60m"]}
        with mock.patch("historical_cot_feature.COT_CONDITIONED_EVENTS", ("CB Consumer Confidence",)), \
             mock.patch("historical_model.DEPLOYED_EVENTS", {"CB Consumer Confidence": "reaction_30m"}), \
             mock.patch("historical_cot_feature.validate_cot_conditioning", return_value=fake_result):
            report = cf.check_cot_conditioned_events_health()
        self.assertTrue(report["CB Consumer Confidence"]["healthy"])

    def test_unhealthy_when_horizon_no_longer_genuine(self):
        fake_result = {"n": 200, "genuine_horizons": ["reaction_60m"]}  # non più reaction_30m
        with mock.patch("historical_cot_feature.COT_CONDITIONED_EVENTS", ("CB Consumer Confidence",)), \
             mock.patch("historical_model.DEPLOYED_EVENTS", {"CB Consumer Confidence": "reaction_30m"}), \
             mock.patch("historical_cot_feature.validate_cot_conditioning", return_value=fake_result):
            report = cf.check_cot_conditioned_events_health()
        self.assertFalse(report["CB Consumer Confidence"]["healthy"])


if __name__ == "__main__":
    unittest.main()
