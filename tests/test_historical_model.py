"""
Test per historical_model.py — copre tre bug trovati nell'audit del
2026-09-04 (progetto dati storici, mai testato prima):

1. _total_n_for_series(): il "Verdetto per serie" calcolava il campione
   totale con .max() indipendente su n_train e n_test, colonne che
   possono appartenere a orizzonte/split diversi tra loro — sovrastimava
   il campione reale (verificato: su Core CPI m/m dava 243 invece di 187,
   +30%), potendo far passare la soglia n>=100 anche per una serie in
   realtà più piccola.
2. summarize(all_series=...): una serie che non supera mai la soglia
   minima per split (n_train>=20/n_test>=10) spariva silenziosamente dal
   report invece di comparire come "dati insufficienti".
3. regenerate_deployed_models(): fittava sempre DEPLOY_HORIZON alla
   cieca, senza verificare che fosse davvero l'orizzonte validato per
   quella serie — ora fallisce rumorosamente se non lo è.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import historical_model as hm


def _split_row(event_name, horizon, train_fraction, n_train, n_test, beats_naive=True):
    return {
        "event_name": event_name, "horizon": horizon, "train_fraction": train_fraction,
        "n_train": n_train, "n_test": n_test, "slope": 0.1,
        "r2_test": 0.1, "r2_naive_test": 0.0, "beats_naive": beats_naive,
        "direction_accuracy": 0.55,
    }


class TestTotalNForSeries(unittest.TestCase):
    def test_does_not_mix_n_train_and_n_test_across_horizons(self):
        """Il bug reale: due orizzonti con campioni diversi (es. reaction_5m
        con più dati validi di reaction_60m) — .max() indipendente su
        n_train e n_test prendeva il train più grande di un orizzonte e il
        test più grande dell'altro, sommando un totale mai esistito
        davvero in nessuno split."""
        group = pd.DataFrame([
            _split_row("X", "reaction_5m", 0.5, 90, 90),   # n=180 (orizzonte con più dati)
            _split_row("X", "reaction_5m", 0.7, 126, 54),  # stesso orizzonte, n=180 invariato
            _split_row("X", "reaction_60m", 0.5, 40, 40),  # n=80 (orizzonte con meno dati)
        ])
        # Prima del fix: max(n_train)=126, max(n_test)=90 -> 216 (mai esistito)
        # Dopo il fix: max tra i totali per-orizzonte -> max(180, 80) = 180
        self.assertEqual(hm._total_n_for_series(group), 180)

    def test_single_horizon_constant_across_splits(self):
        group = pd.DataFrame([
            _split_row("X", "reaction_30m", 0.5, 50, 50),
            _split_row("X", "reaction_30m", 0.8, 80, 20),
        ])
        self.assertEqual(hm._total_n_for_series(group), 100)


class TestSummarizeMissingSeries(unittest.TestCase):
    def test_series_with_no_passing_split_reported_not_silently_dropped(self):
        results = pd.DataFrame([_split_row("HasData", "reaction_30m", 0.5, 50, 50)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            hm.summarize(results, all_series=["HasData", "NeverPassesGate"])
        output = buf.getvalue()
        self.assertIn("NeverPassesGate", output)
        self.assertIn("DATI INSUFFICIENTI", output)

    def test_all_series_none_does_not_report_missing(self):
        """Comportamento invariato se il chiamante non passa all_series
        (retrocompatibile con l'uso precedente)."""
        results = pd.DataFrame([_split_row("HasData", "reaction_30m", 0.5, 50, 50)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            hm.summarize(results, all_series=None)
        self.assertNotIn("NeverPassesGate", buf.getvalue())


class TestRegenerateDeployedModelsValidatesHorizon(unittest.TestCase):
    """I test patchano DEPLOYED_EVENTS esplicitamente con un solo evento di
    prova, cosi' restano indipendenti da quali serie reali sono deployate
    in produzione in questo momento."""

    def test_raises_when_deploy_horizon_never_validated(self):
        fake_results = pd.DataFrame([
            _split_row("Core CPI m/m", hm.DEPLOY_HORIZON, 0.5, 50, 50, beats_naive=False),
        ])
        with mock.patch("historical_model.DEPLOYED_EVENTS", {"Core CPI m/m": hm.DEPLOY_HORIZON}), \
             mock.patch("historical_model.run_all", return_value=fake_results), \
             mock.patch("historical_model.fit_final_model") as mock_fit:
            with self.assertRaises(ValueError):
                hm.regenerate_deployed_models()
            mock_fit.assert_not_called()

    def test_raises_when_deploy_horizon_missing_entirely(self):
        """Il campione non contiene affatto l'orizzonte dichiarato per
        questa serie (es. dati insufficienti proprio a quell'orizzonte)."""
        fake_results = pd.DataFrame([
            _split_row("Core CPI m/m", "reaction_5m", 0.5, 50, 50, beats_naive=True),
        ])
        with mock.patch("historical_model.DEPLOYED_EVENTS", {"Core CPI m/m": hm.DEPLOY_HORIZON}), \
             mock.patch("historical_model.run_all", return_value=fake_results), \
             mock.patch("historical_model.fit_final_model") as mock_fit:
            with self.assertRaises(ValueError):
                hm.regenerate_deployed_models()
            mock_fit.assert_not_called()

    def test_proceeds_when_horizon_beats_naive_on_every_split(self):
        fake_results = pd.DataFrame([
            _split_row("Core CPI m/m", hm.DEPLOY_HORIZON, 0.5, 50, 50, beats_naive=True),
            _split_row("Core CPI m/m", hm.DEPLOY_HORIZON, 0.7, 70, 30, beats_naive=True),
        ])
        fake_model = {"event_name": "Core CPI m/m", "horizon": hm.DEPLOY_HORIZON, "slope": 1.0}
        with mock.patch("historical_model.DEPLOYED_EVENTS", {"Core CPI m/m": hm.DEPLOY_HORIZON}), \
             mock.patch("historical_model.run_all", return_value=fake_results), \
             mock.patch("historical_model.fit_final_model", return_value=fake_model) as mock_fit, \
             mock.patch("pathlib.Path.write_text") as mock_write:
            hm.regenerate_deployed_models()
            mock_fit.assert_called_once_with("Core CPI m/m", horizon=hm.DEPLOY_HORIZON, db_path=hm.HIST_DB_PATH)
            mock_write.assert_called_once()

    def test_each_series_validated_at_its_own_horizon_not_a_global_one(self):
        """Il bug che questo fix risolve: una serie che valida SOLO a un
        orizzonte diverso da quello di un'altra serie deployata deve
        comunque passare, fittata al SUO orizzonte — non deve fallire solo
        perche' non coincide con l'orizzonte di un'altra serie nello
        stesso DEPLOYED_EVENTS."""
        fake_results = pd.DataFrame([
            _split_row("Core CPI m/m", "reaction_30m", 0.5, 50, 50, beats_naive=True),
            _split_row("Core CPI m/m", "reaction_30m", 0.7, 70, 30, beats_naive=True),
            _split_row("Unemployment Claims", "reaction_1m", 0.5, 50, 50, beats_naive=True),
            _split_row("Unemployment Claims", "reaction_1m", 0.7, 70, 30, beats_naive=True),
            # Unemployment Claims NON valida a reaction_30m — non deve
            # importare, dato che il suo orizzonte dichiarato e' un altro.
            _split_row("Unemployment Claims", "reaction_30m", 0.5, 50, 50, beats_naive=False),
        ])

        def fake_fit(name, horizon, db_path):
            return {"event_name": name, "horizon": horizon, "slope": 1.0}

        with mock.patch("historical_model.DEPLOYED_EVENTS",
                        {"Core CPI m/m": "reaction_30m", "Unemployment Claims": "reaction_1m"}), \
             mock.patch("historical_model.run_all", return_value=fake_results), \
             mock.patch("historical_model.fit_final_model", side_effect=fake_fit) as mock_fit, \
             mock.patch("pathlib.Path.write_text") as mock_write:
            hm.regenerate_deployed_models()
            mock_fit.assert_any_call("Core CPI m/m", horizon="reaction_30m", db_path=hm.HIST_DB_PATH)
            mock_fit.assert_any_call("Unemployment Claims", horizon="reaction_1m", db_path=hm.HIST_DB_PATH)
            mock_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
