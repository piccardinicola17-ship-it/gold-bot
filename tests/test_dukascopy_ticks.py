"""
Test per dukascopy_ticks.run_pilot — bug reale trovato il 2026-09-16 e
corretto lo stesso giorno: un evento con una riga PARZIALE in
event_price_reactions (alcune colonne prezzo NULL perché il rate limit di
Dukascopy aveva bloccato il download di alcune ore .bi5 necessarie, mentre
altre erano comunque riuscite) veniva considerato "già fatto" per sempre —
il NOT IN originale guardava solo l'ESISTENZA della riga, mai la sua
completezza. Aggirato a mano quel giorno cancellando le righe parziali e
rilanciando; qui si verifica il fix nel codice (la query ora richiede che
TUTTE le colonne prezzo siano non-NULL per considerare un evento "fatto").

Nessun accesso di rete reale: compute_reaction_features è sempre mockata.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from historical_events import init_historical_db, _connect
from dukascopy_ticks import run_pilot, init_reactions_table, REACTION_OFFSETS_MIN

_PRICE_COLS = [f"price_t{m:+d}m" for m in REACTION_OFFSETS_MIN]


def _complete_feats(base_price: float = 4300.0) -> dict:
    return {**{c: base_price for c in _PRICE_COLS},
            "max_excursion_up": 5.0, "max_excursion_down": -3.0, "tick_count": 500}


class TestRunPilotRetriesIncompleteReactions(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        init_historical_db(self.db_path)
        init_reactions_table(self.db_path)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    def _insert_event(self, event_uid: str, event_name: str = "Core CPI m/m",
                       dt: str = "2026-01-15T13:30:00+00:00") -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO macro_events (event_uid, datetime_utc, date_utc, currency, "
                "impact, event_name, macro_category, source, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (event_uid, dt, dt[:10], "USD", "HIGH", event_name, "CPI", "test", now_iso),
            )

    def _insert_reaction(self, event_uid: str, feats: dict) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        cols = list(feats.keys())
        quoted = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        with _connect(self.db_path) as conn:
            conn.execute(
                f'INSERT INTO event_price_reactions (event_uid, {quoted}, computed_at) '
                f'VALUES (?, {placeholders}, ?)',
                (event_uid, *[feats[c] for c in cols], now_iso),
            )

    def test_complete_reaction_is_not_reprocessed(self):
        self._insert_event("uid-complete")
        self._insert_reaction("uid-complete", _complete_feats())

        with mock.patch("dukascopy_ticks.compute_reaction_features") as mock_compute:
            result = run_pilot(event_names=("Core CPI m/m",), db_path=self.db_path)

        mock_compute.assert_not_called()
        self.assertEqual(result["total"], 0)

    def test_partial_reaction_is_reprocessed_and_completed(self):
        """Il caso reale del bug: una riga esiste ma con colonne NULL
        (es. price_t+5m..+60m mancanti per un rate limit parziale) — deve
        essere ritentata, non trattata come già fatta."""
        partial = {c: (4300.0 if c in ("price_t-5m", "price_t-1m") else None) for c in _PRICE_COLS}
        partial.update({"max_excursion_up": None, "max_excursion_down": None, "tick_count": 120})
        self._insert_event("uid-partial")
        self._insert_reaction("uid-partial", partial)

        with mock.patch("dukascopy_ticks.compute_reaction_features",
                         return_value=_complete_feats(4305.0)) as mock_compute:
            result = run_pilot(event_names=("Core CPI m/m",), db_path=self.db_path)

        mock_compute.assert_called_once()
        self.assertEqual(result["done"], 1)

        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM event_price_reactions WHERE event_uid='uid-partial'"
            ).fetchone()
        for c in _PRICE_COLS:
            self.assertIsNotNone(row[c], f"{c} doveva essere stato riempito dal retry")
            self.assertEqual(row[c], 4305.0)

    def test_still_partial_after_retry_keeps_previous_data_not_wiped(self):
        """Se il nuovo tentativo fallisce di nuovo (rate limit ancora
        attivo, compute_reaction_features ritorna None per zero tick nella
        finestra), la riga parziale precedente non deve essere cancellata —
        solo non aggiornata, in attesa del prossimo giro."""
        partial = {c: (4300.0 if c == "price_t-1m" else None) for c in _PRICE_COLS}
        partial.update({"max_excursion_up": None, "max_excursion_down": None, "tick_count": 10})
        self._insert_event("uid-still-partial")
        self._insert_reaction("uid-still-partial", partial)

        with mock.patch("dukascopy_ticks.compute_reaction_features", return_value=None):
            result = run_pilot(event_names=("Core CPI m/m",), db_path=self.db_path)

        self.assertEqual(result["empty"], 1)
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM event_price_reactions WHERE event_uid='uid-still-partial'"
            ).fetchone()
        self.assertEqual(row["price_t-1m"], 4300.0)  # dato precedente preservato

    def test_never_attempted_event_is_processed_as_before(self):
        """Comportamento invariato per il caso comune: nessuna riga
        esistente -> viene calcolata la prima volta."""
        self._insert_event("uid-new")

        with mock.patch("dukascopy_ticks.compute_reaction_features",
                         return_value=_complete_feats()) as mock_compute:
            result = run_pilot(event_names=("Core CPI m/m",), db_path=self.db_path)

        mock_compute.assert_called_once()
        self.assertEqual(result["done"], 1)


if __name__ == "__main__":
    unittest.main()
