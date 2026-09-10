"""
historical_cb_text.py — Progetto separato: hawkish/dovish scoring per
BCE (Banca Centrale Europea) e BOJ (Bank of Japan), stesso principio di
historical_fomc_text.py (di cui riusa le funzioni generiche — vedi
save_text_scores/validate_fomc_scores lì, non duplicate qui per evitare
il pattern di bug più comune di questo progetto, vedi
[[feedback-dual-mechanism-drift-pattern]]).

Perché l'oro dovrebbe reagire a BCE/BOJ e non solo alla Fed: sono le
altre due banche centrali di primo piano al mondo, il cui posizionamento
di policy relativo alla Fed muove EUR/USD e USD/JPY — e con essi il
dollaro su base commerciale ponderata (DXY), che è uno dei driver più
diretti del prezzo dell'oro (oro quotato in USD).

Fonti (91+99+37 testi, 2007-2025, fetchati e scorati da sotto-agenti
paralleli con lo STESSO criterio/scala -2/+2 usato per la Fed, con
contesto storico specifico per ciascuna banca centrale nel prompt di
scoring):
1. ECB Press Conference — 91/91 dichiarazione+Q&A del Presidente/
   Presidentessa BCE. Punteggi in data/ecb_presconf_hawkish_scores.json.
2. Monetary Policy Statement (JPY) — 99/99 statement brevi BOJ (stesso
   event_name di un piccolo statement EUR, n=10, per cui NON vale la pena
   scorare: strutturalmente non potrà mai superare n>=100 — vedi
   filtro currency='JPY' sotto). Punteggi in
   data/boj_statement_hawkish_scores.json.
3. BOJ Outlook Report — 37/38 fetchati ma DELIBERATAMENTE NON scorati:
   n=38 non supererà mai n>=100, il lavoro di scoring (report lunghissimi,
   100-177k caratteri ciascuno) non sarebbe ripagato da un verdetto
   possibile.
4. BOJ Press Conference — limite STRUTTURALE confermato (2026-09-10): la
   BOJ non pubblica alcuna trascrizione/riassunto in inglese delle
   proprie conferenze stampa (dichiarato esplicitamente dalla BOJ
   stessa). Rimossa da EXTENDED_EVENT_NAMES in dukascopy_ticks.py.
5. ECB Monetary Policy Meeting Accounts (EUR, n=12) — NON scorata per lo
   stesso motivo di (3): campione strutturalmente troppo piccolo.

Locale, non tocca il bot live — stesso principio di historical_events.py.
"""

from __future__ import annotations

import os

from historical_events import HIST_DB_PATH
from historical_fomc_text import save_text_scores, validate_fomc_scores as validate_text_scores

ECB_PRESCONF_SCORES_PATH = os.path.join(os.path.dirname(__file__), "data", "ecb_presconf_hawkish_scores.json")
BOJ_STATEMENT_SCORES_PATH = os.path.join(os.path.dirname(__file__), "data", "boj_statement_hawkish_scores.json")


def save_all_cb_scores(db_path: str = HIST_DB_PATH) -> dict:
    n_ecb = save_text_scores("ECB Press Conference", ECB_PRESCONF_SCORES_PATH, db_path, currency="EUR")
    n_boj = save_text_scores("Monetary Policy Statement", BOJ_STATEMENT_SCORES_PATH, db_path, currency="JPY")
    return {"ecb_presconf": n_ecb, "boj_statement": n_boj}


if __name__ == "__main__":
    saved = save_all_cb_scores()
    print(f"Punteggi salvati: {saved}")
    for label, ev, cur in (
        ("ECB Press Conference", "ECB Press Conference", "EUR"),
        ("BOJ Statement", "Monetary Policy Statement", "JPY"),
    ):
        result = validate_text_scores(event_name=ev, currency=cur)
        print(f"\n=== {label} ===")
        print(f"n={result.get('n')} disponibile={result.get('available')} "
              f"verdetto={result.get('verdict', result.get('reason'))}")
