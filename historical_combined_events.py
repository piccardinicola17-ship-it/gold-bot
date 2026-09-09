"""
historical_combined_events.py — Progetto separato: sorpresa COMBINATA per
report macro rilasciati insieme.

Idea (2026-09-09, richiesta esplicita di massimizzare l'analisi news):
diversi eventi macro escono nello STESSO istante esatto perché fanno parte
dello stesso comunicato ufficiale — es. Non-Farm Employment Change,
Unemployment Rate e Average Hourly Earnings m/m sono tutti componenti
dell'Employment Situation Report del BLS, stesso giorno stesso minuto
(verificato su datetime_utc esatti nel DB). Il modello esistente
(historical_model.py) tratta ogni serie in isolamento — ma il mercato
reagisce al REPORT nel suo complesso, non a un numero alla volta. Ipotesi
concreta: Unemployment Rate da sola non ha mostrato edge (validato
2026-09-09, n=145), ma potrebbe comunque contribuire informazione utile se
combinata con NFP e Average Hourly Earnings in un punteggio unico.

Metodo: per ogni istante di rilascio condiviso, media dei surprise_zscore
già calcolati da historical_features.py per le serie componenti disponibili
in quel momento (non richiede che TUTTE e 3 siano presenti — alcune
mancano in certi anni). Stessa validazione Theil-Sen + split cronologico
di historical_model.py, sulla stessa reazione di prezzo (condivisa, è lo
stesso istante per costruzione).

Richiede event_features popolata (historical_features.py) con reazioni di
prezzo disponibili per le serie coinvolte — la copertura sta ancora
crescendo mentre il job Fase 2 esteso gira in background.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_events import HIST_DB_PATH, _connect
from historical_model import _theil_sen_fit, _r2, TRAIN_FRACTIONS, HORIZONS

# Componenti dell'Employment Situation Report BLS (stesso giorno stesso
# minuto, verificato su datetime_utc esatti). Non-Farm Employment Change è
# la piu' seguita dal mercato, le altre due meno mai isolatamente
# significative finora (ADP simile, Unemployment Rate senza edge).
EMPLOYMENT_SITUATION_COMPONENTS = (
    "Non-Farm Employment Change", "Unemployment Rate", "Average Hourly Earnings m/m",
)


def build_combined_score(components: tuple, db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    """Un punteggio combinato per ogni istante di rilascio condiviso: media
    dei surprise_zscore delle componenti disponibili in quel momento (min 2
    su 3, altrimenti non è davvero "combinato")."""
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT datetime_utc, event_name, surprise_zscore, reaction_1m, reaction_5m,
                   reaction_15m, reaction_30m, reaction_60m
            FROM event_features
            WHERE event_name IN ({",".join("?" for _ in components)})
              AND surprise_zscore IS NOT NULL
            """,
            conn, params=components,
        )
    if df.empty:
        return df

    grouped = df.groupby("datetime_utc")
    rows = []
    for dt, g in grouped:
        if len(g) < 2:
            continue  # serve almeno 2 componenti per parlare di "combinato"
        row = {
            "datetime_utc": dt,
            "combined_zscore": g["surprise_zscore"].mean(),
            "n_components": len(g),
            "components": ",".join(sorted(g["event_name"])),
        }
        for h in HORIZONS:
            row[h] = g[h].iloc[0]  # stessa reazione per tutte le componenti, stesso istante
        rows.append(row)
    return pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)


def _evaluate_split(df: pd.DataFrame, horizon: str, train_fraction: float) -> dict | None:
    """Stessa logica di historical_model._evaluate_split, adattata al
    punteggio combinato invece del surprise_zscore di una singola serie."""
    valid = df.dropna(subset=["combined_zscore", horizon])
    if len(valid) < 20:
        return None
    split_idx = int(len(valid) * train_fraction)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]
    if len(train) < 10 or len(test) < 5:
        return None

    slope, intercept = _theil_sen_fit(train["combined_zscore"].to_numpy(), train[horizon].to_numpy())
    pred_test = slope * test["combined_zscore"].to_numpy() + intercept
    r2_test = _r2(test[horizon].to_numpy(), pred_test)

    naive_pred = np.full(len(test), train[horizon].mean())
    r2_naive = _r2(test[horizon].to_numpy(), naive_pred)

    direction_correct = np.sign(pred_test) == np.sign(test[horizon].to_numpy())
    direction_acc = float(direction_correct.mean())

    return {
        "train_fraction": train_fraction, "n_train": len(train), "n_test": len(test),
        "r2_test": round(r2_test, 3), "r2_naive_test": round(r2_naive, 3),
        "beats_naive": bool(r2_test > r2_naive),
        "direction_accuracy": round(direction_acc, 3),
    }


def validate_combined_score(components: tuple = EMPLOYMENT_SITUATION_COMPONENTS, db_path: str = HIST_DB_PATH) -> dict:
    df = build_combined_score(components, db_path)
    if df.empty or len(df) < 20:
        return {
            "available": len(df) > 0,
            "n": len(df),
            "verdict": f"DATI INSUFFICIENTI (n={len(df)}, serve almeno 20 rilasci condivisi con reazione — "
                       f"la copertura sta ancora crescendo, vedi job dukascopy_ticks.py in corso)",
        }

    results = []
    for horizon in HORIZONS:
        for frac in TRAIN_FRACTIONS:
            r = _evaluate_split(df, horizon, frac)
            if r:
                r["horizon"] = horizon
                results.append(r)

    if not results:
        return {"available": True, "n": len(df), "verdict": "DATI INSUFFICIENTI per ogni combinazione orizzonte/split"}

    res_df = pd.DataFrame(results)
    beats_all_by_horizon = (
        res_df.groupby("horizon")["beats_naive"].agg(lambda s: s.all())
    )
    n_horizons_ok = int(beats_all_by_horizon.sum())
    # Stessa soglia n>=100 usata in historical_model.summarize() per ogni
    # altra serie di questo progetto — sotto quella soglia un "batte naive
    # su tutti gli split" è spesso solo rumore su un campione piccolo (qui
    # verificato: succede con n=35, ma la direction accuracy salta tra 14%
    # e 64% a seconda dello split, incoerente con un segnale vero). Mai
    # dichiarare edge sotto la stessa soglia usata ovunque nel progetto.
    if len(df) < 100:
        verdict = f"DATI INSUFFICIENTI PER UN VERDETTO AFFIDABILE (n={len(df)}, serve n>=100 — segnale preliminare non ancora fidabile)"
    elif n_horizons_ok > 0:
        verdict = f"EDGE VALIDATO su {n_horizons_ok}/{len(HORIZONS)} orizzonti (n={len(df)})"
    else:
        verdict = f"NESSUN EDGE — risultato pulito (n={len(df)})"

    return {
        "available": True, "n": len(df), "components": components,
        "verdict": verdict, "details": results,
    }


if __name__ == "__main__":
    result = validate_combined_score()
    print(f"Componenti: {EMPLOYMENT_SITUATION_COMPONENTS}")
    print(f"n rilasci condivisi con >=2 componenti disponibili: {result.get('n')}")
    print(f"Verdetto: {result.get('verdict')}")
    if result.get("details"):
        print("\nDettaglio per orizzonte/split:")
        for r in result["details"]:
            print(f"  {r['horizon']:14s} split={r['train_fraction']:.1f} n_test={r['n_test']:3d} "
                  f"R2={r['r2_test']:+.3f} R2_naive={r['r2_naive_test']:+.3f} "
                  f"batte_naive={r['beats_naive']} dir_acc={r['direction_accuracy']:.1%}")
