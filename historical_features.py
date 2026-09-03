"""
historical_features.py — Fase 3 del progetto dati storici.

Trasforma macro_events + event_price_reactions (Fasi 1-2) in una tabella
di feature/label pronta per il training (Fase 4): per ogni evento con un
dato numerico reale (actual vs forecast) e una reazione di prezzo valida,
calcola la "sorpresa" normalizzata e le variazioni di prezzo a vari
orizzonti.

ATTENZIONE AL LOOK-AHEAD BIAS — il punto più delicato di questa fase:
la sorpresa (actual - forecast) va normalizzata per essere comparabile tra
serie diverse (NFP si misura in centinaia di migliaia, CPI in decimi di
punto percentuale), ma la normalizzazione (media/deviazione standard) deve
usare SOLO gli eventi della stessa serie già avvenuti PRIMA di quello in
esame — mai eventi futuri. Se si usasse tutta la storia (passata e futura)
per normalizzare ogni evento, il modello vedrebbe implicitamente "quanto
sarà volatile questa serie in futuro", un'informazione che nella realtà
non avrebbe mai al momento della previsione. Qui si usa una statistica
espandente (expanding mean/std) calcolata evento per evento in ordine
cronologico, con un minimo di occorrenze precedenti richieste prima di
fidarsi della normalizzazione (MIN_PRIOR_OCCURRENCES).

Solo le serie con un valore numerico reale (actual_num/forecast_num non
NULL) entrano in questa fase — eventi qualitativi come "FOMC Meeting
Minutes" o i discorsi Fed non hanno una "sorpresa" calcolabile e restano
fuori (lo screening a bias qualitativo dell'LLM in news_analyst.py resta
l'unico strumento per quelli, invariato).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from historical_events import HIST_DB_PATH, _connect

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Occorrenze precedenti minime della stessa serie richieste prima di fidarsi
# della sorpresa normalizzata (expanding mean/std troppo instabile con pochi
# campioni). Le prime occorrenze di ogni serie restano nel dataset con
# surprise_zscore NULL — visibili ma non utilizzabili come feature.
MIN_PRIOR_OCCURRENCES = 5

REACTION_HORIZONS_MIN = [1, 5, 15, 30, 60]


def _load_raw(db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                e.event_uid, e.event_name, e.macro_category, e.datetime_utc,
                e.actual_num, e.forecast_num, e.previous_num,
                r."price_t-1m" AS price_base,
                r."price_t+1m" AS price_t1,
                r."price_t+5m" AS price_t5,
                r."price_t+15m" AS price_t15,
                r."price_t+30m" AS price_t30,
                r."price_t+60m" AS price_t60,
                r.max_excursion_up, r.max_excursion_down, r.tick_count
            FROM macro_events e
            JOIN event_price_reactions r ON e.event_uid = r.event_uid
            WHERE e.actual_num IS NOT NULL AND e.forecast_num IS NOT NULL
              AND r."price_t-1m" IS NOT NULL
            ORDER BY e.datetime_utc ASC
            """,
            conn,
        )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def _add_expanding_surprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola surprise_raw (actual-forecast) e surprise_zscore per ciascuna
    serie (event_name), usando solo la storia precedente di quella stessa
    serie — vedi il commento in testa al file sul look-ahead bias.
    """
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["surprise_raw"] = df["actual_num"] - df["forecast_num"]

    zscores = pd.Series(index=df.index, dtype=float)
    n_prior_col = pd.Series(index=df.index, dtype="Int64")

    for name, group in df.groupby("event_name", sort=False):
        group = group.sort_values("datetime_utc")
        surprises = group["surprise_raw"]
        # shift(1) esclude l'evento corrente dalla propria statistica —
        # solo eventi strettamente precedenti della stessa serie.
        expanding_mean = surprises.shift(1).expanding().mean()
        expanding_std  = surprises.shift(1).expanding().std()
        n_prior = surprises.shift(1).expanding().count()

        z = (surprises - expanding_mean) / expanding_std
        z[(n_prior < MIN_PRIOR_OCCURRENCES) | (expanding_std == 0) | expanding_std.isna()] = float("nan")

        zscores.loc[group.index] = z
        n_prior_col.loc[group.index] = n_prior.fillna(0).astype(int)

    df["surprise_zscore"] = zscores
    df["n_prior_occurrences"] = n_prior_col
    return df


def _add_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """Variazione di prezzo dalla baseline (t-1m, appena prima dell'uscita) a ogni orizzonte."""
    for m, col in zip(REACTION_HORIZONS_MIN, ["price_t1", "price_t5", "price_t15", "price_t30", "price_t60"]):
        df[f"reaction_{m}m"] = df[col] - df["price_base"]
    return df


def init_features_table(db_path: str = HIST_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS event_features (
                event_uid           TEXT PRIMARY KEY REFERENCES macro_events(event_uid),
                event_name          TEXT NOT NULL,
                macro_category      TEXT,
                datetime_utc        TEXT NOT NULL,
                actual_num          REAL,
                forecast_num        REAL,
                previous_num        REAL,
                surprise_raw        REAL,
                surprise_zscore     REAL,
                n_prior_occurrences INTEGER,
                reaction_1m         REAL,
                reaction_5m         REAL,
                reaction_15m        REAL,
                reaction_30m        REAL,
                reaction_60m        REAL,
                max_excursion_up    REAL,
                max_excursion_down  REAL,
                tick_count          INTEGER,
                computed_at         TEXT NOT NULL
            );
            """
        )


def build_features(db_path: str = HIST_DB_PATH) -> dict:
    init_features_table(db_path)
    df = _load_raw(db_path)
    if df.empty:
        logger.warning("Nessun evento con actual/forecast e reazione di prezzo valida trovato.")
        return {"total": 0}

    df = _add_expanding_surprise(df)
    df = _add_reactions(df)

    now_iso = datetime.now(timezone.utc).isoformat()
    cols = [
        "event_uid", "event_name", "macro_category", "datetime_utc",
        "actual_num", "forecast_num", "previous_num",
        "surprise_raw", "surprise_zscore", "n_prior_occurrences",
        "reaction_1m", "reaction_5m", "reaction_15m", "reaction_30m", "reaction_60m",
        "max_excursion_up", "max_excursion_down", "tick_count",
    ]
    with _connect(db_path) as conn:
        for _, row in df.iterrows():
            values = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    values.append(None)
                elif c == "datetime_utc":
                    values.append(v.isoformat())
                else:
                    values.append(v)
            placeholders = ", ".join("?" for _ in cols) + ", ?"
            conn.execute(
                f"INSERT OR REPLACE INTO event_features ({', '.join(cols)}, computed_at) "
                f"VALUES ({placeholders})",
                (*values, now_iso),
            )

    usable = df["surprise_zscore"].notna().sum()
    logger.info(f"Feature calcolate: {len(df)} eventi totali, {usable} con surprise_zscore valido (n_prior>={MIN_PRIOR_OCCURRENCES})")
    return {"total": len(df), "usable": int(usable)}


def summarize(db_path: str = HIST_DB_PATH) -> None:
    """Statistiche descrittive per serie — NON un modello, solo un controllo di sanità dei dati prima della Fase 4."""
    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM event_features WHERE surprise_zscore IS NOT NULL", conn)
    if df.empty:
        print("Nessuna feature con surprise_zscore valido.")
        return

    print(f"Eventi utilizzabili (surprise_zscore valido): {len(df)}\n")
    for name, group in df.groupby("event_name"):
        corr30 = group["surprise_zscore"].corr(group["reaction_30m"])
        print(
            f"{name:35s} n={len(group):4d}  "
            f"corr(surprise, reaction_30m)={corr30:+.3f}  "
            f"media|reaction_30m|=${group['reaction_30m'].abs().mean():.2f}"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fase 3: feature engineering dai dati storici")
    parser.add_argument("--verify-only", action="store_true", help="Salta il ricalcolo, mostra solo le statistiche")
    args = parser.parse_args()
    if not args.verify_only:
        result = build_features()
        print(result)
    summarize()
