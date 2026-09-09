"""
historical_cot_feature.py — Progetto separato: posizionamento speculativo
(COT) come modificatore della sorpresa macro.

Idea (2026-09-09, richiesta esplicita di massimizzare l'analisi news):
historical_model.py testa "sorpresa -> reazione" nello stesso modo per
ogni evento della stessa serie, indipendentemente da quanto il mercato sia
già posizionato in una direzione. Ipotesi: una sorpresa hawkish quando gli
speculativi sono già molto lunghi (posizione "affollata") potrebbe causare
una reazione più ampia (chiusura forzata di posizioni) di una sorpresa
identica con posizionamento neutro — potrebbe spiegare perché alcune serie
(es. Unemployment Rate, validata SENZA edge il 2026-09-09) non mostrano
nulla se il segnale reale dipende dal posizionamento e viene annegato
nella media.

Fonte: report CFTC Commitment of Traders, Comex Gold, non-commercial
(speculativo) — già scaricato in una sessione precedente
(scratchpad/cot_gold.json, 1986-2026, settimanale). Pubblicato ogni
venerdì con i dati del martedì precedente (verificato sul pattern delle
date più recenti: sempre martedì) — LOOKAHEAD: per un evento del giorno D,
è "noto" solo il report il cui venerdì di pubblicazione (report_date + 3
giorni) è già passato rispetto a D.

Metrica: net_long_pct = (long - short) / open_interest, poi z-score
espandente (stessa logica look-ahead-safe di historical_features.py) per
renderlo comparabile nel tempo (l'open interest è cresciuto ~3x dal 1986).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_events import HIST_DB_PATH, _connect
from historical_model import _theil_sen_fit, _r2, TRAIN_FRACTIONS, HORIZONS

COT_JSON_PATH = Path(
    "/private/tmp/claude-501/-Users-nico-piccardi-Desktop-gold-bot/"
    "c68a52aa-24a5-496a-8fa7-f9c008a7a9a7/scratchpad/cot_gold.json"
)
COT_RELEASE_LAG_DAYS = 3  # report del martedi', pubblicato il venerdi' successivo


def load_cot_series(path: Path = COT_JSON_PATH) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    for col in ("noncomm_positions_long_all", "noncomm_positions_short_all", "open_interest_all"):
        df[col] = df[col].astype(float)
    df["net_long_pct"] = (
        (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / df["open_interest_all"]
    )
    df = df.sort_values("report_date").reset_index(drop=True)
    # z-score espandente look-ahead-safe, stessa logica di historical_features.py
    shifted = df["net_long_pct"].shift(1)
    df["net_long_zscore"] = (df["net_long_pct"] - shifted.expanding().mean()) / shifted.expanding().std()
    df["release_date"] = df["report_date"] + pd.Timedelta(days=COT_RELEASE_LAG_DAYS)
    return df[["report_date", "release_date", "net_long_pct", "net_long_zscore"]]


def cot_asof(event_dates: pd.Series, cot_df: pd.DataFrame) -> pd.Series:
    """Per ogni data evento, l'ultimo net_long_zscore GIA' PUBBLICO
    (release_date < data evento) — merge_asof rispetta l'ordine, nessun
    lookahead."""
    events = pd.DataFrame({"event_date": pd.to_datetime(event_dates).dt.tz_localize(None)}).sort_values("event_date")
    merged = pd.merge_asof(
        events, cot_df.sort_values("release_date"),
        left_on="event_date", right_on="release_date", direction="backward",
    )
    return merged.set_index(events.index)["net_long_zscore"]


def _evaluate_interaction_split(df: pd.DataFrame, horizon: str, train_fraction: float) -> dict | None:
    """Confronta due modelli sullo stesso split: sorpresa da sola vs
    sorpresa*posizionamento come feature aggiuntiva (regressione robusta
    a 2 variabili non disponibile via Theil-Sen univariato - qui si separa
    in due gruppi, posizionamento sopra/sotto la mediana, e si guarda se
    la relazione sorpresa->reazione e' diversa tra i due gruppi)."""
    valid = df.dropna(subset=["surprise_zscore", horizon, "cot_zscore"])
    if len(valid) < 20:
        return None
    split_idx = int(len(valid) * train_fraction)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]
    if len(train) < 10 or len(test) < 5:
        return None

    train_crowded = train[train["cot_zscore"].abs() >= train["cot_zscore"].abs().median()]
    train_calm = train[train["cot_zscore"].abs() < train["cot_zscore"].abs().median()]
    if len(train_crowded) < 5 or len(train_calm) < 5:
        return None

    slope_c, intercept_c = _theil_sen_fit(train_crowded["surprise_zscore"].to_numpy(), train_crowded[horizon].to_numpy())
    slope_k, intercept_k = _theil_sen_fit(train_calm["surprise_zscore"].to_numpy(), train_calm[horizon].to_numpy())
    slope_all, intercept_all = _theil_sen_fit(train["surprise_zscore"].to_numpy(), train[horizon].to_numpy())

    crowded_threshold = train["cot_zscore"].abs().median()
    test_crowded = test[test["cot_zscore"].abs() >= crowded_threshold]
    test_calm = test[~test.index.isin(test_crowded.index)]

    r2_baseline = _r2(test[horizon].to_numpy(), slope_all * test["surprise_zscore"].to_numpy() + intercept_all)

    preds_conditional = pd.Series(index=test.index, dtype=float)
    if len(test_crowded):
        preds_conditional.loc[test_crowded.index] = slope_c * test_crowded["surprise_zscore"].to_numpy() + intercept_c
    if len(test_calm):
        preds_conditional.loc[test_calm.index] = slope_k * test_calm["surprise_zscore"].to_numpy() + intercept_k
    r2_conditional = _r2(test[horizon].to_numpy(), preds_conditional.to_numpy())

    return {
        "train_fraction": train_fraction, "n_train": len(train), "n_test": len(test),
        "r2_baseline": round(r2_baseline, 3), "r2_conditional_on_cot": round(r2_conditional, 3),
        "conditioning_helps": bool(r2_conditional > r2_baseline),
        "slope_crowded": round(slope_c, 3), "slope_calm": round(slope_k, 3),
    }


def validate_cot_conditioning(event_name: str, db_path: str = HIST_DB_PATH) -> dict:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT datetime_utc, surprise_zscore, reaction_1m, reaction_5m, reaction_15m, "
            "reaction_30m, reaction_60m FROM event_features "
            "WHERE event_name=? AND surprise_zscore IS NOT NULL",
            conn, params=(event_name,),
        )
    if df.empty:
        return {"available": False, "reason": "nessun dato in event_features per questa serie"}

    cot_df = load_cot_series()
    df["cot_zscore"] = cot_asof(pd.to_datetime(df["datetime_utc"]), cot_df).to_numpy()
    df = df.dropna(subset=["cot_zscore"])

    results = []
    for horizon in HORIZONS:
        for frac in TRAIN_FRACTIONS:
            r = _evaluate_interaction_split(df, horizon, frac)
            if r:
                r["horizon"] = horizon
                results.append(r)

    if not results:
        return {"available": True, "n": len(df), "verdict": "DATI INSUFFICIENTI per il confronto condizionato"}

    res_df = pd.DataFrame(results)
    helps_all_by_horizon = res_df.groupby("horizon")["conditioning_helps"].agg(lambda s: s.all())
    n_helps = int(helps_all_by_horizon.sum())

    if len(df) < 100:
        verdict = f"DATI INSUFFICIENTI PER UN VERDETTO AFFIDABILE (n={len(df)}, serve n>=100)"
    elif n_helps > 0:
        verdict = f"IL POSIZIONAMENTO COT MIGLIORA LA PREVISIONE su {n_helps}/{len(HORIZONS)} orizzonti (n={len(df)})"
    else:
        verdict = f"Il posizionamento COT non aggiunge nulla — nessun miglioramento consistente (n={len(df)})"

    return {"available": True, "n": len(df), "verdict": verdict, "details": results}


if __name__ == "__main__":
    import sys
    event_name = sys.argv[1] if len(sys.argv) > 1 else "Unemployment Rate"
    result = validate_cot_conditioning(event_name)
    print(f"Serie: {event_name}")
    print(f"n: {result.get('n')}")
    print(f"Verdetto: {result.get('verdict')}")
    if result.get("details"):
        for r in result["details"]:
            print(f"  {r['horizon']:14s} split={r['train_fraction']:.1f} n_test={r['n_test']:3d} "
                  f"R2_base={r['r2_baseline']:+.3f} R2_cond={r['r2_conditional_on_cot']:+.3f} "
                  f"aiuta={r['conditioning_helps']} slope_crowded={r['slope_crowded']:+.3f} slope_calm={r['slope_calm']:+.3f}")
