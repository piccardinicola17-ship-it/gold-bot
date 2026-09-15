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
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from historical_events import HIST_DB_PATH, _connect
from historical_model import _theil_sen_fit, _r2, TRAIN_FRACTIONS, HORIZONS, SURPRISE_ZSCORE_CLIP
from trade_manager import BOT_DIR

logger = logging.getLogger(__name__)

# FIX (2026-09-15): il path originale puntava a un file nello scratchpad di
# una sessione precedente — sparito, esattamente lo stesso problema di
# project_sniper_candidates_not_reproducible.md (dati/risultati vissuti
# solo in scratchpad, persi a fine sessione). Path permanente in data/
# (gitignored ma persistente sul disco locale, non serve committarlo: si
# riscarica gratis in pochi secondi con fetch_cot_gold(), a differenza dei
# punteggi FOMC/BCE/BOJ che invece SONO committati perché costati lavoro
# reale — vedi nota in historical_events.py sui punteggi hawkish/dovish).
COT_JSON_PATH = BOT_DIR / "cot_gold.json"
COT_RELEASE_LAG_DAYS = 3  # report del martedi', pubblicato il venerdi' successivo

# CFTC Commitment of Traders (Futures Only), dataset pubblico Socrata —
# nessuna chiave richiesta. Filtrato su Gold/Comex al fetch, non sull'intero
# dataset (che copre tutte le materie prime).
_CFTC_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_CFTC_PAGE_SIZE = 1000  # limite Socrata per richiesta, serve paginare


def fetch_cot_gold(dest: Path = COT_JSON_PATH) -> int:
    """Scarica l'intero storico COT Gold/Comex (1986-oggi, settimanale) e
    lo salva in dest. Rieseguibile in qualunque momento per aggiornare i
    dati più recenti (idempotente: sovrascrive con lo storico completo)."""
    rows: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            _CFTC_API_URL,
            params={
                # cftc_market_code per Comex e' "CMX" nelle righe piu' vecchie
                # e "CMX " (spazio finale) da un certo punto in poi - trovato
                # empiricamente (2026-09-15): un filtro sul solo valore con
                # spazio dava 206 righe invece delle ~1930 attese, tagliando
                # fuori tutto lo storico prima del 2022.
                "$where": "contract_market_name='GOLD' AND trim(cftc_market_code)='CMX'",
                "$order": "report_date_as_yyyy_mm_dd ASC",
                "$limit": _CFTC_PAGE_SIZE,
                "$offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        rows.extend(page)
        offset += _CFTC_PAGE_SIZE
        if len(page) < _CFTC_PAGE_SIZE:
            break

    if len(rows) < 500:  # 1986-oggi settimanale sono ~2000+ righe attese
        raise ValueError(f"Righe COT sospettosamente poche: {len(rows)} (atteso >= 500)")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows))
    logger.info(f"COT Gold: {len(rows):,} righe scaricate -> {dest}")
    return len(rows)


def load_cot_series(path: Path = COT_JSON_PATH) -> pd.DataFrame:
    if not path.exists():
        fetch_cot_gold(path)
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

    direction_correct = np.sign(preds_conditional.to_numpy()) == np.sign(test[horizon].to_numpy())
    direction_acc = float(np.mean(direction_correct))

    return {
        "train_fraction": train_fraction, "n_train": len(train), "n_test": len(test),
        "r2_baseline": round(r2_baseline, 3), "r2_conditional_on_cot": round(r2_conditional, 3),
        # FIX (2026-09-15, stesso bug già trovato e corretto in
        # historical_fomc_text.py/historical_model.py l'11/09): senza "and
        # r2_conditional > 0" un modello "meno negativo" di un baseline
        # ANCH'ESSO negativo (es. -0.1 contro -1.3) passava come "aiuta",
        # nonostante nessuno dei due abbia vera capacità predittiva —
        # trovato qui rifacendo il verdetto COT da zero il 2026-09-15,
        # quasi tutte le righe "conditioning_helps=True" di prima
        # avevano r2_conditional_on_cot negativo.
        "conditioning_helps": bool(r2_conditional > r2_baseline and r2_conditional > 0),
        "direction_accuracy": round(direction_acc, 3),
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

    # DIRECTION_ACCURACY_FLOOR = 0.55, stessa soglia/stesso nome usato in
    # historical_fomc_text.py — un orizzonte conta come genuino solo se
    # "aiuta" (r2_conditional > r2_baseline E > 0) su TUTTI gli split E ha
    # direction_accuracy media >= 55%, altrimenti anche un R² tecnicamente
    # positivo ma minuscolo passerebbe come falso segnale.
    DIRECTION_ACCURACY_FLOOR = 0.55
    res_df = pd.DataFrame(results)
    helps_all_by_horizon = res_df.groupby("horizon")["conditioning_helps"].agg(lambda s: s.all())
    dir_acc_by_horizon = res_df.groupby("horizon")["direction_accuracy"].mean()
    genuine_horizons = [
        h for h in helps_all_by_horizon.index
        if helps_all_by_horizon[h] and dir_acc_by_horizon[h] >= DIRECTION_ACCURACY_FLOOR
    ]
    n_genuine = len(genuine_horizons)

    if len(df) < 100:
        verdict = f"DATI INSUFFICIENTI PER UN VERDETTO AFFIDABILE (n={len(df)}, serve n>=100)"
    elif n_genuine > 0:
        verdict = (
            f"IL POSIZIONAMENTO COT MIGLIORA LA PREVISIONE, GENUINO, su "
            f"{n_genuine}/{len(HORIZONS)} orizzonti ({', '.join(genuine_horizons)}, n={len(df)})"
        )
    else:
        verdict = f"Il posizionamento COT non aggiunge nulla di genuino — nessun miglioramento consistente (n={len(df)})"

    return {
        "available": True, "n": len(df), "verdict": verdict,
        "genuine_horizons": genuine_horizons, "details": results,
    }


# Solo gli eventi dove l'orizzonte GIA' deployato in produzione
# (historical_model.DEPLOYED_EVENTS) coincide con un orizzonte "genuino"
# trovato da validate_cot_conditioning() — validato il 2026-09-15.
# ISM Services PMI (genuino solo a 5m/15m, deployato a 1m) e PPI y/y
# (genuino a 1m/15m, deployato a 30m) mostrano un effetto COT reale ma NON
# sull'orizzonte che il bot mostra oggi: applicarlo li' richiederebbe
# prima cambiare l'orizzonte deployato di quegli eventi, una decisione
# separata non presa qui — lasciati fuori deliberatamente, non dimenticati.
COT_CONDITIONED_EVENTS = ("CB Consumer Confidence", "ISM Manufacturing PMI")

COT_MODELS_PATH = Path(__file__).resolve().parent / "cot_models.json"


def fit_final_cot_model(event_name: str, horizon: str, db_path: str = HIST_DB_PATH) -> dict:
    """Calibra il modello condizionato-COT da usare in produzione, stesso
    principio di historical_model.fit_final_model() ma con due pendenze
    (crowded/calm) invece di una — fittate su TUTTI i dati disponibili,
    non su uno split di validazione (quello lo fa già
    validate_cot_conditioning()). La soglia crowded/calm è la mediana di
    |cot_zscore| sull'INTERO storico disponibile, non ricalcolata ad ogni
    split come in validate_cot_conditioning (li' serviva per non guardare
    il futuro nel training di ogni split; qui stiamo preparando l'unico
    artefatto finale, non misurando quanto regge)."""
    with _connect(db_path) as conn:
        raw = pd.read_sql_query(
            "SELECT datetime_utc, surprise_raw, surprise_zscore, " + horizon + " AS y "
            "FROM event_features WHERE event_name=? AND surprise_zscore IS NOT NULL",
            conn, params=(event_name,),
        )
    raw = raw.dropna(subset=["surprise_zscore", "y"])
    if len(raw) < 20:
        raise ValueError(f"Dati insufficienti per calibrare {event_name} (n={len(raw)})")

    cot_df = load_cot_series()
    raw["cot_zscore"] = cot_asof(pd.to_datetime(raw["datetime_utc"]), cot_df).to_numpy()
    raw = raw.dropna(subset=["cot_zscore"])
    if len(raw) < 20:
        raise ValueError(f"Dati insufficienti dopo il merge COT per {event_name} (n={len(raw)})")

    threshold = float(raw["cot_zscore"].abs().median())
    crowded = raw[raw["cot_zscore"].abs() >= threshold]
    calm = raw[raw["cot_zscore"].abs() < threshold]
    if len(crowded) < 10 or len(calm) < 10:
        raise ValueError(f"Split crowded/calm troppo sbilanciato per {event_name} ({len(crowded)}/{len(calm)})")

    slope_crowded, intercept_crowded = _theil_sen_fit(
        np.clip(crowded["surprise_zscore"].to_numpy(), -SURPRISE_ZSCORE_CLIP, SURPRISE_ZSCORE_CLIP),
        crowded["y"].to_numpy(),
    )
    slope_calm, intercept_calm = _theil_sen_fit(
        np.clip(calm["surprise_zscore"].to_numpy(), -SURPRISE_ZSCORE_CLIP, SURPRISE_ZSCORE_CLIP),
        calm["y"].to_numpy(),
    )

    return {
        "event_name": event_name,
        "horizon": horizon,
        "cot_zscore_threshold": threshold,
        "slope_crowded": slope_crowded, "intercept_crowded": intercept_crowded,
        "slope_calm": slope_calm, "intercept_calm": intercept_calm,
        "surprise_mean": float(raw["surprise_raw"].mean()),
        "surprise_std": float(raw["surprise_raw"].std()),
        "surprise_zscore_clip": SURPRISE_ZSCORE_CLIP,
        "n_crowded": int(len(crowded)), "n_calm": int(len(calm)),
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
    }


def current_cot_zscore() -> float | None:
    """Ultimo net_long_zscore GIA' pubblico OGGI — usato dal bot live per
    scegliere crowded/calm sul prossimo evento in arrivo. None se i dati
    COT non sono disponibili (mai un valore indovinato)."""
    try:
        cot_df = load_cot_series()
    except Exception as e:
        logger.warning(f"COT non disponibile: {e}")
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    published = cot_df[cot_df["release_date"] <= now]
    if published.empty:
        return None
    return float(published.iloc[-1]["net_long_zscore"])


def regenerate_cot_models(db_path: str = HIST_DB_PATH) -> None:
    """Rigenera cot_models.json (root del progetto, committato e
    deployato col codice, stesso trattamento di macro_models.json) —
    fallisce rumorosamente se un evento in COT_CONDITIONED_EVENTS non
    risulta genuino su TUTTI gli split per il suo orizzonte deployato,
    stesso principio di historical_model.regenerate_deployed_models()."""
    from historical_model import DEPLOYED_EVENTS

    models = {}
    for name in COT_CONDITIONED_EVENTS:
        horizon = DEPLOYED_EVENTS[name]
        result = validate_cot_conditioning(name, db_path=db_path)
        if horizon not in result.get("genuine_horizons", []):
            raise ValueError(
                f"{name} @ {horizon}: il condizionamento COT non risulta genuino su "
                f"questo orizzonte (vedi validate_cot_conditioning) — non lo deploy alla cieca."
            )
        models[name] = fit_final_cot_model(name, horizon=horizon, db_path=db_path)

    COT_MODELS_PATH.write_text(json.dumps(models, indent=2))
    logger.info(f"Modelli COT deployati rigenerati: {list(models)} -> {COT_MODELS_PATH}")


def check_cot_conditioned_events_health(db_path: str = HIST_DB_PATH) -> dict:
    """Come historical_model.check_deployed_events_health, per i modelli
    condizionati-COT — sola lettura, mai rigenera cot_models.json."""
    from historical_model import DEPLOYED_EVENTS

    report = {}
    for name in COT_CONDITIONED_EVENTS:
        horizon = DEPLOYED_EVENTS[name]
        result = validate_cot_conditioning(name, db_path=db_path)
        healthy = horizon in result.get("genuine_horizons", [])
        report[name] = {"horizon": horizon, "healthy": healthy, "n": result.get("n")}
    return report


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
