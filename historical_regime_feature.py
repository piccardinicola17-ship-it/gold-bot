"""
historical_regime_feature.py — Progetto separato: condizionamento dei
modelli di sorpresa esistenti al regime di mercato (trending/ranging).

Idea (2026-09-09, richiesta esplicita di completare il discorso news):
historical_model.py assume che la relazione sorpresa->reazione sia
COSTANTE nel tempo. Ma la reazione dell'oro a una sorpresa macro potrebbe
essere sistematicamente diversa in un mercato in trend forte (l'oro si
muove con il trend, "assorbe" rumore di breve termine) rispetto a un
mercato laterale (più sensibile a ogni notizia, meno direzione dominante
da vincere). Stesso pattern di historical_cot_feature.py (condizionamento
al posizionamento) applicato al regime di prezzo invece che al
posizionamento speculativo.

Regime: ADX(14) giornaliero sull'oro, STESSA soglia (20) e STESSA libreria
(ta.trend.ADXIndicator) già usate dal bot LIVE in analyzer.py
(trend_following_strategy) — non una soglia nuova inventata per
l'occasione, per restare coerenti con cosa il bot considera già
"trending" nell'operatività quotidiana. ADX >= 20 sulla chiusura del
giorno PRECEDENTE l'evento = regime TRENDING, ADX < 20 = RANGING. Mai il
giorno stesso dell'evento (la barra giornaliera non è ancora chiusa
quando l'evento avviene durante la sessione — look-ahead safe).

Dati: candele giornaliere XAU/USD 2006-2025 da dukascopy-python
(data/xauusd_daily_2006_2025.csv, 6371 barre) — vedi memoria
reference_dukascopy_python.md per come sono generate (libreria richiede
Python 3.10+, mai installata nell'ambiente del bot che resta su 3.9).

Testato sulle 6 serie già DEPLOYATE in macro_predictor.py (FRED_SERIES +
RSS_SERIES) — se il regime rivela un edge nascosto o rafforza un edge
già noto su una di queste, è direttamente applicabile al bot live.

Stesso standard Theil-Sen + split cronologico multiplo + soglia n>=100
PER REGIME usato ovunque nel progetto (historical_model.py,
historical_combined_events.py, historical_cot_feature.py,
historical_fomc_text.py) — condizionare su regime DIMEZZA il campione
disponibile per ciascun sotto-test, quindi n>=100 per regime è un
traguardo lontano per la maggior parte di queste serie nel breve termine,
ma il codice è pronto per quando la copertura storica crescerà.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from historical_events import HIST_DB_PATH, _connect
from historical_model import HORIZONS, TRAIN_FRACTIONS, _r2, _theil_sen_fit

DAILY_PRICES_PATH = "data/xauusd_daily_2006_2025.csv"
ADX_WINDOW = 14
ADX_TREND_THRESHOLD = 20  # stessa soglia di analyzer.trend_following_strategy

DEPLOYED_EVENTS = (
    "Core CPI m/m", "Unemployment Claims", "ADP Non-Farm Employment Change",
    "ISM Manufacturing PMI", "ISM Services PMI", "CB Consumer Confidence",
)


def load_regime_series(path: str = DAILY_PRICES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_WINDOW)
    df["adx"] = adx.adx()
    df["date"] = df["timestamp"].dt.date
    return df[["date", "adx"]].dropna()


def _regime_asof(regime_df: pd.DataFrame, event_dt: pd.Timestamp) -> str | None:
    event_date = event_dt.date()
    prior = regime_df[regime_df["date"] < event_date]
    if prior.empty:
        return None
    adx_val = prior.iloc[-1]["adx"]
    return "trending" if adx_val >= ADX_TREND_THRESHOLD else "ranging"


def build_regime_features(event_names: tuple, db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    regime_df = load_regime_series()
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT datetime_utc, event_name, surprise_zscore, reaction_1m, reaction_5m,
                   reaction_15m, reaction_30m, reaction_60m
            FROM event_features
            WHERE event_name IN ({",".join("?" for _ in event_names)})
              AND surprise_zscore IS NOT NULL
            """,
            conn, params=event_names,
        )
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["regime"] = df["datetime_utc"].apply(lambda dt: _regime_asof(regime_df, dt))
    return df.dropna(subset=["regime"]).sort_values("datetime_utc").reset_index(drop=True)


def _evaluate_regime_split(df: pd.DataFrame, horizon: str, regime: str, train_fraction: float) -> dict | None:
    """Come historical_combined_events._evaluate_split, ristretto al
    sottoinsieme di righe nel regime dato."""
    sub = df[df["regime"] == regime].dropna(subset=["surprise_zscore", horizon])
    if len(sub) < 20:
        return None
    split_idx = int(len(sub) * train_fraction)
    train, test = sub.iloc[:split_idx], sub.iloc[split_idx:]
    if len(train) < 10 or len(test) < 5:
        return None

    slope, intercept = _theil_sen_fit(train["surprise_zscore"].to_numpy(dtype=float), train[horizon].to_numpy(dtype=float))
    pred_test = slope * test["surprise_zscore"].to_numpy(dtype=float) + intercept
    r2_test = _r2(test[horizon].to_numpy(dtype=float), pred_test)

    naive_pred = np.full(len(test), train[horizon].mean())
    r2_naive = _r2(test[horizon].to_numpy(dtype=float), naive_pred)

    direction_correct = np.sign(pred_test) == np.sign(test[horizon].to_numpy(dtype=float))

    return {
        "regime": regime, "train_fraction": train_fraction,
        "n_train": len(train), "n_test": len(test),
        "r2_test": round(r2_test, 3), "r2_naive_test": round(r2_naive, 3),
        # "and r2_test > 0" — criterio originale di historical_model.py,
        # perso in questa copia (trovato 2026-09-11, vedi lo stesso fix in
        # historical_fomc_text.py per i dettagli).
        "beats_naive": bool(r2_test > r2_naive and r2_test > 0),
        "direction_accuracy": round(float(direction_correct.mean()), 3),
    }


def validate_regime_conditioning(event_name: str, db_path: str = HIST_DB_PATH) -> dict:
    df = build_regime_features((event_name,), db_path)
    if df.empty:
        return {"available": False, "event_name": event_name, "reason": "nessuna riga con regime disponibile"}

    n_trend = int((df["regime"] == "trending").sum())
    n_range = int((df["regime"] == "ranging").sum())

    results = []
    for regime in ("trending", "ranging"):
        for horizon in HORIZONS:
            for frac in TRAIN_FRACTIONS:
                r = _evaluate_regime_split(df, horizon, regime, frac)
                if r:
                    r["horizon"] = horizon
                    results.append(r)

    if not results:
        return {
            "available": True, "event_name": event_name, "n": len(df),
            "n_trending": n_trend, "n_ranging": n_range,
            "by_regime": [], "details": [],
            "verdict": "DATI INSUFFICIENTI per ogni combinazione regime/orizzonte/split (serve >=20 per regime/split)",
        }

    res_df = pd.DataFrame(results)
    by_regime = []
    for regime in ("trending", "ranging"):
        sub = res_df[res_df["regime"] == regime]
        n_reg = n_trend if regime == "trending" else n_range
        if sub.empty:
            by_regime.append({"regime": regime, "n": n_reg, "verdict": "DATI INSUFFICIENTI (nessuno split valido)"})
            continue
        beats_all = sub.groupby("horizon")["beats_naive"].agg(lambda s: s.all())
        n_ok = int(beats_all.sum())
        if n_reg < 100:
            v = f"DATI INSUFFICIENTI PER UN VERDETTO AFFIDABILE (n={n_reg}, serve n>=100)"
        elif n_ok > 0:
            v = f"EDGE VALIDATO su {n_ok}/{len(HORIZONS)} orizzonti (n={n_reg})"
        else:
            v = f"NESSUN EDGE (n={n_reg})"
        by_regime.append({"regime": regime, "n": n_reg, "verdict": v})

    return {
        "available": True, "event_name": event_name, "n": len(df),
        "n_trending": n_trend, "n_ranging": n_range,
        "by_regime": by_regime, "details": results,
    }


if __name__ == "__main__":
    for ev in DEPLOYED_EVENTS:
        result = validate_regime_conditioning(ev)
        print(f"\n=== {ev} ===")
        if not result.get("available"):
            print(result.get("reason"))
            continue
        print(f"n totale={result['n']} (trending={result['n_trending']}, ranging={result['n_ranging']})")
        if not result.get("by_regime"):
            print(f"  {result.get('verdict')}")
        for s in result["by_regime"]:
            print(f"  {s['regime']:10s}: {s['verdict']}")
