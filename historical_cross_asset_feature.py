"""
historical_cross_asset_feature.py — Ricerca (2026-09-15, wishlist "multi-
asset"): il rendimento di ieri di un asset correlato (argento, miniere
d'oro, VIX, petrolio) predice il rendimento di oggi di XAU/USD?

Stesso standard di rigore di tutto il resto del progetto (Theil-Sen,
split cronologico in due metà indipendenti, mai fidarsi di un solo
periodo) — vedi feedback_conservative_validation_standard. Puro script di
ricerca, non tocca il bot live: se qualcosa qui sotto validasse davvero,
andrebbe integrato con una PR separata (nuova strategia o nuovo feature
per stat_arb), non deployato automaticamente da questo file.

Dati: yfinance, 10 anni giornalieri (già una dipendenza del progetto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from historical_model import TRAIN_FRACTIONS, _r2, _theil_sen_fit

CANDIDATES = {
    "XAGUSD (argento)": "SI=F",
    "GDX (miniere oro)": "GDX",
    "VIX": "^VIX",
    "OIL (WTI)": "CL=F",
}
GOLD_TICKER = "GC=F"


def _daily_returns(ticker: str, period: str = "10y") -> pd.Series:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.pct_change().dropna()


def _evaluate_lagged_predictor(pred_ret: pd.Series, gold_ret: pd.Series, train_fraction: float) -> dict | None:
    """pred_ret[t-1] -> gold_ret[t]: allinea per data, sposta il predittore
    di un giorno (mai il valore dello stesso giorno, che sarebbe lookahead
    — il predittore di oggi non è ancora noto quando si apre la barra di
    oggi)."""
    aligned = pd.DataFrame({"pred": pred_ret.shift(1), "gold": gold_ret}).dropna()
    if len(aligned) < 100:
        return None
    split_idx = int(len(aligned) * train_fraction)
    train, test = aligned.iloc[:split_idx], aligned.iloc[split_idx:]
    if len(train) < 50 or len(test) < 20:
        return None

    slope, intercept = _theil_sen_fit(train["pred"].to_numpy(), train["gold"].to_numpy())
    pred_test = slope * test["pred"].to_numpy() + intercept
    r2_test = _r2(test["gold"].to_numpy(), pred_test)
    r2_naive = _r2(test["gold"].to_numpy(), np.full(len(test), train["gold"].mean()))
    direction_acc = float(np.mean(np.sign(pred_test) == np.sign(test["gold"].to_numpy())))

    return {
        "train_fraction": train_fraction, "n_train": len(train), "n_test": len(test),
        "r2_test": round(r2_test, 4), "r2_naive": round(r2_naive, 4),
        "beats_naive": bool(r2_test > r2_naive and r2_test > 0),
        "direction_accuracy": round(direction_acc, 3),
    }


def run() -> dict:
    gold_ret = _daily_returns(GOLD_TICKER)
    results = {}
    for name, ticker in CANDIDATES.items():
        pred_ret = _daily_returns(ticker)
        splits = [
            r for frac in TRAIN_FRACTIONS
            if (r := _evaluate_lagged_predictor(pred_ret, gold_ret, frac)) is not None
        ]
        if not splits:
            results[name] = {"verdict": "dati insufficienti"}
            continue
        beats_all = all(s["beats_naive"] for s in splits)
        mean_dir_acc = float(np.mean([s["direction_accuracy"] for s in splits]))
        genuine = beats_all and mean_dir_acc >= 0.55
        results[name] = {
            "n": splits[0]["n_train"] + splits[-1]["n_test"],
            "beats_naive_on_all_splits": beats_all,
            "mean_direction_accuracy": round(mean_dir_acc, 3),
            "verdict": "GENUINO" if genuine else "nessun edge genuino",
            "details": splits,
        }
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
