"""
sniper_backtest.py — harness PERMANENTE per backtestare i motori "cecchino"
(confluenza SMC + un secondo segnale, o SMC pura — vedi analyzer.SNIPER_CONFIGS)
su dati storici a barre (Dukascopy o altra fonte OHLC).

Nasce dal problema trovato il 2026-09-14 (vedi memoria di progetto
"cecchini 15min/4h/1day non riproducibili"): ogni backtest di questo tipo
viveva in uno script ad-hoc nello scratchpad di sessione, sparito a fine
sessione — impossibile poi capire perché un numero non si riproduceva più.
Questo file sostituisce quegli script: stessa logica, ma versionata,
testata, riusabile.

Riusa deliberatamente i primitivi già testati di backtest.py
(_check_trade_bar, _r_result, _execution_entry, _stat_arb_for_date) e le
funzioni di segnale già live in analyzer.py (smc_generic_zone_signal,
candlestick_strategy, calculate_risk_levels) — MAI una reimplementazione
parallela, stesso principio di [[feedback-dual-mechanism-drift-pattern]].

Non gira in produzione: è uno strumento di ricerca, come backtest.py.
"""

from __future__ import annotations

import pandas as pd

from analyzer import (
    calculate_risk_levels,
    candlestick_strategy,
    compute_indicators,
    detect_market_regime,
    detect_swing_points,
    smc_generic_zone_signal,
)
from backtest import (
    MAX_PENDING_BARS,
    BarResult,
    _check_trade_bar,
    _execution_entry,
    _r_result,
    _stat_arb_for_date,
)
from risk_manager import MAX_CONSECUTIVE_LOSS

# Stessa finestra di backtest.SETUP_WINDOW — quante barre passate alla
# rilevazione SMC ad ogni bar. Il 5min non è qui perché il cecchino 5min è
# già deployato/validato (n=811) con questi stessi identici parametri.
SETUP_WINDOW = {"5min": 150, "15min": 150, "1h": 200, "4h": 200, "1day": 200}
MIN_LOOKBACK = 220

SECOND_SIGNAL_TYPES = ("pure", "stat_arb", "candlestick")


def load_ohlc_csv(path: str) -> pd.DataFrame:
    """Carica un CSV con colonne timestamp,open,high,low,close[,volume]
    (formato dukascopy-python, vedi reference_dukascopy_python) in un
    DataFrame Open/High/Low/Close indicizzato per data — stesso formato
    atteso da compute_indicators()."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
    return df[cols]


def _second_signal(second_signal_type: str, window: pd.DataFrame, window_last_ts: pd.Timestamp) -> dict:
    if second_signal_type == "stat_arb":
        # _stat_arb_for_date confronta con uno storico DXY/TLT
        # timezone-naive (yfinance) — un Timestamp tz-aware (tipico di dati
        # Dukascopy in UTC) fallisce silenziosamente nel suo try/except e
        # ritorna sempre NEUTRAL. Bug reale trovato il 2026-09-14 con
        # questo stesso harness (allora ancora in scratchpad) — fixato qui
        # una volta sola, non ad ogni nuovo script.
        lookup_date = window_last_ts.tz_localize(None) if window_last_ts.tzinfo is not None else window_last_ts
        return _stat_arb_for_date(lookup_date)
    if second_signal_type == "candlestick":
        return candlestick_strategy(window)
    raise ValueError(f"second_signal_type sconosciuto: {second_signal_type!r}")


def run_sniper_backtest(
    df_full: pd.DataFrame,
    interval: str,
    second_signal_type: str,
    progress_every: int | None = None,
) -> list[dict]:
    """
    Simula un motore cecchino (SMC pura se second_signal_type='pure',
    altrimenti SMC in confluenza con Stat Arb o Candlestick) bar-per-bar
    su df_full, un trade alla volta, con lo stop di sicurezza a
    MAX_CONSECUTIVE_LOSS perdite consecutive nella stessa sessione
    (giornata) — stessa logica di backtest.run_backtest(), qui applicata
    al motore cecchino invece che all'aggregato a 8 strategie.

    Entry sempre a mercato (stessa semplificazione di backtest._make_setup:
    il backtest valuta la qualità del segnale, non l'ottimizzazione
    dell'entry). Segnale edge-triggered: conta solo il bar in cui SMC
    CAMBIA direzione, non ogni bar in cui la struttura persiste — senza
    questo un trade chiuso rapidamente (es. SL) riaprirebbe subito sulla
    stessa identica struttura ancora valida, gonfiando artificialmente n
    (trovato il 2026-09-14: n a 2-3x il valore atteso, PF molto peggiore).
    """
    if second_signal_type not in SECOND_SIGNAL_TYPES:
        raise ValueError(f"second_signal_type deve essere uno di {SECOND_SIGNAL_TYPES}")

    df_full = compute_indicators(df_full.copy())
    trades: list[dict] = []
    open_trade: dict | None = None
    window_size = SETUP_WINDOW.get(interval, 200)
    n = len(df_full)
    prev_raw_signal = "NEUTRAL"
    current_day = None
    consecutive_losses = 0
    session_stopped = False

    for index in range(MIN_LOOKBACK, n):
        if progress_every and index % progress_every == 0:
            print(f"  ...{index}/{n}", flush=True)

        timestamp = pd.Timestamp(df_full.index[index])
        day = timestamp.date().isoformat()
        if day != current_day:
            current_day = day
            consecutive_losses = 0
            session_stopped = False

        if open_trade is not None:
            open_trade["tp1_hit_before_bar"] = bool(open_trade.get("tp1_hit"))
            result = _check_trade_bar(open_trade, df_full.iloc[index])
            pending_age = index - open_trade["bar_open"]
            if not open_trade["activated"] and pending_age >= MAX_PENDING_BARS.get(interval, 6):
                result = BarResult("NEVER_TRIGGERED", None)
            if result.outcome:
                open_trade["outcome"] = result.outcome
                open_trade["exit_price"] = result.exit_price
                open_trade["r_result"] = _r_result(open_trade, result.outcome, result.exit_price)
                open_trade["time_close"] = timestamp
                trades.append(open_trade)
                if result.outcome == "LOSS":
                    consecutive_losses += 1
                    session_stopped = consecutive_losses >= MAX_CONSECUTIVE_LOSS
                elif result.outcome.startswith("WIN"):
                    consecutive_losses = 0
                open_trade = None
            continue

        if session_stopped:
            continue

        window_start = max(0, index + 1 - window_size)
        window = df_full.iloc[window_start: index + 1].copy()
        window = detect_swing_points(window)

        smc_signal = smc_generic_zone_signal(window)
        this_raw_signal = smc_signal["signal"]
        is_new_edge = this_raw_signal in ("BUY", "SELL") and this_raw_signal != prev_raw_signal
        prev_raw_signal = this_raw_signal
        if not is_new_edge:
            continue

        if second_signal_type == "pure":
            signal = this_raw_signal
        else:
            second = _second_signal(second_signal_type, window, window.index[-1])
            signal = this_raw_signal if this_raw_signal == second.get("signal") else "NEUTRAL"

        if signal not in ("BUY", "SELL"):
            continue

        row = window.iloc[-1]
        price = float(row["Close"])
        atr = max(float(row["atr"]) if not pd.isna(row["atr"]) else 5.0, 2.0)
        regime_data = detect_market_regime(window)
        regime = regime_data["regime"]
        risk = calculate_risk_levels(signal, price, atr, regime)
        entry = _execution_entry(signal, price)
        initial_risk = abs(entry - float(risk["sl"]))
        if initial_risk <= 0:
            continue

        open_trade = {
            "signal": signal, "order_type": signal, "raw_entry": price,
            "entry": entry, "sl": float(risk["sl"]), "tp1": float(risk["tp1"]),
            "tp2": float(risk["tp2"]), "tp3": float(risk["tp3"]),
            "initial_risk": initial_risk, "activated": True,
            "tp1_hit": False, "bar_open": index, "time": timestamp,
        }

    if open_trade is not None:
        open_trade["outcome"] = "NO_OUTCOME"
        open_trade["exit_price"] = None
        open_trade["r_result"] = 0.0
        open_trade["time_close"] = pd.Timestamp(df_full.index[-1])
        trades.append(open_trade)

    return trades


def stats_for(trades: list[dict]) -> dict:
    wins = [t for t in trades if t["r_result"] > 0]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    decisive = wins + losses
    gross_win = sum(t["r_result"] for t in wins)
    gross_loss = abs(sum(t["r_result"] for t in losses))
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf")
    wr = round(len(wins) / len(decisive) * 100, 1) if decisive else 0.0
    total_r = round(sum(t["r_result"] for t in trades), 2)
    return {"n": len(trades), "pf": pf, "wr": wr, "total_r": total_r}


def split_stats(trades: list[dict]) -> tuple[dict, dict]:
    """Split cronologico in due metà per la verifica anti-overfitting
    standard del progetto — un edge deve reggere su ENTRAMBE, non solo
    nell'aggregato (vedi feedback_conservative_validation_standard)."""
    trades_sorted = sorted(trades, key=lambda t: t["time"])
    mid = len(trades_sorted) // 2
    return stats_for(trades_sorted[:mid]), stats_for(trades_sorted[mid:])


def validated(trades: list[dict], min_n: int = 20) -> dict:
    """Verdetto unico e onesto: PF>1 sull'aggregato E su entrambe le metà
    (quando n>=min_n per metà, altrimenti il verdetto sulla metà è
    'campione troppo piccolo' e non conta ne' a favore ne' contro)."""
    full = stats_for(trades)
    half1, half2 = split_stats(trades)
    halves_ok = []
    for half in (half1, half2):
        if half["n"] < min_n:
            continue
        halves_ok.append(half["pf"] > 1.0)
    split_holds = bool(halves_ok) and all(halves_ok)
    return {
        "full": full, "half1": half1, "half2": half2,
        "split_holds": split_holds,
        "verdict": (
            "VALIDATO" if full["pf"] > 1.0 and split_holds
            else "campione insufficiente per uno split affidabile" if not halves_ok
            else "NON VALIDATO"
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest di un motore cecchino su un CSV OHLC")
    parser.add_argument("csv_path")
    parser.add_argument("--interval", required=True, choices=list(SETUP_WINDOW))
    parser.add_argument("--second-signal", required=True, choices=SECOND_SIGNAL_TYPES)
    args = parser.parse_args()

    df = load_ohlc_csv(args.csv_path)
    trades = run_sniper_backtest(df, args.interval, args.second_signal, progress_every=20000)
    result = validated(trades)
    import json
    print(json.dumps(result, indent=2))
