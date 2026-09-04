"""Backtest causale e conservativo per la versione paper-trading di GoldMind."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analyzer import (
    aggregate_strategies,
    calculate_risk_levels,
    candlestick_strategy,
    compute_indicators,
    detect_bos_choch,
    detect_fvg,
    detect_liquidity,
    detect_market_regime,
    detect_order_blocks,
    detect_premium_discount,
    detect_swing_points,
    determine_order_type,
    estimate_probability,
    get_data,
    get_support_resistance,
    mean_reversion_strategy,
    ml_alpha_strategy,
    momentum_strategy,
    order_flow_strategy,
    trend_following_strategy,
)

logger = logging.getLogger(__name__)

SPREAD_PTS = 0.30
SLIPPAGE_PTS = 0.10
COMMISSION_PTS = 0.05
ENTRY_COST = SPREAD_PTS / 2 + SLIPPAGE_PTS + COMMISSION_PTS / 2
EXIT_COST = SPREAD_PTS / 2 + COMMISSION_PTS / 2

# Finestra di candele passata a _make_setup ad ogni bar — la stessa usata
# dalla pipeline live (agent_orchestrator.OUTPUTSIZE). Senza questo limite
# _make_setup ririceveva df.iloc[:index+1], cioè l'INTERA storia accumulata
# fino a quel bar: su backtest lunghi (es. 5200 candele/10y) il costo cresce
# quadraticamente (ogni bar rianalizza swing/order block/FVG/S-R/strategie
# su migliaia di righe) e un /backtest 1day 10y non finiva più in tempi
# ragionevoli. Con la finestra fissa il costo torna lineare nel numero di bar.
SETUP_WINDOW = {"5min": 150, "15min": 150, "1h": 200, "4h": 200, "1day": 200}
MIN_LOOKBACK = 220
MAX_PENDING_BARS = {
    "1min": 30,
    "5min": 6,
    "15min": 6,
    "1h": 6,
    "4h": 6,
    "1day": 3,
}
MAX_TRADES_PER_DAY = 3
MAX_CONSECUTIVE_LOSS = 3

# Mirror ESATTO di agent_orchestrator.py (fonte di verità — vedi i commenti
# lì per il ragionamento e i dati dietro ogni blocco, non duplicati qui per
# evitare che i due file divergano nel commento pur restando allineati nel
# valore). FIX (audit 2026-09-04): _make_setup() non applicava questi filtri
# regime/direzione — agent_orchestrator.py li applica in produzione dal
# 2026-09-03/04, ma il backtest non ne sapeva nulla. Ogni numero di backtest
# fatto finora (incluso quello usato per giustificare il blocco SELL+NORMAL
# su 1day) includeva trade che il bot live rifiuta categoricamente: stesso
# pattern di "doppio meccanismo che diverge" trovato altre volte oggi.
_BLOCKED_REGIMES_BY_TF = {
    "1h":   ("RANGING", "TRENDING_UP"),
    "4h":   ("RANGING",),
    "1day": ("TRENDING_DOWN",),
}
_BLOCKED_REGIME_DIRECTION_BY_TF = {
    "1day": {"SELL": ("NORMAL",)},
}


@dataclass
class BarResult:
    outcome: str | None = None
    exit_price: float | None = None
    activated_now: bool = False


def _is_pending(order_type: str) -> bool:
    value = str(order_type).upper()
    return "LIMIT" in value or "STOP" in value


def _order_touched(trade: dict, high: float, low: float) -> bool:
    signal = trade["signal"]
    order_type = str(trade["order_type"]).upper()
    entry = float(trade["raw_entry"])
    if "LIMIT" in order_type:
        return (signal == "BUY" and low <= entry) or (signal == "SELL" and high >= entry)
    if "STOP" in order_type:
        return (signal == "BUY" and high >= entry) or (signal == "SELL" and low <= entry)
    return True


def _execution_entry(signal: str, raw_entry: float) -> float:
    return raw_entry + ENTRY_COST if signal == "BUY" else raw_entry - ENTRY_COST


def _execution_exit(signal: str, raw_exit: float) -> float:
    return raw_exit - EXIT_COST if signal == "BUY" else raw_exit + EXIT_COST


def _check_trade_bar(trade: dict, bar) -> BarResult:
    """
    Valuta una candela senza inventare la sequenza intrabar.

    Quando SL e target sono entrambi toccati, prevale sempre l'esito peggiore.
    TP1 attiva la protezione a BE soltanto dalla candela successiva.

    FIX: prima la protezione BE scattava anche a soli $10 di movimento a
    favore (trade["be"]), ben prima di TP1. Il bot live (trade_manager.py,
    arm_break_even) non lo fa mai: arma la protezione SOLO quando TP1 viene
    davvero raggiunto. Con il trigger anticipato il backtest chiudeva a
    pareggio trade che live avrebbe lasciato correre fino al SL originale
    (o fino a un vero target) — un esito diverso da quello reale. Rimosso:
    ora la protezione scatta solo su TP1, come nel bot live.
    """
    high = float(bar["High"])
    low = float(bar["Low"])
    signal = trade["signal"]
    sl = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp3 = float(trade["tp3"])
    raw_entry = float(trade["raw_entry"])
    activated_now = False

    if not trade["activated"]:
        if not _order_touched(trade, high, low):
            return BarResult()
        trade["activated"] = True
        activated_now = True

    # Stop originale: viene controllato per primo, quindi una candela ambigua
    # non può trasformarsi artificialmente in una vittoria.
    if signal == "BUY" and low <= sl:
        return BarResult("LOSS", _execution_exit(signal, sl), activated_now)
    if signal == "SELL" and high >= sl:
        return BarResult("LOSS", _execution_exit(signal, sl), activated_now)

    # Protezione attivata da una candela precedente: TP1 già raggiunto, il
    # prezzo torna all'entry. Come nel bot live questo chiude a pareggio
    # (outcome "WIN_BE", 0R) — nessuna chiusura parziale reale avviene su
    # TP1, quindi non va contato come vittoria piena.
    if trade.get("tp1_hit_before_bar"):
        if signal == "BUY" and low <= raw_entry:
            return BarResult("WIN_BE", _execution_exit(signal, raw_entry), activated_now)
        if signal == "SELL" and high >= raw_entry:
            return BarResult("WIN_BE", _execution_exit(signal, raw_entry), activated_now)

    if signal == "BUY":
        if high >= tp3:
            return BarResult("WIN_TP3", _execution_exit(signal, tp3), activated_now)
        if high >= tp2:
            return BarResult("WIN_TP2", _execution_exit(signal, tp2), activated_now)
        if high >= tp1:
            trade["tp1_hit"] = True
    else:
        if low <= tp3:
            return BarResult("WIN_TP3", _execution_exit(signal, tp3), activated_now)
        if low <= tp2:
            return BarResult("WIN_TP2", _execution_exit(signal, tp2), activated_now)
        if low <= tp1:
            trade["tp1_hit"] = True

    return BarResult(activated_now=activated_now)


def _make_setup(window: pd.DataFrame, interval: str, min_prob: int) -> dict | None:
    window = detect_swing_points(window.copy())
    row = window.iloc[-1]
    price = float(row["Close"])
    atr = max(float(row["atr"]) if not pd.isna(row["atr"]) else 5.0, 2.0)
    rsi = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0
    adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0

    smc = detect_bos_choch(window)
    order_blocks = detect_order_blocks(window)
    fvg = detect_fvg(window)
    detect_liquidity(window)
    pd_zone = detect_premium_discount(window, smc)
    support_resistance = get_support_resistance(window)
    regime_data = detect_market_regime(window)
    regime = regime_data["regime"]
    mtf_neutral = {
        timeframe: "NEUTRAL"
        for timeframe in ("1min", "5min", "15min", "1h", "4h", "1day")
    }

    strategies = {
        "trend": trend_following_strategy(window),
        "mean_rev": mean_reversion_strategy(window),
        "momentum": momentum_strategy(window, mtf_neutral),
        "candle": candlestick_strategy(window),
        "order_flow": order_flow_strategy(window),
        "ml": ml_alpha_strategy(window, mtf_neutral, smc),
        "smc": {"signal": "NEUTRAL", "score": 0},
        "event": {"signal": "NEUTRAL", "score": 0},
        "stat_arb": {"signal": "NEUTRAL", "score": 0},
    }
    aggregated = aggregate_strategies(strategies, regime_data, timeframe=interval)
    if aggregated["signal"] == "NEUTRAL":
        return None

    signal = aggregated["signal"]
    _tf = interval.lower()
    _regime_up = str(regime).upper().replace(" ", "_")
    if _regime_up in _BLOCKED_REGIMES_BY_TF.get(_tf, ()):
        return None
    if _regime_up in _BLOCKED_REGIME_DIRECTION_BY_TF.get(_tf, {}).get(signal, ()):
        return None
    # Nel backtest usiamo sempre MARKET entry:
    # il backtest valuta la qualità del segnale, non l'ottimizzazione dell'entry.
    # I LIMIT/STOP nel trading live aggiungono alpha sull'entry ma rendono
    # i backtest non confrontabili (troppi "mai attivati").
    order_type, raw_entry = "BUY" if signal == "BUY" else "SELL", price
    # Salva l'order_type originale solo per riferimento
    _original_order_type = determine_order_type(
        signal, price, support_resistance, atr, adx, rsi,
        smc["structure"], order_blocks, fvg, regime, pd_zone,
    )[0]
    risk = calculate_risk_levels(signal, raw_entry, atr, regime)
    probability = estimate_probability(
        aggregated["total_score"],
        aggregated["buy_count"],
        aggregated["sell_count"],
        False,
        regime,
        smc["structure"],
        False,
    )
    if probability < min_prob:
        return None

    execution_entry = _execution_entry(signal, float(raw_entry))
    initial_risk = abs(execution_entry - float(risk["sl"]))
    if initial_risk <= 0:
        return None

    return {
        "signal": signal,
        "order_type": order_type,
        "raw_entry": float(raw_entry),
        "entry": execution_entry,
        "sl": float(risk["sl"]),
        "tp1": float(risk["tp1"]),
        "tp2": float(risk["tp2"]),
        "tp3": float(risk["tp3"]),
        "be": float(risk["be"]),
        "prob": probability,
        "regime": regime,
        "initial_risk": initial_risk,
        "activated": not _is_pending(order_type),
        "tp1_hit": False,
        "contributing_strategies": [
            name
            for name, contribution in strategies.items()
            if contribution.get("signal") == signal and contribution.get("score", 0) > 0
        ],
    }


def _r_result(trade: dict, outcome: str, exit_price: float | None) -> float:
    if outcome in ("WIN_BE", "NEVER_TRIGGERED", "NO_OUTCOME"):
        return 0.0
    if exit_price is None:
        return 0.0
    direction = 1 if trade["signal"] == "BUY" else -1
    return round(
        direction * (float(exit_price) - float(trade["entry"])) / trade["initial_risk"],
        3,
    )


def run_backtest(interval: str = "5min", bars: int = 2000, min_prob: int = 60) -> dict:
    bars = max(100, min(int(bars), 5000))  # max 5000 candele per backtest lungo
    try:
        df = get_data(interval=interval, outputsize=bars, bypass_cache=True)
        # Tutti gli indicatori qui usati sono trailing; gli swing vengono invece
        # ricalcolati dentro ogni finestra per evitare conferme dal futuro.
        df = compute_indicators(df.copy())
    except Exception as exc:
        logger.exception("Download/indicatori backtest falliti")
        return {"total": 0, "message": f"Errore dati: {exc}"}

    if len(df) <= MIN_LOOKBACK + 2:
        return {"total": 0, "message": f"Dati insufficienti: {len(df)} candele."}

    trades: list[dict] = []
    open_trade: dict | None = None
    errors_count = 0
    current_day = None
    trades_today = 0
    consecutive_losses = 0
    session_stopped = False

    for index in range(MIN_LOOKBACK, len(df)):
        timestamp = pd.Timestamp(df.index[index])
        day = timestamp.date().isoformat()
        if day != current_day:
            current_day = day
            trades_today = 0
            consecutive_losses = 0
            session_stopped = False

        if open_trade is not None:
            open_trade["tp1_hit_before_bar"] = bool(open_trade.get("tp1_hit"))
            result = _check_trade_bar(open_trade, df.iloc[index])
            if result.activated_now:
                trades_today += 1
                open_trade["activated_at"] = timestamp

            pending_age = index - open_trade["bar_open"]
            if (
                not open_trade["activated"]
                and pending_age >= MAX_PENDING_BARS.get(interval, 6)
            ):
                result = BarResult("NEVER_TRIGGERED", None)

            if result.outcome:
                open_trade["outcome"] = result.outcome
                open_trade["exit_price"] = result.exit_price
                open_trade["bars_to_outcome"] = index - open_trade["bar_open"]
                open_trade["r_result"] = _r_result(
                    open_trade, result.outcome, result.exit_price
                )
                trades.append(open_trade)
                if result.outcome == "LOSS":
                    consecutive_losses += 1
                    session_stopped = consecutive_losses >= MAX_CONSECUTIVE_LOSS
                elif result.outcome.startswith("WIN"):
                    consecutive_losses = 0
                open_trade = None
            continue

        if session_stopped or trades_today >= MAX_TRADES_PER_DAY:
            continue

        try:
            window_size = SETUP_WINDOW.get(interval, 200)
            window_start = max(0, index + 1 - window_size)
            setup = _make_setup(df.iloc[window_start : index + 1].copy(), interval, min_prob)
            if setup is None:
                continue
            setup.update(
                {
                    "bar_open": index,
                    "time": timestamp,
                    "outcome": None,
                    "exit_price": None,
                    "bars_to_outcome": 0,
                }
            )
            open_trade = setup
            if setup["activated"]:
                trades_today += 1
                setup["activated_at"] = timestamp
        except Exception as exc:
            errors_count += 1
            if errors_count <= 10:
                logger.warning("Errore backtest alla candela %s: %s", index, exc)

    if open_trade is not None:
        open_trade["outcome"] = "NO_OUTCOME"
        open_trade["exit_price"] = None
        open_trade["bars_to_outcome"] = len(df) - open_trade["bar_open"]
        open_trade["r_result"] = 0.0
        trades.append(open_trade)

    return _compute_backtest_stats(trades, interval, bars, errors_count)


def _compute_backtest_stats(
    trades: list[dict], interval: str, bars: int, errors_count: int
) -> dict:
    if not trades:
        return {"total": 0, "message": "Nessun setup generato nel periodo."}
    frame = pd.DataFrame(trades)
    concluded = frame[
        frame["outcome"].isin(("WIN_TP1", "WIN_TP2", "WIN_TP3", "WIN_BE", "LOSS"))
    ].copy()
    decisive = concluded[concluded["outcome"] != "WIN_BE"].copy()
    wins = decisive[decisive["outcome"].str.startswith("WIN")]
    losses = decisive[decisive["outcome"] == "LOSS"]

    total_r = float(concluded["r_result"].sum()) if len(concluded) else 0.0
    gross_win = float(concluded.loc[concluded["r_result"] > 0, "r_result"].sum())
    gross_loss = abs(float(concluded.loc[concluded["r_result"] < 0, "r_result"].sum()))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    equity = concluded.sort_values("time")["r_result"].cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    current_drawdown = float(drawdown.iloc[-1]) if len(drawdown) else 0.0

    def grouped_stats(column: str) -> dict:
        output = {}
        if not len(decisive):
            return output
        for key, group in decisive.groupby(column):
            if len(group) < 3:
                continue
            output[str(key)] = {
                "win_rate": round(group["outcome"].str.startswith("WIN").mean() * 100, 1),
                "count": len(group),
                "avg_r": round(float(group["r_result"].mean()), 3),
            }
        return output

    strategy_stats = {}
    if len(decisive):
        all_strategies = sorted(
            {
                strategy
                for values in decisive["contributing_strategies"]
                if isinstance(values, list)
                for strategy in values
            }
        )
        for strategy in all_strategies:
            group = decisive[
                decisive["contributing_strategies"].apply(
                    lambda values: isinstance(values, list) and strategy in values
                )
            ]
            if len(group) >= 3:
                strategy_stats[strategy] = {
                    "win_rate": round(
                        group["outcome"].str.startswith("WIN").mean() * 100, 1
                    ),
                    "count": len(group),
                    "avg_r": round(float(group["r_result"].mean()), 3),
                }

    return {
        "interval": interval,
        "bars_analyzed": bars,
        "model": "single_timeframe_causal_proxy",
        "total": len(frame),
        "concluded": len(concluded),
        "wins": len(wins),
        "losses": len(losses),
        "be_count": int((concluded["outcome"] == "WIN_BE").sum()) if len(concluded) else 0,
        "no_outcome": int((frame["outcome"] == "NO_OUTCOME").sum()),
        "never_triggered": int((frame["outcome"] == "NEVER_TRIGGERED").sum()),
        "win_rate": round(len(wins) / len(decisive) * 100, 1) if len(decisive) else 0,
        "total_r": round(total_r, 2),
        "profit_factor": profit_factor,
        "max_drawdown_r": round(max_drawdown, 2),
        "current_dd_r": round(current_drawdown, 2),
        "tp_distribution": frame["outcome"].value_counts().to_dict(),
        "win_rate_by_regime": grouped_stats("regime"),
        "win_rate_by_direction": grouped_stats("signal"),
        "win_rate_by_strategy": strategy_stats,
        "avg_bars_to_outcome": round(
            float(concluded["bars_to_outcome"].mean()), 1
        ) if len(concluded) else 0,
        "spread_pts": SPREAD_PTS,
        "slippage_pts": SLIPPAGE_PTS,
        "commission_pts": COMMISSION_PTS,
        "errors_count": 0,
        "tf_stats": {},
        "errors_count": errors_count,
    }


def _safe(text: str) -> str:
    """Rimuove caratteri che rompono Markdown Telegram."""
    return str(text).replace("_", " ").replace("*", "").replace("`", "").replace("[", "").replace("]", "")


def format_backtest_report(stats: dict, interval: str) -> str:
    if stats.get("total", 0) == 0:
        return f"Backtest {interval}\n\n{stats.get('message', 'Nessun risultato.')}"

    pf     = stats["profit_factor"]
    pf_txt = "inf" if pf == float("inf") else str(pf)

    strat_labels = {
        "smc": "SMC v3.0", "trend": "Trend Following", "mean_rev": "Mean Reversion",
        "momentum": "Momentum", "event": "Event-Driven", "stat_arb": "Stat Arb",
        "ml": "ML Alpha", "candle": "Candlestick", "order_flow": "Order Flow"
    }

    regime_txt = "\n".join(
        f"  {_safe(r)}: {d['win_rate']}% | avg {d.get('avg_r',0):+.1f}R (n={d['count']})"
        for r, d in sorted(stats.get("win_rate_by_regime", {}).items(), key=lambda x: -x[1]["count"])
    ) or "  Dati insufficienti (min 3 trade)"

    dir_txt = "\n".join(
        f"  {_safe(d)}: {v['win_rate']}% | avg {v.get('avg_r',0):+.1f}R (n={v['count']})"
        for d, v in stats.get("win_rate_by_direction", {}).items()
    ) or "  Nessun dato"

    strat_txt = "\n".join(
        f"  {_safe(strat_labels.get(s,s))}: {v['win_rate']}% | avg {v.get('avg_r',0):+.1f}R (n={v['count']})"
        for s, v in sorted(stats.get("win_rate_by_strategy", {}).items(), key=lambda x: -x[1]["count"])
    ) or "  Dati insufficienti (min 3 trade)"

    tp_dist    = stats.get("tp_distribution", {})
    tp_txt     = " | ".join(f"{k}: {v}" for k, v in tp_dist.items()) or "-"
    cost_total = stats.get("spread_pts",0) + stats.get("slippage_pts",0) + stats.get("commission_pts",0)
    errors     = stats.get("errors_count", 0)

    lines = [
        f"BACKTEST XAU/USD - {_safe(interval)}",
        "=" * 30,
        f"Segnali: {stats['total']} | Conclusi: {stats.get('concluded', stats['total'])}",
        f"Mai attivati: {stats.get('never_triggered',0)} | Non conclusi: {stats.get('no_outcome',0)}",
        f"Win: {stats['wins']} | Loss: {stats['losses']}",
        "-" * 30,
        f"Win Rate: {stats['win_rate']}%",
        f"Profit Factor: {pf_txt}",
        f"Max DD: {stats.get('max_drawdown_r',0)}R | P&L: {stats['total_r']:+.1f}R",
        f"Barre medie a esito: {stats.get('avg_bars_to_outcome',0)}",
        "-" * 30,
        f"Esiti: {tp_txt}",
        "-" * 30,
        "Per direzione:",
        dir_txt,
        "-" * 30,
        "Per regime:",
        regime_txt,
        "-" * 30,
        "Per strategia:",
        strat_txt,
        "-" * 30,
        f"Costi: {cost_total:.2f}$/trade",
    ]
    if errors > 0:
        lines.append(f"Attenzione: {errors} candele con errori")
    # Statistiche per TF se disponibili (backtest multi-TF)
    tf_stats = stats.get("tf_stats", {})
    if tf_stats and len(tf_stats) > 1:
        lines.append("-" * 30)
        lines.append("Segnali per timeframe:")
        for tf, d in sorted(tf_stats.items()):
            wr_tf = round(d["wins"] / max(d["wins"]+d["losses"],1) * 100, 1)
            lines.append(f"  {tf}: {d['total']} segnali | {d['wins']}W/{d['losses']}L | WR {wr_tf}% | {d['pnl_r']:+.1f}R")

    lines.append("Walk-forward: /backtest wf 5min 2000")
    lines.append("Multi-TF: /backtest tutti 3m")

    return "\n".join(lines)[:4000]


def run_walkforward_backtest(
    interval: str = "5min",
    bars: int = 2000,
    min_prob: int = 60,
    n_windows: int = 4,
) -> dict:
    """
    Walk-Forward Backtest reale.

    Divide i dati in N finestre. Per ogni finestra:
      - Train: 75% delle candele → ottimizza min_prob
      - Test:  25% delle candele → valida out-of-sample

    Questo è il modo corretto per testare una strategia senza
    overfitting: i parametri vengono ottimizzati su dati passati
    e poi validati su dati futuri mai visti.

    Uso: /backtest wf [interval] [bars] [n_windows]
    """
    bars = max(600, min(int(bars), 5000))

    try:
        df = get_data(interval=interval, outputsize=bars, bypass_cache=True)
        df = compute_indicators(df.copy())
    except Exception as exc:
        return {"total": 0, "message": f"Errore dati: {exc}"}

    if len(df) < MIN_LOOKBACK * 2 + 50:
        return {"total": 0, "message": f"Dati insufficienti per walk-forward ({len(df)} candele)."}

    window_size  = (len(df) - MIN_LOOKBACK) // n_windows
    train_size   = int(window_size * 0.75)
    test_size    = window_size - train_size

    all_test_trades = []
    window_results  = []

    for w in range(n_windows):
        w_start = MIN_LOOKBACK + w * window_size
        w_end   = w_start + window_size
        if w_end > len(df):
            break

        train_end = w_start + train_size
        test_end  = min(w_end, len(df) - 1)

        # ── TRAIN: trova il min_prob ottimale su questo segmento ──────────────
        best_prob   = min_prob
        best_pf     = 0.0
        train_df    = df.iloc[:train_end].copy()

        for prob_try in [50, 55, 60, 65, 70]:
            # Backtest veloce sul segmento train
            train_trades = _run_segment(
                train_df, interval, prob_try,
                start_idx=w_start, end_idx=train_end
            )
            if not train_trades:
                continue
            concluded = [t for t in train_trades
                        if t.get("outcome") in ("WIN_TP1","WIN_TP2","WIN_TP3","LOSS")]
            if len(concluded) < 3:
                continue
            wins = sum(1 for t in concluded if "WIN" in t["outcome"])
            losses = sum(1 for t in concluded if t["outcome"] == "LOSS")
            if losses == 0:
                continue
            pf = (wins * 1.5) / losses  # profit factor semplificato
            if pf > best_pf:
                best_pf   = pf
                best_prob = prob_try

        # ── TEST: valida il best_prob sul segmento out-of-sample ─────────────
        test_df     = df.iloc[:test_end].copy()
        test_trades = _run_segment(
            test_df, interval, best_prob,
            start_idx=train_end, end_idx=test_end
        )

        concluded = [t for t in test_trades
                    if t.get("outcome") in ("WIN_TP1","WIN_TP2","WIN_TP3","LOSS")]
        wins_t  = sum(1 for t in concluded if "WIN" in t.get("outcome",""))
        losses_t = sum(1 for t in concluded if t.get("outcome") == "LOSS")
        wr_t    = round(wins_t / len(concluded) * 100, 1) if concluded else 0
        pnl_t   = sum(t.get("r_result", 0) for t in test_trades)

        window_results.append({
            "window":    w + 1,
            "best_prob": best_prob,
            "trades":    len(concluded),
            "wins":      wins_t,
            "losses":    losses_t,
            "win_rate":  wr_t,
            "pnl_r":     round(pnl_t, 2),
        })

        all_test_trades.extend(test_trades)

    # ── Stats aggregate ───────────────────────────────────────────────────────
    all_concluded = [t for t in all_test_trades
                    if t.get("outcome") in ("WIN_TP1","WIN_TP2","WIN_TP3","LOSS")]
    total_wins   = sum(1 for t in all_concluded if "WIN" in t.get("outcome",""))
    total_losses = sum(1 for t in all_concluded if t.get("outcome") == "LOSS")
    total_wr     = round(total_wins / len(all_concluded) * 100, 1) if all_concluded else 0
    total_pnl    = round(sum(t.get("r_result", 0) for t in all_test_trades), 2)
    gross_win    = sum(t["r_result"] for t in all_concluded if t.get("r_result", 0) > 0)
    gross_loss   = abs(sum(t["r_result"] for t in all_concluded if t.get("r_result", 0) < 0))
    pf           = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "type":       "walk-forward",
        "interval":   interval,
        "n_windows":  len(window_results),
        "total":      len(all_test_trades),
        "concluded":  len(all_concluded),
        "wins":       total_wins,
        "losses":     total_losses,
        "win_rate":   total_wr,
        "total_r":    total_pnl,
        "profit_factor": pf,
        "window_results": window_results,
    }


def _run_segment(
    df: pd.DataFrame,
    interval: str,
    min_prob: int,
    start_idx: int,
    end_idx: int,
) -> list:
    """Esegue il backtest su un segmento specifico del DataFrame."""
    trades      = []
    open_trade  = None
    consec_loss = 0
    stopped     = False

    for index in range(start_idx, min(end_idx, len(df))):
        if open_trade is not None:
            # FIX: mancava l'aggiornamento di tp1_hit_before_bar (presente
            # invece in run_backtest) — senza, _check_trade_bar non rileva
            # mai il ritorno a breakeven dopo TP1 nel walk-forward.
            open_trade["tp1_hit_before_bar"] = bool(open_trade.get("tp1_hit"))
            result = _check_trade_bar(open_trade, df.iloc[index])
            pending_age = index - open_trade["bar_open"]
            if not open_trade["activated"] and pending_age >= MAX_PENDING_BARS.get(interval, 6):
                result = BarResult("NEVER_TRIGGERED", None)
            if result.outcome:
                open_trade["outcome"]     = result.outcome
                open_trade["exit_price"]  = result.exit_price
                open_trade["bars_to_outcome"] = index - open_trade["bar_open"]
                open_trade["r_result"]    = _r_result(open_trade, result.outcome, result.exit_price)
                trades.append(open_trade)
                if result.outcome == "LOSS":
                    consec_loss += 1
                    if consec_loss >= MAX_CONSECUTIVE_LOSS:
                        stopped = True
                elif result.outcome.startswith("WIN"):
                    consec_loss = 0
                open_trade = None
            continue

        if stopped:
            continue

        try:
            window_size = SETUP_WINDOW.get(interval, 200)
            window_start = max(0, index + 1 - window_size)
            setup = _make_setup(df.iloc[window_start:index + 1].copy(), interval, min_prob)
            if setup is None:
                continue
            setup.update({
                "bar_open": index,
                "time":     pd.Timestamp(df.index[index]),
                "outcome":  None,
                "exit_price": None,
                "bars_to_outcome": 0,
            })
            open_trade = setup
        except Exception:
            continue

    return trades


def format_walkforward_report(stats: dict, interval: str) -> str:
    """Formatta il report walk-forward per Telegram."""
    if stats.get("total", 0) == 0:
        return f"📊 Walk-Forward {interval}\n\n{stats.get('message','Nessun risultato.')}"

    pf = stats["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else str(pf)

    windows_txt = "\n".join(
        f"  W{w['window']}: prob={w['best_prob']}% | "
        f"{w['wins']}W/{w['losses']}L | WR {w['win_rate']}% | {w['pnl_r']:+.1f}R"
        for w in stats.get("window_results", [])
    )

    return (
        f"📊 *WALK-FORWARD — {interval} ({stats['n_windows']} finestre)*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Trade out-of-sample: *{stats['concluded']}*\n"
        f"Win: *{stats['wins']}* | Loss: *{stats['losses']}*\n"
        f"📈 WR: *{stats['win_rate']}%*\n"
        f"💰 P&L: *{stats['total_r']:+.1f}R* | PF: *{pf_txt}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Per finestra (out-of-sample):*\n{windows_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Walk-forward: ogni finestra ottimizza il minprob sul train\n"
        f"e valida su dati futuri mai visti (out-of-sample)."
    )
    if stats.get("total", 0) == 0:
        return f"📊 Backtest {interval}\n\n{stats.get('message', 'Nessun risultato.')}"
    profit_factor = stats["profit_factor"]
    pf_text = "∞" if profit_factor == float("inf") else str(profit_factor)
    return (
        f"📊 *BACKTEST CAUSALE — {interval}*\n"
        f"Modello: `{stats.get('model')}`\n"
        f"Barre: *{stats['bars_analyzed']}* | Setup: *{stats['total']}*\n"
        f"Conclusi: *{stats['concluded']}* | Pending mai attivati: *{stats['never_triggered']}*\n"
        f"Win: *{stats['wins']}* | Loss: *{stats['losses']}* | BE: *{stats['be_count']}*\n"
        f"WR: *{stats['win_rate']}%* | P&L: *{stats['total_r']:+.2f}R*\n"
        f"Profit factor: *{pf_text}* | Max DD: *{stats['max_drawdown_r']:.2f}R*\n"
        f"Costi: spread {stats['spread_pts']}, slippage {stats['slippage_pts']}, "
        f"commissione {stats['commission_pts']}\n"
        f"Errori simulazione: *{stats['errors_count']}*\n\n"
        "_È un proxy single-timeframe: prima del live servono ancora walk-forward "
        "multi-timeframe e paper trading._"
    )
