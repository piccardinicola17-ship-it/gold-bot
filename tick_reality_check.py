"""
tick_reality_check.py — verifica un'assunzione centrale del motore di
backtest (backtest._check_trade_bar): quando una candela tocca sia lo
stop che un target, vince sempre l'esito peggiore (scelta prudente per
non inventare la sequenza intrabar, mai una vittoria "fortunata" da
un'ambiguità) — ma non era mai stato verificato se questa scelta
coincide con cosa sarebbe successo DAVVERO, tick per tick.

NON è un backtest di strategia multi-anno (richiederebbe settimane di
storico candele prima di ogni evento per calcolare gli indicatori — ATR,
medie mobili, regime — che non abbiamo: i tick Dukascopy scaricati in
Fase 2 coprono solo finestre strette [-10min,+65min] attorno a ~2100
eventi macro 2007-2025, non uno storico continuo). È invece un test
MECCANICO del motore stesso, sugli stessi tick reali già scaricati:
costruisce trade sintetici con SL/TP realistici (analyzer.
calculate_risk_levels, stessa funzione usata in produzione) attorno a
momenti storicamente volatili (i rilasci macro), e confronta:

- "tick-truth": backtest._check_trade_bar chiamato tick per tick (barra
  di larghezza zero, High=Low=prezzo medio) — la sequenza VERA.
- "candle-approx": la STESSA _check_trade_bar chiamata sulle candele
  ricostruite dagli stessi tick a un intervallo dato (5min, 1h) — quello
  che fa oggi backtest.py.

Nessuna logica duplicata: le due colonne usano la stessa identica
funzione di produzione, l'unica variabile è la granularità delle barre
che le vengono date in pasto. Se le due divergono spesso, la scelta
"vince il peggiore" del motore di backtest non è neutra — se
concordano quasi sempre, è una conferma diretta (mai fatta finora) che
tutti i numeri prodotti da backtest.py (Monte Carlo, calibrazione,
sensitivity) non sono distorti da questo dettaglio meccanico.
"""

from __future__ import annotations

import logging

import pandas as pd

from historical_events import HIST_DB_PATH, _connect
from dukascopy_ticks import get_ticks_window, _mid_price_asof
from backtest import _check_trade_bar
from analyzer import calculate_risk_levels

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Due regimi (multiplier SL/TP diversi) e due ATR (stop stretto/largo):
# copertura di geometrie di trade realistiche senza esplodere i tempi di
# calcolo. L'ATR è in dollari assoluti ma per l'oro resta un ordine di
# grandezza ragionevole sull'intero intervallo storico 2007-2025 (da
# ~$650 a ~$2400): a differenza di EUR/USD (vedi backtest esplorativo
# multi-mercato del 2026-09-05), il prezzo dell'oro non cambia mai di
# ordini di grandezza, quindi qui non serve scalare l'ATR per data.
TEST_ATRS = (3.0, 8.0)
TEST_REGIMES = ("NORMAL", "VOLATILE")
CANDLE_INTERVALS = ("5min", "1h")

MAX_DISAGREE_EXAMPLES = 15


def _events_with_ticks(db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT e.event_uid, e.event_name, e.datetime_utc "
            "FROM macro_events e JOIN event_price_reactions r ON e.event_uid = r.event_uid "
            "WHERE r.tick_count > 0 ORDER BY e.datetime_utc",
            conn,
        )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def _make_trade(signal: str, entry: float, sl: float, tp1: float, tp2: float, tp3: float) -> dict:
    # order_type = segnale stesso (non LIMIT/STOP): market entry, come nel
    # motore di backtest principale — qui testiamo la gestione SL/TP
    # intrabar, non l'attivazione di un pending.
    return {
        "signal": signal, "order_type": signal, "raw_entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "activated": True, "tp1_hit": False,
    }


def _walk_bars(trade: dict, bars) -> str:
    """Richiama _check_trade_bar (la funzione di produzione, non una
    copia) bar per bar finché non produce un esito terminale."""
    for bar in bars:
        trade["tp1_hit_before_bar"] = trade["tp1_hit"]
        result = _check_trade_bar(trade, bar)
        if result.outcome:
            return result.outcome
    return "NO_OUTCOME"


def _tick_bars(mids) -> list:
    return [{"High": m, "Low": m} for m in mids]


def _resample_candles(ticks: pd.DataFrame, interval: str) -> pd.DataFrame:
    mid = (ticks["ask"] + ticks["bid"]) / 2.0
    s = pd.Series(mid.to_numpy(), index=pd.DatetimeIndex(ticks["datetime_utc"]))
    ohlc = s.resample(interval).ohlc().dropna()
    ohlc.columns = ["Open", "High", "Low", "Close"]
    return ohlc


def run_check(db_path: str = HIST_DB_PATH,
              candle_intervals=CANDLE_INTERVALS,
              atrs=TEST_ATRS, regimes=TEST_REGIMES) -> dict:
    events = _events_with_ticks(db_path)
    logger.info(f"{len(events)} eventi con tick reali da testare")

    results = {
        iv: {"agree": 0, "disagree": 0, "both_no_outcome": 0, "disagree_examples": []}
        for iv in candle_intervals
    }
    events_used = 0

    for _, ev in events.iterrows():
        window_start = (ev["datetime_utc"] - pd.Timedelta(minutes=10)).to_pydatetime()
        window_end = (ev["datetime_utc"] + pd.Timedelta(minutes=65)).to_pydatetime()
        ticks = get_ticks_window(window_start, window_end)
        if ticks.empty or len(ticks) < 5:
            continue

        entry = _mid_price_asof(ticks, ev["datetime_utc"] - pd.Timedelta(minutes=1))
        if entry is None:
            continue

        # Il trade "parte" all'orario dell'evento — mai usare tick precedenti
        # come se fossero il futuro rispetto all'entry.
        ticks_after = ticks[ticks["datetime_utc"] >= ev["datetime_utc"]]
        if len(ticks_after) < 3:
            continue
        events_used += 1

        mids = ((ticks_after["ask"] + ticks_after["bid"]) / 2.0).to_numpy()
        tick_bars_cache = _tick_bars(mids)
        candle_bars_cache = {
            iv: list(_resample_candles(ticks_after, iv).itertuples(index=False))
            for iv in candle_intervals
        }
        # namedtuple da itertuples non supporta bar["High"] — servono
        # oggetti indicizzabili per stringa, coerenti con _check_trade_bar.
        candle_bars_cache = {
            iv: [{"High": b.High, "Low": b.Low} for b in rows]
            for iv, rows in candle_bars_cache.items()
        }

        for signal in ("BUY", "SELL"):
            for regime in regimes:
                for atr in atrs:
                    risk = calculate_risk_levels(signal, entry, atr, regime)
                    tick_trade = _make_trade(signal, entry, risk["sl"], risk["tp1"], risk["tp2"], risk["tp3"])
                    tick_outcome = _walk_bars(tick_trade, tick_bars_cache)

                    for interval in candle_intervals:
                        candle_trade = _make_trade(signal, entry, risk["sl"], risk["tp1"], risk["tp2"], risk["tp3"])
                        candle_outcome = _walk_bars(candle_trade, candle_bars_cache[interval])

                        r = results[interval]
                        if tick_outcome == "NO_OUTCOME" and candle_outcome == "NO_OUTCOME":
                            r["both_no_outcome"] += 1
                        elif tick_outcome == candle_outcome:
                            r["agree"] += 1
                        else:
                            r["disagree"] += 1
                            if len(r["disagree_examples"]) < MAX_DISAGREE_EXAMPLES:
                                r["disagree_examples"].append({
                                    "event": ev["event_name"], "date": str(ev["datetime_utc"].date()),
                                    "signal": signal, "regime": regime, "atr": atr,
                                    "tick_outcome": tick_outcome, "candle_outcome": candle_outcome,
                                })

    return {"events_used": events_used, "by_interval": results}


def format_report(result: dict) -> str:
    lines = [f"=== Verifica realtà tick vs candele — {result['events_used']} eventi con tick reali ===\n"]
    for interval, r in result["by_interval"].items():
        total_compared = r["agree"] + r["disagree"]
        total_all = total_compared + r["both_no_outcome"]
        agree_pct = (r["agree"] / total_compared * 100) if total_compared else float("nan")
        lines.append(
            f"Candele {interval}: {total_all} trade sintetici testati, {total_compared} risolti da entrambi i metodi\n"
            f"  Concordano: {r['agree']} ({agree_pct:.1f}%)\n"
            f"  Discordano: {r['disagree']} ({100 - agree_pct:.1f}%)\n"
            f"  Nessun esito in nessuno dei due (finestra troppo corta): {r['both_no_outcome']}\n"
        )
        if r["disagree_examples"]:
            lines.append("  Esempi di discordanza:")
            for ex in r["disagree_examples"]:
                lines.append(
                    f"    {ex['date']} {ex['event']:30s} {ex['signal']:4s} {ex['regime']:8s} atr={ex['atr']:>4.1f}  "
                    f"tick={ex['tick_outcome']:10s} candela={ex['candle_outcome']}"
                )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_check()
    print(format_report(result))
