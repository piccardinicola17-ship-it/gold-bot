"""
dukascopy_ticks.py — Fase 2 del progetto dati storici.

Scarica i tick XAU/USD da Dukascopy attorno a ogni evento macro storico
(vedi historical_events.py, Fase 1) e ne ricava feature di reazione-prezzo
semplici (prezzo prima/dopo a vari orizzonti, escursione massima) — la
base su cui la Fase 3 costruirà le feature/etichette vere per il modello.

Formato Dukascopy verificato scaricando e decomprimendo un file reale in
questa sessione: LZMA-compresso, 20 byte/tick (3x uint32 big-endian
[offset_ms nell'ora, ask*1000, bid*1000] + 2x float32 [volume ask, volume
bid]). L'URL usa il MESE 0-indicizzato (00=gennaio, 11=dicembre) — un
dettaglio facile da sbagliare, verificato esplicitamente.

Il servizio è a tratti lento sul primo tentativo (osservato: timeout dopo
~25-30s poi riuscito al retry) — non è un blocco, serve solo pazienza e
un backoff, stesso pattern già usato altrove nel bot per Yahoo/Twelve Data.
"""

from __future__ import annotations

import io
import lzma
import logging
import sqlite3
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from trade_manager import BOT_DIR
from historical_events import HIST_DB_PATH, _connect

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TICK_CACHE_DIR = BOT_DIR / "dukascopy_cache"
BASE_URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD"

# Rispetto verso un servizio gratuito non documentato ufficialmente: pausa
# tra le richieste, mai in parallelo. Scoperto testando in questa sessione:
# con pausa di 1.5s il pilota ha comunque preso un 429/cooldown quasi ogni
# 3-4 richieste (25 eventi in 28 minuti reali — al ritmo osservato 1.344
# eventi avrebbero richiesto ~25 ORE). Il throttling sembra sull'intero
# servizio/IP con soglia più bassa del previsto, non sul singolo file:
# un rate limit blocca TUTTE le richieste successive per un po', non solo
# quella corrente. Pausa alzata a 5s per verificare se evita il problema
# invece di limitarsi a gestirlo dopo che è già scattato.
REQUEST_DELAY_SECONDS = 5.0
MAX_RETRIES = 2
COOLDOWN_SECONDS_ON_RATE_LIMIT = (90, 240)  # scala se il problema persiste

_cooldown_until = 0.0


def _respect_cooldown() -> None:
    global _cooldown_until
    now = time.time()
    if now < _cooldown_until:
        wait = _cooldown_until - now
        logger.info(f"In cooldown per rate limit Dukascopy, aspetto {wait:.0f}s")
        time.sleep(wait)


def _trigger_cooldown(attempt: int) -> None:
    global _cooldown_until
    seconds = COOLDOWN_SECONDS_ON_RATE_LIMIT[min(attempt, len(COOLDOWN_SECONDS_ON_RATE_LIMIT) - 1)]
    _cooldown_until = max(_cooldown_until, time.time() + seconds)

# Orizzonti di reazione calcolati per ogni evento (minuti relativi all'orario
# dell'evento). Negativi = prima, positivi = dopo. Tengono conto di finestre
# brevi (per catturare lo spike immediato) e più lunghe (per il movimento
# "digerito").
REACTION_OFFSETS_MIN = [-5, -1, 0, 1, 5, 15, 30, 60]

PILOT_CATEGORIES = ("NFP", "CPI", "FOMC")


def _hour_url(dt_utc: datetime) -> str:
    # Dukascopy indicizza il mese da 0: gennaio=00, dicembre=11.
    month0 = dt_utc.month - 1
    return f"{BASE_URL}/{dt_utc.year:04d}/{month0:02d}/{dt_utc.day:02d}/{dt_utc.hour:02d}h_ticks.bi5"


def _cache_path(dt_utc: datetime) -> Path:
    return TICK_CACHE_DIR / f"{dt_utc.year:04d}{dt_utc.month:02d}{dt_utc.day:02d}_{dt_utc.hour:02d}.bi5"


def _download_hour(dt_utc: datetime) -> bytes | None:
    """Scarica (o riusa dalla cache) il file .bi5 di una singola ora. None se quell'ora non ha avuto tick (mercato chiuso, es. weekend)."""
    cache_file = _cache_path(dt_utc)
    if cache_file.exists():
        return cache_file.read_bytes()

    url = _hour_url(dt_utc)
    last_error = None
    for attempt in range(MAX_RETRIES):
        _respect_cooldown()
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 404:
                # Nessun tick in quell'ora (weekend/festivo/mercato chiuso) — non è un errore.
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(b"")
                return None
            if r.status_code in (429, 503):
                # Throttling sull'intero servizio, non solo su questo file:
                # blocca ogni richiesta successiva per un po', non solo ritenta questa.
                _trigger_cooldown(attempt)
                last_error = f"{r.status_code} rate limit"
                continue
            r.raise_for_status()
            if not r.content:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(b"")
                return None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(r.content)
            return r.content
        except Exception as e:
            last_error = e
        finally:
            time.sleep(REQUEST_DELAY_SECONDS)

    logger.warning(f"Impossibile scaricare {url} dopo {MAX_RETRIES} tentativi: {last_error}")
    return None


def _parse_bi5(raw: bytes, hour_start_utc: datetime) -> pd.DataFrame:
    """Decomprime e traduce in DataFrame [datetime_utc, ask, bid]. Cache vuota (weekend) -> DataFrame vuoto."""
    if not raw:
        return pd.DataFrame(columns=["datetime_utc", "ask", "bid"])
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError as e:
        # File scaricato a metà per un intoppo di rete — non fidarsi di dati corrotti.
        raise ValueError(f"Decompressione fallita (file probabilmente troncato): {e}")

    n = len(data) // 20
    rows = []
    for i in range(n):
        offset_ms, ask_x1000, bid_x1000, _vol_ask, _vol_bid = struct.unpack(">IIIff", data[i * 20:(i + 1) * 20])
        ts = hour_start_utc + timedelta(milliseconds=offset_ms)
        rows.append((ts, ask_x1000 / 1000.0, bid_x1000 / 1000.0))
    return pd.DataFrame(rows, columns=["datetime_utc", "ask", "bid"])


def get_ticks_window(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Tutti i tick tra start_utc ed end_utc (incluso), scaricando/riusando ogni ora coinvolta."""
    frames = []
    hour = start_utc.replace(minute=0, second=0, microsecond=0)
    last_hour = end_utc.replace(minute=0, second=0, microsecond=0)
    while hour <= last_hour:
        raw = _download_hour(hour)
        if raw:
            frames.append(_parse_bi5(raw, hour))
        hour += timedelta(hours=1)
    if not frames:
        return pd.DataFrame(columns=["datetime_utc", "ask", "bid"])
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["datetime_utc"] >= start_utc) & (df["datetime_utc"] <= end_utc)]
    return df.sort_values("datetime_utc").reset_index(drop=True)


def _mid_price_asof(ticks: pd.DataFrame, target: pd.Timestamp) -> float | None:
    """Prezzo medio (ask+bid)/2 del tick più recente NON successivo a target. None se non c'è nessun tick prima."""
    eligible = ticks[ticks["datetime_utc"] <= target]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return (row["ask"] + row["bid"]) / 2.0


def compute_reaction_features(event_dt_utc: pd.Timestamp) -> dict | None:
    """
    Scarica i tick nella finestra [-10min, +65min] attorno all'evento e
    calcola prezzo agli orizzonti definiti in REACTION_OFFSETS_MIN, più
    l'escursione massima nell'ora successiva. Ritorna None se non ci sono
    proprio tick nella finestra (es. evento caduto in un giorno senza dati).
    """
    window_start = event_dt_utc - timedelta(minutes=10)
    window_end = event_dt_utc + timedelta(minutes=65)
    ticks = get_ticks_window(window_start.to_pydatetime(), window_end.to_pydatetime())
    if ticks.empty:
        return None

    prices = {}
    for m in REACTION_OFFSETS_MIN:
        target = event_dt_utc + timedelta(minutes=m)
        prices[f"price_t{m:+d}m"] = _mid_price_asof(ticks, target)

    after = ticks[ticks["datetime_utc"] >= event_dt_utc]
    max_up = max_down = None
    baseline = prices.get("price_t-1m")
    if not after.empty and baseline:
        mids = (after["ask"] + after["bid"]) / 2.0
        max_up = float(mids.max() - baseline)
        max_down = float(mids.min() - baseline)

    return {**prices, "max_excursion_up": max_up, "max_excursion_down": max_down, "tick_count": len(ticks)}


def init_reactions_table(db_path: str = HIST_DB_PATH) -> None:
    cols = ", ".join(f'"{f}" REAL' for f in [f"price_t{m:+d}m" for m in REACTION_OFFSETS_MIN])
    with _connect(db_path) as conn:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS event_price_reactions (
                event_uid          TEXT PRIMARY KEY REFERENCES macro_events(event_uid),
                {cols},
                max_excursion_up   REAL,
                max_excursion_down REAL,
                tick_count         INTEGER,
                computed_at        TEXT NOT NULL
            );
            """
        )


# Copertura estesa (3 settembre 2026): tutte le serie con numero reale
# (actual+forecast) e un campione statisticamente utilizzabile (n>=20 righe
# in macro_events) — così la Fase 2 non va rifatta una serie alla volta se
# la Fase 3/4 rivela un edge altrove. Filtrato per event_name esatto, non
# per macro_category: molte di queste serie sono taggate "OTHER" insieme a
# eventi che non vogliamo (Crude Oil Inventories escluso deliberatamente,
# più rilevante per il petrolio che per l'oro — l'utente ha confermato di
# escluderlo). Le serie con n<20 (Housing Starts, Industrial Production,
# ecc.) restano fuori: troppo poche occorrenze "high impact" nella fonte
# per validare comunque, indipendentemente da quanti tick si scaricano.
EXTENDED_EVENT_NAMES = (
    "Unemployment Claims", "Unemployment Rate", "Core Retail Sales m/m",
    "Retail Sales m/m", "ISM Manufacturing PMI", "ISM Services PMI",
    "Core CPI m/m", "PPI m/m", "ADP Non-Farm Employment Change",
    "CB Consumer Confidence", "Core Durable Goods Orders m/m",
    "Prelim UoM Consumer Sentiment", "Building Permits", "CPI m/m",
    "Average Hourly Earnings m/m", "Trade Balance", "Pending Home Sales m/m",
    "New Home Sales", "Philly Fed Manufacturing Index", "Existing Home Sales",
    "Advance GDP q/q", "Federal Funds Rate", "Prelim GDP q/q",
    "TIC Long-Term Purchases", "Core PCE Price Index m/m", "JOLTS Job Openings",
    "Flash Manufacturing PMI", "Flash Services PMI", "Final GDP q/q",
    "Core PPI m/m", "Empire State Manufacturing Index", "CPI y/y",
    "Non-Farm Employment Change",
)


def run_pilot(categories=None, event_names: tuple | None = None,
              limit: int | None = None, db_path: str = HIST_DB_PATH) -> dict:
    """
    Calcola le feature di reazione per gli eventi indicati, saltando quelli
    già fatti (ripartibile). event_names filtra per nome esatto (usato per
    la copertura estesa, dato che molte serie sono taggate macro_category
    "OTHER" insieme ad altre che non interessano); categories filtra per
    macro_category (comportamento originale del pilota NFP+CPI+FOMC). Se
    entrambi sono None usa PILOT_CATEGORIES per compatibilità.
    """
    init_reactions_table(db_path)
    if event_names:
        placeholders = ",".join("?" for _ in event_names)
        filter_clause = f"event_name IN ({placeholders})"
        filter_params = event_names
        filter_desc = f"{len(event_names)} serie per nome"
    else:
        categories = categories or PILOT_CATEGORIES
        placeholders = ",".join("?" for _ in categories)
        filter_clause = f"macro_category IN ({placeholders})"
        filter_params = categories
        filter_desc = f"categorie: {categories}"

    with _connect(db_path) as conn:
        query = (
            f"SELECT event_uid, datetime_utc, event_name FROM macro_events "
            f"WHERE {filter_clause} "
            f"AND event_uid NOT IN (SELECT event_uid FROM event_price_reactions) "
            f"ORDER BY datetime_utc"
        )
        if limit:
            query += f" LIMIT {int(limit)}"
        todo = conn.execute(query, filter_params).fetchall()

    total = len(todo)
    logger.info(f"Da processare: {total} eventi ({filter_desc})")
    done, empty, errors = 0, 0, 0

    for i, row in enumerate(todo):
        event_uid = row["event_uid"]
        event_dt = pd.Timestamp(row["datetime_utc"])
        try:
            feats = compute_reaction_features(event_dt)
        except Exception as e:
            logger.warning(f"Errore su {row['event_name']} @ {row['datetime_utc']}: {e}")
            errors += 1
            continue

        if feats is None:
            empty += 1
            continue

        now_iso = datetime.now(timezone.utc).isoformat()
        cols = list(feats.keys())
        quoted_cols = ", ".join('"{}"'.format(c) for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        with _connect(db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO event_price_reactions "
                f"(event_uid, {quoted_cols}, computed_at) "
                f"VALUES (?, {placeholders}, ?)",
                (event_uid, *[feats[c] for c in cols], now_iso),
            )
        done += 1

        if (i + 1) % 25 == 0 or (i + 1) == total:
            logger.info(f"Progresso: {i+1}/{total} (fatti={done}, senza tick={empty}, errori={errors})")

    return {"total": total, "done": done, "empty": empty, "errors": errors}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fase 2: feature di reazione-prezzo dai tick Dukascopy")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--extended", action="store_true",
                         help="Copertura estesa: tutte le serie con n>=20 (vedi EXTENDED_EVENT_NAMES), non solo NFP/CPI/FOMC")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.extended:
        result = run_pilot(event_names=EXTENDED_EVENT_NAMES, limit=args.limit)
    else:
        result = run_pilot(categories=tuple(args.categories) if args.categories else None, limit=args.limit)
    print(result)
