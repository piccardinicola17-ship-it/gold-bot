"""
Gestione completa del ciclo di vita dei trade.

Il database SQLite è l'unica fonte di verità. ``active_trades.json`` è soltanto
uno snapshot compatibile con le vecchie versioni e viene rigenerato dal DB.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytz
import requests

logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")
XAUUSD_PIP_SIZE = float(os.environ.get("XAUUSD_PIP_SIZE", "0.10"))

PROJECT_DIR = Path(__file__).resolve().parent
BOT_DIR = Path(os.environ.get("BOT_DIR", PROJECT_DIR / "data")).expanduser().resolve()
BOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(Path(os.environ.get("DB_PATH", BOT_DIR / "goldbot.db")).expanduser().resolve())
ACTIVE_FILE = str(BOT_DIR / "active_trades.json")

TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "")
AUTHORIZED_ID = os.environ.get("CHAT_ID", "").strip()

TF_LABEL = {"5min": "M5", "15min": "M15", "1h": "H1", "4h": "H4", "1day": "D1"}
PENDING_TTL_MINUTES = {
    "5min": 30,
    "15min": 90,
    "1h": 360,
    "4h": 1440,
    "1day": 4320,
}

RESULT_PNL = {
    "WIN_TP1": 1.0,
    "WIN_TP2": 2.0,
    "WIN_TP3": 3.0,
    "WIN_BE": 0.0,
    "LOSS": -1.0,
    "CANCELLED": 0.0,
}

_write_lock = threading.RLock()
_price_cache = {"price": 0.0, "ts": 0.0}


class DuplicateSetupError(RuntimeError):
    """Il medesimo setup è già stato registrato."""


def is_authorized(update) -> bool:
    """Autorizzazione fail-closed: senza CHAT_ID nessun comando è accettato."""
    if not AUTHORIZED_ID:
        logger.error("CHAT_ID non configurato: comando Telegram rifiutato")
        return False
    chat = getattr(update, "effective_chat", None)
    return chat is not None and str(chat.id) == AUTHORIZED_ID


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_trade_columns(conn: sqlite3.Connection) -> None:
    """Aggiorna in-place qualunque schema ``trades`` precedente.

    Alcune installazioni storiche usavano una tabella ``trades`` più piccola
    (o creata da un altro modulo).  ``CREATE TABLE IF NOT EXISTS`` non aggiunge
    le colonne mancanti, quindi ogni colonna usata dalla versione corrente deve
    essere verificata prima di creare indici o lanciare le migrazioni.
    """
    additions = {
        "trade_id": "TEXT",
        "setup_key": "TEXT",
        "timestamp": "TEXT",
        "activated_at": "TEXT",
        "closed_at": "TEXT",
        "signal": "TEXT DEFAULT 'NEUTRAL'",
        "order_type": "TEXT DEFAULT 'UNKNOWN'",
        "timeframe": "TEXT DEFAULT '?'",
        "regime": "TEXT DEFAULT ''",
        "entry": "REAL DEFAULT 0",
        "sl": "REAL DEFAULT 0",
        "tp1": "REAL DEFAULT 0",
        "tp2": "REAL DEFAULT 0",
        "tp3": "REAL DEFAULT 0",
        "be_price": "REAL",
        "prob": "INTEGER DEFAULT 0",
        "risk_pct": "REAL DEFAULT 0",
        "lot_size": "REAL DEFAULT 0",
        "status": "TEXT DEFAULT 'OPEN'",
        "activated": "INTEGER DEFAULT 0",
        "entry_filled": "INTEGER DEFAULT 0",
        "counted_open": "INTEGER DEFAULT 0",
        "counted_close": "INTEGER DEFAULT 0",
        "be_armed": "INTEGER DEFAULT 0",
        "be_hit": "INTEGER DEFAULT 0",
        "tp1_hit": "INTEGER DEFAULT 0",
        "tp2_hit": "INTEGER DEFAULT 0",
        "tp3_hit": "INTEGER DEFAULT 0",
        "notified_json": "TEXT DEFAULT '{}'",
        "strategies_json": "TEXT DEFAULT '{}'",
        "data_timestamp": "TEXT",
        "result": "TEXT",
        "exit_price": "REAL",
        "pips": "REAL",
        "pnl_r": "REAL",
        "notes": "TEXT DEFAULT ''",
        # Scarto GC=F (futures, usato per calcolare i livelli) - spot (quello
        # che l'utente vede sul suo broker) catturato all'apertura del trade.
        # Applicato per tradurre entry/sl/tp1/tp2/tp3 in "equivalente spot" e
        # per convertire il prezzo live durante il monitoraggio. Vedi
        # open_trade() e _monitor_single().
        "price_basis": "REAL DEFAULT 0",
    }
    existing = _column_names(conn, "trades")
    status_was_missing = "status" not in existing
    trade_id_was_missing = "trade_id" not in existing
    timestamp_was_missing = "timestamp" not in existing
    activated_was_missing = "activated" not in existing
    entry_filled_was_missing = "entry_filled" not in existing
    counted_open_was_missing = "counted_open" not in existing
    counted_close_was_missing = "counted_close" not in existing

    for name, ddl in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")

    # Rende identificabili anche eventuali righe provenienti da schemi molto
    # vecchi. ``rowid`` è disponibile nelle normali tabelle SQLite.
    if trade_id_was_missing:
        conn.execute(
            "UPDATE trades SET trade_id='legacy-trade-' || rowid "
            "WHERE trade_id IS NULL OR TRIM(trade_id)=''"
        )
    if timestamp_was_missing:
        conn.execute(
            "UPDATE trades SET timestamp=? "
            "WHERE timestamp IS NULL OR TRIM(timestamp)=''",
            (datetime.now(TIMEZONE).isoformat(),),
        )

    # Se ``status`` non esisteva, ricostruisce lo stato dai risultati già
    # registrati invece di marcare erroneamente tutto come OPEN.
    if status_was_missing:
        conn.execute(
            "UPDATE trades SET status=CASE "
            "WHEN result IS NOT NULL AND TRIM(result)<>'' THEN 'CLOSED' "
            "ELSE 'OPEN' END"
        )
    else:
        conn.execute(
            "UPDATE trades SET status='OPEN' "
            "WHERE status IS NULL OR TRIM(status)=''"
        )

    # Le vecchie versioni registravano soltanto trade già attivati.
    if activated_was_missing:
        conn.execute("UPDATE trades SET activated=1")
    if entry_filled_was_missing:
        conn.execute("UPDATE trades SET entry_filled=1")
    if counted_open_was_missing:
        conn.execute("UPDATE trades SET counted_open=1")
    if counted_close_was_missing:
        conn.execute(
            "UPDATE trades SET counted_close=CASE "
            "WHEN status='CLOSED' THEN 1 ELSE 0 END"
        )


def _ensure_session_columns(conn: sqlite3.Connection) -> None:
    """Completa senza perdita dati una tabella ``sessions`` legacy."""
    additions = {
        "date": "TEXT",
        "trades_count": "INTEGER DEFAULT 0",
        "wins": "INTEGER DEFAULT 0",
        "losses": "INTEGER DEFAULT 0",
        "consecutive_losses": "INTEGER DEFAULT 0",
        "pnl_r": "REAL DEFAULT 0",
        "session_stopped": "INTEGER DEFAULT 0",
        "session_stopped_at": "TEXT",   # ora blocco per cooldown automatico
    }
    existing = _column_names(conn, "sessions")
    for name, ddl in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {ddl}")


def init_db() -> None:
    """Crea lo schema, migra lo storico legacy e riconcilia lo stato attivo."""
    with _write_lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id        TEXT UNIQUE NOT NULL,
                setup_key       TEXT,
                timestamp       TEXT NOT NULL,
                activated_at    TEXT,
                closed_at       TEXT,
                signal          TEXT NOT NULL,
                order_type      TEXT NOT NULL,
                timeframe       TEXT NOT NULL,
                regime          TEXT DEFAULT '',
                entry           REAL NOT NULL,
                sl              REAL NOT NULL,
                tp1             REAL NOT NULL,
                tp2             REAL NOT NULL,
                tp3             REAL NOT NULL,
                be_price        REAL,
                prob            INTEGER DEFAULT 0,
                risk_pct        REAL DEFAULT 0,
                lot_size        REAL DEFAULT 0,
                status          TEXT DEFAULT 'OPEN',
                activated       INTEGER DEFAULT 0,
                entry_filled    INTEGER DEFAULT 0,
                counted_open    INTEGER DEFAULT 0,
                counted_close   INTEGER DEFAULT 0,
                be_armed        INTEGER DEFAULT 0,
                be_hit          INTEGER DEFAULT 0,
                tp1_hit         INTEGER DEFAULT 0,
                tp2_hit         INTEGER DEFAULT 0,
                tp3_hit         INTEGER DEFAULT 0,
                notified_json   TEXT DEFAULT '{}',
                strategies_json TEXT DEFAULT '{}',
                data_timestamp  TEXT,
                result          TEXT,
                exit_price      REAL,
                pips            REAL,
                pnl_r           REAL,
                notes           TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                date                TEXT UNIQUE NOT NULL,
                trades_count        INTEGER DEFAULT 0,
                wins                INTEGER DEFAULT 0,
                losses              INTEGER DEFAULT 0,
                consecutive_losses  INTEGER DEFAULT 0,
                pnl_r               REAL DEFAULT 0,
                session_stopped     INTEGER DEFAULT 0,
                session_stopped_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        _ensure_trade_columns(conn)
        _ensure_session_columns(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_trade_id "
            "ON trades(trade_id) WHERE trade_id IS NOT NULL"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_setup_key "
            "ON trades(setup_key) WHERE setup_key IS NOT NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"
        )

        migrated = _migrate_legacy_signals(conn)
        migrated = _migrate_legacy_active_file(conn) or migrated
        if not _migration_done(conn, "fix_be_semantics_v1"):
            conn.execute(
                """
                UPDATE trades
                   SET be_armed=CASE
                       WHEN tp1_hit=1 OR be_hit=1 THEN 1
                       ELSE COALESCE(be_armed, 0)
                   END
                 WHERE status='OPEN'
                """
            )
            conn.execute(
                """
                UPDATE trades
                   SET be_hit=CASE WHEN result='WIN_BE' THEN 1 ELSE 0 END
                 WHERE status='CLOSED'
                """
            )
            conn.execute("UPDATE trades SET be_hit=0 WHERE status='OPEN'")
            _mark_migration(conn, "fix_be_semantics_v1")
        if not _migration_done(conn, "backfill_pips_v1"):
            rows = conn.execute(
                """
                SELECT id, signal, entry, exit_price
                  FROM trades
                 WHERE status='CLOSED' AND exit_price IS NOT NULL
                """
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE trades SET pips=? WHERE id=?",
                    (
                        calculate_trade_pips(
                            row["signal"], row["entry"], row["exit_price"]
                        ),
                        row["id"],
                    ),
                )
            _mark_migration(conn, "backfill_pips_v1")
        if not _migration_done(conn, "fix_pip_size_v2"):
            # Ricalcola tutti i pips con il pip_size corretto (0.10 invece di 0.01)
            rows = conn.execute(
                "SELECT id, signal, entry, exit_price FROM trades "
                "WHERE status='CLOSED' AND exit_price IS NOT NULL"
            ).fetchall()
            for row in rows:
                correct_pips = calculate_trade_pips(row["signal"], row["entry"], row["exit_price"])
                conn.execute("UPDATE trades SET pips=? WHERE id=?", (correct_pips, row["id"]))
            _mark_migration(conn, "fix_pip_size_v2")

            _rebuild_sessions(conn)
            _mark_migration(conn, "rebuild_sessions_from_trades_v2")
        conn.commit()

    _sync_active_snapshot()
    logger.info("Database inizializzato: %s", DB_PATH)


def _migration_done(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM schema_migrations WHERE name=?", (name,)
    ).fetchone() is not None


def _mark_migration(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(name, applied_at) VALUES (?,?)",
        (name, datetime.now(TIMEZONE).isoformat()),
    )


def _parse_legacy_time(value: str) -> str:
    value = value or ""
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = TIMEZONE.localize(dt)
            return dt.astimezone(TIMEZONE).isoformat()
        except ValueError:
            continue
    return datetime.now(TIMEZONE).isoformat()


def _migrate_legacy_signals(conn: sqlite3.Connection) -> bool:
    name = "legacy_signals_to_trades_v1"
    if _migration_done(conn, name) or not _table_exists(conn, "signals"):
        return False

    rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    for row in rows:
        data = dict(row)
        raw_result = (data.get("result") or "").upper()
        pending = raw_result in ("", "PENDING", "OPEN")
        result = "CANCELLED" if pending else raw_result
        pnl_r = RESULT_PNL.get(result, 0.0)
        timestamp = _parse_legacy_time(data.get("time", ""))
        trade_id = f"legacy-signal-{data['id']}"
        conn.execute(
            """
            INSERT OR IGNORE INTO trades(
                trade_id, setup_key, timestamp, closed_at, signal, order_type,
                timeframe, regime, entry, sl, tp1, tp2, tp3, prob, status,
                activated, entry_filled, counted_open, counted_close,
                strategies_json, result, exit_price, pnl_r, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade_id,
                f"legacy-signal:{data['id']}",
                timestamp,
                timestamp,
                data.get("signal") or "NEUTRAL",
                data.get("order_type") or data.get("signal") or "UNKNOWN",
                data.get("timeframe") or "?",
                data.get("regime") or "",
                float(data.get("entry") or 0),
                float(data.get("sl") or 0),
                float(data.get("tp1") or 0),
                float(data.get("tp2") or 0),
                float(data.get("tp3") or 0),
                int(data.get("prob") or 0),
                "CLOSED",
                0 if pending else 1,
                0 if pending else 1,
                0 if pending else 1,
                1,
                data.get("strategies") or "{}",
                result,
                None,
                pnl_r,
                "Migrato dal DB legacy; i vecchi pending sono stati archiviati come CANCELLED.",
            ),
        )

    _mark_migration(conn, name)
    logger.info("Migrati %s record dalla tabella legacy signals", len(rows))
    return True


def _migrate_legacy_active_file(conn: sqlite3.Connection) -> bool:
    name = "legacy_active_json_to_trades_v1"
    if _migration_done(conn, name) or not os.path.exists(ACTIVE_FILE):
        return False

    try:
        with open(ACTIVE_FILE, encoding="utf-8") as handle:
            active = json.load(handle)
    except (OSError, json.JSONDecodeError):
        active = []

    imported = 0
    for old in active if isinstance(active, list) else []:
        if old.get("auto_closed"):
            continue
        old_id = str(old.get("trade_id") or "")
        trade_id = old_id if old_id else str(uuid.uuid4())
        if conn.execute("SELECT 1 FROM trades WHERE trade_id=?", (trade_id,)).fetchone():
            trade_id = str(uuid.uuid4())
        timestamp = _parse_legacy_time(old.get("timestamp", ""))
        activated = bool(old.get("activated") or old.get("entry_filled"))
        setup_key = f"legacy-active:{old_id or trade_id}"
        conn.execute(
            """
            INSERT OR IGNORE INTO trades(
                trade_id, setup_key, timestamp, activated_at, signal, order_type,
                timeframe, regime, entry, sl, tp1, tp2, tp3, be_price, prob,
                risk_pct, status, activated, entry_filled, counted_open,
                notified_json, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade_id,
                setup_key,
                timestamp,
                timestamp if activated else None,
                old.get("signal") or "NEUTRAL",
                old.get("order_type") or old.get("signal") or "UNKNOWN",
                old.get("timeframe") or "?",
                old.get("regime") or "",
                float(old.get("entry") or 0),
                float(old.get("sl") or 0),
                float(old.get("tp1") or 0),
                float(old.get("tp2") or 0),
                float(old.get("tp3") or 0),
                float(old.get("be") or 0) or None,
                int(old.get("prob") or 0),
                float(old.get("risk_pct") or 1.0),
                "OPEN",
                int(activated),
                int(activated),
                int(activated),
                json.dumps(old.get("notified") or {}),
                "Importato da active_trades.json legacy.",
            ),
        )
        imported += 1

    _mark_migration(conn, name)
    logger.info("Importati %s trade attivi legacy; record auto_closed rimossi", imported)
    return True


def _rebuild_sessions(conn: sqlite3.Connection) -> None:
    """Ricostruisce le statistiche giornaliere dai trade migrati."""
    conn.execute("DELETE FROM sessions")
    rows = conn.execute("SELECT * FROM trades ORDER BY COALESCE(closed_at,timestamp), id").fetchall()
    for row in rows:
        trade = dict(row)
        open_date = (trade.get("activated_at") or trade.get("timestamp") or "")[:10]
        if trade.get("counted_open") and open_date:
            conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (open_date,))
            conn.execute(
                "UPDATE sessions SET trades_count=trades_count+1 WHERE date=?",
                (open_date,),
            )
        result = trade.get("result")
        close_date = (trade.get("closed_at") or "")[:10]
        if trade.get("counted_close") and result and close_date:
            conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (close_date,))
            win = int(
                result in ("WIN_TP1", "WIN_TP2", "WIN_TP3")
                or bool(trade.get("tp1_hit"))
            )
            loss = int(result == "LOSS")
            conn.execute(
                """
                UPDATE sessions SET
                    wins=wins+?, losses=losses+?, pnl_r=pnl_r+?,
                    consecutive_losses=CASE
                        WHEN ?=1 THEN 0
                        WHEN ?=1 THEN consecutive_losses+1
                        ELSE consecutive_losses
                    END
                WHERE date=?
                """,
                (win, loss, float(trade.get("pnl_r") or 0), win, loss, close_date),
            )


def build_setup_key(data: dict) -> str:
    """Fingerprint persistente del setup/candela, usato contro i duplicati."""
    timestamp = str(data.get("data_timestamp") or "")
    if not timestamp:
        now = datetime.now(TIMEZONE)
        timestamp = now.strftime("%Y-%m-%dT%H:%M")
    payload = "|".join(
        [
            str(data.get("symbol", "XAU/USD")),
            str(data.get("timeframe", "?")),
            str(data.get("signal", "")),
            str(data.get("order_type", "")),
            f"{float(data.get('entry') or 0):.5f}",
            timestamp,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def was_setup_seen(setup_key: str) -> bool:
    if not setup_key:
        return False
    with _connect() as conn:
        return conn.execute(
            "SELECT 1 FROM trades WHERE setup_key=?", (setup_key,)
        ).fetchone() is not None


def is_pending_order(order_type: str) -> bool:
    value = str(order_type).upper()
    return "LIMIT" in value or "STOP" in value


def open_trade(data: dict) -> str:
    """Registra un setup; i MARKET vengono attivati immediatamente."""
    now = datetime.now(TIMEZONE)
    trade_id = str(uuid.uuid4())
    setup_key = str(data.get("setup_key") or build_setup_key(data))
    pending = is_pending_order(data.get("order_type", data.get("signal", "")))

    notified = data.get("notified") or {}
    strategies = data.get("strategies") or {}

    try:
        with _write_lock, _connect() as conn:
            conn.execute(
                """
                INSERT INTO trades(
                    trade_id, setup_key, timestamp, signal, order_type, timeframe,
                    regime, entry, sl, tp1, tp2, tp3, be_price, prob, risk_pct,
                    lot_size, status, notified_json, strategies_json, data_timestamp,
                    price_basis
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trade_id,
                    setup_key,
                    now.isoformat(),
                    data.get("signal"),
                    data.get("order_type", data.get("signal")),
                    data.get("timeframe"),
                    data.get("regime", ""),
                    float(data.get("entry", 0)),
                    float(data.get("sl", 0)),
                    float(data.get("tp1", 0)),
                    float(data.get("tp2", 0)),
                    float(data.get("tp3", 0)),
                    float(data.get("be", 0)) or None,
                    int(data.get("prob", 0)),
                    float(data.get("risk_pct", 0)),
                    float(data.get("lot_size", 0)),
                    "OPEN",
                    json.dumps(notified),
                    json.dumps(strategies),
                    data.get("data_timestamp"),
                    float(data.get("price_basis", 0) or 0),
                ),
            )
    except sqlite3.IntegrityError as exc:
        if "setup_key" in str(exc):
            raise DuplicateSetupError("Setup già registrato per questa candela") from exc
        raise

    if not pending:
        activate_trade(trade_id)
    else:
        _sync_active_snapshot()

    logger.info("Trade registrato: %s", trade_id)
    return trade_id


def _today() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d")


def calculate_trade_pips(signal: str, entry: float, exit_price: float) -> float:
    """
    Calcola i pips virtuali XAU/USD.
    Per XAU/USD: 1 pip = $0.10 (non $0.01 come nel forex standard).
    Es: entry=4075.25, exit=4064.62, SELL → (4075.25-4064.62)/0.10 = 106.3 pip
    """
    try:
        entry_value = float(entry)
        exit_value = float(exit_price)
    except (TypeError, ValueError):
        return 0.0
    if XAUUSD_PIP_SIZE <= 0:
        return 0.0
    direction = 1.0 if str(signal).upper() == "BUY" else -1.0
    return round(((exit_value - entry_value) * direction) / XAUUSD_PIP_SIZE, 1)


def activate_trade(trade_id: str) -> bool:
    """Attiva il trade e incrementa la sessione esattamente una volta."""
    now = datetime.now(TIMEZONE)
    changed = False
    with _write_lock, _connect() as conn:
        row = conn.execute(
            "SELECT status, counted_open FROM trades WHERE trade_id=?", (trade_id,)
        ).fetchone()
        if not row or row["status"] != "OPEN":
            return False
        if not row["counted_open"]:
            conn.execute(
                """
                UPDATE trades SET activated=1, entry_filled=1, counted_open=1,
                                  activated_at=?
                WHERE trade_id=? AND status='OPEN'
                """,
                (now.isoformat(), trade_id),
            )
            today = now.strftime("%Y-%m-%d")
            conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (today,))
            conn.execute(
                "UPDATE sessions SET trades_count=trades_count+1 WHERE date=?",
                (today,),
            )
            changed = True
        else:
            conn.execute(
                "UPDATE trades SET activated=1, entry_filled=1 WHERE trade_id=?",
                (trade_id,),
            )
    _sync_active_snapshot()
    return changed


def close_trade(trade_id: str, result: str, exit_price: float, notes: str = "") -> bool:
    """Chiude e contabilizza il trade una sola volta.

    Il risultato finale aggiorna anche i flag dei livelli. In questo modo una
    chiusura manuale e una chiusura automatica producono esattamente gli stessi
    dati per Telegram e dashboard.
    """
    result = result.upper()
    if result not in RESULT_PNL:
        raise ValueError(f"Risultato non valido: {result}")
    pnl_r = RESULT_PNL[result]
    now = datetime.now(TIMEZONE)

    with _write_lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE trade_id=? AND status='OPEN'", (trade_id,)
        ).fetchone()
        if not row:
            return False

        tp1_hit = int(bool(row["tp1_hit"]))
        tp2_hit = int(bool(row["tp2_hit"]))
        tp3_hit = int(bool(row["tp3_hit"]))
        be_armed = int(bool(row["be_armed"]))
        be_hit = int(bool(row["be_hit"]))

        if result == "WIN_TP1":
            tp1_hit = 1
        elif result == "WIN_TP2":
            tp1_hit = 1
            tp2_hit = 1
        elif result == "WIN_TP3":
            tp1_hit = 1
            tp2_hit = 1
            tp3_hit = 1
        elif result == "WIN_BE":
            be_hit = 1

        if tp1_hit:
            be_armed = 1

        # WIN_BE: il trade ha raggiunto TP1 prima di tornare a BE.
        # I pips devono essere calcolati fino a TP1 (il massimo raggiunto),
        # non fino all'exit price (che è l'entry = 0 pips).
        if result == "WIN_BE" and row.get("tp1") and float(row["tp1"]) > 0:
            pips = calculate_trade_pips(row["signal"], row["entry"], float(row["tp1"]))
        else:
            pips = calculate_trade_pips(row["signal"], row["entry"], exit_price)

        conn.execute(
            """
            UPDATE trades SET status='CLOSED', result=?, exit_price=?, pnl_r=?,
                              pips=?, closed_at=?, notes=?, counted_close=1,
                              tp1_hit=?, tp2_hit=?, tp3_hit=?,
                              be_armed=?, be_hit=?
            WHERE trade_id=? AND status='OPEN'
            """,
            (
                result,
                float(exit_price),
                pnl_r,
                pips,
                now.isoformat(),
                notes,
                tp1_hit,
                tp2_hit,
                tp3_hit,
                be_armed,
                be_hit,
                trade_id,
            ),
        )

        if row["activated"] and result != "CANCELLED":
            today = now.strftime("%Y-%m-%d")
            conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (today,))
            win = int(
                result in ("WIN_TP1", "WIN_TP2", "WIN_TP3")
                or bool(tp1_hit)
            )
            loss = int(result == "LOSS")
            conn.execute(
                """
                UPDATE sessions SET
                    wins=wins+?, losses=losses+?, pnl_r=pnl_r+?,
                    consecutive_losses=CASE
                        WHEN ?=1 THEN 0
                        WHEN ?=1 THEN consecutive_losses+1
                        ELSE consecutive_losses
                    END
                WHERE date=?
                """,
                (win, loss, pnl_r, win, loss, today),
            )

    _sync_active_snapshot()
    logger.info("Trade chiuso: %s -> %s (%+.1fR)", trade_id, result, pnl_r)
    return True


def _decode_row(row: sqlite3.Row | dict) -> dict:
    data = dict(row)
    for source, target, default in (
        ("notified_json", "notified", {}),
        ("strategies_json", "strategies", {}),
    ):
        try:
            data[target] = json.loads(data.get(source) or "{}")
        except json.JSONDecodeError:
            data[target] = default
    data["be"] = data.get("be_price")
    data["activated"] = bool(data.get("activated"))
    data["entry_filled"] = bool(data.get("entry_filled"))
    data["be_armed"] = bool(data.get("be_armed"))
    data["be_hit"] = bool(data.get("be_hit"))
    data["tp1_hit"] = bool(data.get("tp1_hit"))
    data["tp2_hit"] = bool(data.get("tp2_hit"))
    data["tp3_hit"] = bool(data.get("tp3_hit"))
    return data


def load_all_active_trades() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY id"
        ).fetchall()
    return [_decode_row(row) for row in rows]


def load_active_trade() -> dict:
    trades = load_all_active_trades()
    return trades[-1] if trades else {}


def get_trade_by_id(trade_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
    return _decode_row(row) if row else {}


def get_open_trade_by_timeframe(timeframe: str) -> dict:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM trades
            WHERE status='OPEN' AND timeframe=?
            ORDER BY id DESC LIMIT 1
            """,
            (timeframe,),
        ).fetchone()
    return _decode_row(row) if row else {}


def has_open_trade_on_timeframe(timeframe: str) -> bool:
    return bool(get_open_trade_by_timeframe(timeframe))


def _atomic_write_snapshot(data: list[dict]) -> None:
    directory = os.path.dirname(ACTIVE_FILE) or "."
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, default=str)
        temp_path = handle.name
    os.replace(temp_path, ACTIVE_FILE)


def _sync_active_snapshot() -> None:
    with _write_lock:
        _atomic_write_snapshot(load_all_active_trades())


def _update_trade(trade_id: str, **fields) -> None:
    allowed = {
        "be_armed",
        "be_hit",
        "tp1_hit",
        "tp2_hit",
        "tp3_hit",
        "notified_json",
        "activated",
        "entry_filled",
    }
    clean = {key: value for key, value in fields.items() if key in allowed}
    if not clean:
        return
    assignments = ", ".join(f"{key}=?" for key in clean)
    with _write_lock, _connect() as conn:
        conn.execute(
            f"UPDATE trades SET {assignments} WHERE trade_id=?",
            (*clean.values(), trade_id),
        )
    _sync_active_snapshot()


def mark_tp1_hit(trade_id: str) -> None:
    _update_trade(trade_id, tp1_hit=1)


def mark_tp2_hit(trade_id: str) -> None:
    _update_trade(trade_id, tp2_hit=1)


def mark_tp3_hit(trade_id: str) -> None:
    _update_trade(trade_id, tp3_hit=1)


def arm_break_even(trade_id: str) -> None:
    _update_trade(trade_id, be_armed=1)


def mark_be_hit(trade_id: str) -> None:
    _update_trade(trade_id, be_hit=1)


def update_notified_json(trade_id: str, notified: dict) -> None:
    _update_trade(trade_id, notified_json=json.dumps(notified))


def update_trade_field_json(trade_id: str, field: str, value) -> None:
    mapping = {
        "activated": "activated",
        "entry_filled": "entry_filled",
    }
    column = mapping.get(field)
    if column:
        _update_trade(trade_id, **{column: int(bool(value))})


_PRICE_CACHE_TTL = 45  # secondi — Twelve Data free: 8 req/min → 1 req ogni 7.5s minimo


def _fetch_price_goldapi() -> float:
    """
    gold-api.com — gratuita, nessuna API key, nessun limite noto.
    Aggiunta come fonte primaria dopo che Twelve Data (quota giornaliera) e
    Yahoo/yfinance (rate limit) sono risultati entrambi giù insieme in
    produzione, lasciando il monitor senza nessun prezzo per minuti.
    """
    response = requests.get(
        "https://api.gold-api.com/price/XAU",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=6,
    )
    response.raise_for_status()
    return float(response.json()["price"])


def _fetch_price_twelvedata() -> float:
    """Twelve Data — primario se i crediti sono disponibili."""
    if not TWELVE_API_KEY:
        raise RuntimeError("No API key")
    response = requests.get(
        "https://api.twelvedata.com/price",
        params={"symbol": "XAU/USD", "apikey": TWELVE_API_KEY},
        timeout=5,
    )
    if response.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests")
    response.raise_for_status()
    return float(response.json()["price"])


def _fetch_price_stooq() -> float:
    """
    Stooq.com — fonte gratuita, nessuna API key, funziona da server.
    Simbolo XAU/USD su Stooq: XAUUSD
    """
    response = requests.get(
        "https://stooq.com/q/l/?s=xauusd&f=sd2t2ohlcv&h&e=csv",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=8,
    )
    response.raise_for_status()
    lines = response.text.strip().split("\n")
    if len(lines) < 2:
        raise ValueError("Stooq: risposta vuota")
    # CSV: Symbol,Date,Time,Open,High,Low,Close,Volume
    parts = lines[1].split(",")
    return float(parts[6])  # Close


def _fetch_price_metals_api() -> float:
    """
    metals.live API — gratuita, nessun rate limit noto.
    Ritorna prezzo troy ounce in USD.
    """
    response = requests.get(
        "https://metals.live/api/spot",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    # Risposta: [{"metal":"gold","price":XXXX,...},...]
    for item in data:
        if str(item.get("metal", "")).lower() in ("gold", "xau"):
            return float(item["price"])
    raise ValueError("metals.live: oro non trovato nella risposta")


def _fetch_price_yahoo() -> float:
    """
    Yahoo Finance — GC=F (futures oro COMEX). Il simbolo forex-spot
    "XAUUSD=X" è stato rimosso da Yahoo (404 su tutte le richieste, non un
    blocco IP): GC=F è l'unico proxy gold ancora attivo su questa API.
    """
    response = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
        params={"interval": "1m", "range": "1m"},
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
        timeout=6,
    )
    response.raise_for_status()
    data = response.json()
    price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
    return float(price)


def _fetch_price_sync() -> float:
    """
    Prezzo XAU/USD con cache 45s e 5 fonti in cascata.

    Ordine: Yahoo Finance (GC=F) → gold-api.com → Twelve Data → Stooq →
    metals.live → cache stale.

    IMPORTANTE — coerenza di scala: entry/SL/TP di ogni trade sono calcolati
    da analyzer.get_data(), che usa GC=F (futures oro COMEX) come fonte
    primaria. gold-api.com e Twelve Data (simbolo "XAU/USD") sono invece
    prezzo SPOT — e lo scarto reale osservato tra i due può superare i $40
    su ~$4400 (oltre l'1%). Se il monitor usa una fonte spot mentre i livelli
    del trade sono in scala futures, TP1/TP2/TP3/SL/BE possono scattare nel
    momento sbagliato — trovato in produzione: gold-api.com era stata messa
    per prima per un'emergenza (Twelve Data+Yahoo giù insieme), ma così il
    monitor confrontava sistematicamente prezzo spot con livelli futures.
    Yahoo/GC=F torna quindi primario (stessa scala dei livelli); gold-api.com
    e Twelve Data restano come fallback per quando Yahoo è irraggiungibile —
    meglio un prezzo con basis diverso che nessun prezzo, ma è un compromesso
    consapevole, non la norma. Stooq (quote live) e metals.live rispondono
    404 su tutti gli endpoint (servizi dismessi) — restano in fondo alla
    cascata come tentativo extra a costo quasi nullo.
    Se tutte le fonti live falliscono usa il prezzo più recente in cache.
    Il monitor NON si ferma mai per mancanza di prezzo — usa il dato più vecchio.
    """
    now = time.time()

    if now - _price_cache["ts"] < _PRICE_CACHE_TTL and _price_cache["price"] > 0:
        return float(_price_cache["price"])

    sources = [
        ("Yahoo Finance", _fetch_price_yahoo),
        ("gold-api.com",  _fetch_price_goldapi),
        ("Twelve Data",   _fetch_price_twelvedata),
        ("Stooq",         _fetch_price_stooq),
        ("metals.live",   _fetch_price_metals_api),
    ]

    for name, fetch_fn in sources:
        try:
            price = fetch_fn()
            if price > 0:
                _price_cache.update(price=price, ts=now)
                logger.debug(f"Prezzo da {name}: ${price}")
                return price
        except requests.HTTPError as e:
            if "429" in str(e):
                logger.debug(f"{name}: rate limit, prossima fonte")
            else:
                logger.debug(f"{name}: HTTP error {e}")
        except Exception as e:
            logger.debug(f"{name}: {e}")

    # Tutte le fonti live fallite — usa cache stale
    if _price_cache["price"] > 0:
        age = (now - _price_cache["ts"]) / 60
        logger.warning(f"Prezzo: tutte le fonti fallite, uso cache stale ${_price_cache['price']} (età: {age:.0f} min)")
        return float(_price_cache["price"])

    logger.error("Prezzo non disponibile da nessuna fonte e cache vuota")
    return 0.0


async def get_current_price_async() -> float:
    return await asyncio.to_thread(_fetch_price_sync)


def get_current_price() -> float:
    return _fetch_price_sync()


# Tracker in memoria del range (high/low) della candela M5 corrente,
# costruito dai campionamenti di prezzo che il monitor fa già ogni 10s.
# PRIMA questa funzione scaricava una candela fresca da get_data() ad ogni
# giro (bypass_cache=True) per non perdere ombre intracandle: >300
# chiamate/ora 24/7, che di fatto ha contribuito a far scattare il rate
# limit di Yahoo/yfinance. Il tracker ottiene lo stesso risultato (anzi più
# preciso, aggiorna ogni 10s invece che sulla candela del vendor) a costo
# di rete zero, perché riusa il prezzo che get_current_price_async() prende
# comunque ad ogni ciclo del monitor.
_CANDLE_PERIOD_SECONDS = 300  # M5
_live_candle = {"period_start": 0, "high": 0.0, "low": 0.0}


def _update_live_candle(price: float) -> None:
    if price <= 0:
        return
    period_start = int(time.time() // _CANDLE_PERIOD_SECONDS) * _CANDLE_PERIOD_SECONDS
    if period_start != _live_candle["period_start"]:
        _live_candle["period_start"] = period_start
        _live_candle["high"] = price
        _live_candle["low"] = price
    else:
        _live_candle["high"] = max(_live_candle["high"], price)
        _live_candle["low"] = min(_live_candle["low"], price)


def get_current_candle_range() -> tuple:
    """
    Ritorna (high, low) della candela M5 corrente, ricostruita dai
    campionamenti di prezzo del monitor (vedi _update_live_candle).
    Usato per catturare ombre intracandle che toccano SL/TP/BE senza che
    il prezzo dell'ultimo campionamento le rilevi.
    """
    if _live_candle["high"] > 0 and _live_candle["low"] > 0:
        return _live_candle["high"], _live_candle["low"]
    return 0.0, 0.0


def is_bot_paused() -> bool:
    """True se il bot è in pausa manuale (/pausa), fino a /riattiva."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key='paused'"
            ).fetchone()
        return bool(row) and row["value"] == "1"
    except Exception:
        return False


def set_bot_paused(paused: bool) -> None:
    """Imposta/rimuove la pausa manuale, persistita nel DB (sopravvive ai riavvii)."""
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO bot_state(key, value) VALUES('paused', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if paused else "0",),
        )


def load_breaking_news_seen() -> tuple[set, bool]:
    """
    Ritorna (seen_ids, is_first_run). is_first_run=True se non è mai stato
    salvato nulla prima — usato da gold_bot.py per non spammare tutto lo
    storico dei comunicati Fed al primo avvio.
    """
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key='breaking_news_seen'"
            ).fetchone()
        if not row:
            return set(), True
        return set(json.loads(row["value"])), False
    except Exception:
        return set(), True


def save_breaking_news_seen(seen_ids: set) -> None:
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO bot_state(key, value) VALUES('breaking_news_seen', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(list(seen_ids)),),
        )


def check_order_activation(trade: dict, price: float) -> bool:
    signal = trade.get("signal")
    order_type = str(trade.get("order_type", signal)).upper()
    entry = float(trade.get("entry", 0))
    if "LIMIT" in order_type:
        return (signal == "BUY" and price <= entry) or (signal == "SELL" and price >= entry)
    if "STOP" in order_type:
        return (signal == "BUY" and price >= entry) or (signal == "SELL" and price <= entry)
    return False


def check_limit_invalidation(trade: dict, _price: float) -> bool:
    """I pending scadono per tempo, non perché inizialmente distanti dall'entry."""
    timestamp = trade.get("timestamp") or ""
    try:
        created = datetime.fromisoformat(timestamp)
        if created.tzinfo is None:
            created = TIMEZONE.localize(created)
    except ValueError:
        return True
    ttl = PENDING_TTL_MINUTES.get(trade.get("timeframe", "15min"), 90)
    age_minutes = (datetime.now(TIMEZONE) - created.astimezone(TIMEZONE)).total_seconds() / 60
    return age_minutes >= ttl


def _tf_label(timeframe: str) -> str:
    return TF_LABEL.get(timeframe, (timeframe or "").upper())


def msg_order_activated(order_type, entry, price, timeframe="") -> str:
    return (
        f"✅ *ORDINE ATTIVATO — {_tf_label(timeframe)}*\n"
        f"{order_type} @ ${entry} eseguito\n"
        f"Prezzo rilevato: *${price}*"
    )


def msg_limit_cancelled(entry, price, _distance, signal="", timeframe="") -> str:
    return (
        f"⌛ *PENDING SCADUTO — {signal} {_tf_label(timeframe)}*\n"
        f"Entry: ${entry} | Prezzo: ${price}\n"
        "Setup non più valido: ordine archiviato."
    )


def msg_be_closed(entry, signal="", timeframe="") -> str:
    return (
        f"⚖️ *BREAK EVEN RAGGIUNTO — {signal} {_tf_label(timeframe)}*\n"
        f"Il prezzo è tornato all'entry *${entry}*.\n"
        "Trade virtuale chiuso a pareggio; i TP già raggiunti restano registrati."
    )


def msg_tp1(entry, tp2, price, signal="", timeframe="") -> str:
    return (
        f"🎯 *TP1 — {signal} {_tf_label(timeframe)}*\n"
        f"Entry: ${entry} | Prezzo: *${price}*\n"
        f"Trade monitorato verso TP2 @ ${tp2}; protezione a BE attiva."
    )


def msg_tp2(entry, tp3, price, signal="", timeframe="") -> str:
    return (
        f"🎯🎯 *TP2 — {signal} {_tf_label(timeframe)}*\n"
        f"Entry: ${entry} | Prezzo: *${price}*\n"
        f"Livello registrato; monitoraggio attivo verso TP3 @ ${tp3}."
    )


def msg_tp3(entry, price, signal="", timeframe="") -> str:
    return (
        f"🏆 *TP3 — {signal} {_tf_label(timeframe)}*\n"
        f"Entry: ${entry} | Exit virtuale: *${price}* | Risultato: +3R"
    )


def msg_sl(entry, price, signal="", timeframe="", after_tp1=False) -> str:
    return (
        f"❌ *STOP LOSS — {signal} {_tf_label(timeframe)}*\n"
        f"Entry: ${entry} | SL raggiunto: *${price}* | Risultato: -1R"
    )


def _target_reached(signal: str, price: float, target: float) -> bool:
    return price >= target if signal == "BUY" else price <= target


def _stop_reached(signal: str, price: float, stop: float) -> bool:
    return price <= stop if signal == "BUY" else price >= stop


async def _send_monitor_messages(bot, chat_id: str, messages: list[tuple[str, str]],
                                 notified: dict, trade_id: str) -> None:
    """Invia ogni evento una sola volta e salva l'esito della notifica."""
    changed = False
    for event_key, message in messages:
        if notified.get(event_key):
            continue
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="Markdown",
            )
            notified[event_key] = True
            changed = True
        except Exception:
            # Non marca l'evento come notificato: se il trade resta aperto il
            # monitor riproverà al ciclo successivo.
            logger.exception("Invio alert %s fallito per %s", event_key, trade_id)
    if changed:
        update_notified_json(trade_id, notified)


def _adverse_extreme(signal: str, price: float, candle_high: float, candle_low: float) -> float:
    """
    Prezzo peggiore raggiunto in questo ciclo — quello da usare per i check
    di SL e di ritorno a BE. BUY: il minimo (Low della candela se disponibile,
    altrimenti il prezzo campionato). SELL: il massimo.
    """
    if signal == "BUY":
        return min(price, candle_low) if candle_low > 0 else price
    return max(price, candle_high) if candle_high > 0 else price


def _favorable_extreme(signal: str, price: float, candle_high: float, candle_low: float) -> float:
    """
    Prezzo migliore raggiunto in questo ciclo — quello da usare per i check
    TP1/TP2/TP3. BUY: il massimo. SELL: il minimo.
    """
    if signal == "BUY":
        return max(price, candle_high) if candle_high > 0 else price
    return min(price, candle_low) if candle_low > 0 else price


async def monitor_active_trade(bot, chat_id: str) -> None:
    trades = load_all_active_trades()
    if not trades:
        return
    price = await get_current_price_async()
    if price <= 0:
        return
    # High/Low candela M5 in formazione: cattura le ombre intracandle
    # (il prezzo tocca SL/TP/BE e torna indietro prima del prossimo ciclo).
    # Aggiornato in memoria dal prezzo appena campionato, nessuna chiamata
    # di rete aggiuntiva (vedi _update_live_candle).
    _update_live_candle(price)
    candle_high, candle_low = get_current_candle_range()
    for trade in trades:
        try:
            await _monitor_single(bot, chat_id, trade, price, candle_high, candle_low)
        except Exception:
            logger.exception("Errore monitor sul trade %s", trade.get("trade_id"))


async def _monitor_single(bot, chat_id: str, trade: dict, price: float,
                           candle_high: float = 0.0, candle_low: float = 0.0) -> None:
    fresh = get_trade_by_id(trade.get("trade_id", ""))
    if not fresh or fresh.get("status") != "OPEN":
        return
    trade = fresh
    trade_id = trade["trade_id"]
    signal = trade["signal"]
    order_type = trade["order_type"]
    entry = float(trade["entry"])
    sl = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp3 = float(trade["tp3"])
    notified = dict(trade.get("notified") or {})
    timeframe = trade.get("timeframe", "")

    # entry/sl/tp1/tp2/tp3 sono salvati in "equivalente spot" (vedi
    # open_trade in gold_bot.py: price_basis = GC=F - spot catturato
    # all'apertura). Il prezzo live e il range candela arrivano invece in
    # scala GC=F (fonte primaria del monitor) — li riportiamo alla stessa
    # scala del trade sottraendo lo stesso basis, altrimenti si confronta
    # un prezzo futures con livelli spot (il bug scoperto in produzione:
    # il monitor perdeva TP1/SL/BE quando i due prezzi divergevano).
    basis = float(trade.get("price_basis") or 0)
    if basis:
        price = price - basis
        if candle_high > 0:
            candle_high = candle_high - basis
        if candle_low > 0:
            candle_low = candle_low - basis

    # Estremi del ciclo corrente (candela M5 in formazione): usati per
    # rilevare ombre che toccano un livello e rientrano prima del prossimo
    # campionamento. adverse_price → check SL/BE (peggiore); favorable_price
    # → check TP1/2/3 (migliore). Prima si usava un solo "effective_price"
    # calcolato solo per l'SL: le ombre che toccavano BE o TP passavano
    # inosservate.
    adverse_price = _adverse_extreme(signal, price, candle_high, candle_low)
    favorable_price = _favorable_extreme(signal, price, candle_high, candle_low)

    if is_pending_order(order_type) and not trade.get("entry_filled"):
        if check_limit_invalidation(trade, price):
            if close_trade(trade_id, "CANCELLED", price, "Pending scaduto"):
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg_limit_cancelled(entry, price, 0, signal, timeframe),
                    parse_mode="Markdown",
                )
            return
        if check_order_activation(trade, price):
            activate_trade(trade_id)
            await bot.send_message(
                chat_id=chat_id,
                text=msg_order_activated(order_type, entry, price, timeframe),
                parse_mode="Markdown",
            )
        return

    if not trade.get("activated"):
        activate_trade(trade_id)

    # Prima gestisce le uscite avverse usando lo stato già consolidato dal
    # ciclo precedente. Dopo TP1 lo stop virtuale è a entry: un ritorno
    # all'entry è BE, mai SL.
    tp1_hit = bool(trade.get("tp1_hit"))
    be_armed = bool(trade.get("be_armed"))
    if tp1_hit and be_armed and _stop_reached(signal, adverse_price, entry):
        mark_be_hit(trade_id)
        closed = close_trade(
            trade_id,
            "WIN_BE",
            entry,
            "Chiuso automaticamente a break even dal monitor virtuale",
        )
        if closed:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg_be_closed(entry, signal, timeframe),
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Invio alert BE fallito per %s", trade_id)
            asyncio.create_task(_post_trade_analysis(bot, chat_id))
        return

    if _stop_reached(signal, adverse_price, sl):
        # Chiudi PRIMA nel DB, poi notifica.
        # Se la notifica fallisce il trade è già CLOSED → non viene riprocessato.
        # Se close_trade fallisce (es. già chiuso) non mandiamo notifiche false.
        closed = close_trade(
            trade_id,
            "LOSS",
            sl,
            "Stop loss raggiunto dal monitor virtuale",
        )
        if closed:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg_sl(entry, sl, signal, timeframe),
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Invio alert SL fallito per %s", trade_id)
            asyncio.create_task(_post_trade_analysis(bot, chat_id))
        return

    # Registra tutti i target attraversati nello stesso campionamento. Non usa
    # ``elif``: un salto diretto a TP3 deve salvare e notificare TP1, TP2 e TP3.
    messages: list[tuple[str, str]] = []
    if _target_reached(signal, favorable_price, tp1):
        if not trade.get("tp1_hit"):
            mark_tp1_hit(trade_id)
            arm_break_even(trade_id)
        messages.append(
            ("tp1", msg_tp1(entry, tp2, favorable_price, signal, timeframe))
        )

    if _target_reached(signal, favorable_price, tp2):
        if not trade.get("tp1_hit"):
            mark_tp1_hit(trade_id)
            arm_break_even(trade_id)
        if not trade.get("tp2_hit"):
            mark_tp2_hit(trade_id)
        messages.append(
            ("tp2", msg_tp2(entry, tp3, favorable_price, signal, timeframe))
        )

    tp3_reached = _target_reached(signal, favorable_price, tp3)
    if tp3_reached:
        if not trade.get("tp1_hit"):
            mark_tp1_hit(trade_id)
            arm_break_even(trade_id)
        if not trade.get("tp2_hit"):
            mark_tp2_hit(trade_id)
        if not trade.get("tp3_hit"):
            mark_tp3_hit(trade_id)
        # NON aggiungere a messages — TP3 viene notificato dopo close_trade

    # Manda TP1/TP2 (non chiudono il trade)
    await _send_monitor_messages(bot, chat_id, messages, notified, trade_id)

    # TP3 chiude il trade — chiudi DB prima, notifica dopo (stesso pattern SL/BE)
    if tp3_reached:
        closed = close_trade(
            trade_id,
            "WIN_TP3",
            tp3,
            "TP3 raggiunto dal monitor virtuale",
        )
        if closed:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg_tp3(entry, tp3, signal, timeframe),
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Invio alert TP3 fallito per %s", trade_id)
            asyncio.create_task(_post_trade_analysis(bot, chat_id))


async def _post_trade_analysis(bot, chat_id: str) -> None:
    await asyncio.sleep(1)
    try:
        from self_learning import analyze_last_trade, format_learning_report, optimize_strategy_weights

        analysis = await asyncio.to_thread(analyze_last_trade)
        await bot.send_message(
            chat_id=chat_id,
            text=("🤖 *ANALISI POST-TRADE*\n" + analysis)[:4000],
            parse_mode="Markdown",
        )
        with _connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND result!='CANCELLED'"
            ).fetchone()[0]
        if total >= 20 and total % 10 == 0:
            result = await asyncio.to_thread(optimize_strategy_weights)
            if result.get("status") == "optimized":
                await bot.send_message(
                    chat_id=chat_id,
                    text=format_learning_report(result)[:4000],
                    parse_mode="Markdown",
                )
    except Exception:
        logger.exception("Analisi post-trade fallita")


# Compatibilità con chiamate di versioni precedenti.
def save_active_trade(data: dict) -> str:
    return open_trade(data)


def clear_active_trade() -> None:
    logger.warning("clear_active_trade è disabilitata: specificare sempre trade_id")
