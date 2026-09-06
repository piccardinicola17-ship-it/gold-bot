"""Risk management centralizzato per GoldMind.

FIX rispetto alla versione precedente:
  - SESSION COOLDOWN: dopo 3 SL la sessione si ferma per COOLDOWN_HOURS ore
    (default 5h), poi riprende automaticamente con risk ridotto al 50%
  - _stop_session() salva l'ora del blocco in session_stopped_at
  - check_can_trade() verifica se il cooldown è scaduto e riattiva la sessione
  - _resume_session() riattiva e azzera i contatori loss consecutive
  - Aggiunto comando /riprendisessione per sblocco manuale immediato
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz

from trade_manager import DB_PATH, _connect

logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

MAX_TRADES_PER_DAY     = 999
MAX_TRADES_PER_SESSION = 999
MAX_CONSECUTIVE_LOSS   = 3
RISK_PCT_PER_TRADE     = 1.0
MAX_RISK_PCT           = 1.0
MAX_TOTAL_RESERVED_RISK_PCT = 6.0
MAX_SAME_DIRECTION     = 3  # max 3 trade nella stessa direzione (era 2 — bloccava con M5+M15+H4)
MIN_RR                 = 2.0
NEWS_BUFFER_MINUTES    = 30

# Timeframe con soglia di probabilita' minima piu' alta (65% invece del
# default 55%) - unica fonte di verita' per questo valore (FIX 2026-09-05:
# due copie separate della stessa soglia rischiavano di disallinearsi tra
# agent_orchestrator.py e questo modulo).
# 1min/5min/15min: nessun edge dimostrato a probabilita' basse (vedi
# agent_orchestrator._BLOCKED_REGIMES_BY_TF per il contesto completo).
# 4h aggiunto il 2026-09-06: il PF migliora passando da 55 a 65 in ENTRAMBE
# le meta' cronologiche indipendenti del campione 2021-2026 (0.96->1.09 e
# 1.08->1.22) - validato con uno split train/test dedicato, non solo
# osservato sull'aggregato. 1day mostrava lo stesso miglioramento
# nell'aggregato (PF 1.10->1.22) ma NON regge in entrambe le meta':
# peggiora (1.02->0.94) nel periodo 2003-2015, migliora solo nel piu'
# recente - scartato, resta al default (era guidato da un solo
# sottoperiodo, non un edge reale).
MIN_PROB_HIGH_THRESHOLD_TFS = ("1min", "5min", "15min", "4h")
MIN_PROB_HIGH_THRESHOLD     = 65


def min_prob_for_timeframe(timeframe: str | None) -> int:
    """Soglia minima di probabilita' per aprire un trade sul timeframe dato."""
    tf = timeframe.lower() if timeframe else ""
    if tf in MIN_PROB_HIGH_THRESHOLD_TFS:
        return MIN_PROB_HIGH_THRESHOLD
    import os as _os
    return int(_os.environ.get("MIN_PROB", "55"))


DD_REDUCE_AT_R  = 3.0
DD_PAUSE_AT_R   = 10.0   # soglia alzata — non blocca con DD storico da paper trading
MIN_KELLY_TRADES = 30
KELLY_FRACTION   = 0.25

XAUUSD_OZ_PER_LOT = 100.0
MIN_LOT   = 0.01
MAX_LOT   = 10.0
LOT_STEP  = 0.01

# Ore di cooldown dopo 3 SL consecutivi prima della ripresa automatica
COOLDOWN_HOURS = 5
# Risk ridotto dopo il cooldown (50% del normale)
COOLDOWN_RISK_MULTIPLIER = 0.5


def init_db() -> None:
    from trade_manager import init_db as trade_init_db
    # session_stopped_at è già definita in trade_manager._ensure_session_columns
    # e nel CREATE TABLE sessions — nessuna migration manuale necessaria
    trade_init_db()


def get_current_session() -> str:
    hour = datetime.now(TIMEZONE).hour
    if hour < 9:
        return "ASIA"
    elif hour < 15:
        return "LONDRA"
    else:
        return "NY"


def get_trades_this_session() -> int:
    session = get_current_session()
    now = datetime.now(TIMEZONE)
    if session == "ASIA":
        start_h = 0
    elif session == "LONDRA":
        start_h = 9
    else:
        start_h = 15
    start = now.replace(hour=start_h, minute=0, second=0, microsecond=0)
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) as n FROM trades "
                "WHERE timestamp >= ? AND status != 'CANCELLED'",
                (start.isoformat(),)
            ).fetchone()
        return int(rows["n"]) if rows else 0
    except Exception:
        return 0


def _today() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d")


def _get_or_create_session() -> dict:
    today = _today()
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (today,))
        row = conn.execute("SELECT * FROM sessions WHERE date=?", (today,)).fetchone()
    return dict(row) if row else {}


def get_session_stats() -> dict:
    return _get_or_create_session()


def get_all_closed_trades() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM trades
            WHERE status='CLOSED' AND result IS NOT NULL AND result!='CANCELLED'
            ORDER BY COALESCE(closed_at,timestamp), id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_open_trades(*, activated_only: bool = False) -> list[dict]:
    query = "SELECT * FROM trades WHERE status='OPEN'"
    if activated_only:
        query += " AND activated=1"
    query += " ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_consecutive_losses() -> int:
    """Conta le loss consecutive della sessione odierna."""
    today = _today()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT result FROM trades
            WHERE status='CLOSED'
              AND result IN ('LOSS','WIN_TP1','WIN_TP2','WIN_TP3')
              AND closed_at LIKE ?
            ORDER BY COALESCE(closed_at,timestamp) DESC, id DESC
            LIMIT 20
            """,
            (f"{today}%",),
        ).fetchall()
    count = 0
    for row in rows:
        if row["result"] == "LOSS":
            count += 1
        else:
            break
    return count


def _cooldown_expired(session: dict) -> bool:
    """
    Ritorna True se il cooldown di COOLDOWN_HOURS ore è scaduto
    rispetto all'ora in cui la sessione è stata fermata.
    """
    stopped_at_str = session.get("session_stopped_at") or ""
    if not stopped_at_str:
        # Colonna non ancora popolata (sessione bloccata con versione vecchia)
        # → considera il blocco avvenuto a mezzanotte di oggi (comportamento safe)
        today_midnight = datetime.now(TIMEZONE).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        stopped_at = today_midnight
    else:
        try:
            stopped_at = datetime.fromisoformat(stopped_at_str)
            if stopped_at.tzinfo is None:
                stopped_at = TIMEZONE.localize(stopped_at)
        except (ValueError, TypeError):
            return False

    now = datetime.now(TIMEZONE)
    elapsed_hours = (now - stopped_at.astimezone(TIMEZONE)).total_seconds() / 3600
    return elapsed_hours >= COOLDOWN_HOURS


def _resume_session() -> None:
    """
    Riattiva la sessione dopo il cooldown:
    - session_stopped → 0
    - session_stopped_at → NULL
    - consecutive_losses → 0 (il cooldown azzera il contatore)
    """
    today = _today()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET session_stopped=0,
                session_stopped_at=NULL,
                consecutive_losses=0
            WHERE date=?
            """,
            (today,),
        )
    logger.info(
        "Sessione ripresa automaticamente dopo %dh di cooldown", COOLDOWN_HOURS
    )


def resume_session_manual() -> str:
    """
    Sblocco manuale immediato via /riprendisessione.
    Ritorna un messaggio di conferma.
    """
    today = _today()
    session = _get_or_create_session()
    if not session.get("session_stopped"):
        return "✅ La sessione è già attiva, nessuna azione necessaria."
    _resume_session()
    risk = round(calculate_kelly_size() * COOLDOWN_RISK_MULTIPLIER, 2)
    return (
        f"✅ *Sessione riattivata manualmente*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Risk prossimo trade: *{risk}%* (ridotto al 50%)\n"
        f"I contatori loss consecutive sono stati azzerati.\n"
        f"_Il risk tornerà al 100% dopo un trade vincente._"
    )


def calculate_lot_size(
    account_balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
) -> dict:
    risk_amount = max(0.0, account_balance) * max(0.0, risk_pct) / 100
    sl_distance = abs(entry - sl)
    if account_balance <= 0 or risk_pct <= 0 or sl_distance <= 0:
        return {
            "tradable": False,
            "lot_size": 0.0,
            "risk_amount": round(risk_amount, 2),
            "actual_risk_amount": 0.0,
            "actual_risk_pct": 0.0,
            "sl_distance_usd": round(sl_distance, 4),
            "error": "Capitale, rischio o distanza SL non validi.",
        }
    value_per_lot = sl_distance * XAUUSD_OZ_PER_LOT
    raw_lot = risk_amount / value_per_lot
    stepped_lot = int(raw_lot / LOT_STEP) * LOT_STEP
    if stepped_lot < MIN_LOT:
        min_risk = MIN_LOT * value_per_lot
        return {
            "tradable": False,
            "lot_size": 0.0,
            "risk_amount": round(risk_amount, 2),
            "actual_risk_amount": 0.0,
            "actual_risk_pct": 0.0,
            "minimum_lot_risk": round(min_risk, 2),
            "sl_distance_usd": round(sl_distance, 4),
            "error": (
                f"Il lotto minimo {MIN_LOT} rischierebbe ${min_risk:.2f}, "
                f"oltre il budget ${risk_amount:.2f}."
            ),
        }
    lot_size = min(round(stepped_lot, 2), MAX_LOT)
    actual_risk = lot_size * value_per_lot
    actual_risk_pct = actual_risk / account_balance * 100
    if actual_risk_pct > risk_pct + 0.01:
        return {
            "tradable": False,
            "lot_size": 0.0,
            "risk_amount": round(risk_amount, 2),
            "actual_risk_amount": round(actual_risk, 2),
            "actual_risk_pct": round(actual_risk_pct, 3),
            "sl_distance_usd": round(sl_distance, 4),
            "error": "Arrotondamento lotto oltre il rischio massimo.",
        }
    return {
        "tradable": True,
        "lot_size": lot_size,
        "risk_amount": round(risk_amount, 2),
        "actual_risk_amount": round(actual_risk, 2),
        "actual_risk_pct": round(actual_risk_pct, 3),
        "sl_distance_usd": round(sl_distance, 4),
        "value_per_lot": round(value_per_lot, 2),
    }


def calculate_kelly_size(base_risk_pct: float = RISK_PCT_PER_TRADE) -> float:
    trades = [
        t for t in get_all_closed_trades()
        if t.get("result") in ("WIN_TP1", "WIN_TP2", "WIN_TP3", "LOSS")
    ]
    if len(trades) < MIN_KELLY_TRADES:
        return min(base_risk_pct, MAX_RISK_PCT)
    wins   = [t for t in trades if str(t["result"]).startswith("WIN")]
    losses = [t for t in trades if t["result"] == "LOSS"]
    if not wins or not losses:
        return 0.25
    avg_win  = sum(max(0.0, float(t.get("pnl_r") or 0)) for t in wins)  / len(wins)
    avg_loss = sum(abs(float(t.get("pnl_r") or -1)) for t in losses) / len(losses)
    if avg_win <= 0 or avg_loss <= 0:
        return 0.25
    p = len(wins) / len(trades)
    q = 1 - p
    payoff = avg_win / avg_loss
    kelly_full = (p * payoff - q) / payoff
    if kelly_full <= 0:
        return 0.25
    scaled = base_risk_pct * min(1.0, (kelly_full * KELLY_FRACTION) / 0.10)
    return round(max(0.25, min(scaled, MAX_RISK_PCT)), 2)


def get_equity_stats() -> dict:
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for trade in get_all_closed_trades():
        equity += float(trade.get("pnl_r") or 0)
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)
    current_dd = max(0.0, peak - equity)
    return {
        "equity_r":    round(equity, 2),
        "peak_r":      round(peak, 2),
        "current_dd_r":round(current_dd, 2),
        "max_dd_r":    round(max_dd, 2),
    }


def get_current_drawdown() -> float:
    return float(get_equity_stats()["current_dd_r"])


def get_drawdown_multiplier() -> float:
    """
    Riduce il risk in base al DD ma NON lo azzera mai.
    Il blocco avviene SOLO per 3 SL consecutivi, non per DD.
    """
    drawdown = get_current_drawdown()
    if drawdown >= DD_PAUSE_AT_R:
        return 0.25   # risk minimo, mai zero
    if drawdown >= DD_REDUCE_AT_R:
        return 0.5
    return 1.0


def get_reserved_risk_pct() -> float:
    return round(sum(float(t.get("risk_pct") or 0) for t in get_open_trades()), 2)


@dataclass
class RiskCheckResult:
    allowed: bool
    reason:  str
    risk_pct: float = 0.0
    size_multiplier: float = 1.0


def check_can_trade(
    prob: int = 0,
    rr: float = 0.0,
    is_weekend: bool = False,
    near_news: bool = False,
    signal: str = "",
    news_error: bool = False,
    timeframe: str = "",
) -> RiskCheckResult:
    session = _get_or_create_session()

    if is_weekend:
        return RiskCheckResult(False, "Weekend — trading disabilitato")
    if news_error:
        return RiskCheckResult(False, "Calendario non disponibile — blocco prudenziale")

    # ── COOLDOWN AUTOMATICO ───────────────────────────────────────────────────
    # Se la sessione è fermata, controlla se il cooldown è scaduto.
    # Se sì: riattiva automaticamente con risk ridotto.
    # Se no: blocca e comunica quante ore mancano.
    post_cooldown = False
    if session.get("session_stopped"):
        if _cooldown_expired(session):
            _resume_session()
            session = _get_or_create_session()   # rilegge dopo il reset
            post_cooldown = True                 # useremo risk ridotto
            logger.info("Cooldown scaduto — sessione riattivata automaticamente")
        else:
            # Calcola ore rimanenti per il messaggio
            stopped_at_str = session.get("session_stopped_at") or ""
            try:
                stopped_at = datetime.fromisoformat(stopped_at_str)
                if stopped_at.tzinfo is None:
                    stopped_at = TIMEZONE.localize(stopped_at)
                elapsed = (datetime.now(TIMEZONE) - stopped_at.astimezone(TIMEZONE)).total_seconds() / 3600
                remaining = max(0.0, COOLDOWN_HOURS - elapsed)
                reason = (
                    f"Sessione in cooldown — ripresa automatica tra "
                    f"{remaining:.1f}h (alle "
                    f"{(stopped_at.astimezone(TIMEZONE) + timedelta(hours=COOLDOWN_HOURS)).strftime('%H:%M')} IT)"
                )
            except Exception:
                reason = f"Sessione in cooldown — ripresa automatica dopo {COOLDOWN_HOURS}h"
            return RiskCheckResult(False, reason)
    # ─────────────────────────────────────────────────────────────────────────

    # Controlla 3 SL consecutivi DOPO il cooldown (i contatori sono stati azzerati)
    consecutive = get_consecutive_losses()
    if consecutive >= MAX_CONSECUTIVE_LOSS:
        _stop_session(f"{consecutive} perdite consecutive")
        return RiskCheckResult(
            False,
            f"{consecutive} SL consecutivi — cooldown {COOLDOWN_HOURS}h attivato"
        )

    if near_news:
        return RiskCheckResult(False, f"Evento macro entro ±{NEWS_BUFFER_MINUTES} minuti")
    # Soglia differenziata per timeframe - vedi min_prob_for_timeframe()
    # sopra (unica fonte di verita', condivisa con agent_orchestrator.py).
    _min_prob = min_prob_for_timeframe(timeframe)
    if prob < _min_prob:
        return RiskCheckResult(False, f"Confidence {prob}% sotto soglia {_min_prob}% [{timeframe or 'tutti i TF'}]")
    if rr < MIN_RR:
        return RiskCheckResult(False, f"R:R {rr:.2f} sotto {MIN_RR:.2f}")
    if signal not in ("BUY", "SELL"):
        return RiskCheckResult(False, "Direzione del segnale non valida")

    open_and_pending = get_open_trades()
    same_direction = sum(1 for t in open_and_pending if t.get("signal") == signal)
    if same_direction >= MAX_SAME_DIRECTION:
        return RiskCheckResult(
            False,
            f"Esposizione {signal} massima: {same_direction} trade/pending già riservati",
        )

    dd_multiplier = get_drawdown_multiplier()
    # dd_multiplier non è mai 0 — il DD riduce il risk ma NON blocca la sessione
    # Solo 3 SL consecutivi attivano il cooldown

    base_risk = calculate_kelly_size()

    # Dopo il cooldown: risk dimezzato per rientrare gradualmente
    if post_cooldown:
        base_risk = round(base_risk * COOLDOWN_RISK_MULTIPLIER, 2)

    risk_pct = round(min(base_risk * dd_multiplier, MAX_RISK_PCT), 2)
    reserved = get_reserved_risk_pct()
    if reserved + risk_pct > MAX_TOTAL_RESERVED_RISK_PCT + 1e-9:
        return RiskCheckResult(
            False,
            (
                f"Rischio globale: {reserved:.2f}% già riservato + {risk_pct:.2f}% "
                f"> {MAX_TOTAL_RESERVED_RISK_PCT:.2f}%"
            ),
        )

    equity = get_equity_stats()
    cooldown_note = " [risk ridotto post-cooldown]" if post_cooldown else ""
    return RiskCheckResult(
        True,
        (
            f"Consentito{cooldown_note} | oggi {session.get('trades_count',0)}/{MAX_TRADES_PER_DAY} | "
            f"risk {risk_pct:.2f}% | riservato {reserved:.2f}% | "
            f"DD {equity['current_dd_r']:.1f}R"
        ),
        risk_pct=risk_pct,
        size_multiplier=dd_multiplier,
    )


def _stop_session(reason: str) -> None:
    """Ferma la sessione e salva l'ora esatta del blocco per il cooldown."""
    today = _today()
    now   = datetime.now(TIMEZONE).isoformat()
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(date) VALUES (?)", (today,))
        # session_stopped_at potrebbe non esistere su DB vecchi — usa try/except
        try:
            conn.execute(
                "UPDATE sessions SET session_stopped=1, session_stopped_at=? WHERE date=?",
                (now, today),
            )
        except Exception:
            conn.execute(
                "UPDATE sessions SET session_stopped=1 WHERE date=?",
                (today,),
            )
    logger.warning("Sessione fermata: %s (cooldown %dh)", reason, COOLDOWN_HOURS)


def is_weekend_now() -> bool:
    return datetime.now(TIMEZONE).weekday() >= 5


def is_near_news(events: list, news_error: bool = False) -> bool:
    if news_error:
        return True
    now = datetime.now(TIMEZONE)
    for event in events:
        if str(event.get("impact", "HIGH")).upper() not in ("HIGH", "MEDIUM"):
            continue
        try:
            if event.get("datetime"):
                event_dt = datetime.fromisoformat(event["datetime"])
                if event_dt.tzinfo is None:
                    event_dt = TIMEZONE.localize(event_dt)
            else:
                date_text = event.get("date") or now.strftime("%Y-%m-%d")
                event_dt  = TIMEZONE.localize(
                    datetime.strptime(
                        f"{date_text} {event.get('time','')}",
                        "%Y-%m-%d %H:%M",
                    )
                )
            delta = abs((event_dt.astimezone(TIMEZONE) - now).total_seconds()) / 60
            if delta <= NEWS_BUFFER_MINUTES:
                return True
        except (ValueError, TypeError):
            logger.warning("Evento calendario non interpretabile: %s", event)
            return True
    return False


def register_trade_open(*_args, **_kwargs) -> None:
    logger.warning("register_trade_open ignorata: lifecycle centralizzato")


def register_trade_close(*_args, **_kwargs) -> None:
    logger.warning("register_trade_close ignorata: lifecycle centralizzato")


def format_risk_report() -> str:
    session      = get_session_stats()
    equity       = get_equity_stats()
    active       = get_open_trades(activated_only=True)
    pending      = [t for t in get_open_trades() if not t.get("activated")]
    risk         = calculate_kelly_size() * get_drawdown_multiplier()
    curr_session = get_current_session()
    trades_sess  = get_trades_this_session()

    stopped      = bool(session.get("session_stopped"))
    cooldown_txt = ""
    if stopped:
        stopped_at_str = session.get("session_stopped_at") or ""
        try:
            stopped_at = datetime.fromisoformat(stopped_at_str)
            if stopped_at.tzinfo is None:
                stopped_at = TIMEZONE.localize(stopped_at)
            elapsed    = (datetime.now(TIMEZONE) - stopped_at.astimezone(TIMEZONE)).total_seconds() / 3600
            remaining  = max(0.0, COOLDOWN_HOURS - elapsed)
            resume_at  = (stopped_at.astimezone(TIMEZONE) + timedelta(hours=COOLDOWN_HOURS)).strftime("%H:%M")
            cooldown_txt = f"\n⏱ Ripresa automatica tra *{remaining:.1f}h* (alle {resume_at} IT)\n_Usa /riprendisessione per sblocco immediato_"
        except Exception:
            cooldown_txt = f"\n⏱ Ripresa automatica dopo {COOLDOWN_HOURS}h dal blocco"

    return (
        "⚖️ *RISK MANAGEMENT — XAU/USD*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Stato: {'🛑 COOLDOWN' if stopped else '✅ ATTIVA'}\n"
        f"{cooldown_txt}"
        f"Trade oggi: *{session.get('trades_count', 0)}*\n"
        f"Sessione corrente: *{curr_session}* ({trades_sess} trade)\n"
        f"Loss consecutive: *{get_consecutive_losses()}/{MAX_CONSECUTIVE_LOSS}*\n"
        f"Posizioni: *{len(active)}* | Pending: *{len(pending)}*\n"
        f"Rischio riservato: *{get_reserved_risk_pct():.2f}%*\n"
        f"Rischio prossimo trade: *{risk:.2f}%* (cap {MAX_RISK_PCT:.2f}%)\n"
        f"Equity: *{equity['equity_r']:+.1f}R*\n"
        f"DD corrente: *{equity['current_dd_r']:.1f}R* | massimo: *{equity['max_dd_r']:.1f}R*"
    )
