"""
gold_bot.py — GoldMind v2
============================
Fix rispetto alla versione precedente:
  - DB UNICO: usa trade_manager.open_trade() / close_trade()
  - Autenticazione Telegram: ogni comando verifica is_authorized()
  - has_open_trade_on_timeframe() controlla TUTTI i trade attivi (non solo l'ultimo)
  - /chiudi usa trade_id esplicito, non clear_active_trade()
  - contatori sessione gestiti atomicamente dal lifecycle dei trade
  - _async_posttrade_and_learn legge da goldbot.db unico
  - Notizie fail-safe: errore calendario = news_error=True → blocco
  - Ora legale NY: usa pytz per conversione dinamica
  - cmd_stats / cmd_report leggono dal DB unico
  - Monitor chiamato da _fast_monitor solo se ci sono trade attivi
"""

import logging
import asyncio
import os
import sqlite3
import statistics
from datetime import datetime, time as dt_time
import pytz
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from analyzer import get_news_sentiment, get_extended_news, seconds_since_last_data_success, SNIPER_CONFIGS
from agent_orchestrator import run_pipeline, format_pipeline_report
from news_analyst import format_news_message, analyze_macro_event, analyze_combined_macro_event, get_macro_briefing, analyze_breaking_news, get_bias_briefing, _escape_md
from market_structure import get_market_structure_snapshot
# ORB rimosso — gestito manualmente dall'utente
from self_learning import analyze_last_trade, weekly_review, optimize_strategy_weights, format_learning_report
from risk_manager import format_risk_report, calculate_lot_size, resume_session_manual, min_prob_for_timeframe
from trade_manager import (
    init_db,
    open_trade, close_trade,
    load_all_active_trades, load_active_trade,
    get_open_trade_by_timeframe, has_open_trade_on_timeframe,
    recently_cancelled_on_timeframe,
    monitor_active_trade,
    get_current_price, get_current_price_async,
    is_authorized, build_setup_key, was_setup_seen, DuplicateSetupError,
    is_decisive_win, calculate_trade_pips,
    get_recent_decisions as load_recent_decisions,
    _fmt,
    is_bot_paused, set_bot_paused,
    load_macro_alert_state, save_macro_alert_state,
    save_macro_event_pre, save_macro_event_post,
    DB_PATH as CORE_DB_PATH, BOT_DIR as CORE_BOT_DIR,
)
from ai_assistant import ask_ai
# backtest importato lazy dentro cmd_backtest (evita crash all'avvio)

# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
BOT_DIR        = str(CORE_BOT_DIR)
DB_PATH        = CORE_DB_PATH
os.environ.setdefault("DB_PATH", DB_PATH)

NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
TIMEZONE       = pytz.timezone("Europe/Rome")
MIN_PROB       = 55
# Sotto questa soglia una variazione di prezzo (bias pre-evento vs reale, o
# previsione statistica vs trade aperto) è rumore, non un movimento vero —
# unica fonte di verità (2026-09-09: prima due copie locali identiche,
# stesso pattern di bug già visto altrove in questo file).
CONFIRM_THRESHOLD_USD = 2.0

# Finestra di silenzio notturno (2026-09-17, richiesta esplicita
# dell'utente dopo un alert macro su un GDP neozelandese delle 00:45:
# "questi eventi a mezzanotte non me ne frega nulla") — sopprime SOLO i
# push Telegram puramente informativi degli alert macro/breaking-news
# (ALERT MACRO, POST-EVENTO, previsione statistica, digest notizie) tra
# le 23:00 e le 7:30. Non tocca: il tracciamento sottostante (bias,
# stato per il confronto successivo, storico dashboard — continuano a
# scrivere silenziosamente, così l'evento resta comunque visibile in
# dashboard al mattino), le chiusure protettive pre-evento (un'azione
# reale sul trade, non rumore: l'utente deve saperlo subito) e la
# generazione dei segnali di trading vera e propria (auto_check_all_
# timeframes), che gira indipendentemente 24/5.
QUIET_HOURS_START = dt_time(23, 0)
QUIET_HOURS_END   = dt_time(7, 30)


def _in_quiet_hours(now: datetime) -> bool:
    t = now.time()
    return t >= QUIET_HOURS_START or t <= QUIET_HOURS_END
# ─────────────────────────────────────────────


def _confirm_bias(bias: str, change: float) -> tuple:
    """Esito CONFERMATO/NON CONFERMATO/NEUTRO + segno per un confronto
    bias-previsto vs prezzo-reale — stessa logica ripetuta 3 volte identica
    (post-breaking-news, bias evento, bias post-evento) prima di questa
    funzione, con un solo dettaglio a fare la differenza: il ramo NEUTRO
    esisteva solo per i due macro-alert (dove l'AI può davvero rispondere
    NEUTRO) e non per il post-breaking-news, dove non serve perché quella
    chiamata scarta già a monte le risposte non direzionali (vedi
    check_breaking_news_job) — qui resta comunque gestito per sicurezza,
    così le 3 chiamate condividono la stessa unica implementazione.

    Terzo valore (status): stessa classificazione ma come codice semplice
    (CONFERMATO/NON_CONFERMATO/NEUTRO/NON_SIGNIFICATIVO), aggiunto il
    2026-09-16 per popolare il tracciamento macro_event_outcomes in
    dashboard senza dover ri-derivare lo stato dal testo con emoji/markdown
    già pensato per Telegram — stessa soglia, nessuna logica duplicata."""
    if abs(change) < CONFIRM_THRESHOLD_USD:
        esito = "➖ *Movimento non significativo* — prezzo praticamente invariato"
        status = "NON_SIGNIFICATIVO"
    elif bias == "BUY":
        confirmed = change > 0
        esito = "✅ *CONFERMATO*" if confirmed else "❌ *NON CONFERMATO* — mosso al contrario"
        status = "CONFERMATO" if confirmed else "NON_CONFERMATO"
    elif bias == "SELL":
        confirmed = change < 0
        esito = "✅ *CONFERMATO*" if confirmed else "❌ *NON CONFERMATO* — mosso al contrario"
        status = "CONFERMATO" if confirmed else "NON_CONFERMATO"
    else:
        esito = "➖ Bias pre-evento era NEUTRO — nessuna previsione da verificare"
        status = "NEUTRO"
    segno = "+" if change >= 0 else ""
    return esito, segno, status


# Finestra entro cui un evento macro precedente (stessa valuta) viene
# considerato "correlato" — bug reale trovato il 2026-09-16: la FOMC Press
# Conference delle 20:30 ha ricevuto bias NEUTRO con motivazione "in attesa
# della decisione della Fed", nonostante Federal Funds Rate/FOMC Statement
# della STESSA riunione fossero già usciti 30 minuti prima con bias SELL
# confermato — ogni gruppo (raggruppato per data+ora+valuta) veniva
# analizzato in totale isolamento, senza memoria di eventi appena successi.
# 90 minuti copre comodamente lo schema FOMC (Statement/Rate/Projections
# alle 14:00 ET, Press Conference 30 min dopo) e analoghi per BCE/BOJ,
# senza risalire a eventi macro del tutto slegati di prima mattina.
_RELATED_MACRO_WINDOW_MIN = 90


def _find_related_macro_context(currency: str, ev_dt: datetime, exclude_group_key: str) -> str:
    """Cerca l'evento macro più recente della STESSA valuta, uscito nelle
    ultime _RELATED_MACRO_WINDOW_MIN minuti (mai nel futuro rispetto a
    ev_dt), tra quelli già processati in _pre_event_bias. Ritorna una riga
    di contesto pronta per il prompt LLM, o stringa vuota se non trova
    nulla — vedi analyze_macro_event/analyze_combined_macro_event in
    news_analyst.py per come viene usata."""
    best_info = None
    best_dt = None
    for gk, info in _pre_event_bias.items():
        if gk == exclude_group_key:
            continue
        if info.get("currency") != currency:
            continue
        other_dt_str = info.get("event_dt")
        if not other_dt_str:
            continue
        try:
            other_dt = datetime.fromisoformat(other_dt_str)
        except (ValueError, TypeError):
            continue
        if other_dt >= ev_dt:
            continue  # solo eventi già usciti PRIMA di questo, mai dopo
        delta_min = (ev_dt - other_dt).total_seconds() / 60
        if delta_min > _RELATED_MACRO_WINDOW_MIN:
            continue
        if best_dt is None or other_dt > best_dt:
            best_info, best_dt = info, other_dt

    if best_info is None:
        return ""

    minutes_ago = round((ev_dt - best_dt).total_seconds() / 60)
    change = None
    if best_info.get("price_immediate") is not None:
        change = best_info["price_immediate"] - best_info["price"]
    change_txt = f", prezzo oro {change:+.2f}$" if change is not None else ""
    title = best_info.get("title", "un evento correlato")
    bias = best_info.get("bias", "NEUTRO")
    return (
        f"Contesto correlato: {minutes_ago} minuti fa è già uscito \"{title}\" "
        f"per la stessa valuta (bias {bias}{change_txt}) — non è più un esito incerto, "
        f"è la STESSA riunione/decisione vista da un angolo diverso."
    )


def _live_min_prob_for_tf(interval: str) -> int:
    """Stessa soglia usata dalla pipeline live — delega a
    risk_manager.min_prob_for_timeframe() (unica fonte di verità, vedi
    MIN_PROB_HIGH_THRESHOLD_TFS) invece di una copia propria.
    FIX 2026-09-05: /backtest passava sempre MIN_PROB=55 fisso a
    run_backtest per qualunque timeframe, mentre il testo mostrato a
    schermo dichiarava soglie differenziate — il backtest valutava quindi
    una popolazione di setup più ampia e di qualità inferiore di quella
    che il bot live prende davvero. FIX 2026-09-06: questa funzione aveva
    una copia propria della soglia (solo M1/M5/M15 a 65%) rimasta
    indietro quando H4 è stato aggiunto in risk_manager - lo stesso
    pattern di bug (due copie che divergono) risolto altrove stanotte."""
    return min_prob_for_timeframe(interval)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# M5 escluso dalla generazione automatica di segnali: backtest su 5 anni
# reali (Dukascopy) mostra PF 0.80 — nessun edge reale trovato finora
# nonostante piu' tentativi (vedi memoria progetto
# project-m5-unprofitable-open-problem), il bot perderebbe soldi in media.
# Resta disponibile a mano con /signal (con un avviso), per chi vuole
# comunque testarlo.
#
# M15 RIATTIVATO il 2026-09-07 (era escluso dal 1 settembre 2026 per lo
# stesso motivo): un filtro di allineamento col trend 4h (vedi
# _HTF_ALIGNMENT_REQUIRED_TF in agent_orchestrator.py) porta il PF sopra
# il pareggio su ENTRAMBE le meta' cronologiche indipendenti testate
# (0.81->1.05 e 0.96->1.07) - primo fix validato con lo stesso rigore
# degli altri timeframe.
#
# H1 escluso dall'aggregato normale il 2026-09-09: backtest su 5 anni
# mostra PF 0.96 (n=2.523, totale -57,6R) — due tentativi di fix mirato
# (blocco SELL, soglia min_prob piu' alta) scartati perche' non reggono
# lo split cronologico (vedi memoria progetto). I due cecchini dedicati
# gia' attivi su H1 (SMC+Candlestick +29,5R n=92, SMC+Stat Arb +22,2R
# n=64, entrambi validati su split indipendente) portano da soli H1 a
# +51,7R — molto meglio che sommarli al bot normale in perdita (-5,9R
# combinato). L'aggregato a 8 strategie resta disponibile a mano con /h1
# (con lo stesso avviso di M5).
ALL_TIMEFRAMES = ["4h", "1day", "15min"]
NO_EDGE_TIMEFRAMES = {"5min", "1h"}

TF_LABEL = {
    "5min": "M5", "15min": "M15", "1h": "H1", "4h": "H4", "1day": "D1",
    "5min_sniper": "M5 🎯", "1h_sniper_stat_arb": "H1 🎯 SA", "1h_sniper_candlestick": "H1 🎯 CS",
}

# Soglia di disaccordo tra due fonti spot indipendenti (gold-api.com e Twelve
# Data) oltre la quale il basis GC=F-spot calcolato in _check_single_timeframe
# viene scartato. Due letture spot per lo stesso istante dovrebbero essere
# vicine tra loro (a differenza di GC=F-spot, che diverge per costruzione) —
# vedi il commento sul controllo incrociato più sotto.
MAX_SPOT_SOURCE_DISAGREEMENT_USD = 20.0


# ═══════════════════════════════════════════════
# DB — usa il DB unico di trade_manager
# ═══════════════════════════════════════════════

def _db():
    """Connessione al DB unico per query di sola lettura (report, stats)."""
    import time
    for attempt in range(5):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 4:
                time.sleep(0.2 * (attempt + 1))
            else:
                raise


def _get_closed_trades() -> list:
    """Ritorna tutti i trade CLOSED con result non nullo."""
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='CLOSED' AND result IS NOT NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _get_trades_today() -> list:
    """
    Recupera i trade di oggi usando DATE() di SQLite invece di LIKE.
    Robusto rispetto a timezone offset nel campo timestamp.
    """
    today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    try:
        conn = _db()
        rows = conn.execute(
            """
            SELECT * FROM trades
            WHERE DATE(
                CASE
                    WHEN INSTR(timestamp, '+') > 10
                    THEN SUBSTR(timestamp, 1, INSTR(timestamp, '+') - 1)
                    WHEN INSTR(timestamp, 'T') > 0
                    THEN REPLACE(timestamp, 'T', ' ')
                    ELSE timestamp
                END
            ) = ?
            """,
            (today,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        # Fallback al LIKE originale
        try:
            conn = _db()
            rows = conn.execute(
                "SELECT * FROM trades WHERE timestamp LIKE ?", (f"{today}%",)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


# ═══════════════════════════════════════════════
# FILTRO ORARIO
# ═══════════════════════════════════════════════

def is_market_open() -> bool:
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5: return False
    if now.hour < 8 or now.hour >= 20: return False
    return True


def market_status_text() -> str:
    """Stesso confine di is_market_open() (chiamata qui, non ricalcolata):
    prima ripeteva le stesse due condizioni in una copia indipendente, col
    rischio che un domani cambiasse una sola delle due copie."""
    if is_market_open():
        return "🟢 Mercato aperto (08:00–20:00)"
    if datetime.now(TIMEZONE).weekday() >= 5:
        return "🔴 Mercato chiuso (weekend)"
    return "🔴 Mercato chiuso (fuori orario 08:00–20:00)"


def _ny_open_time_it() -> str:
    """Ora di apertura NY (9:30 ET) convertita in ora italiana dinamicamente."""
    try:
        ny_tz  = pytz.timezone("America/New_York")
        it_tz  = TIMEZONE
        now_it = datetime.now(it_tz)
        ny_open_naive = now_it.replace(hour=9, minute=30, second=0, microsecond=0, tzinfo=None)
        ny_open_et = ny_tz.localize(ny_open_naive)
        return ny_open_et.astimezone(it_tz).strftime("%H:%M")
    except Exception:
        return "15:30"  # fallback


# ═══════════════════════════════════════════════
# COMANDI TELEGRAM
# ═══════════════════════════════════════════════
# FIX (audit 2026-09-05): format_signal()/format_signal_detail() (114 righe)
# erano definite qui ma mai chiamate da nessuna parte - superate da
# format_pipeline_report() di agent_orchestrator.py, che è quello davvero
# usato per i messaggi SEGNALE. Rimosse.

async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👋 *GoldMind Bot v2*\n\n"
        f"🆔 Chat ID: `{chat_id}`\n\n"
        f"*Comandi:*\n"
        f"/signal — Analisi M5 completa\n"
        f"/m15 — Analisi M15\n"
        f"/h1 — Analisi H1\n"
        f"/h4 — Analisi H4\n"
        f"/d1 — Analisi Daily\n"
        f"/trade — Posizioni attive\n"
        f"/chiudi [tp1|tp2|tp3|loss|be] [prezzo] [tf] — Chiudi trade\n"
        f"/stats — Storico trade\n"
        f"/report — Dashboard performance\n"
        f"/status — Stato bot\n"
        f"/risk — Risk management\n"
        f"/riprendisessione — Sblocca la sessione in cooldown senza aspettare\n"
        f"/news — Notizie oro\n"
        f"/macro — Briefing eventi macro\n"
        f"/regime — Regime di mercato\n"
        f"/decisioni [tf] [n] — Log EXECUTE/WAIT/SKIP con motivazione\n"
        f"/lotto [capitale] [rischio%] — Calcolo lotti\n"
        f"/backtest [tf] [barre]\n"
        f"/posttrade — Analisi ultimo trade\n"
        f"/review — Review settimanale\n"
        f"/learn — Ottimizza pesi strategie\n"
        f"/weekend — Analisi weekend multi-TF (anteprima)\n"
        f"/pausa — Ferma segnali/report (resta monitor SL/TP/BE)\n"
        f"/riattiva — Riprende dopo /pausa",
        parse_mode="Markdown"
    )


async def _run_analysis(update, timeframe: str):
    """I comandi manuali attraversano la stessa pipeline risk/news dell'automazione."""
    if not is_authorized(update):
        return
    tf_label = TF_LABEL.get(timeframe, timeframe.upper())
    await update.message.reply_text(f"⏳ Analisi {tf_label} in corso...")
    try:
        state = await run_pipeline(timeframe=timeframe)
        if state.final_decision != "EXECUTE":
            await update.message.reply_text(
                f"⛔ *Nessuna apertura {tf_label}*\n"
                f"Decisione: *{state.final_decision}*\n"
                # decision_reason può contenere testo di un'eccezione esterna
                # (risk_reason/news_reason su errore) con caratteri Markdown
                # non bilanciati — va escapato come i titoli di notizie,
                # altrimenti l'invio fallisce con "Can't parse entities".
                f"Motivo: {_escape_md(state.decision_reason)}",
                parse_mode="Markdown"
            )
            return
        if has_open_trade_on_timeframe(timeframe):
            await update.message.reply_text(f"⏭️ Esiste già un trade/pending su {tf_label}.")
            return
        if recently_cancelled_on_timeframe(timeframe):
            await update.message.reply_text(
                f"⏳ Un pending su {tf_label} è stato annullato di recente — "
                f"aspetta qualche minuto prima di riprovare (evita segnali ripetuti)."
            )
            return

        data = {
            "signal": state.signal,
            "order_type": state.order_type,
            "entry": state.entry,
            "sl": state.sl,
            "tp1": state.tp1,
            "tp2": state.tp2,
            "tp3": state.tp3,
            "prob": state.prob,
            "regime": state.regime,
            "timeframe": timeframe,
            "price": state.current_price,
            "risk_pct": state.risk_pct,
            "strategies": state.strategies,
            "data_timestamp": state.data_timestamp,
            "early_be_level": state.early_be_level,
        }
        data["setup_key"] = build_setup_key(data)
        if was_setup_seen(data["setup_key"]):
            await update.message.reply_text("⏭️ Questo setup/candela è già stato registrato.")
            return
        trade_id = open_trade(data)
        await update.message.reply_text(format_pipeline_report(state), parse_mode="Markdown")
        if timeframe in NO_EDGE_TIMEFRAMES:
            await update.message.reply_text(
                f"⚠️ Promemoria: il backtest mostra che l'aggregato normale su "
                f"{tf_label} non ha edge positivo (profit factor sotto il "
                f"pareggio) — per questo non genera più segnali automatici. "
                f"Restano attivi eventuali cecchini dedicati su questo "
                f"timeframe. Questo trade è stato aperto solo perché "
                f"richiesto a mano."
            )
        logger.info(f"[MANUAL] Trade aperto: {trade_id}")

    except DuplicateSetupError:
        await update.message.reply_text("⏭️ Setup già registrato.")
    except Exception as e:
        logger.error(f"Errore analisi {timeframe}: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_signal(update, context: ContextTypes.DEFAULT_TYPE):
    await _run_analysis(update, "5min")

async def cmd_h1(update, context: ContextTypes.DEFAULT_TYPE):
    await _run_analysis(update, "1h")

async def cmd_h4(update, context: ContextTypes.DEFAULT_TYPE):
    await _run_analysis(update, "4h")

async def cmd_m15(update, context: ContextTypes.DEFAULT_TYPE):
    await _run_analysis(update, "15min")

async def cmd_d1(update, context: ContextTypes.DEFAULT_TYPE):
    await _run_analysis(update, "1day")


async def cmd_trade(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    trades = load_all_active_trades()
    if not trades:
        await update.message.reply_text("📭 Nessuna posizione attiva.")
        return
    msg = f"📊 *Posizioni Attive ({len(trades)})*\n━━━━━━━━━━━━━━━━━━━━\n"
    for t in trades:
        tf    = TF_LABEL.get(t.get("timeframe",""), t.get("timeframe","?").upper())
        tid   = t.get("trade_id","?")
        flags = []
        if t.get("tp1_hit"): flags.append("TP1✅")
        if t.get("tp2_hit"): flags.append("TP2✅")
        if t.get("tp3_hit"): flags.append("TP3✅")
        if t.get("be_armed") and not t.get("be_hit"): flags.append("BE🛡")
        if t.get("be_hit"): flags.append("BE✅")
        flags_txt = " | ".join(flags) if flags else ""
        msg += (
            f"*{t['signal']} [{tf}]* @ {t.get('entry','?')}\n"
            f"SL: {t.get('sl','?')} | TP1: {t.get('tp1','?')} | "
            f"TP2: {t.get('tp2','?')} | TP3: {t.get('tp3','?')}\n"
            f"ID: `{tid}` {flags_txt}\n"
            f"────────────────────\n"
        )
    if len(msg) > 4000:
        msg = msg[:3950] + "\n_[Troncato]_"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_chiudi(update, context: ContextTypes.DEFAULT_TYPE):
    """
    Chiude un trade manualmente.
    Uso: /chiudi [tp1|tp2|tp3|loss|be] [prezzo] [timeframe_opzionale]
    Esempio: /chiudi tp2 3320.5 1h
    """
    if not is_authorized(update): return

    args = context.args if context.args else []

    trades = load_all_active_trades()
    if not trades:
        await update.message.reply_text("📭 Nessuna posizione attiva.")
        return

    # Senza argomenti → mostra guida
    if not args:
        open_txt = ""
        for t in trades:
            tf = TF_LABEL.get(t.get("timeframe",""), "?")
            open_txt += f"• `{t.get('trade_id','?')}` — {t['signal']} [{tf}] @ {t.get('entry','?')}\n"
        await update.message.reply_text(
            f"📍 *Posizioni attive:*\n{open_txt}\n"
            f"*Uso:* `/chiudi [tp1|tp2|tp3|loss|be] [prezzo] [tf_opzionale]`\n"
            f"Esempio: `/chiudi tp2 3320.5 1h`",
            parse_mode="Markdown"
        )
        return

    result_map = {
        "tp1": "WIN_TP1", "tp2": "WIN_TP2", "tp3": "WIN_TP3",
        "loss": "LOSS", "sl": "LOSS", "be": "WIN_BE", "manual": "CANCELLED"
    }
    result_key = args[0].lower()
    result     = result_map.get(result_key)
    if not result:
        await update.message.reply_text(
            f"❌ Risultato non valido: `{result_key}`\n"
            f"Usa: tp1, tp2, tp3, loss, be, manual",
            parse_mode="Markdown"
        )
        return

    # Prezzo exit
    exit_price = 0.0
    if len(args) > 1:
        try:
            exit_price = float(args[1])
        except ValueError:
            pass
    if exit_price == 0.0:
        exit_price = get_current_price() or float(trades[-1].get("entry", 0))

    # Filtro timeframe opzionale
    tf_filter = args[2] if len(args) > 2 else None
    filtered  = trades
    if tf_filter:
        filtered = [t for t in trades if t.get("timeframe") == tf_filter]
    if not filtered:
        await update.message.reply_text(
            f"❌ Nessun trade attivo su timeframe `{tf_filter}`.", parse_mode="Markdown"
        )
        return

    # Chiudi il trade più recente tra quelli filtrati
    trade    = filtered[-1]
    trade_id = trade.get("trade_id")
    if not close_trade(trade_id, result, exit_price, "Chiusura manuale"):
        await update.message.reply_text("⏭️ Il trade era già stato chiuso.")
        return

    pnl_map = {
        "WIN_TP1": "+1R ✅", "WIN_TP2": "+2R ✅✅", "WIN_TP3": "+3R 🏆",
        "LOSS": "-1R ❌", "WIN_BE": "0R ⚖️", "CANCELLED": "0R (manuale)"
    }
    tf = TF_LABEL.get(trade.get("timeframe",""), "?")
    await update.message.reply_text(
        f"{'✅' if 'WIN' in result else '❌' if result == 'LOSS' else '⚖️'} *TRADE CHIUSO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 {trade.get('signal')} [{tf}] @ ${_fmt(trade.get('entry'))}\n"
        f"💰 Exit: ${_fmt(exit_price)}\n"
        f"📊 Risultato: *{pnl_map.get(result, result)}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"ID: `{trade_id}`",
        parse_mode="Markdown"
    )

    # Post-trade analysis asincrona
    asyncio.create_task(_async_posttrade_and_learn(update, result, trade_id))


async def cmd_news(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Analizzo le notizie con AI... (20-30 sec)")
    try:
        news, price = await asyncio.gather(
            asyncio.to_thread(get_extended_news),
            get_current_price_async(),
        )
        if not news:
            await update.message.reply_text("❌ Nessuna notizia disponibile.")
            return
        msg = await asyncio.to_thread(format_news_message, news, price)
        if len(msg) > 4000:
            msg = msg[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_macro(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Analisi macro in corso... (20-30 sec)")
    try:
        from analyzer import get_economic_events
        price = await get_current_price_async()
        args  = context.args if context.args else []
        if not args:
            cal    = await asyncio.to_thread(get_economic_events, broad=True)
            events = cal.get("events", [])
            msg    = await asyncio.to_thread(get_macro_briefing, events, price)
        else:
            event_name = args[0].upper()
            actual     = args[1] if len(args) > 1 else "N/A"
            cal        = await asyncio.to_thread(get_economic_events, broad=True)
            events     = cal.get("events", [])
            event_data = next(
                (e for e in events if event_name.lower() in e.get("title","").lower()), {}
            )
            # _escape_md: analyze_macro_event ritorna testo AI libero (a
            # differenza di get_macro_briefing sopra, che internamente
            # escapa già solo i titoli evento) — nessun escape qui prima,
            # rischio "can't parse entities" sotto parse_mode="Markdown".
            msg = _escape_md(await asyncio.to_thread(
                analyze_macro_event,
                event_data.get("title", event_name),
                event_data.get("forecast", "N/A"),
                event_data.get("previous", "N/A"),
                actual,
                price,
                event_data.get("currency", ""),
            ))
        if len(msg) > 4000:
            msg = msg[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_stats(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    try:
        trades = _get_closed_trades()
        # FIX: usava un criterio diverso da /report e dalla dashboard
        # (escludeva ogni WIN_BE dal win rate, anche quelli con TP1 già
        # raggiunto) — stesso DB, numeri diversi a seconda del comando.
        # Ora tutti e tre condividono is_decisive_win() da trade_manager.py.
        wins     = [t for t in trades if is_decisive_win(t)]
        losses   = [t for t in trades if t.get("result") == "LOSS"]
        decisivi = wins + losses
        be_t   = [t for t in trades if t.get("result") == "WIN_BE"]
        total  = len(decisivi)
        wr     = round(len(wins) / total * 100, 1) if total > 0 else 0
        pnl_r  = sum(t.get("pnl_r") or 0 for t in trades if t.get("status") == "CLOSED")

        recent_txt = ""
        for t in reversed(trades[-5:]):
            res = t.get("result","?")
            e   = (
                "✅" if "WIN" in res and res != "WIN_BE" else
                "⚖️" if res == "WIN_BE" else
                "🛡️" if res == "CLOSED_EARLY" else
                "❌"
            )
            tf  = TF_LABEL.get(t.get("timeframe",""), "?")
            recent_txt += f"{e} {t.get('signal','?')} [{tf}] @ ${t.get('entry','?')} — {res}\n"
        if not recent_txt:
            recent_txt = "Nessun trade completato ancora.\n"

        msg = (
            f"📊 *STORICO TRADE*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Win: *{len(wins)}* | ❌ Loss: *{len(losses)}* | ⚖️ BE: *{len(be_t)}*\n"
            f"📈 Win Rate: *{wr}%* (su {total} trade decisivi)\n"
            f"💰 P&L totale: *{pnl_r:+.1f}R*\n"
            f"📊 Trade aperti: *{len(load_all_active_trades())}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Ultimi 5:*\n{recent_txt}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Totale trade nel DB: {len(trades)}_"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    status  = market_status_text()
    trades  = load_all_active_trades()
    open_count = len(trades)
    if trades:
        t_info = f"📊 {open_count} posizione/i attiva/e\n"
        for t in trades:
            tf = TF_LABEL.get(t.get("timeframe",""), "?")
            t_info += f"  • {t['signal']} [{tf}] @ {t['entry']}\n"
    else:
        t_info = "📭 Nessun trade attivo"
    paused = is_bot_paused()
    stato_txt = "⏸ *IN PAUSA* (usa /riattiva)" if paused else "🤖 Attivo"
    await update.message.reply_text(
        f"⚙️ *Stato GoldMind v2*\n"
        f"━━━━━━━━━━━━━━\n"
        f"{status}\n"
        f"📊 Soglia minima: *{MIN_PROB}%*\n"
        f"🤖 Strategie: *fino a 9* (order flow solo con volume reale)\n"
        f"🗄 Database: *goldbot.db unico*\n"
        f"{t_info}\n"
        f"Stato: {stato_txt}",
        parse_mode="Markdown"
    )


async def cmd_pausa(update, context: ContextTypes.DEFAULT_TYPE):
    """Ferma la generazione di nuovi segnali/analisi finché non arriva /riattiva.
    Il monitoraggio SL/TP/BE dei trade già aperti resta sempre attivo."""
    if not is_authorized(update): return
    if is_bot_paused():
        await update.message.reply_text("⏸ Il bot è già in pausa. Usa /riattiva per riprendere.")
        return
    set_bot_paused(True)
    await update.message.reply_text(
        "⏸ *Bot in pausa*\n"
        "Niente più nuovi segnali, report o alert macro finché non fai /riattiva.\n"
        "I trade già aperti restano monitorati (SL/TP/BE).",
        parse_mode="Markdown"
    )


async def cmd_riattiva(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    if not is_bot_paused():
        await update.message.reply_text("✅ Il bot è già attivo.")
        return
    set_bot_paused(False)
    await update.message.reply_text("✅ *Bot riattivato* — analisi e segnali ripresi.", parse_mode="Markdown")


async def cmd_report(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    try:
        trades   = _get_closed_trades()

        # Stesso criterio della dashboard/is_decisive_win() (trade_manager.py):
        # un WIN_BE con TP1 già raggiunto conta come vittoria (profitto
        # parziale incassato), non viene escluso come i BE "puri".
        wins_l   = [t for t in trades if is_decisive_win(t)]
        losses_l = [t for t in trades if t.get("result") == "LOSS"]
        decisivi = wins_l + losses_l
        if not decisivi:
            await update.message.reply_text("📭 Nessun trade chiuso nel database.")
            return

        # R-multiple reali (compreso 0R dei BE) su tutti i trade chiusi non
        # annullati, non solo sui decisivi — coerente col total_r della
        # dashboard (che esclude solo i CANCELLED).
        r_values = [
            float(t.get("pnl_r") or 0) for t in trades if t.get("result") != "CANCELLED"
        ]

        total    = len(decisivi)
        wins     = len(wins_l)
        losses   = len(losses_l)
        win_rate = round(wins / total * 100, 1)
        total_r  = round(sum(r_values), 1)

        gross_win  = sum(v for v in r_values if v > 0)
        gross_loss = abs(sum(v for v in r_values if v < 0))
        pf         = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0

        # Equity curve e drawdown
        equity = 0.0
        peak   = 0.0
        max_dd = 0.0
        curr_dd_at_end = 0.0
        for v in r_values:
            equity += v
            if equity > peak: peak = equity
            dd = peak - equity
            if dd > max_dd: max_dd = dd
        curr_dd_at_end = max(0, peak - equity)

        avg_r  = statistics.mean(r_values) if r_values else 0
        std_r  = statistics.stdev(r_values) if len(r_values) > 1 else 0
        sharpe = round(avg_r / std_r, 2) if std_r > 0 else 0

        # Per timeframe
        by_tf = {}
        for t in decisivi:
            tf = TF_LABEL.get(t.get("timeframe",""), t.get("timeframe","?"))
            by_tf.setdefault(tf, {"w": 0, "n": 0})
            by_tf[tf]["n"] += 1
            if is_decisive_win(t): by_tf[tf]["w"] += 1
        tf_txt = "\n".join(
            f"  • {tf}: {d['w']}/{d['n']} ({round(d['w']/d['n']*100,1)}%)"
            for tf, d in sorted(by_tf.items())
        )

        # TP1 e BE hit rate
        tp1_rate = round(sum(1 for t in trades if t.get("tp1_hit")) / len(trades) * 100, 1) if trades else 0
        be_rate  = round(sum(1 for t in trades if t.get("be_hit"))  / len(trades) * 100, 1) if trades else 0

        msg = (
            f"📊 *DASHBOARD PERFORMANCE*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trade decisivi: *{total}* (W={wins} L={losses})\n"
            f"📈 Win Rate: *{win_rate}%*\n"
            f"💰 P&L totale: *{total_r:+.1f}R*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 Profit Factor: *{pf}*\n"
            f"📉 DD corrente: *{curr_dd_at_end:.1f}R* | Max: *{max_dd:.1f}R*\n"
            f"📐 Sharpe (per-trade): *{sharpe}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 TP1 hit: *{tp1_rate}%* | BE hit: *{be_rate}%*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Per timeframe:*\n{tf_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Win Rate: WIN o WIN\\_BE con TP1 raggiunto contano come vittoria; BE puri esclusi._"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore report: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_risk(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    try:
        msg = format_risk_report()
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_lotto(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    args = context.args if context.args else []
    try:
        balance  = float(args[0]) if len(args) > 0 else 1000.0
        risk_pct = float(args[1]) if len(args) > 1 else 1.0
    except ValueError:
        await update.message.reply_text("⚠️ Uso: /lotto 1000 1  (capitale=$1000, rischio=1%)")
        return

    trades = load_all_active_trades()
    if not trades:
        await update.message.reply_text(
            "📭 Nessun trade attivo su cui calcolare il lotto.\n"
            "Fai prima /signal, /h1 o /h4."
        )
        return

    trade  = trades[-1]
    entry  = float(trade["entry"])
    sl     = float(trade["sl"])
    sizing = calculate_lot_size(balance, risk_pct, entry, sl)

    if not sizing.get("tradable"):
        await update.message.reply_text(
            "⛔ *TRADE NON ESEGUIBILE CON QUESTO RISCHIO*\n"
            f"{sizing.get('error','Parametri non validi')}\n"
            f"Budget rischio: ${sizing.get('risk_amount',0):,.2f}",
            parse_mode="Markdown",
        )
        return

    msg = (
        f"💰 *POSITION SIZING — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Capitale: ${balance:,.2f}\n"
        f"Rischio: {risk_pct}% → ${sizing['risk_amount']:,.2f}\n"
        f"Distanza SL: ${sizing['sl_distance_usd']} "
        f"(= ${sizing['value_per_lot']:.0f}/lotto)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📏 *Lotto consigliato: {sizing['lot_size']}*\n"
        f"Rischio effettivo: ${sizing['actual_risk_amount']:,.2f} "
        f"({sizing['actual_risk_pct']:.3f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Trade: {trade['signal']} @ {entry}, SL @ {sl}_\n"
        f"_Formula: rischio$ / (dist$ × 100oz)_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")




async def cmd_riprendisessione(update, context: ContextTypes.DEFAULT_TYPE):
    """Sblocca manualmente la sessione in cooldown senza aspettare le 5 ore."""
    if not is_authorized(update): return
    try:
        msg = resume_session_manual()
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")

async def cmd_regime(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Analisi regime in corso...")
    try:
        from analyzer import get_data, compute_indicators, detect_market_regime, format_live_regime_message
        def _regime_analysis():
            frame = compute_indicators(get_data(interval="5min", outputsize=100))
            return detect_market_regime(frame)
        regime = await asyncio.to_thread(_regime_analysis)
        msg    = format_live_regime_message(regime)
        if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


_DECISION_ICON = {"EXECUTE": "✅", "WAIT": "⏳", "SKIP": "⏭️"}


async def cmd_decisions(update, context: ContextTypes.DEFAULT_TYPE):
    """
    /decisioni [tf] [n] — ultime N decisioni EXECUTE/WAIT/SKIP della
    pipeline, con motivazione. Log strutturato interrogabile (vedi
    trade_manager.log_decision, chiamato da agent_orchestrator.run_pipeline
    in un unico punto per non doverlo agganciare a ogni singolo agente).
    """
    if not is_authorized(update): return
    args = context.args or []
    tf = None
    n = 15
    for arg in args:
        if arg.lower() in ("1h", "4h", "1day", "5min", "15min", "1min"):
            tf = arg.lower()
        elif arg.isdigit():
            n = int(arg)

    rows = load_recent_decisions(timeframe=tf, limit=n)
    if not rows:
        await update.message.reply_text("Nessuna decisione registrata ancora.")
        return

    tf_txt = f" [{TF_LABEL.get(tf, tf.upper())}]" if tf else ""
    lines = [f"📋 *Ultime decisioni{tf_txt}*", "━━━━━━━━━━━━━━━━━━━━"]
    for row in rows:
        icon = _DECISION_ICON.get(row.get("decision"), "•")
        ts = str(row.get("timestamp", ""))[:16].replace("T", " ")
        tf_lbl = TF_LABEL.get(row.get("timeframe"), row.get("timeframe", "?"))
        sig = row.get("signal") or "N/D"
        prob_txt = f" {row['prob']}%" if row.get("prob") else ""
        reason = _escape_md(str(row.get("reason") or "")[:120])
        lines.append(
            f"{icon} `{ts}` [{tf_lbl}] {sig}{prob_txt} — *{row.get('decision')}*\n"
            f"   _{reason}_"
        )
    msg = "\n".join(lines)
    if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
    await update.message.reply_text(msg, parse_mode="Markdown")


# Mapping periodo → barre per ogni timeframe
# M5: 288 candele/giorno | M15: 96/giorno | H1: 24/giorno | H4: 6/giorno | D1: 1/giorno
# Giorni di trading per periodo: 3m=65, 6m=130, 1y=260, 2y=520, 5y=1300, 10y=2600, 20y=5200
_TRADING_DAYS = {"3m": 65, "6m": 130, "1y": 260, "2y": 520, "5y": 1300, "10y": 2600, "20y": 5200}
_BARS_PER_DAY = {"5min": 288, "15min": 96, "1h": 24, "4h": 6, "1day": 1}
_PERIODO_BARRE = {
    tf: {periodo: bpd * days for periodo, days in _TRADING_DAYS.items()}
    for tf, bpd in _BARS_PER_DAY.items()
}
_PERIODI_VALIDI = tuple(_TRADING_DAYS.keys())
# Le fonti gratuite (yfinance/Stooq) hanno storico intraday limitato a
# ~2 anni: 5y/10y/20y sono affidabili solo su 1day (vedi analyzer._interval_to_yf).
_PERIODI_LUNGHI = ("5y", "10y", "20y")
# Tetto barre: 1day può arrivare a 20y (~5200gg), gli intraday sono comunque
# limitati dalla fonte dati quindi un tetto più basso non li penalizza.
_BAR_CAP = {"5min": 4500, "15min": 4500, "1h": 4500, "4h": 4500, "1day": 6000}


async def cmd_backtest(update, context: ContextTypes.DEFAULT_TYPE):
    """
    Uso:
      /backtest [tf] [barre]              — backtest standard
      /backtest [tf] 3m/6m/1y/2y/5y/10y/20y — backtest lungo
      /backtest tutti 3m/6m/1y/2y/5y/10y/20y — tutti i TF con statistiche comparate
      /backtest wf [tf] [barre]           — walk-forward
      /backtest mc [tf] [barre]           — Monte Carlo sul drawdown
      /backtest robust [tf] [barre]       — robustezza ai parametri
      /backtest calib [tf] [barre]        — calibrazione probabilità

    NB: 5y/10y/20y sono storico affidabile solo su 1day/4h — le fonti gratuite
    (yfinance/Stooq) non hanno abbastanza storico intraday oltre ~2 anni.
    """
    if not is_authorized(update): return
    args = context.args if context.args else []

    # /backtest wf [interval] [bars]
    if args and args[0].lower() == "wf":
        from backtest import run_walkforward_backtest, format_walkforward_report
        interval = args[1] if len(args) > 1 else "5min"
        try: bars = int(args[2]) if len(args) > 2 else 2000
        except ValueError: bars = 2000
        valid = ["1min","5min","15min","1h","4h","1day"]
        if interval not in valid:
            await update.message.reply_text(f"TF non valido. Usa: {', '.join(valid)}")
            return
        await update.message.reply_text(f"Walk-Forward {interval} su {bars} candele... (3-6 min)")
        try:
            stats = await asyncio.to_thread(
                run_walkforward_backtest, interval=interval, bars=bars, min_prob=MIN_PROB
            )
            msg = format_walkforward_report(stats, interval)
            await update.message.reply_text(msg[:4000])
        except Exception as e:
            await update.message.reply_text(f"Errore walk-forward: {e}")
        return

    # /backtest mc [interval] [bars] — Monte Carlo sul drawdown
    if args and args[0].lower() == "mc":
        from backtest import run_backtest, monte_carlo_drawdown, format_monte_carlo_report
        interval = args[1] if len(args) > 1 else "1day"
        try: bars = int(args[2]) if len(args) > 2 else 2000
        except ValueError: bars = 2000
        valid = ["1min","5min","15min","1h","4h","1day"]
        if interval not in valid:
            await update.message.reply_text(f"TF non valido. Usa: {', '.join(valid)}")
            return
        await update.message.reply_text(f"Monte Carlo {interval} su {bars} candele... (1-2 min)")
        try:
            min_prob = _live_min_prob_for_tf(interval)
            stats = await asyncio.to_thread(run_backtest, interval=interval, bars=bars, min_prob=min_prob)
            mc = await asyncio.to_thread(monte_carlo_drawdown, stats.get("r_results", []))
            msg = format_monte_carlo_report(mc, interval)
            await update.message.reply_text(msg[:4000], parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Errore Monte Carlo: {e}")
        return

    # /backtest robust [interval] [bars] — robustezza ai parametri
    if args and args[0].lower() == "robust":
        from backtest import parameter_sensitivity_backtest, format_sensitivity_report
        interval = args[1] if len(args) > 1 else "1day"
        try: bars = int(args[2]) if len(args) > 2 else 2000
        except ValueError: bars = 2000
        valid = ["1min","5min","15min","1h","4h","1day"]
        if interval not in valid:
            await update.message.reply_text(f"TF non valido. Usa: {', '.join(valid)}")
            return
        await update.message.reply_text(f"Test robustezza {interval} su {bars} candele (5 soglie)... (3-5 min)")
        try:
            base_prob = _live_min_prob_for_tf(interval)
            sens = await asyncio.to_thread(
                parameter_sensitivity_backtest, interval=interval, bars=bars, base_min_prob=base_prob
            )
            msg = format_sensitivity_report(sens, interval)
            await update.message.reply_text(msg[:4000], parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Errore test robustezza: {e}")
        return

    # /backtest calib [interval] [bars] — calibrazione probabilità
    if args and args[0].lower() == "calib":
        from backtest import run_backtest, calibration_report, format_calibration_report
        interval = args[1] if len(args) > 1 else "1day"
        try: bars = int(args[2]) if len(args) > 2 else 2000
        except ValueError: bars = 2000
        valid = ["1min","5min","15min","1h","4h","1day"]
        if interval not in valid:
            await update.message.reply_text(f"TF non valido. Usa: {', '.join(valid)}")
            return
        await update.message.reply_text(f"Calibrazione probabilità {interval} su {bars} candele... (1-2 min)")
        try:
            min_prob = _live_min_prob_for_tf(interval)
            stats = await asyncio.to_thread(run_backtest, interval=interval, bars=bars, min_prob=min_prob)
            calib = await asyncio.to_thread(calibration_report, stats.get("prob_outcome_pairs", []))
            msg = format_calibration_report(calib, interval)
            await update.message.reply_text(msg[:4000], parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Errore calibrazione: {e}")
        return

    # /backtest tutti 3m/6m/1y/2y/5y/10y/20y — tutti i TF in sequenza
    if args and args[0].lower() == "tutti":
        from backtest import run_backtest, format_backtest_report
        periodo = args[1].lower() if len(args) > 1 else "3m"
        if periodo not in _PERIODI_VALIDI:
            await update.message.reply_text(f"Periodo non valido. Usa: {', '.join(_PERIODI_VALIDI)}")
            return
        tfs = ["5min","15min","1h","4h","1day"]
        avviso_storico = (
            f"\n⚠️ {periodo.upper()}: storico intraday limitato (~2 anni) sulle fonti gratuite; "
            f"solo 1day copre l'intero periodo."
            if periodo in _PERIODI_LUNGHI else ""
        )
        await update.message.reply_text(
            f"Backtest TUTTI i TF - {periodo.upper()}\n"
            f"Elaboro {len(tfs)} timeframe... (5-10 min){avviso_storico}"
        )
        risultati = []
        for tf in tfs:
            barre = _PERIODO_BARRE.get(tf, {}).get(periodo, 1000)
            barre = min(barre, _BAR_CAP.get(tf, 4500))
            try:
                stats = await asyncio.to_thread(
                    run_backtest, interval=tf, bars=barre, min_prob=_live_min_prob_for_tf(tf)
                )
                if stats.get("total", 0) == 0:
                    # Niente trade non è per forza "0 segnali validi": può essere
                    # un fetch dati fallito (es. rate limit Yahoo a metà sequenza).
                    # Mostriamo il motivo reale invece di uno "0 trade" muto che
                    # sembra un risultato valido.
                    risultati.append(f"{tf}: {stats.get('message', 'nessun trade concluso')}")
                else:
                    wr   = stats.get("win_rate", 0)
                    pnl  = stats.get("total_r", 0)
                    wins = stats.get("wins", 0)
                    loss = stats.get("losses", 0)
                    tot  = stats.get("concluded", 0)
                    risultati.append(
                        f"{tf}: WR {wr}% | {wins}W/{loss}L | P&L {pnl:+.1f}R ({tot} trade)"
                    )
            except Exception as e:
                risultati.append(f"{tf}: ERRORE - {str(e)[:60]}")
            # Pausa tra TF per non saturare il rate limit di Yahoo/Twelve Data —
            # ogni TF scarica fino a migliaia di candele, 2s non bastava sempre.
            await asyncio.sleep(5)

        msg = (
            f"BACKTEST TUTTI TF - {periodo.upper()}\n"
            f"{'='*35}\n"
            + "\n".join(risultati) +
            f"\n{'='*35}\n"
            f"Soglia prob: M5/M15 >= 65% | H1/H4/D1 >= 55%"
        )
        await update.message.reply_text(msg[:4000])
        return

    # /backtest [interval] [barre|periodo]
    valid = ["1min","5min","15min","1h","4h","1day"]
    interval = args[0] if len(args) > 0 else "5min"
    if interval not in valid:
        await update.message.reply_text(
            f"Uso:\n"
            f"/backtest [tf] [barre]     es. /backtest 5min 500\n"
            f"/backtest [tf] {'/'.join(_PERIODI_VALIDI)}   es. /backtest 1day 10y\n"
            f"/backtest tutti {'/'.join(_PERIODI_VALIDI)}   tutti i TF\n"
            f"/backtest wf [tf] [barre]  walk-forward\n"
            f"TF validi: {', '.join(valid)}\n"
            f"_5y/10y/20y affidabili solo su 1day (storico intraday gratuito limitato)_"
        )
        return

    # Determina barre: numero intero o periodo (3m/6m/1y/2y/5y/10y/20y)
    periodo_arg = args[1].lower() if len(args) > 1 else "500"
    if periodo_arg in _PERIODI_VALIDI:
        bars = _PERIODO_BARRE.get(interval, {}).get(periodo_arg, 1000)
        bars = min(bars, _BAR_CAP.get(interval, 4500))
        label = periodo_arg.upper()
        if periodo_arg in _PERIODI_LUNGHI and interval != "1day":
            label += " (⚠️ storico intraday limitato ~2 anni sulle fonti gratuite)"
    else:
        try: bars = int(periodo_arg)
        except ValueError: bars = 500
        label = f"{bars} candele"

    from backtest import run_backtest, format_backtest_report
    await update.message.reply_text(
        f"Backtest {interval} - {label}... (1-5 min)"
    )
    try:
        stats = await asyncio.to_thread(
            run_backtest, interval=interval, bars=bars, min_prob=_live_min_prob_for_tf(interval)
        )
        msg = format_backtest_report(stats, interval)
        await update.message.reply_text(msg[:4000])
    except Exception as e:
        await update.message.reply_text(f"Errore backtest: {e}")


async def cmd_posttrade(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Analisi post-trade in corso... (20-30 sec)")
    try:
        analysis = await asyncio.to_thread(analyze_last_trade)
        if len(analysis) > 4000: analysis = analysis[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(
            f"🔍 *ANALISI POST-TRADE*\n━━━━━━━━━━━━━━━━━━━━\n{analysis}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_review(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Review settimanale in corso... (30-40 sec)")
    try:
        review = await asyncio.to_thread(weekly_review)
        if len(review) > 4000: review = review[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(review, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_learn(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Ottimizzazione pesi in corso...")
    try:
        result = await asyncio.to_thread(optimize_strategy_weights)
        msg    = format_learning_report(result)
        if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def handle_free_text(update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    question = update.message.text
    if not question: return
    thinking = await update.message.reply_text("🤔 Ci penso...")
    try:
        answer = await ask_ai(question)
        await thinking.edit_text(answer)
    except Exception as e:
        await thinking.edit_text(f"❌ Errore: {e}")


# ═══════════════════════════════════════════════
# JOB AUTOMATICI
# ═══════════════════════════════════════════════

async def _async_posttrade_and_learn(update, outcome: str, trade_id: str = ""):
    """
    Post-trade analysis + auto-ottimizzazione ogni 10 trade chiusi.

    FIX (audit 2026-09-05): trade_id era già un parametro di questa funzione
    (passato correttamente dal chiamante) ma non veniva inoltrato ad
    analyze_last_trade() — cadeva sempre sul fallback "ultimo trade nel
    DB", lo stesso bug trovato in trade_manager._post_trade_analysis.
    """
    if outcome in ("CANCELLED", "manual"):
        return
    await asyncio.sleep(2)
    try:
        analysis = await asyncio.to_thread(analyze_last_trade, trade_id)
        if len(analysis) > 3800: analysis = analysis[:3750] + "\n_[Troncato]_"
        await update.message.reply_text(
            f"🤖 *ANALISI POST-TRADE*\n━━━━━━━━━━━━━━━━━━━━\n{analysis}",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.debug(f"Post-trade analysis fallita: {e}")

    # Auto-ottimizzazione ogni 10 trade chiusi
    try:
        trades = _get_closed_trades()
        total  = len(trades)
        if total >= 10 and total % 10 == 0:
            await asyncio.sleep(2)
            result = await asyncio.to_thread(optimize_strategy_weights)
            if result.get("status") == "optimized":
                msg = format_learning_report(result)
                if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
                await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.debug(f"Auto-ottimizzazione fallita: {e}")


async def _fast_monitor(bot):
    """Monitor SL/TP ogni 10 secondi. Skip se nessun trade attivo."""
    try:
        trades = load_all_active_trades()
        if trades:
            await monitor_active_trade(bot, CHAT_ID)
    except Exception as e:
        logger.debug(f"Fast monitor errore: {e}")




async def _send_weekly_review(bot):
    """Weekly review automatica domenica 20:30."""
    try:
        review = await asyncio.to_thread(weekly_review)
        if len(review) > 4000: review = review[:3950] + "\n_[Troncato]_"
        await bot.send_message(chat_id=CHAT_ID, text=review, parse_mode="Markdown")
        result = await asyncio.to_thread(optimize_strategy_weights)
        if result.get("status") == "optimized":
            msg = format_learning_report(result)
            if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
            await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore weekly review: {e}")


async def _build_weekend_outlook() -> tuple:
    """
    Analisi multi-timeframe di XAU/USD (M15/H1/H4/D1) per anticipare cosa
    potrebbe accadere alla riapertura di lunedì. Riusa la stessa pipeline
    di analisi dei segnali live (agent_orchestrator.run_pipeline), ma è
    puramente informativa: non apre trade, va letta anche se final_decision
    non è EXECUTE.

    Ritorna (msg, chart_path): chart_path è il PNG con le zone chiave
    (supporto/resistenza/prezzo attuale/trend) generato da weekly_chart,
    calcolate sugli STESSI dati riportati nel testo — oppure None se il
    grafico non è stato generabile (es. dati 4h non disponibili).
    """
    outlook_tfs = ["15min", "1h", "4h", "1day"]
    lines = []
    bias_votes = {"BUY": 0, "SELL": 0}

    for tf in outlook_tfs:
        tf_lbl = TF_LABEL.get(tf, tf.upper())
        try:
            state = await run_pipeline(timeframe=tf)
            regime_label = state.regime or "N/D"
            if state.signal in ("BUY", "SELL"):
                bias_votes[state.signal] += 1
                lines.append(
                    f"*{tf_lbl}* — {state.signal} (prob {state.prob}%, storico ~{state.prob_display}%) | regime {regime_label}\n"
                    f"  Entry {state.entry} | SL {state.sl} | TP1 {state.tp1} | R:R {state.rr}"
                )
            else:
                lines.append(f"*{tf_lbl}* — nessun bias chiaro | regime {regime_label}")
        except Exception as e:
            lines.append(f"*{tf_lbl}* — errore analisi ({str(e)[:60]})")
        await asyncio.sleep(1)

    if bias_votes["BUY"] > bias_votes["SELL"]:
        overall = "🟢 Bias prevalente RIALZISTA sui TF analizzati"
    elif bias_votes["SELL"] > bias_votes["BUY"]:
        overall = "🔴 Bias prevalente RIBASSISTA sui TF analizzati"
    else:
        overall = "⚪ Nessun bias dominante — TF in disaccordo, cautela alla riapertura"

    events_txt = ""
    try:
        from analyzer import get_upcoming_events
        events = await asyncio.to_thread(get_upcoming_events, 7)
        if events:
            ev_lines = [
                f"• {ev.get('date','')} {ev.get('time','')} — "
                f"{'🔴' if ev.get('impact','HIGH')=='HIGH' else '🟠'} "
                f"{ev.get('title','')}"
                f"{' [' + ev.get('currency','') + ']' if ev.get('currency','USD') not in ('USD','US') else ''}"
                for ev in events[:8]
            ]
            events_txt = "\n\n📅 *Eventi macro della settimana:*\n" + "\n".join(ev_lines)
    except Exception:
        pass

    zone_txt = ""
    chart_path = None
    try:
        from analyzer import get_data
        from weekly_chart import compute_weekly_zones, render_weekly_outlook_chart
        df_4h = await asyncio.to_thread(get_data, "4h", 90)
        zones = compute_weekly_zones(df_4h)
        bias_dir = "BUY" if bias_votes["BUY"] > bias_votes["SELL"] else (
            "SELL" if bias_votes["SELL"] > bias_votes["BUY"] else "NEUTRAL"
        )
        chart_path = await asyncio.to_thread(render_weekly_outlook_chart, df_4h, zones, bias_dir)
        zone_txt = (
            "\n\n🎯 *Zone chiave (4H):*\n"
            f"🔴 Resistenza: {zones['resistance']:,.2f} (max settimana {zones['week_high']:,.2f})\n"
            f"🟢 Supporto: {zones['support']:,.2f} (min settimana {zones['week_low']:,.2f})\n"
            f"⚪ Prezzo attuale: {zones['current_price']:,.2f} | Pivot: {zones['pivot']:,.2f}"
        )
    except Exception as e:
        logger.error(f"Errore generazione grafico weekend: {e}")

    narrative_txt = ""
    if chart_path:
        try:
            from weekly_chart import compute_smc_context
            from news_analyst import get_weekly_smc_narrative
            smc_ctx = await asyncio.to_thread(compute_smc_context, df_4h)
            narrative = await asyncio.to_thread(get_weekly_smc_narrative, smc_ctx, zones["current_price"])
            narrative_txt = f"\n\n🧠 *Lettura tecnica (4H):*\n{_escape_md(narrative)}"
        except Exception as e:
            logger.error(f"Errore narrativa SMC weekend: {e}")

    msg = (
        f"🔮 *ANALISI WEEKEND — XAU/USD*\n"
        f"_In vista della riapertura di lunedì_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{overall}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        + "\n\n".join(lines)
        + zone_txt
        + narrative_txt
        + events_txt
    )
    return msg, chart_path


async def _send_chart_then_message(bot_or_message, chart_path: str, msg: str, *, is_update_reply: bool):
    """Invia prima la foto (se disponibile) e poi il messaggio testuale —
    stesso ordine dell'esempio a cui si è ispirata la richiesta: prima le
    zone chiave a colpo d'occhio, poi il dettaglio scritto."""
    if chart_path:
        try:
            with open(chart_path, "rb") as photo:
                if is_update_reply:
                    await bot_or_message.reply_photo(photo)
                else:
                    await bot_or_message.send_photo(chat_id=CHAT_ID, photo=photo)
        except Exception as e:
            logger.error(f"Errore invio grafico weekend: {e}")
        finally:
            try:
                os.remove(chart_path)
            except OSError:
                pass
    if is_update_reply:
        await bot_or_message.reply_text(msg, parse_mode="Markdown")
    else:
        await bot_or_message.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")


async def _send_weekend_outlook(bot):
    """Analisi weekend automatica domenica sera, in vista di lunedì."""
    if is_bot_paused():
        return
    try:
        msg, chart_path = await _build_weekend_outlook()
        if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
        await _send_chart_then_message(bot, chart_path, msg, is_update_reply=False)
    except Exception as e:
        logger.error(f"Errore weekend outlook: {e}")


async def cmd_weekend(update, context: ContextTypes.DEFAULT_TYPE):
    """Anteprima manuale dell'analisi weekend (utile per testarla senza aspettare domenica)."""
    if not is_authorized(update): return
    await update.message.reply_text("⏳ Analisi multi-timeframe in corso... (1-2 min)")
    try:
        msg, chart_path = await _build_weekend_outlook()
        if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
        await _send_chart_then_message(update.message, chart_path, msg, is_update_reply=True)
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


# Traccia alert macro già inviati. Sono dict usati come "set ordinato"
# (chiave = alert già inviato, valore ignorato), non veri set: un set non
# preserva l'ordine di inserimento, quindi il tetto di sicurezza "tieni solo
# gli ultimi 30" più sotto scartava un elemento arbitrario invece del più
# vecchio, rischiando di far sparire una chiave appena aggiunta e reinviare
# lo stesso alert — vedi anche trade_manager.load_breaking_news_seen.
_sent_event_alerts      = {}
_sent_post_event_alerts = {}
# Bias e prezzo al momento dell'alert pre-evento, per verificare nel
# post-evento se la previsione si è avverata (confronto oggettivo sul
# prezzo reale, non un'altra opinione dell'AI).
_pre_event_bias = {}
# Previsioni statistiche Fase 5 (macro_predictor) già inviate — dedup
# separato da _sent_post_event_alerts perché la finestra di retry (3 ore)
# è indipendente da quella del resoconto post-evento (7 minuti).
_sent_stat_prediction = {}


async def check_breaking_news_job(bot):
    """
    Controlla ogni 10 minuti comunicati Fed non programmati (press release +
    discorsi) — cosa che il calendario eventi (check_macro_alerts) non copre
    perché tratta solo eventi SCHEDULATI. Vedi breaking_news.py per le fonti
    usate e perché il Treasury è stato escluso.

    Le press release vengono filtrate: molte sono azioni amministrative di
    routine (approvazioni bancarie, azioni disciplinari) senza alcuna
    rilevanza per l'oro — si notificano solo se il classificatore rileva un
    tono hawkish/dovish. I discorsi vengono sempre notificati: sono meno
    frequenti e tipicamente più rilevanti per il mercato.
    """
    if is_bot_paused():
        return
    try:
        import re
        from trade_manager import (
            load_breaking_news_seen, save_breaking_news_seen,
            load_breaking_news_pending, save_breaking_news_pending,
        )
        import breaking_news

        seen_ids, is_first_run = await asyncio.to_thread(load_breaking_news_seen)
        alerts, new_seen = await asyncio.to_thread(breaking_news.check_breaking_news, seen_ids)

        if is_first_run:
            # Primo avvio: lo storico esistente (decine di comunicati) non è
            # "breaking" — si salva come baseline senza notificare nulla.
            await asyncio.to_thread(save_breaking_news_seen, new_seen)
            logger.info(f"Breaking news: baseline iniziale salvata ({len(new_seen)} item)")
            return

        pending = await asyncio.to_thread(load_breaking_news_pending)

        for alert in alerts:
            if alert["source"] == "fed_press" and alert["classification"]["label"] == "NEUTRO":
                continue
            try:
                price = await get_current_price_async()
            except Exception:
                price = 0.0

            source_label = {
                "fed_press": "Comunicato Fed",
                "fed_speech": "Discorso di un membro Fed",
                "treasury": "Comunicato Treasury",
            }.get(alert["source"], alert["source"])
            try:
                ai_analysis = await asyncio.to_thread(
                    analyze_breaking_news,
                    source_label, alert["title"], alert.get("summary", ""),
                    alert.get("classification", {}).get("xau_bias", "N/D"), price,
                )
            except Exception:
                logger.exception("Analisi AI breaking news fallita")
                ai_analysis = None

            try:
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=breaking_news.format_breaking_alert(alert, current_price=price, ai_analysis=ai_analysis),
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Invio breaking news fallito")

            # Cattura bias+prezzo per il confronto oggettivo ~10 minuti dopo
            # (stesso principio del post-evento macro in check_macro_alerts,
            # ma qui l'orario non è noto in anticipo — si parte dall'invio
            # dell'alert). Solo se l'AI ha dato un bias direzionale reale:
            # NEUTRO/nessuna risposta non ha nulla da confermare.
            if ai_analysis and price > 0:
                m = re.search(r"Per l'oro:\s*(BUY|SELL)", ai_analysis)
                if m:
                    item_id = breaking_news._item_id(alert["source"], alert)
                    pending[item_id] = {
                        "title": alert["title"],
                        "bias": m.group(1),
                        "price": price,
                        "sent_at": datetime.now(TIMEZONE).isoformat(),
                    }

        # Confronto oggettivo ~10 minuti dopo: prezzo reale vs bias previsto.
        # Calcolo aritmetico sui prezzi, non una seconda opinione dell'AI —
        # stesso principio del POST-EVENTO macro. Scarta anche le voci più
        # vecchie di 60 minuti mai risolte (es. prezzo live irraggiungibile
        # per un'ora), per non far crescere pending all'infinito.
        resolved = []
        for item_id, info in pending.items():
            try:
                sent_at = datetime.fromisoformat(info["sent_at"])
                if sent_at.tzinfo is None:
                    sent_at = TIMEZONE.localize(sent_at)
                age_minutes = (datetime.now(TIMEZONE) - sent_at.astimezone(TIMEZONE)).total_seconds() / 60
            except Exception:
                resolved.append(item_id)
                continue

            if age_minutes >= 60:
                resolved.append(item_id)
                continue
            if age_minutes < 10:
                continue

            try:
                price_now = await get_current_price_async()
            except Exception:
                price_now = 0.0
            if price_now <= 0:
                continue

            change = price_now - info["price"]
            bias = info["bias"]
            esito, segno, _status = _confirm_bias(bias, change)

            msg_post = (
                f"📊 *POST-BREAKING NEWS — {_escape_md(info['title'])}*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 XAU/USD: *${_fmt(price_now)}*\n"
                f"🎯 Bias previsto: *{bias}* (era ${_fmt(info['price'])})\n"
                f"{esito}\n"
                f"📐 Variazione: *{segno}{change:.2f}$*"
            )
            try:
                await bot.send_message(chat_id=CHAT_ID, text=msg_post, parse_mode="Markdown")
            except Exception:
                logger.exception("Invio post-breaking-news fallito")
            resolved.append(item_id)

        for item_id in resolved:
            pending.pop(item_id, None)

        await asyncio.to_thread(save_breaking_news_pending, pending)
        await asyncio.to_thread(save_breaking_news_seen, new_seen)
    except Exception as e:
        logger.error(f"Errore breaking news: {e}")


async def check_macro_alerts(bot):
    """Controlla ogni 5 minuti se ci sono eventi macro entro 30 minuti."""
    global _sent_event_alerts, _sent_post_event_alerts, _pre_event_bias, _sent_stat_prediction
    if is_bot_paused():
        return
    try:
        from analyzer import get_upcoming_events
        now    = datetime.now(TIMEZONE)
        # hours_lookback=3.5: senza, get_upcoming_events() scarta gli eventi
        # già avvenuti prima ancora che si possa controllare la finestra
        # POST-evento (8-15 minuti dopo) — il resoconto post-evento non
        # scattava mai, per nessun evento (bug trovato il 1 settembre 2026).
        # Alzato da 0.5 a 3.5 (FIX 2026-09-06): la previsione statistica
        # (sotto) deve poter ricontrollare FRED fino a 3 ore dopo l'evento
        # — se l'evento sparisse dalla lista dopo 30 minuti come prima,
        # quel blocco non avrebbe mai la possibilità di ritentare. Le
        # finestre pre/post-evento più strette restano invariate, sono
        # condizioni indipendenti più a valle nello stesso ciclo.
        events = await asyncio.to_thread(get_upcoming_events, 1, 3.5)

        # Raggruppa gli eventi con stesso giorno+ora (es. CPI m/m + CPI y/y +
        # Core CPI m/m + Core CPI y/y, tutti alle 14:30 — stesso rilascio,
        # quattro angolazioni dello stesso dato): un solo alert con un bias
        # combinato invece di N alert scollegati e a volte contraddittori
        # (bug segnalato dall'utente l'11/09/2026: 4 messaggi consecutivi con
        # bias NEUTRO/BUY/SELL/SELL tutti per le 14:30, screenshot Telegram).
        # get_upcoming_events() ordina già per date+time, il raggruppamento
        # preserva l'ordine.
        # Raggruppa anche per valuta, non solo data+ora: dal 2026-09-16 gli
        # eventi non sono più tutti USD (vedi analyzer._passes_news_filter)
        # — un evento USD e uno GBP che cadono nello stesso minuto IT non
        # sono lo stesso rilascio e non vanno mai combinati in un solo bias
        # (analyze_combined_macro_event assume esplicitamente che il gruppo
        # sia "lo stesso rilascio", vero solo se è anche la stessa valuta).
        groups: dict = {}
        for ev in events:
            gk = f"{ev['date']}_{ev['time']}_{ev.get('currency', 'USD')}"
            groups.setdefault(gk, []).append(ev)

        for group_key, group_events in groups.items():
            ev = group_events[0]  # rappresentante del gruppo per data/ora/titolo singolo
            combined_title = " + ".join(e["title"] for e in group_events)
            try:
                ev_dt     = TIMEZONE.localize(
                    datetime.strptime(f"{ev['date']} {ev['time']}", "%Y-%m-%d %H:%M")
                )
                mins_away = (ev_dt - now).total_seconds() / 60
            except Exception:
                continue

            # PRE-EVENTO (30 min prima)
            if 25 <= mins_away <= 35 and group_key not in _sent_event_alerts:
                price = await get_current_price_async()
                event_currency = ev.get("currency", "USD")
                related_context = _find_related_macro_context(event_currency, ev_dt, group_key)
                analysis = await asyncio.to_thread(
                    analyze_combined_macro_event, group_events, price, related_context
                )
                # analyze_combined_macro_event ora ritorna solo "Bias: X\nMotivo: Y"
                # (niente più pip/livelli/TP/SL inventati — vedi news_analyst.py),
                # UN SOLO bias anche quando il gruppo ha più indicatori.
                bias, motivo = "NEUTRO", ""
                for line in analysis.splitlines():
                    if line.lower().startswith("bias:"):
                        bias = line.split(":", 1)[1].strip().upper()
                    elif line.lower().startswith("motivo:"):
                        motivo = line.split(":", 1)[1].strip()

                # Bias evento (reazione immediata) vs bias post-evento
                # (assestamento ~10-15 min) — richiesta esplicita dell'utente
                # 2026-09-10 da un caso reale (Core PPI m/m): il bot aveva UN
                # solo bias e un solo controllo, e ha detto "NON CONFERMATO"
                # anche quando la reazione immediata era stata corretta (+100
                # pip nella direzione giusta) e solo l'assestamento successivo
                # si era invertito. Stesso valore di bias per ora (nessun
                # modello statistico valida oggi una vera inversione — vedi
                # feedback_conservative_validation_standard) ma verificati
                # SEPARATAMENTE nel messaggio POST-evento con due finestre di
                # misurazione indipendenti (vedi snapshot immediata sotto).
                bias_line = f"⚡ Bias evento (reazione immediata): *{bias}*"
                if motivo:
                    bias_line += f"\n_{_escape_md(motivo)}_"
                bias_line += f"\n🕒 Bias post-evento (assestamento ~10-15 min): *{bias}*"

                _pre_event_bias[group_key] = {
                    "bias": bias, "price": price, "price_immediate": None,
                    "title": combined_title, "currency": event_currency,
                    "event_dt": ev_dt.isoformat(),
                }

                # Tracciamento permanente per la dashboard (2026-09-16,
                # richiesta esplicita: "aggiungi nella dashboard gli eventi
                # che si verificano a mercato e se il bot li prende") —
                # separato da _pre_event_bias sopra, che è solo stato di
                # lavoro temporaneo (pruned a 50 voci, vive in bot_state).
                # Fuori dal try/except principale non serve: un fallimento
                # qui non deve mai bloccare l'invio dell'alert Telegram,
                # che è la parte che conta davvero per l'utente in tempo
                # reale — l'evento indicizzato in dashboard è un di più.
                try:
                    await asyncio.to_thread(
                        save_macro_event_pre, group_key, combined_title, event_currency,
                        ev.get("impact", "HIGH"), ev_dt.isoformat(), price, bias,
                    )
                except Exception as e:
                    logger.warning(f"save_macro_event_pre fallita per {group_key}: {e}")

                # Chiusura protettiva dei trade aperti in direzione opposta
                # al bias dell'evento in arrivo. Trovato in diretta il
                # 2026-09-04 sull'NFP: un BUY 4h aperto dal giorno prima è
                # stato preso in pieno dallo SL 2 minuti dopo il rilascio,
                # mentre il bias pre-evento (SELL, poi confermato dal
                # movimento reale) andava nella direzione opposta — non ha
                # senso restare esposti controtrend a un catalizzatore
                # programmato quando il bot ha già una view direzionale su
                # quell'evento. Un pending non ancora attivato si cancella
                # (nessun capitale reale a rischio); un trade già attivo si
                # chiude al prezzo attuale con un R reale calcolato in
                # proporzione a entry/sl/exit (non forzato a 0 come un
                # CANCELLED — c'era rischio vero, va contabilizzato). Solo
                # se il bias è direzionale: un NEUTRO non dà alcuna
                # indicazione su cui agire, restano tutti aperti.
                if bias in ("BUY", "SELL"):
                    for open_trade in load_all_active_trades():
                        trade_signal = open_trade.get("signal")
                        if trade_signal not in ("BUY", "SELL") or trade_signal == bias:
                            continue
                        trade_id = open_trade.get("trade_id")
                        tf_label = TF_LABEL.get(open_trade.get("timeframe", ""), "?")
                        title_safe = _escape_md(combined_title)
                        if open_trade.get("activated"):
                            if not close_trade(
                                trade_id, "CLOSED_EARLY", price,
                                f"Chiuso in anticipo: controtrend rispetto al bias pre-evento ({combined_title})",
                            ):
                                continue
                            pips = calculate_trade_pips(trade_signal, open_trade.get("entry"), price)
                            protect_msg = (
                                f"🛡️ *CHIUSURA PROTETTIVA PRE-EVENTO*\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"📍 {trade_signal} [{tf_label}] @ ${_fmt(open_trade.get('entry'))}\n"
                                f"💰 Chiuso a: ${_fmt(price)} ({pips:+.1f} pips)\n"
                                f"📅 Evento tra ~30 min: *{title_safe}*\n"
                                f"🎯 Bias evento: *{bias}* — il trade era controtrend\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"ID: `{trade_id}`"
                            )
                        else:
                            if not close_trade(
                                trade_id, "CANCELLED", price,
                                f"Pending cancellato: controtrend rispetto al bias pre-evento ({combined_title})",
                            ):
                                continue
                            protect_msg = (
                                f"🛡️ *PENDING CANCELLATO PRE-EVENTO*\n"
                                f"📍 {trade_signal} [{tf_label}] @ ${_fmt(open_trade.get('entry'))}\n"
                                f"📅 Evento tra ~30 min: *{title_safe}* — bias {bias}, trade controtrend\n"
                                f"ID: `{trade_id}`"
                            )
                        try:
                            await bot.send_message(chat_id=CHAT_ID, text=protect_msg, parse_mode="Markdown")
                        except Exception as e:
                            logger.error(f"Notifica chiusura protettiva fallita per {trade_id}: {e}")

                # FIX: ev['title'] (dal calendario esterno) non era mai
                # escapato qui, a differenza di send_morning_report e
                # get_macro_briefing che lo fanno già per lo stesso campo —
                # un titolo con caratteri Markdown speciali poteva far
                # fallire il send. E il dedup veniva marcato PRIMA del send:
                # un fallimento (es. proprio per questo) perdeva l'alert
                # (incluso il blackout trading) per sempre, senza retry al
                # giro successivo. Ora si marca solo dopo un send riuscito.
                #
                # Con più indicatori nello stesso gruppo, ciascuno mostra la
                # propria previsione/precedente (il dato grezzo per
                # indicatore resta utile), ma sotto c'è un solo bias combinato.
                if len(group_events) == 1:
                    dettagli = f"📊 Prev: `{ev.get('forecast','N/A')}` | Prec: `{ev.get('previous','N/A')}`\n"
                else:
                    dettagli = "".join(
                        f"📊 *{_escape_md(e['title'])}*: Prev `{e.get('forecast','N/A')}` | Prec `{e.get('previous','N/A')}`\n"
                        for e in group_events
                    )
                # Emoji impatto + valuta: dal 2026-09-16 il gruppo non è più
                # sempre "USD rosso" (vedi analyzer._passes_news_filter), va
                # detto esplicitamente cosa si sta guardando.
                impact_emoji = "🔴" if ev.get("impact", "HIGH") == "HIGH" else "🟠"
                currency_tag = ev.get("currency", "USD")
                # Struttura di mercato (liquidità/FVG/order block, 2026-09-17,
                # richiesta esplicita dell'utente): SOLO testo informativo in
                # più, calcolato in modo deterministico su dati OHLC reali —
                # vedi market_structure.py. Non entra nel prompt che genera
                # bias_line sopra: non deve mai influenzare il bias né le
                # chiusure protettive, solo arricchire quello che si legge.
                tech_snapshot = (
                    "" if _in_quiet_hours(now)
                    else await asyncio.to_thread(get_market_structure_snapshot, price)
                )
                tech_block = f"\n━━━━━━━━━━━━━━━━━━━━\n{tech_snapshot}" if tech_snapshot else ""
                msg = (
                    f"⚠️ *ALERT MACRO — TRA 30 MINUTI*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 *{_escape_md(combined_title)}*\n"
                    f"{impact_emoji} Impatto: *{ev.get('impact','HIGH')}* | Valuta: *{currency_tag}*\n"
                    f"🕐 Orario: *{ev['time']} IT*\n"
                    f"{dettagli}"
                    f"💰 XAU/USD: *${_fmt(price)}*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🚫 *BLACKOUT TRADING ATTIVO*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"{bias_line}"
                    f"{tech_block}"
                )
                if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
                if not _in_quiet_hours(now):
                    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                _sent_event_alerts[group_key] = True

            # SNAPSHOT IMMEDIATA (per verificare "bias evento" nel post-evento)
            # Finestra -7/-1 (1-7 minuti dopo il rilascio) invece di "1-2
            # minuti" come discusso inizialmente: check_macro_alerts gira
            # ogni 5 minuti (vedi scheduler in main()), una finestra più
            # stretta di 5 minuti rischierebbe di cadere per intero tra due
            # cicli consecutivi e non catturare mai nulla. -7/-1 resta
            # comunque nettamente prima della finestra di assestamento
            # (-15/-8), distinguendo davvero le due fasi. Silenziosa (nessun
            # messaggio), aggiorna solo lo stato in memoria.
            pre_for_snapshot = _pre_event_bias.get(group_key)
            if (pre_for_snapshot is not None and pre_for_snapshot.get("price_immediate") is None
                    and -7 <= mins_away <= -1):
                pre_for_snapshot["price_immediate"] = await get_current_price_async()

            # POST-EVENTO (8-15 min dopo, tollerante fino a 60 min)
            #
            # BUG REALE TROVATO IL 2026-09-16: la finestra stretta originale
            # (-15/-8, larga solo 7 minuti) è stata saltata del tutto per un
            # FOMC delle 20:00 — un deploy (il bot gira su Railway, un
            # riavvio ricarica lo stato ma azzera anche il timer dello
            # scheduler APScheduler a intervalli fissi) è avvenuto alle
            # 20:11:56, e il primo giro di check_macro_alerts dopo il
            # riavvio è scattato solo alle 20:16:56 (l'intervallo è di 5
            # minuti) — a quel punto mins_away era già -16.9, fuori dalla
            # finestra per soli ~2 minuti. Nessuna eccezione, nessun bug di
            # calcolo: solo una finestra troppo stretta per sopravvivere a
            # un riavvio proprio nel mezzo. Stesso identico pattern già
            # risolto il 2026-09-06 per la previsione statistica sotto
            # (allargata da 7 minuti a 3 ore per lo stesso motivo) — qui non
            # era mai stato applicato. Allargata a 60 minuti (via
            # POST_EVENT_MAX_DELAY_MIN sotto): stesso post_key quindi invia
            # comunque una sola volta, il testo del resoconto ora riporta i
            # minuti trascorsi REALI invece di "10-15 min" fisso (che
            # sarebbe stato fuorviante se il messaggio arriva in ritardo).
            # 30 minuti (non di più): abbondante contro il gap di 5 minuti
            # dello scheduler dopo un riavvio, ma abbastanza corto da restare
            # ben distinto dalla finestra di 3 ore della previsione
            # statistica sotto (le due finestre si sovrappongono di
            # proposito solo per un margine ridotto, non del tutto).
            POST_EVENT_MAX_DELAY_MIN = 30
            post_key = f"POST_{group_key}"
            if -POST_EVENT_MAX_DELAY_MIN <= mins_away <= -8 and post_key not in _sent_post_event_alerts:
                price = await get_current_price_async()
                minutes_elapsed = round(-mins_away)
                # Resoconto oggettivo: bias previsto pre-evento vs movimento
                # di prezzo reale — confronto aritmetico sui prezzi, non una
                # seconda opinione dell'AI. Due blocchi separati (bias
                # evento / bias post-evento), ciascuno con la propria
                # finestra di misurazione, invece di un singolo verdetto che
                # confondeva le due fasi (vedi commento sopra).
                # .get() e non .pop(): con il dedup marcato solo dopo un
                # send riuscito, un fallimento fa ritentare questo blocco al
                # giro successivo — un .pop() qui l'avrebbe già consumato al
                # primo tentativo fallito, perdendo il bias pre-evento anche
                # se poi disponibile per il retry.
                pre = _pre_event_bias.get(group_key)
                if pre:
                    pre_bias        = pre["bias"]
                    price_immediate = pre.get("price_immediate")
                    change_evento   = None
                    status_evento   = None

                    if price_immediate is None:
                        blocco_evento = (
                            f"⚡ Bias evento: *{pre_bias}* (era ${pre['price']})\n"
                            f"_Snapshot immediata non disponibile (bot riavviato nel frattempo)._"
                        )
                    else:
                        change_evento = price_immediate - pre["price"]
                        esito_evento, segno_ev, status_evento = _confirm_bias(pre_bias, change_evento)
                        blocco_evento = (
                            f"⚡ Bias evento: *{pre_bias}* (era ${pre['price']})\n"
                            f"{esito_evento}\n"
                            f"📐 Variazione (1-7 min): *{segno_ev}{change_evento:.2f}$*"
                        )

                    change_post = price - pre["price"]
                    esito_post, segno_post, status_post = _confirm_bias(pre_bias, change_post)
                    blocco_post = (
                        f"🕒 Bias post-evento: *{pre_bias}* (era ${pre['price']})\n"
                        f"{esito_post}\n"
                        f"📐 Variazione a ~{minutes_elapsed} min: *{segno_post}{change_post:.2f}$*"
                    )

                    resoconto = f"{blocco_evento}\n\n{blocco_post}"
                else:
                    resoconto = "_Bias pre-evento non disponibile (bot riavviato nel frattempo)._"

                # Struttura di mercato DOPO l'evento (vedi stesso blocco nel
                # pre-evento sopra per il perché resta solo testo informativo).
                tech_snapshot_post = (
                    "" if _in_quiet_hours(now)
                    else await asyncio.to_thread(get_market_structure_snapshot, price)
                )
                tech_block_post = f"\n━━━━━━━━━━━━━━━━━━━━\n{tech_snapshot_post}" if tech_snapshot_post else ""

                # Stesso fix del blocco pre-evento: titolo escapato, dedup
                # marcato solo dopo il send principale riuscito (i due send
                # successivi — previsione statistica e news — sono già
                # protetti da try/except propri e non condizionano post_key).
                msg_post = (
                    f"📊 *POST-EVENTO — {_escape_md(combined_title)}*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 XAU/USD: *${_fmt(price)}*\n"
                    f"🚦 *Blackout terminato — trading riaperto*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"{resoconto}"
                    f"{tech_block_post}"
                )
                if len(msg_post) > 4000: msg_post = msg_post[:3950] + "\n_[Troncato]_"
                if not _in_quiet_hours(now):
                    await bot.send_message(chat_id=CHAT_ID, text=msg_post, parse_mode="Markdown")
                _sent_post_event_alerts[post_key] = True

                # Tracciamento permanente per la dashboard, simmetrico al
                # save_macro_event_pre nel blocco pre-evento sopra — anche
                # qui fuori da qualsiasi percorso che condiziona l'invio
                # Telegram: un fallimento di scrittura sul DB non deve mai
                # far perdere o ritardare il resoconto già inviato.
                if pre:
                    try:
                        await asyncio.to_thread(
                            save_macro_event_post, group_key, price_immediate, change_evento,
                            status_evento, price, change_post, status_post, minutes_elapsed,
                        )
                    except Exception as e:
                        logger.warning(f"save_macro_event_post fallita per {group_key}: {e}")

                _pre_event_bias.pop(group_key, None)

            # Fase 5 progetto dati storici: previsione statistica reale, solo
            # per le serie validate in Fase 4 (oggi: Core CPI m/m). Silenziosa
            # se non applicabile (evento diverso, forecast non numerico) —
            # mai forzata. Resta PER-SINGOLO-EVENTO (non per gruppo): il
            # modello è specifico a un singolo event_name — es. solo Core CPI
            # m/m ha un modello deployato anche quando esce insieme ad altri
            # 3 indicatori CPI nello stesso gruppo.
            #
            # FIX (2026-09-06): prima questo controllo viveva DENTRO il blocco
            # post-evento sopra, con la stessa finestra stretta di 7 minuti
            # (-15/-8) e marcato "fatto" (post_key) subito dopo l'invio del
            # resoconto principale — anche se FRED non aveva ancora pubblicato
            # il dato (aggiornamento reale fino a un'ora, non minuti, per
            # CPI/Unemployment Claims), il controllo non veniva MAI ripetuto:
            # la previsione rischiava di non scattare quasi mai nella pratica.
            # Ora è una condizione indipendente, con una finestra propria di
            # 3 ore e una chiave di dedup separata (stat_key): ricontrolla
            # FRED a ogni giro dello scheduler (5 min) finché non trova il
            # dato o la finestra scade — mai bloccata dall'invio (o mancato
            # invio) del resoconto principale.
            for ev_single in group_events:
                stat_key = f"STAT_{ev_single['date']}_{ev_single['time']}_{ev_single['title']}"
                if -180 <= mins_away <= 0 and stat_key not in _sent_stat_prediction:
                    try:
                        from macro_predictor import predict_reaction, format_prediction
                        prediction = await asyncio.to_thread(
                            predict_reaction, ev_single["title"], ev_single.get("forecast", "N/A")
                        )
                        if prediction:
                            if not _in_quiet_hours(now):
                                await bot.send_message(
                                    chat_id=CHAT_ID, text=format_prediction(prediction), parse_mode="Markdown"
                                )
                            _sent_stat_prediction[stat_key] = True

                            # Chiusura protettiva basata sulla previsione statistica
                            # (2026-09-09, richiesta esplicita dell'utente — "solo
                            # protezione, niente nuovi trade"): stesso principio
                            # già in produzione per il bias LLM pre-evento (vedi
                            # sopra), qui esteso al caso in cui il bias LLM sia
                            # NEUTRO/debole ma la previsione statistica (calibrata
                            # su dati storici reali, non un'opinione dell'AI) sia
                            # forte e discorde da un trade già aperto. MAI apre un
                            # nuovo trade da sola — solo protegge quelli esistenti.
                            # CONFIRM_THRESHOLD_USD (stessa soglia usata sopra per
                            # il resoconto post-evento) evita di agire su previsioni
                            # troppo piccole per essere significative.
                            predicted_usd = prediction.get("predicted_reaction_usd", 0.0)
                            if abs(predicted_usd) >= CONFIRM_THRESHOLD_USD:
                                predicted_signal = "BUY" if predicted_usd > 0 else "SELL"
                                price = await get_current_price_async()
                                for open_trade in load_all_active_trades():
                                    trade_signal = open_trade.get("signal")
                                    if trade_signal not in ("BUY", "SELL") or trade_signal == predicted_signal:
                                        continue
                                    trade_id = open_trade.get("trade_id")
                                    tf_label = TF_LABEL.get(open_trade.get("timeframe", ""), "?")
                                    title_safe = _escape_md(ev_single["title"])
                                    reason_note = (
                                        f"Chiuso in anticipo: previsione statistica ({ev_single['title']}) "
                                        f"discorde dal trade — {predicted_signal} atteso {predicted_usd:+.2f}$"
                                    )
                                    if open_trade.get("activated"):
                                        if not close_trade(trade_id, "CLOSED_EARLY", price, reason_note):
                                            continue
                                        pips = calculate_trade_pips(trade_signal, open_trade.get("entry"), price)
                                        protect_msg = (
                                            f"🛡️ *CHIUSURA PROTETTIVA — PREVISIONE STATISTICA*\n"
                                            f"━━━━━━━━━━━━━━━━━━━━\n"
                                            f"📍 {trade_signal} [{tf_label}] @ ${_fmt(open_trade.get('entry'))}\n"
                                            f"💰 Chiuso a: ${_fmt(price)} ({pips:+.1f} pips)\n"
                                            f"📐 *{title_safe}*: reazione attesa *{predicted_signal}* ({predicted_usd:+.2f}$, n={prediction.get('n_historical')})\n"
                                            f"━━━━━━━━━━━━━━━━━━━━\n"
                                            f"ID: `{trade_id}`"
                                        )
                                    else:
                                        if not close_trade(trade_id, "CANCELLED", price, reason_note):
                                            continue
                                        protect_msg = (
                                            f"🛡️ *PENDING CANCELLATO — PREVISIONE STATISTICA*\n"
                                            f"📍 {trade_signal} [{tf_label}] @ ${_fmt(open_trade.get('entry'))}\n"
                                            f"📐 *{title_safe}*: reazione attesa *{predicted_signal}* ({predicted_usd:+.2f}$, n={prediction.get('n_historical')})\n"
                                            f"ID: `{trade_id}`"
                                        )
                                    try:
                                        await bot.send_message(chat_id=CHAT_ID, text=protect_msg, parse_mode="Markdown")
                                    except Exception as e:
                                        logger.error(f"Notifica chiusura protettiva (previsione statistica) fallita per {trade_id}: {e}")
                    except Exception as e:
                        logger.debug(f"[{ev_single['title']}] Previsione statistica non disponibile: {e}")

        # Pulizia memory leak
        if len(_sent_event_alerts) > 50:
            _sent_event_alerts = dict(list(_sent_event_alerts.items())[-30:])
        if len(_sent_post_event_alerts) > 50:
            _sent_post_event_alerts = dict(list(_sent_post_event_alerts.items())[-30:])
        if len(_sent_stat_prediction) > 50:
            _sent_stat_prediction = dict(list(_sent_stat_prediction.items())[-30:])
        if len(_pre_event_bias) > 50:
            for k in list(_pre_event_bias.keys())[:-30]:
                _pre_event_bias.pop(k, None)

        # Persisti lo stato aggiornato: sopravvive a un riavvio/deploy nel
        # mezzo di una finestra pre/post-evento (vedi load_macro_alert_state).
        await asyncio.to_thread(
            save_macro_alert_state,
            _sent_event_alerts, _sent_post_event_alerts, _pre_event_bias,
            _sent_stat_prediction,
        )

    except Exception as e:
        logger.error(f"Errore check_macro_alerts: {e}")


async def send_morning_report(bot: Bot):
    try:
        from analyzer import get_economic_events
        today     = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
        price, news, sentiment, cal = await asyncio.gather(
            get_current_price_async(),
            asyncio.to_thread(get_extended_news),
            asyncio.to_thread(get_news_sentiment),
            asyncio.to_thread(get_economic_events, broad=True),
        )
        s_label = sentiment.get("label","NEUTRAL")
        s_emoji = "🟢" if s_label == "BULLISH" else "🔴" if s_label == "BEARISH" else "⚪"
        events  = cal.get("events",[])

        if events:
            events_txt = "📅 *EVENTI MACRO OGGI:*\n" + "\n".join(
                f"• {'🔴' if ev.get('impact','HIGH')=='HIGH' else '🟠'} "
                f"{_escape_md(ev.get('title','?'))}"
                f"{' [' + ev.get('currency','') + ']' if ev.get('currency','USD') not in ('USD','US') else ''}"
                f" — {ev.get('time','?')} IT"
                for ev in events[:8]
            )
        else:
            events_txt = "Nessun evento macro ad alto impatto oggi."

        # Testo generato da Groq: può contenere caratteri Markdown non
        # bilanciati (underscore, asterisco...) — va escapato prima di
        # finire in un messaggio parse_mode="Markdown", altrimenti l'intero
        # report mattutino unico non arriva.
        bias_txt = _escape_md(await asyncio.to_thread(get_bias_briefing, news, price))

        ny_time = _ny_open_time_it()
        msg = (
            f"🌅 *BUONGIORNO — {today}*\n"
            f"💰 XAU/USD: *${_fmt(price)}*\n"
            f"{s_emoji} Sentiment: *{s_label}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{events_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{bias_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *KILL ZONES OGGI:*\n"
            f"• 🇬🇧 Londra: 09:00–11:00\n"
            f"• 🇺🇸 NY Open: {ny_time}–16:00\n"
            f"• 🎯 Primaria: 15:30–17:30\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Usa /macro per analisi evento specifico_\n"
            f"_Buon trading! 📈_"
        )
        if len(msg) > 4000: msg = msg[:3950] + "\n_[Troncato]_"
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        logger.info("Report mattutino inviato")
    except Exception as e:
        logger.error(f"Errore report mattutino: {e}")


async def send_daily_report(bot: Bot):
    try:
        today  = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
        all_t  = _get_closed_trades()
        today_t = _get_trades_today()

        # Stesso criterio di /report e della dashboard (is_decisive_win in
        # trade_manager.py) — prima escludeva ogni WIN_BE, anche quelli con
        # TP1 già raggiunto, e divergeva dagli altri due.
        wins_today_l   = [t for t in today_t if is_decisive_win(t)]
        losses_today_l = [t for t in today_t if t.get("result") == "LOSS"]
        decisivi_today = wins_today_l + losses_today_l
        wins_t  = len(wins_today_l)
        loss_t  = len(losses_today_l)
        pnl_t   = sum(t.get("pnl_r") or 0 for t in today_t if t.get("status") == "CLOSED")
        wr_t    = round(wins_t / len(decisivi_today) * 100, 1) if decisivi_today else 0

        wins_all_l   = [t for t in all_t if is_decisive_win(t)]
        losses_all_l = [t for t in all_t if t.get("result") == "LOSS"]
        decisivi_all = wins_all_l + losses_all_l
        wins_all = len(wins_all_l)
        wr_all   = round(wins_all / len(decisivi_all) * 100, 1) if decisivi_all else 0
        pnl_all  = sum(t.get("pnl_r") or 0 for t in all_t if t.get("status") == "CLOSED")

        signals_txt = ""
        for t in today_t:
            res = t.get("result","?")
            e   = (
                "✅" if "WIN" in res and res != "WIN_BE" else
                "⚖️" if res == "WIN_BE" else
                "🛡️" if res == "CLOSED_EARLY" else
                "❌" if res == "LOSS" else
                "🔵"
            )
            tf  = TF_LABEL.get(t.get("timeframe",""), "?")
            signals_txt += f"{e} {t.get('signal','?')} [{tf}] @ ${t.get('entry','?')} — {res}\n"
        if not signals_txt:
            signals_txt = "Nessun trade oggi.\n"

        msg = (
            f"🌙 *REPORT GIORNALIERO — {today}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ {wins_t} Win | ❌ {loss_t} Loss | P&L: *{pnl_t:+.1f}R*\n"
            f"📈 Win Rate oggi: *{wr_t}%*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Trade di oggi:*\n{signals_txt}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Storico:* WR *{wr_all}%* | P&L *{pnl_all:+.1f}R* | Tot. *{len(decisivi_all)}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Buona notte! 🌙_"
        )
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore report serale: {e}")


# ═══════════════════════════════════════════════
# AUTO CHECK — Pipeline multi-timeframe
# ═══════════════════════════════════════════════

# Sopra questa soglia senza NESSUN fetch candele riuscito (su nessun
# timeframe, nessuna fonte), il bot avvisa che non può generare segnali —
# prima andava semplicemente in SKIP in silenzio: un blackout Yahoo+Stooq
# poteva durare ore senza che nessuno se ne accorgesse (visto in produzione
# il 1 settembre 2026, notato solo controllando i log a mano).
BLIND_THRESHOLD_SECONDS = 900  # 15 minuti
_blind_alert_sent = False


async def _check_data_blindness(bot: Bot):
    """Avvisa una volta se il bot resta senza dati candele per troppo tempo, e una volta quando si riprende."""
    global _blind_alert_sent
    elapsed = seconds_since_last_data_success()

    if elapsed > BLIND_THRESHOLD_SECONDS:
        if not _blind_alert_sent:
            _blind_alert_sent = True
            minuti = int(elapsed / 60)
            try:
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"⚠️ *Nessun dato candele da {minuti} minuti*\n"
                        f"Yahoo, Stooq e Twelve Data sembrano irraggiungibili insieme — "
                        f"nessun nuovo segnale possibile finché non tornano disponibili. "
                        f"Il monitoraggio dei trade già aperti (SL/TP/BE) non è influenzato, "
                        f"usa una fonte prezzo separata."
                    ),
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Invio alert blackout dati fallito")
    elif _blind_alert_sent:
        _blind_alert_sent = False
        try:
            await bot.send_message(
                chat_id=CHAT_ID,
                text="✅ Dati candele ripristinati — i segnali riprendono normalmente.",
            )
        except Exception:
            logger.exception("Invio alert ripristino dati fallito")


async def auto_check_all_timeframes(bot: Bot):
    """
    Controlla tutti i timeframe ogni 5 minuti IN SEQUENZA.

    NOTA: la pipeline parallela (asyncio.gather) bruciava ~40 API credits
    in 1 secondo contro il limite di 8/min di Twelve Data gratuito,
    causando il fallimento di tutti i TF. Torniamo alla pipeline sequenziale
    ma con stagger di 2 secondi tra TF per non colpire il rate limit.

    La cache delle candele in get_data() evita chiamate duplicate sullo stesso TF.
    """
    if not is_market_open():
        return
    if is_bot_paused():
        return

    for tf in ALL_TIMEFRAMES:
        try:
            await _check_single_timeframe(bot, tf)
        except DuplicateSetupError:
            logger.debug("[%s] setup duplicato intercettato dal DB", tf)
        except Exception as e:
            logger.error(f"[{tf}] Errore pipeline: {e}")

        # Stagger 2s tra TF per non esaurire il rate limit API
        await asyncio.sleep(2)

    # Cecchini (2026-09-09) — motori separati, non fanno parte di ALL_
    # TIMEFRAMES perche' non sono veri timeframe/interval dati, e le loro
    # regole (NO_EDGE_TIMEFRAMES, ecc.) non gli si applicano. Stessa
    # funzione generica _check_single_timeframe per ognuno, vedi analyzer.
    # SNIPER_CONFIGS/_sniper_analyze per la logica di segnale separata di
    # ciascuno. Due cecchini possono condividere lo stesso vero timeframe
    # (es. i due su 1h) senza mai interferire tra loro: ognuno ha la sua
    # chiave sintetica propria per dedup/cooldown/trade aperti.
    for sniper_tf in SNIPER_CONFIGS:
        try:
            await _check_single_timeframe(bot, sniper_tf)
        except DuplicateSetupError:
            logger.debug("[%s] setup duplicato intercettato dal DB", sniper_tf)
        except Exception as e:
            logger.error(f"[{sniper_tf}] Errore pipeline: {e}")
        await asyncio.sleep(2)

    await _check_data_blindness(bot)


async def _check_single_timeframe(bot: Bot, tf: str):
    """Esegue la pipeline su un singolo timeframe e invia il segnale se valido."""
    try:
        state = await run_pipeline(timeframe=tf)

        if state.final_decision != "EXECUTE":
            if state.final_decision == "WAIT":
                logger.info(f"[{tf}] WAIT — {state.decision_reason}")
            return

        # Trade già aperto su questo TF?
        if has_open_trade_on_timeframe(tf):
            logger.debug(f"[{tf}] Trade già aperto, skip")
            return

        # FIX (2026-09-07): senza questo, lo stesso setup (struttura
        # invariata, entry ricalcolata ad ogni giro con un prezzo
        # leggermente diverso) veniva riproposto e ricancellato ogni 5
        # minuti (frequenza di questo stesso job) - spam di segnali-poi-
        # annullati in chat e dashboard, segnalato dall'utente.
        if recently_cancelled_on_timeframe(tf):
            logger.debug(f"[{tf}] Annullato di recente, cooldown attivo, skip")
            return

        # Entry/SL/TP sono calcolati da candele GC=F (futures) — il prezzo
        # che l'utente vede sul suo broker (spot) può divergere anche di
        # decine di dollari, e lo scarto si muove nel tempo (osservato in
        # produzione: un trade Daily con TP1 mai segnalato perché GC=F non
        # era ancora arrivato al livello, mentre lo spot l'aveva già superato
        # da un pezzo). Catturiamo qui lo scarto GC=F-spot e traduciamo tutti
        # i livelli in "equivalente spot": messaggio Telegram e monitoraggio
        # (trade_manager._monitor_single) restano coerenti con quello che
        # l'utente vede davvero sul suo grafico. Se gold-api.com non risponde
        # in questo istante, basis resta 0 — nessuna traduzione, comportamento
        # invariato rispetto a prima.
        basis = 0.0
        try:
            from trade_manager import (
                _fetch_price_goldapi, _fetch_price_twelvedata, is_twelvedata_price_blocked,
            )
            spot_price = await asyncio.to_thread(_fetch_price_goldapi)
            if spot_price > 0 and state.current_price > 0:
                candidate_basis = state.current_price - spot_price

                # Verifica incrociata prima di fidarsi del basis: gold-api.com
                # è una fonte piccola, gratuita e senza SLA, e una sua lettura
                # anomala isolata resterebbe congelata per tutta la vita del
                # trade (visto in produzione l'1 settembre 2026 — trade 76:
                # basis di $54.60 rivelatosi sbagliato rispetto al prezzo
                # reale sul broker dell'utente, con errore ereditato sia dal
                # messaggio SEGNALE sia dal monitoraggio). Due fonti spot
                # indipendenti per lo stesso istante dovrebbero essere vicine
                # tra loro (a differenza di GC=F-spot, che diverge per
                # costruzione): se non lo sono, la lettura non è affidabile —
                # meglio nessuna traduzione (comportamento originale) che una
                # traduzione probabilmente sbagliata. Se Twelve Data non è
                # disponibile (quota esaurita o errore), nessun controllo è
                # possibile: si procede come prima, senza bloccare la feature.
                confirmed = True
                if not is_twelvedata_price_blocked():
                    try:
                        spot_price_2 = await asyncio.to_thread(_fetch_price_twelvedata)
                        disagreement = abs(spot_price - spot_price_2)
                        if spot_price_2 > 0 and disagreement > MAX_SPOT_SOURCE_DISAGREEMENT_USD:
                            confirmed = False
                            logger.warning(
                                f"[{tf}] Basis GC=F-spot scartato: gold-api "
                                f"${spot_price:.2f} e Twelve Data ${spot_price_2:.2f} "
                                f"disaccordo di ${disagreement:.2f} (soglia "
                                f"${MAX_SPOT_SOURCE_DISAGREEMENT_USD:.0f})"
                            )
                    except Exception as e:
                        logger.debug(f"[{tf}] Controllo incrociato basis non riuscito: {e}")

                if confirmed:
                    basis = candidate_basis
                    state.entry -= basis
                    state.sl    -= basis
                    state.tp1   -= basis
                    state.tp2   -= basis
                    state.tp3   -= basis
                    if state.early_be_level:
                        state.early_be_level -= basis
        except Exception as e:
            logger.debug(f"[{tf}] Basis GC=F-spot non calcolabile: {e}")

        # Costruisci dict trade
        data = {
            "signal":     state.signal,
            "order_type": state.order_type,
            "entry":      state.entry,
            "sl":         state.sl,
            "tp1":        state.tp1,
            "tp2":        state.tp2,
            "tp3":        state.tp3,
            "prob":       state.prob,
            "regime":     state.regime,
            "timeframe":  tf,
            "price":      state.current_price,
            "risk_pct":   state.risk_pct,
            "strategies": state.strategies,
            "data_timestamp": state.data_timestamp,
            "price_basis": basis,
            "early_be_level": state.early_be_level,
        }
        # Tag distinto per i trade dei cecchini (2026-09-09): permette di
        # filtrarli separatamente nel DB/dashboard (WHERE strategy_version=
        # 'sniper_<tf>_v1') senza una migrazione di schema — open_trade()
        # calcola il fingerprint normale solo se questo campo manca.
        if tf in SNIPER_CONFIGS:
            data["strategy_version"] = f"sniper_{tf}_v1"
        data["setup_key"] = build_setup_key(data)
        if was_setup_seen(data["setup_key"]):
            logger.debug("[%s] setup già registrato, skip", tf)
            return

        # Costruisci messaggio PRIMA di aprire il trade
        tf_label_str = TF_LABEL.get(tf, tf.upper())
        msg = f"🚨 *SEGNALE {tf_label_str}!*\n\n" + format_pipeline_report(state)

        # Tronca se troppo lungo per Telegram
        if len(msg) > 4000:
            msg = msg[:3950] + "\n_[Troncato]_"

        # Prova a mandare il messaggio con retry (max 3 tentativi)
        sent = False
        last_err = None
        for attempt in range(3):
            try:
                await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                sent = True
                break
            except Exception as e:
                last_err = e
                logger.warning(f"[{tf_label_str}] Tentativo {attempt+1}/3 invio segnale fallito: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)

        if not sent:
            logger.error(f"[{tf_label_str}] Impossibile inviare segnale dopo 3 tentativi: {last_err}")
            # Non apriamo il trade se non riusciamo a notificarlo
            return

        # Apri trade nel DB solo DOPO conferma invio messaggio
        try:
            trade_id = open_trade(data)
            logger.info(
                f"[{tf_label_str}] EXECUTE: {state.signal} @ {state.entry} | "
                f"confidence={state.prob} | risk={state.risk_pct}% | id={trade_id}"
            )
        except Exception as e:
            logger.error(f"[{tf_label_str}] Segnale inviato ma open_trade fallito: {e}")
            # Notifica l'errore in chat
            try:
                # FIX: {e} (testo di eccezione, può contenere setup_key con
                # underscore/altri caratteri Markdown) andava in un messaggio
                # parse_mode="Markdown" senza escape — proprio questa
                # notifica (segnale annunciato ma trade mai aperto nel DB,
                # il caso "segnale fantasma" della sessione del 3 settembre)
                # poteva fallire silenziosamente nell'except sotto, lasciando
                # l'operatore senza alcun avviso del disallineamento.
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"⚠️ Segnale {tf_label_str} inviato ma errore apertura trade nel DB: {_escape_md(str(e))}",
                    parse_mode="Markdown"
                )
            except Exception:
                pass

    except DuplicateSetupError:
        logger.debug("[%s] setup duplicato intercettato dal DB", tf)
    except Exception as e:
        logger.error(f"[{tf}] Errore: {e}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

async def main():
    missing = [
        name
        for name, value in (
            ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
            ("CHAT_ID", CHAT_ID),
            ("TWELVE_API_KEY", os.environ.get("TWELVE_API_KEY", "")),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Configurazione mancante: {', '.join(missing)}")

    # Inizializza DB unico
    init_db()

    # Ripristina lo stato degli alert macro (pre/post-evento + bias
    # catturato) da prima di un eventuale riavvio/deploy — senza questo,
    # un deploy a metà finestra evento perde il bias pre-evento e il
    # post-evento non può più confermare/smentire la previsione.
    global _sent_event_alerts, _sent_post_event_alerts, _pre_event_bias, _sent_stat_prediction
    _macro_state = load_macro_alert_state()
    _sent_event_alerts      = _macro_state["sent_event_alerts"]
    _sent_post_event_alerts = _macro_state["sent_post_event_alerts"]
    _pre_event_bias         = _macro_state["pre_event_bias"]
    _sent_stat_prediction   = _macro_state["sent_stat_prediction"]

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("signal",    cmd_signal))
    app.add_handler(CommandHandler("m15",       cmd_m15))
    app.add_handler(CommandHandler("h1",        cmd_h1))
    app.add_handler(CommandHandler("h4",        cmd_h4))
    app.add_handler(CommandHandler("d1",        cmd_d1))
    app.add_handler(CommandHandler("trade",     cmd_trade))
    app.add_handler(CommandHandler("chiudi",    cmd_chiudi))
    app.add_handler(CommandHandler("news",      cmd_news))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("report",    cmd_report))
    app.add_handler(CommandHandler("risk",      cmd_risk))
    app.add_handler(CommandHandler("riprendisessione", cmd_riprendisessione))
    app.add_handler(CommandHandler("lotto",     cmd_lotto))
    app.add_handler(CommandHandler("regime",    cmd_regime))
    app.add_handler(CommandHandler("decisioni", cmd_decisions))
    # Weekly review domenica 20:30
    app.add_handler(CommandHandler("macro",     cmd_macro))
    app.add_handler(CommandHandler("backtest",  cmd_backtest))
    app.add_handler(CommandHandler("posttrade", cmd_posttrade))
    app.add_handler(CommandHandler("review",    cmd_review))
    app.add_handler(CommandHandler("learn",     cmd_learn))
    app.add_handler(CommandHandler("weekend",   cmd_weekend))
    app.add_handler(CommandHandler("pausa",     cmd_pausa))
    app.add_handler(CommandHandler("riattiva",  cmd_riattiva))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_free_text))

    scheduler = AsyncIOScheduler(
        timezone=TIMEZONE,
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        },
    )

    # Pipeline multi-TF ogni 5 minuti
    scheduler.add_job(
        auto_check_all_timeframes, "interval", minutes=5, args=[app.bot],
        id="auto_timeframes"
    )

    # Monitor SL/TP ogni 10 secondi (prezzo cachato → max 1 API call/10s)
    scheduler.add_job(
        _fast_monitor, "interval", seconds=10, args=[app.bot], id="trade_monitor"
    )

    # Report mattutino 8:30 — misfire_grace_time=1800 garantisce l'invio
    # anche se il bot si riavvia entro 30 minuti dalle 8:30
    scheduler.add_job(
        send_morning_report, "cron", hour=8, minute=30, args=[app.bot],
        id="morning_report",
        misfire_grace_time=1800,
    )

    # Report serale 20:00 — misfire_grace_time=3600 garantisce l'invio
    # anche se il bot era offline alle 20:00 (lo manda appena si riavvia entro 1h)
    scheduler.add_job(
        send_daily_report, "cron", hour=20, minute=0, args=[app.bot],
        id="daily_report",
        misfire_grace_time=3600,
    )

    # Alert macro ogni 5 minuti
    scheduler.add_job(
        check_macro_alerts, "interval", minutes=5, args=[app.bot],
        id="macro_alerts"
    )

    # Breaking news Fed (non programmate) ogni 2 minuti — prima 10, troppo
    # lento: un discorso Fed non programmato può muovere il prezzo in pochi
    # minuti e il bot lo segnalava solo al giro successivo, a mossa già
    # fatta (visto in produzione il 2026-09-03, discorso Waller). La
    # classificazione è a parole chiave (nessuna chiamata Groq se non trova
    # nulla di nuovo), quindi accorciare l'intervallo non aumenta i costi.
    scheduler.add_job(
        check_breaking_news_job, "interval", minutes=2, args=[app.bot],
        id="breaking_news"
    )

    # Weekly review domenica 20:30
    scheduler.add_job(
        _send_weekly_review, "cron", day_of_week="sun", hour=20, minute=30,
        args=[app.bot], id="weekly_review"
    )

    # Analisi weekend multi-TF domenica 19:00 — anteprima riapertura lunedì
    scheduler.add_job(
        _send_weekend_outlook, "cron", day_of_week="sun", hour=19, minute=0,
        args=[app.bot], id="weekend_outlook"
    )

    scheduler.start()

    logger.info("✅ GoldMind v2 avviato")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
