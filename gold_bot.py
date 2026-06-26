"""
Gold Trading Bot — Sistema Completo XAU/USD
Nuovo formato segnale: 3 TP, tipo ordine automatico, BE, indicazioni operative
SMC completo, multi-timeframe, sentiment, calendario economico
"""

import logging
import asyncio
import os
from datetime import datetime
import pytz
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import requests as req

from analyzer import full_analyze, get_news_sentiment
from trade_manager import (save_active_trade, load_active_trade,
                            clear_active_trade, monitor_active_trade)

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
DATABASE_URL   = os.environ.get("DATABASE_URL", "")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "d929b1d0334e4160872bbb1bef9fbb15")
CHECK_INTERVAL = 2
TIMEZONE       = pytz.timezone("Europe/Rome")
# ─────────────────────────────────────────────

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
last_signal = None


# ═══════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════

def get_db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    time TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    entry REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    sl REAL NOT NULL,
                    score INTEGER DEFAULT 0,
                    prob INTEGER DEFAULT 0,
                    regime TEXT DEFAULT '',
                    result TEXT DEFAULT 'pending'
                )
            """)
        conn.commit()
    logger.info("Database inizializzato")


def add_signal_to_db(data: dict):
    time = datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals
                   (time, signal, order_type, entry, tp1, tp2, tp3, sl, score, prob, regime)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (time, data["signal"], data["order_type"], data["entry"],
                 data["tp1"], data["tp2"], data["tp3"], data["sl"],
                 data.get("score", 0), data.get("prob", 0), data.get("regime", ""))
            )
        conn.commit()


def update_db_results(current_price: float):
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM signals WHERE result = 'pending'")
            for e in cur.fetchall():
                new_result = None
                if e["signal"] == "BUY":
                    if current_price >= e["tp3"]:   new_result = "WIN_TP3"
                    elif current_price >= e["tp2"]:  new_result = "WIN_TP2"
                    elif current_price >= e["tp1"]:  new_result = "WIN_TP1"
                    elif current_price <= e["sl"]:   new_result = "LOSS"
                elif e["signal"] == "SELL":
                    if current_price <= e["tp3"]:    new_result = "WIN_TP3"
                    elif current_price <= e["tp2"]:  new_result = "WIN_TP2"
                    elif current_price <= e["tp1"]:  new_result = "WIN_TP1"
                    elif current_price >= e["sl"]:   new_result = "LOSS"
                if new_result:
                    cur.execute("UPDATE signals SET result=%s WHERE id=%s", (new_result, e["id"]))
        conn.commit()


def compute_stats() -> dict:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT result FROM signals")
            rows = cur.fetchall()
    total   = len([r for r in rows if r["result"] != "pending"])
    wins    = len([r for r in rows if "WIN" in str(r["result"])])
    losses  = len([r for r in rows if r["result"] == "LOSS"])
    pending = len([r for r in rows if r["result"] == "pending"])
    winrate = round(wins / total * 100, 1) if total > 0 else 0
    return {"total": total, "wins": wins, "losses": losses, "pending": pending, "winrate": winrate}


def compute_daily_stats() -> dict:
    today = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM signals WHERE time LIKE %s", (f"{today}%",))
            rows = cur.fetchall()
    total   = len([r for r in rows if r["result"] != "pending"])
    wins    = len([r for r in rows if "WIN" in str(r["result"])])
    losses  = len([r for r in rows if r["result"] == "LOSS"])
    pending = len([r for r in rows if r["result"] == "pending"])
    winrate = round(wins / total * 100, 1) if total > 0 else 0
    return {"total": total, "wins": wins, "losses": losses,
            "pending": pending, "winrate": winrate, "signals": rows}


def get_recent_signals(limit: int = 5) -> list:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM signals WHERE result != 'pending' ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cur.fetchall()


# ═══════════════════════════════════════════════
# FILTRO ORARIO
# ═══════════════════════════════════════════════

def is_market_open() -> bool:
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5: return False
    if now.hour < 8 or now.hour >= 20: return False
    return True


def market_status_text() -> str:
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5: return "🔴 Mercato chiuso (weekend)"
    if now.hour < 8 or now.hour >= 20: return "🔴 Mercato chiuso (fuori orario 08:00–20:00)"
    return "🟢 Mercato aperto (08:00–20:00)"


# ═══════════════════════════════════════════════
# NOTIZIE
# ═══════════════════════════════════════════════

def get_gold_news() -> list:
    try:
        url    = "https://newsapi.org/v2/everything"
        params = {
            "q": "gold XAU price OR gold market OR fed interest rates",
            "language": "en", "sortBy": "publishedAt",
            "pageSize": 3, "apiKey": NEWS_API_KEY
        }
        r    = req.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok": return []
        news = []
        for a in data.get("articles", []):
            title = a.get("title", "")
            src   = a.get("source", {}).get("name", "")
            date  = a.get("publishedAt", "")[:10]
            if title and src:
                news.append(f"📰 *{src}* ({date})\n_{title}_")
        return news
    except Exception as e:
        logger.error(f"Errore notizie: {e}")
        return []


# ═══════════════════════════════════════════════
# FORMATO MESSAGGIO SEGNALE
# ═══════════════════════════════════════════════

def format_signal(data: dict) -> str:
    signal     = data["signal"]
    order_type = data["order_type"]
    entry      = data["entry"]
    sl         = data["sl"]
    tp1        = data["tp1"]
    tp2        = data["tp2"]
    tp3        = data["tp3"]
    be         = data["be"]
    prob       = data["prob"]
    score      = data["score"]
    max_score  = data.get("max_score", 80)
    rr1        = data["rr1"]
    rr2        = data["rr2"]
    rr3        = data["rr3"]
    adx        = data["adx"]
    rsi        = data["rsi"]
    regime     = data["regime"]
    structure  = data["structure"]
    pd_zone    = data.get("pd_zone", "N/A")
    mtf        = data.get("mtf_trends", {})
    notes      = data.get("notes", [])
    bos        = data.get("bos")
    choch      = data.get("choch")
    candle     = data.get("candle", "")
    sentiment  = data.get("sentiment", {})
    calendar   = data.get("calendar", {})
    sr         = data.get("sr", {})
    time       = data["time"]

    emoji = "🟢" if signal == "BUY" else "🔴"

    regime_map = {
        "TRENDING_UP":   "📈 Trending Up",
        "TRENDING_DOWN": "📉 Trending Down",
        "RANGING":       "📦 Ranging",
        "VOLATILE":      "🌪 Volatile",
        "NORMAL":        "➡️ Normale"
    }
    struct_map = {
        "BULLISH": "🟢 Bullish",
        "BEARISH": "🔴 Bearish",
        "NEUTRAL": "⚪ Neutrale"
    }
    pd_map = {
        "PREMIUM":     "⬆️ Premium",
        "DISCOUNT":    "⬇️ Discount",
        "EQUILIBRIUM": "⚖️ Equilibrio"
    }

    # Riga MTF compatta
    mtf_str = ""
    for tf in ["1min", "5min", "15min", "1h", "4h", "1day"]:
        t = mtf.get(tf, "N/A")
        e = "🟢" if t == "BUY" else "🔴" if t == "SELL" else "⚪"
        mtf_str += f"{e}{tf} "

    msg = (
        f"{emoji} *Trade: XAUUSD {order_type} @ {entry}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🛑 SL @ {sl}\n"
        f"🎯 TP1 @ {tp1}  |  R:R 1:{rr1}\n"
        f"🎯 TP2 @ {tp2}  |  R:R 1:{rr2}\n"
        f"🎯 TP3 @ {tp3}  |  R:R 1:{rr3}\n"
        f"⚖️ BE @ {be} (+10 pips)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Score: {score}/{max_score} | Prob: {prob}%\n"
        f"📐 ADX: {adx} | RSI: {rsi}\n"
        f"🌍 Regime: {regime_map.get(regime, regime)}\n"
        f"🏗 Struttura: {struct_map.get(structure, structure)}\n"
        f"💰 Zona: {pd_map.get(pd_zone, pd_zone)}\n"
    )

    # SMC
    if bos:
        msg += f"🔓 BOS: {bos}\n"
    if choch:
        msg += f"🔄 CHoCH: {choch}\n"
    if candle:
        msg += f"🕯 {candle}\n"

    # OB e FVG
    ob = data.get("ob", {})
    fvg = data.get("fvg", {})
    if signal == "BUY" and ob.get("bullish_ob"):
        msg += f"📦 OB Bullish: {ob['bullish_ob']['low']}–{ob['bullish_ob']['high']}\n"
    if signal == "SELL" and ob.get("bearish_ob"):
        msg += f"📦 OB Bearish: {ob['bearish_ob']['low']}–{ob['bearish_ob']['high']}\n"
    if signal == "BUY" and fvg.get("bullish_fvg"):
        msg += f"🔵 FVG Bullish: {fvg['bullish_fvg']['bottom']}–{fvg['bullish_fvg']['top']}\n"
    if signal == "SELL" and fvg.get("bearish_fvg"):
        msg += f"🔴 FVG Bearish: {fvg['bearish_fvg']['bottom']}–{fvg['bearish_fvg']['top']}\n"

    # S/R
    if sr:
        msg += f"📍 S: {sr.get('support')} | R: {sr.get('resistance')} | Pivot: {sr.get('pivot')}\n"

    # MTF
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🔭 MTF: {mtf_str}\n"

    # Sentiment
    s_label = sentiment.get("label", "NEUTRAL")
    s_score = sentiment.get("score", 0)
    s_emoji = "🟢" if s_label == "BULLISH" else "🔴" if s_label == "BEARISH" else "⚪"
    msg += f"{s_emoji} Sentiment: {s_label} ({s_score:+d})\n"

    # Calendario
    if calendar.get("high_impact_today"):
        events = calendar.get("events", [])
        msg += f"⚠️ NEWS ALTO IMPATTO OGGI:\n"
        for ev in events[:2]:
            msg += f"   • {ev['title']} @ {ev['time']}\n"

    # Indicazioni
    if notes:
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"💡 *Indicazioni:*\n"
        for note in notes:
            msg += f"{note}\n"

    msg += (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {time}"
    )

    return msg


# ═══════════════════════════════════════════════
# COMANDI TELEGRAM
# ═══════════════════════════════════════════════

async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👋 *Gold Trading Bot — Sistema Avanzato*\n\n"
        f"🆔 Chat ID: `{chat_id}`\n\n"
        f"*Comandi:*\n"
        f"/signal — Analisi manuale immediata\n"
        f"/trade  — Posizione attiva\n"
        f"/stats  — Storico segnali\n"
        f"/news   — Ultime notizie oro\n"
        f"/status — Stato del bot\n"
        f"/chiudi — Chiudi posizione attiva\n",
        parse_mode="Markdown"
    )


async def cmd_signal(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Analisi completa in corso...")
    try:
        data = full_analyze()
        if data["signal"] == "NEUTRAL":
            buy_s  = data.get("buy_score", 0)
            sell_s = data.get("sell_score", 0)
            regime = data.get("regime", "N/A")
            await update.message.reply_text(
                f"⚪ *Nessun segnale — Mercato in attesa*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Score BUY: {buy_s} | SELL: {sell_s}\n"
                f"🌍 Regime: {regime}\n"
                f"_Aspetta una confluenza più chiara_",
                parse_mode="Markdown"
            )
            return
        msg = format_signal(data)
        await update.message.reply_text(msg, parse_mode="Markdown")
        save_active_trade(data)
    except Exception as e:
        logger.error(f"Errore /signal: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_trade(update, context: ContextTypes.DEFAULT_TYPE):
    trade = load_active_trade()
    if not trade:
        await update.message.reply_text("📭 Nessuna posizione attiva.")
        return
    msg = (
        f"📊 *Posizione Attiva — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 {trade['signal']} {trade['order_type']} @ {trade['entry']}\n"
        f"🛑 SL: {trade['sl']}\n"
        f"🎯 TP1: {trade['tp1']} | TP2: {trade['tp2']} | TP3: {trade['tp3']}\n"
        f"⚖️ BE: {trade.get('be', 'N/A')}\n"
        f"📊 Score: {trade.get('score', 'N/A')} | Prob: {trade.get('prob', 'N/A')}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Aperto: {trade.get('time', 'N/A')}"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_chiudi(update, context: ContextTypes.DEFAULT_TYPE):
    trade = load_active_trade()
    if not trade:
        await update.message.reply_text("📭 Nessuna posizione attiva da chiudere.")
        return
    clear_active_trade()
    await update.message.reply_text(
        f"✅ *Posizione chiusa manualmente*\n"
        f"Trade {trade['signal']} @ {trade['entry']} rimosso dal monitoraggio.",
        parse_mode="Markdown"
    )


async def cmd_news(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Carico le notizie...")
    news = get_gold_news()
    sentiment = get_news_sentiment()
    s_label = sentiment.get("label", "NEUTRAL")
    s_score = sentiment.get("score", 0)
    s_emoji = "🟢" if s_label == "BULLISH" else "🔴" if s_label == "BEARISH" else "⚪"
    if not news:
        await update.message.reply_text("❌ Nessuna notizia disponibile.")
        return
    msg = (
        f"📰 *ULTIME NOTIZIE ORO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{s_emoji} Sentiment: *{s_label}* ({s_score:+d})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(news) +
        f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
        f"_⚠️ News ad alto impatto possono invalidare i segnali tecnici_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_stats(update, context: ContextTypes.DEFAULT_TYPE):
    try:
        stats  = compute_stats()
        recent = get_recent_signals(5)
        recent_txt = ""
        for h in recent:
            e = "✅" if "WIN" in str(h["result"]) else "❌"
            recent_txt += f"{e} {h['signal']} {h['order_type']} @ ${h['entry']} — {h['time']}\n"
        if not recent_txt:
            recent_txt = "Nessun segnale completato ancora.\n"
        msg = (
            f"📊 *STORICO SEGNALI*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Vincenti: *{stats['wins']}*\n"
            f"❌ Perdenti: *{stats['losses']}*\n"
            f"⏳ In attesa: *{stats['pending']}*\n"
            f"📈 Win Rate: *{stats['winrate']}%*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Ultimi 5:*\n{recent_txt}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Totale completati: {stats['total']}_"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    status = market_status_text()
    trade  = load_active_trade()
    trade_info = f"📊 Trade attivo: {trade['signal']} @ {trade['entry']}" if trade else "📭 Nessun trade attivo"
    await update.message.reply_text(
        f"⚙️ *Stato Gold Bot*\n"
        f"━━━━━━━━━━━━━━\n"
        f"{status}\n"
        f"🔁 Controllo: ogni *{CHECK_INTERVAL} min*\n"
        f"📊 Fonte: *Twelve Data*\n"
        f"⏱ Timeframe: *1min/5min/15min/1h/4h/1d*\n"
        f"🗄 Database: *PostgreSQL*\n"
        f"{trade_info}\n"
        f"🤖 Stato: *Attivo*",
        parse_mode="Markdown"
    )


# ═══════════════════════════════════════════════
# REPORT E NOTIZIE MATTINO
# ═══════════════════════════════════════════════

async def send_morning_news(bot: Bot):
    news      = get_gold_news()
    sentiment = get_news_sentiment()
    s_label   = sentiment.get("label", "NEUTRAL")
    s_emoji   = "🟢" if s_label == "BULLISH" else "🔴" if s_label == "BEARISH" else "⚪"
    if not news:
        return
    msg = (
        f"🌅 *BUONGIORNO — NOTIZIE ORO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{s_emoji} Sentiment mattutino: *{s_label}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(news) +
        f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
        f"_Buon trading oggi!_ 📈"
    )
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")


async def send_daily_report(bot: Bot):
    try:
        stats   = compute_daily_stats()
        overall = compute_stats()
        today   = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
        news    = get_gold_news()
        news_txt = "\n\n".join(news[:2]) if news else "Nessuna notizia."
        signals_txt = ""
        for h in stats["signals"]:
            e = "✅" if "WIN" in str(h["result"]) else "❌" if h["result"] == "LOSS" else "⏳"
            signals_txt += f"{e} {h['signal']} @ ${h['entry']} — {h['time']}\n"
        if not signals_txt:
            signals_txt = "Nessun segnale oggi.\n"
        msg = (
            f"🌙 *REPORT GIORNALIERO — {today}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Vincenti: *{stats['wins']}* | ❌ Perdenti: *{stats['losses']}*\n"
            f"📈 Win Rate: *{stats['winrate']}%*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Segnali:*\n{signals_txt}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Storico totale:*\n"
            f"📊 Win Rate: *{overall['winrate']}%* | Totale: *{overall['total']}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Notizie:*\n{news_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Buona notte! Bot riprende domani alle 08:00_ 🌙"
        )
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore report: {e}")


# ═══════════════════════════════════════════════
# JOB AUTOMATICO
# ═══════════════════════════════════════════════

async def auto_check(bot: Bot):
    global last_signal
    if not is_market_open():
        return
    try:
        data = full_analyze()

        # Aggiorna risultati DB
        update_db_results(data["price"])

        # Monitora trade attivo (BE, TP, SL)
        await monitor_active_trade(bot, CHAT_ID)

        if data["signal"] == "NEUTRAL":
            return

        is_new = data["signal"] != last_signal and data["prob"] >= 60

        if data["prob"] >= 60:
            last_signal = data["signal"]
            if is_new:
                add_signal_to_db(data)
                save_active_trade(data)
                msg = "🚨 *NUOVO SEGNALE!*\n\n" + format_signal(data)
                await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                logger.info(f"Segnale: {data['signal']} {data['order_type']} @ {data['entry']}")

    except Exception as e:
        logger.error(f"Errore auto_check: {e}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

async def main():
    init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("trade",  cmd_trade))
    app.add_handler(CommandHandler("chiudi", cmd_chiudi))
    app.add_handler(CommandHandler("news",   cmd_news))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("status", cmd_status))

    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    scheduler.add_job(auto_check,        "interval", minutes=CHECK_INTERVAL, args=[app.bot])
    scheduler.add_job(send_daily_report, "cron",     hour=20, minute=0,      args=[app.bot])
    scheduler.add_job(send_morning_news, "cron",     hour=8,  minute=0,      args=[app.bot])
    scheduler.start()

    logger.info("✅ Gold Bot Sistema Avanzato avviato")
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
