"""
Gold Trading Bot per Telegram
Analizza XAU/USD con Bollinger, Stocastico, notizie mercato, filtro orario e report
"""

import logging
import asyncio
import json
import os
from datetime import datetime
import pandas as pd
import ta
import requests as req
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "85f2bac59bb24b3a8e55551a3337f844")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "d929b1d0334e4160872bbb1bef9fbb15")
CHECK_INTERVAL = 3
ATR_SL_MULT    = 0.6
ATR_TP_MULT    = 0.6
HISTORY_FILE   = "signals_history.json"
TIMEZONE       = pytz.timezone("Europe/Rome")
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

last_signal    = None
last_news_time = None


# ─────────────────────────────────────────────
# FILTRO ORARIO
# ─────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5:
        return False
    if now.hour == 0:
        return False
    return True


def market_status_text() -> str:
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5:
        return "🔴 Mercato chiuso (weekend)"
    if now.hour == 0:
        return "🔴 Mercato chiuso (pausa notturna)"
    return "🟢 Mercato aperto"


# ─────────────────────────────────────────────
# NOTIZIE DI MERCATO
# ─────────────────────────────────────────────

def get_gold_news() -> list:
    """Scarica ultime notizie sull'oro da NewsAPI."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        "gold XAU price OR gold market OR fed interest rates",
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 3,
            "apiKey":   NEWS_API_KEY
        }
        r = req.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        news = []
        for a in articles:
            title = a.get("title", "")
            source = a.get("source", {}).get("name", "")
            published = a.get("publishedAt", "")[:10]
            if title and source:
                news.append(f"📰 *{source}* ({published})\n_{title}_")
        return news
    except Exception as e:
        logger.error(f"Errore notizie: {e}")
        return []


# ─────────────────────────────────────────────
# STORICO SEGNALI
# ─────────────────────────────────────────────

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_signal_to_history(signal: str, price: float, tp: float, sl: float):
    history = load_history()
    history.append({
        "time":   datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M"),
        "signal": signal,
        "price":  price,
        "tp":     tp,
        "sl":     sl,
        "result": "pending"
    })
    save_history(history)


def update_history_results(current_price: float):
    history = load_history()
    updated = False
    for entry in history:
        if entry["result"] != "pending":
            continue
        if entry["signal"] == "BUY":
            if current_price >= entry["tp"]:
                entry["result"] = "WIN"
                updated = True
            elif current_price <= entry["sl"]:
                entry["result"] = "LOSS"
                updated = True
        elif entry["signal"] == "SELL":
            if current_price <= entry["tp"]:
                entry["result"] = "WIN"
                updated = True
            elif current_price >= entry["sl"]:
                entry["result"] = "LOSS"
                updated = True
    if updated:
        save_history(history)
    return history


def compute_stats() -> dict:
    history = load_history()
    total   = len([h for h in history if h["result"] != "pending"])
    wins    = len([h for h in history if h["result"] == "WIN"])
    losses  = len([h for h in history if h["result"] == "LOSS"])
    pending = len([h for h in history if h["result"] == "pending"])
    winrate = round((wins / total * 100), 1) if total > 0 else 0
    return {
        "total":   total,
        "wins":    wins,
        "losses":  losses,
        "pending": pending,
        "winrate": winrate
    }


def compute_daily_stats() -> dict:
    history = load_history()
    today   = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
    today_h = [h for h in history if h["time"].startswith(today)]
    total   = len([h for h in today_h if h["result"] != "pending"])
    wins    = len([h for h in today_h if h["result"] == "WIN"])
    losses  = len([h for h in today_h if h["result"] == "LOSS"])
    pending = len([h for h in today_h if h["result"] == "pending"])
    winrate = round((wins / total * 100), 1) if total > 0 else 0
    return {
        "total":   total,
        "wins":    wins,
        "losses":  losses,
        "pending": pending,
        "winrate": winrate,
        "signals": today_h
    }


# ─────────────────────────────────────────────
# DATI E INDICATORI
# ─────────────────────────────────────────────

def get_gold_data() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     "XAU/USD",
        "interval":   "5min",
        "outputsize": 500,
        "apikey":     TWELVE_API_KEY
    }
    r = req.get(url, params=params, timeout=10)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Nessun dato ricevuto: {data}")
    rows = data["values"]
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["datetime"])
    df = df.rename(columns={
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Close"
    })
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df["Volume"] = 0
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["rsi"]   = ta.momentum.rsi(df["Close"], window=14)

    macd_obj          = ta.trend.MACD(df["Close"])
    df["macd"]        = macd_obj.macd()
    df["signal_line"] = macd_obj.macd_signal()

    df["atr"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    # Bande di Bollinger
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()

    # Stocastico
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    return df


def stars(score: int) -> str:
    if score >= 7: return "⭐⭐⭐ FORTISSIMO"
    if score >= 6: return "⭐⭐⭐ FORTE"
    if score >= 4: return "⭐⭐ MODERATO"
    if score == 3: return "⭐ DEBOLE"
    return ""


def estimate_probability(score: int, rsi: float, atr: float) -> int:
    base = {3: 50, 4: 60, 5: 68, 6: 76, 7: 84, 8: 90, 9: 93}.get(score, 50)
    if rsi < 25 or rsi > 75:
        base += 4
    if atr < 10:
        base += 3
    return min(base, 94)


def analyze(df: pd.DataFrame) -> dict:
    row      = df.iloc[-1]
    price    = round(float(row["Close"]), 2)
    atr      = float(row["atr"])
    rsi      = float(row["rsi"])
    ema20    = float(row["ema20"])
    ema50    = float(row["ema50"])
    macd     = float(row["macd"])
    sig      = float(row["signal_line"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])
    stoch_k  = float(row["stoch_k"])
    stoch_d  = float(row["stoch_d"])

    sl_dist = round(atr * ATR_SL_MULT, 2)
    tp_dist = round(atr * ATR_TP_MULT, 2)

    # Punteggio BUY (0-9)
    buy_score = 0
    if ema20 > ema50:        buy_score += 1
    if macd > sig:           buy_score += 1
    if rsi < 50:             buy_score += 1
    if rsi < 40:             buy_score += 1
    if rsi < 30:             buy_score += 1
    if price <= bb_lower:    buy_score += 1
    if price < bb_lower:     buy_score += 1
    if stoch_k < 30:         buy_score += 1  # Stocastico ipervenduto
    if stoch_k > stoch_d:    buy_score += 1  # Stocastico in rialzo

    # Punteggio SELL (0-9)
    sell_score = 0
    if ema20 < ema50:        sell_score += 1
    if macd < sig:           sell_score += 1
    if rsi > 50:             sell_score += 1
    if rsi > 60:             sell_score += 1
    if rsi > 70:             sell_score += 1
    if price >= bb_upper:    sell_score += 1
    if price > bb_upper:     sell_score += 1
    if stoch_k > 70:         sell_score += 1  # Stocastico ipercomprato
    if stoch_k < stoch_d:    sell_score += 1  # Stocastico in ribasso

    # Testo Bollinger
    if price <= bb_lower:
        bb_txt = f"📉 Prezzo sotto banda inferiore BB (${round(bb_lower, 2)})"
    elif price >= bb_upper:
        bb_txt = f"📈 Prezzo sopra banda superiore BB (${round(bb_upper, 2)})"
    else:
        bb_txt = f"📊 BB: {round(bb_lower, 2)} — {round(bb_upper, 2)}"

    # Testo Stocastico
    if stoch_k < 30:
        stoch_txt = f"📉 Stocastico ipervenduto ({round(stoch_k, 1)})"
    elif stoch_k > 70:
        stoch_txt = f"📈 Stocastico ipercomprato ({round(stoch_k, 1)})"
    else:
        stoch_txt = f"📊 Stocastico: {round(stoch_k, 1)}"

    if buy_score >= 3:
        signal   = "BUY"
        sl       = round(price - sl_dist, 2)
        tp       = round(price + tp_dist, 2)
        strength = stars(buy_score)
        prob     = estimate_probability(buy_score, rsi, atr)
        reason   = (
            f"EMA20 > EMA50 (trend rialzista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {'sopra' if macd > sig else 'sotto'} la signal line\n"
            f"{bb_txt}\n"
            f"{stoch_txt}\n"
            f"Punteggio: {buy_score}/9"
        )
    elif sell_score >= 3:
        signal   = "SELL"
        sl       = round(price + sl_dist, 2)
        tp       = round(price - tp_dist, 2)
        strength = stars(sell_score)
        prob     = estimate_probability(sell_score, rsi, atr)
        reason   = (
            f"EMA20 < EMA50 (trend ribassista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {'sotto' if macd < sig else 'sopra'} la signal line\n"
            f"{bb_txt}\n"
            f"{stoch_txt}\n"
            f"Punteggio: {sell_score}/9"
        )
    else:
        signal   = "NEUTRAL"
        sl       = None
        tp       = None
        strength = ""
        prob     = 0
        reason   = (
            f"Nessuna confluenza chiara tra gli indicatori.\n"
            f"RSI: {round(rsi, 1)} | "
            f"EMA20: {round(ema20, 2)} | EMA50: {round(ema50, 2)}\n"
            f"{bb_txt}\n"
            f"{stoch_txt}"
        )

    return {
        "signal":   signal,
        "strength": strength,
        "price":    price,
        "sl":       sl,
        "tp":       tp,
        "prob":     prob,
        "rsi":      round(rsi, 1),
        "atr":      round(atr, 2),
        "reason":   reason,
        "time":     datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M"),
    }


def format_message(data: dict) -> str:
    emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(data["signal"], "⚪")

    msg = (
        f"{emoji} *SEGNALE ORO (XAU/USD)*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {data['time']}\n"
        f"📊 Segnale: *{data['signal']}* {data['strength']}\n"
        f"💰 Prezzo attuale: *${data['price']}*\n"
    )

    if data["sl"] and data["tp"]:
        rr = abs(data["tp"] - data["price"]) / abs(data["sl"] - data["price"])
        msg += (
            f"\n🎲 *Probabilità stimata:* {data['prob']}%\n"
            f"🎯 *Take Profit:* ${data['tp']}\n"
            f"🛑 *Stop Loss:* ${data['sl']}\n"
            f"⚖️ Risk/Reward: *1:{round(rr, 1)}*\n"
        )

    msg += (
        f"\n📈 *Analisi:*\n"
        f"{data['reason']}\n"
        f"\n📉 ATR (14): ${data['atr']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Fonte: Twelve Data | Timeframe: 5min_"
    )
    return msg


# ─────────────────────────────────────────────
# COMANDI TELEGRAM
# ─────────────────────────────────────────────

async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👋 Benvenuto nel Gold Trading Bot!\n\n"
        f"🆔 Il tuo Chat ID è: `{chat_id}`\n\n"
        f"Comandi disponibili:\n"
        f"/signal — Analisi manuale immediata\n"
        f"/news — Ultime notizie sull'oro\n"
        f"/stats — Storico segnali e % successo\n"
        f"/status — Stato del bot e parametri\n"
        f"/start — Mostra questo messaggio",
        parse_mode="Markdown"
    )


async def cmd_signal(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Analisi in corso...")
    try:
        df   = get_gold_data()
        df   = compute_indicators(df)
        data = analyze(df)
        msg  = format_message(data)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore /signal: {e}")
        await update.message.reply_text(f"❌ Errore nell'analisi: {e}")


async def cmd_news(update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /news — ultime notizie sull'oro."""
    await update.message.reply_text("⏳ Carico le notizie...")
    news = get_gold_news()
    if not news:
        await update.message.reply_text("❌ Nessuna notizia disponibile al momento.")
        return
    msg = (
        f"📰 *ULTIME NOTIZIE ORO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(news) +
        f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
        f"_⚠️ Le notizie importanti possono invalidare i segnali tecnici_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_stats(update, context: ContextTypes.DEFAULT_TYPE):
    stats   = compute_stats()
    history = load_history()

    recent = [h for h in history if h["result"] != "pending"][-5:]
    recent_txt = ""
    for h in reversed(recent):
        emoji = "✅" if h["result"] == "WIN" else "❌"
        recent_txt += f"{emoji} {h['signal']} @ ${h['price']} — {h['time']}\n"

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
        f"*Ultimi 5 segnali:*\n"
        f"{recent_txt}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Totale segnali completati: {stats['total']}_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    status = market_status_text()
    await update.message.reply_text(
        f"⚙️ *Stato Gold Bot*\n"
        f"━━━━━━━━━━━━━━\n"
        f"{status}\n"
        f"🔁 Controllo automatico: ogni *{CHECK_INTERVAL} min*\n"
        f"📐 SL moltiplicatore ATR: *{ATR_SL_MULT}x*\n"
        f"🎯 TP moltiplicatore ATR: *{ATR_TP_MULT}x*\n"
        f"📊 Fonte: *Twelve Data*\n"
        f"⏱ Timeframe: *5min*\n"
        f"🤖 Stato: *Attivo*",
        parse_mode="Markdown"
    )


# ─────────────────────────────────────────────
# REPORT GIORNALIERO
# ─────────────────────────────────────────────

async def send_daily_report(bot: Bot):
    stats = compute_daily_stats()
    today = datetime.now(TIMEZONE).strftime("%d/%m/%Y")

    signals_txt = ""
    for h in stats["signals"]:
        if h["result"] == "WIN":
            emoji = "✅"
        elif h["result"] == "LOSS":
            emoji = "❌"
        else:
            emoji = "⏳"
        signals_txt += f"{emoji} {h['signal']} @ ${h['price']} — {h['time']}\n"

    if not signals_txt:
        signals_txt = "Nessun segnale oggi.\n"

    overall = compute_stats()
    news    = get_gold_news()
    news_txt = "\n\n".join(news[:2]) if news else "Nessuna notizia disponibile."

    msg = (
        f"🌙 *REPORT GIORNALIERO — {today}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Oggi:*\n"
        f"✅ Vincenti: *{stats['wins']}*\n"
        f"❌ Perdenti: *{stats['losses']}*\n"
        f"⏳ In attesa: *{stats['pending']}*\n"
        f"📈 Win Rate oggi: *{stats['winrate']}%*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Segnali di oggi:*\n"
        f"{signals_txt}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Totale storico:*\n"
        f"📊 Win Rate totale: *{overall['winrate']}%*\n"
        f"🏆 Totale segnali: *{overall['total']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Ultime notizie oro:*\n"
        f"{news_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Buona notte! Il bot riprende domani alle 01:00_ 🌙"
    )

    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    logger.info("Report giornaliero inviato")


# ─────────────────────────────────────────────
# JOB AUTOMATICO
# ─────────────────────────────────────────────

async def auto_check(bot: Bot):
    global last_signal, last_news_time

    if not is_market_open():
        logger.info("Mercato chiuso — segnale saltato")
        return

    try:
        df   = get_gold_data()
        df   = compute_indicators(df)
        data = analyze(df)

        update_history_results(data["price"])

        is_new = data["signal"] != "NEUTRAL" and data["signal"] != last_signal
        if data["signal"] != "NEUTRAL":
            last_signal = data["signal"]
            if is_new:
                add_signal_to_history(
                    data["signal"], data["price"], data["tp"], data["sl"]
                )

        prefix = "🚨 *NUOVO SEGNALE RILEVATO!*\n\n" if is_new else ""
        msg = prefix + format_message(data)
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        logger.info(f"Segnale inviato: {data['signal']} @ {data['price']}")

        # Invia notizie ogni ora
        now = datetime.now(TIMEZONE)
        if last_news_time is None or (now - last_news_time).seconds >= 3600:
            news = get_gold_news()
            if news:
                news_msg = (
                    f"📰 *AGGIORNAMENTO NOTIZIE ORO*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    + "\n\n".join(news) +
                    f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
                    f"_⚠️ Notizie importanti possono invalidare i segnali tecnici_"
                )
                await bot.send_message(chat_id=CHAT_ID, text=news_msg, parse_mode="Markdown")
                last_news_time = now

    except Exception as e:
        logger.error(f"Errore job automatico: {e}")
        await bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Errore analisi automatica: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("status", cmd_status))

    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    scheduler.add_job(auto_check, "interval", minutes=CHECK_INTERVAL, args=[app.bot])
    scheduler.add_job(send_daily_report, "cron", hour=22, minute=0, args=[app.bot])
    scheduler.start()
    logger.info(f"✅ Bot avviato — controllo ogni {CHECK_INTERVAL} minuti")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot fermato.")
    finally:
        scheduler.shutdown()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
