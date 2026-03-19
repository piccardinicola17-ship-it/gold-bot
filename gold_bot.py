"""
Gold Trading Bot per Telegram
Analizza XAU/USD e invia segnali BUY/SELL con TP, SL e punteggio a stelle
"""

import logging
import asyncio
from datetime import datetime
import requests as req
import pandas as pd
import ta
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ─────────────────────────────────────────────
# CONFIGURAZIONE — modifica solo questi valori
# ─────────────────────────────────────────────
import os
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
CHECK_INTERVAL = 5
ATR_SL_MULT    = 1.5
ATR_TP_MULT    = 3.0
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

last_signal = None


def get_gold_data() -> pd.DataFrame:
    """Scarica dati oro XAU/USD da Twelve Data."""
    API_KEY = os.environ.get("TWELVE_API_KEY", "85f2bac59bb24b3a8e55551a3337f844")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": 500,
        "apikey": API_KEY
    }
    r = req.get(url, params=params, timeout=10)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Nessun dato ricevuto: {data}")
    rows = data["values"]
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["datetime"])
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df["Volume"] = 0
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola EMA, RSI, MACD e ATR."""
    df["ema20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["rsi"]   = ta.momentum.rsi(df["Close"], window=14)

    macd_obj          = ta.trend.MACD(df["Close"])
    df["macd"]        = macd_obj.macd()
    df["signal_line"] = macd_obj.macd_signal()

    df["atr"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )
    return df


def stars(score: int) -> str:
    if score >= 5: return "⭐⭐⭐ FORTISSIMO"
    if score == 4: return "⭐⭐⭐ FORTE"
    if score == 3: return "⭐⭐ MODERATO"
    if score == 2: return "⭐ DEBOLE"
    return ""


def analyze(df: pd.DataFrame) -> dict:
    """Genera il segnale di trading con punteggio a stelle."""
    row   = df.iloc[-1]
    price = round(float(row["Close"]), 2)
    atr   = float(row["atr"])
    rsi   = float(row["rsi"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    macd  = float(row["macd"])
    sig   = float(row["signal_line"])

    sl_dist = round(atr * ATR_SL_MULT, 2)
    tp_dist = round(atr * ATR_TP_MULT, 2)

    # Punteggio BUY (0-5)
    buy_score = 0
    if ema20 > ema50:  buy_score += 1
    if macd > sig:     buy_score += 1
    if rsi < 50:       buy_score += 1
    if rsi < 40:       buy_score += 1
    if rsi < 30:       buy_score += 1

    # Punteggio SELL (0-5)
    sell_score = 0
    if ema20 < ema50:  sell_score += 1
    if macd < sig:     sell_score += 1
    if rsi > 50:       sell_score += 1
    if rsi > 60:       sell_score += 1
    if rsi > 70:       sell_score += 1

    if buy_score >= 2:
        signal   = "BUY"
        sl       = round(price - sl_dist, 2)
        tp       = round(price + tp_dist, 2)
        strength = stars(buy_score)
        macd_txt = "sopra" if macd > sig else "sotto"
        reason   = (
            f"EMA20 > EMA50 (trend rialzista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {macd_txt} la signal line\n"
            f"Punteggio: {buy_score}/5"
        )
    elif sell_score >= 2:
        signal   = "SELL"
        sl       = round(price + sl_dist, 2)
        tp       = round(price - tp_dist, 2)
        strength = stars(sell_score)
        macd_txt = "sotto" if macd < sig else "sopra"
        reason   = (
            f"EMA20 < EMA50 (trend ribassista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {macd_txt} la signal line\n"
            f"Punteggio: {sell_score}/5"
        )
    else:
        signal   = "NEUTRAL"
        sl       = None
        tp       = None
        strength = ""
        reason   = (
            f"Nessuna confluenza chiara tra gli indicatori.\n"
            f"RSI: {round(rsi, 1)} | "
            f"EMA20: {round(ema20, 2)} | EMA50: {round(ema50, 2)}"
        )

    return {
        "signal":   signal,
        "strength": strength,
        "price":    price,
        "sl":       sl,
        "tp":       tp,
        "rsi":      round(rsi, 1),
        "atr":      round(atr, 2),
        "reason":   reason,
        "time":     datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


def format_message(data: dict) -> str:
    """Formatta il messaggio da inviare su Telegram."""
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
            f"\n🎯 *Take Profit:* ${data['tp']}\n"
            f"🛑 *Stop Loss:* ${data['sl']}\n"
            f"⚖️ Risk/Reward: *1:{round(rr, 1)}*\n"
        )

    msg += (
        f"\n📈 *Analisi:*\n"
        f"{data['reason']}\n"
        f"\n📉 ATR (14): ${data['atr']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Fonte: yfinance | Timeframe: 1H_"
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


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"⚙️ *Stato Gold Bot*\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔁 Controllo automatico: ogni *{CHECK_INTERVAL} min*\n"
        f"📐 SL moltiplicatore ATR: *{ATR_SL_MULT}x*\n"
        f"🎯 TP moltiplicatore ATR: *{ATR_TP_MULT}x*\n"
        f"📊 Ticker: *GC=F (Gold Futures)*\n"
        f"⏱ Timeframe analisi: *1H*\n"
        f"🤖 Stato: *Attivo*",
        parse_mode="Markdown"
    )


# ─────────────────────────────────────────────
# JOB AUTOMATICO
# ─────────────────────────────────────────────

async def auto_check(bot: Bot):
    global last_signal
    try:
        df   = get_gold_data()
        df   = compute_indicators(df)
        data = analyze(df)

        is_new = data["signal"] != "NEUTRAL" and data["signal"] != last_signal
        if data["signal"] != "NEUTRAL":
            last_signal = data["signal"]

        prefix = "🚨 *NUOVO SEGNALE RILEVATO!*\n\n" if is_new else ""
        msg = prefix + format_message(data)
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        logger.info(f"Segnale inviato: {data['signal']} @ {data['price']}")

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
    app.add_handler(CommandHandler("status", cmd_status))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(auto_check, "interval", minutes=CHECK_INTERVAL, args=[app.bot])
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
