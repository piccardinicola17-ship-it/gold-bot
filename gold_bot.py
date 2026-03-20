"""
Gold Trading Bot per Telegram
Analizza XAU/USD con database persistente, timeframe multipli 5min+1H,
Bollinger, Stocastico, filtro orario, notizie e report giornaliero
"""

import logging
import asyncio
import os
from datetime import datetime
import pandas as pd
import ta
import requests as req
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
import psycopg2
from psycopg2.extras import RealDictCursor

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "85f2bac59bb24b3a8e55551a3337f844")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "d929b1d0334e4160872bbb1bef9fbb15")
DATABASE_URL   = os.environ.get("DATABASE_URL", "")
CHECK_INTERVAL = 3
ATR_SL_MULT    = 0.6
ATR_TP_MULT    = 0.6
TIMEZONE       = pytz.timezone("Europe/Rome")
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

last_signal = None


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Crea la tabella segnali se non esiste."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    time TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    price REAL NOT NULL,
                    tp REAL NOT NULL,
                    sl REAL NOT NULL,
                    result TEXT DEFAULT 'pending'
                )
            """)
        conn.commit()
    logger.info("Database inizializzato")


def add_signal_to_db(signal: str, price: float, tp: float, sl: float):
    time = datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (time, signal, price, tp, sl) VALUES (%s, %s, %s, %s, %s)",
                (time, signal, price, tp, sl)
            )
        conn.commit()


def update_db_results(current_price: float):
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM signals WHERE result = 'pending'")
            pending = cur.fetchall()
            for entry in pending:
                new_result = None
                if entry["signal"] == "BUY":
                    if current_price >= entry["tp"]:
                        new_result = "WIN"
                    elif current_price <= entry["sl"]:
                        new_result = "LOSS"
                elif entry["signal"] == "SELL":
                    if current_price <= entry["tp"]:
                        new_result = "WIN"
                    elif current_price >= entry["sl"]:
                        new_result = "LOSS"
                if new_result:
                    cur.execute(
                        "UPDATE signals SET result = %s WHERE id = %s",
                        (new_result, entry["id"])
                    )
        conn.commit()


def compute_stats() -> dict:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT result FROM signals")
            rows = cur.fetchall()
    total   = len([r for r in rows if r["result"] != "pending"])
    wins    = len([r for r in rows if r["result"] == "WIN"])
    losses  = len([r for r in rows if r["result"] == "LOSS"])
    pending = len([r for r in rows if r["result"] == "pending"])
    winrate = round((wins / total * 100), 1) if total > 0 else 0
    return {"total": total, "wins": wins, "losses": losses, "pending": pending, "winrate": winrate}


def compute_daily_stats() -> dict:
    today = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM signals WHERE time LIKE %s", (f"{today}%",))
            rows = cur.fetchall()
    total   = len([r for r in rows if r["result"] != "pending"])
    wins    = len([r for r in rows if r["result"] == "WIN"])
    losses  = len([r for r in rows if r["result"] == "LOSS"])
    pending = len([r for r in rows if r["result"] == "pending"])
    winrate = round((wins / total * 100), 1) if total > 0 else 0
    return {"total": total, "wins": wins, "losses": losses, "pending": pending, "winrate": winrate, "signals": rows}


def get_recent_signals(limit: int = 5) -> list:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM signals WHERE result != 'pending' ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cur.fetchall()


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
# NOTIZIE
# ─────────────────────────────────────────────

def get_gold_news() -> list:
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
            title  = a.get("title", "")
            source = a.get("source", {}).get("name", "")
            published = a.get("publishedAt", "")[:10]
            if title and source:
                news.append(f"📰 *{source}* ({published})\n_{title}_")
        return news
    except Exception as e:
        logger.error(f"Errore notizie: {e}")
        return []


# ─────────────────────────────────────────────
# DATI E INDICATORI
# ─────────────────────────────────────────────

def get_gold_data(interval: str = "5min", outputsize: int = 500) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     "XAU/USD",
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVE_API_KEY
    }
    r = req.get(url, params=params, timeout=10)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Nessun dato ricevuto: {data}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
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

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volume medio
    df["vol_avg"] = df["Volume"].rolling(window=20).mean()

    # ADX — forza del trend
    adx_obj   = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx_obj.adx()

    return df


def get_support_resistance(df):
    """Calcola livelli di supporto e resistenza dagli ultimi 50 periodi."""
    recent     = df.tail(50)
    support    = round(float(recent["Low"].min()), 2)
    resistance = round(float(recent["High"].max()), 2)
    pivot      = round(float((recent["High"].iloc[-1] + recent["Low"].iloc[-1] + recent["Close"].iloc[-1]) / 3), 2)
    return support, resistance, pivot


def detect_candle_pattern(df: pd.DataFrame) -> tuple:
    """Rileva pattern candele giapponesi. Restituisce (pattern_name, direction)."""
    if len(df) < 2:
        return ("", "NEUTRAL")

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = curr["Open"], curr["High"], curr["Low"], curr["Close"]
    po, pc     = prev["Open"], prev["Close"]

    body      = abs(c - o)
    candle_range = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if candle_range == 0:
        return ("", "NEUTRAL")

    # Doji — corpo molto piccolo
    if body <= candle_range * 0.1:
        return ("🕯 Doji (indecisione)", "NEUTRAL")

    # Hammer — corpo piccolo in alto, wick lungo in basso (rialzista)
    if lower_wick >= body * 2 and upper_wick <= body * 0.3 and c > o:
        return ("🔨 Hammer (rialzista)", "BUY")

    # Shooting Star — corpo piccolo in basso, wick lungo in alto (ribassista)
    if upper_wick >= body * 2 and lower_wick <= body * 0.3 and c < o:
        return ("⭐ Shooting Star (ribassista)", "SELL")

    # Engulfing Bullish — candela verde che ingloba quella rossa precedente
    if c > o and pc < po and c > po and o < pc:
        return ("📈 Engulfing Bullish (rialzista)", "BUY")

    # Engulfing Bearish — candela rossa che ingloba quella verde precedente
    if c < o and pc > po and c < po and o > pc:
        return ("📉 Engulfing Bearish (ribassista)", "SELL")

    return ("", "NEUTRAL")


def get_trend_1h() -> str:
    """Controlla il trend sull'1H — restituisce BUY, SELL o NEUTRAL."""
    try:
        df = get_gold_data(interval="1h", outputsize=100)
        df = compute_indicators(df)
        row = df.iloc[-1]
        if float(row["ema20"]) > float(row["ema50"]) and float(row["macd"]) > float(row["signal_line"]):
            return "BUY"
        elif float(row["ema20"]) < float(row["ema50"]) and float(row["macd"]) < float(row["signal_line"]):
            return "SELL"
        return "NEUTRAL"
    except Exception as e:
        logger.error(f"Errore trend 1H: {e}")
        return "NEUTRAL"


def stars(score: int) -> str:
    if score >= 12: return "⭐⭐⭐ FORTISSIMO"
    if score >= 9:  return "⭐⭐⭐ FORTE"
    if score >= 6:  return "⭐⭐ MODERATO"
    if score >= 3:  return "⭐ DEBOLE"
    return ""


def estimate_probability(score: int, rsi: float, atr: float, trend_confirmed: bool) -> int:
    base = {3: 45, 4: 50, 5: 55, 6: 60, 7: 65, 8: 70, 9: 75, 10: 80, 11: 85, 12: 88, 13: 91, 14: 93, 15: 95}.get(score, 45)
    if rsi < 25 or rsi > 75:
        base += 3
    if atr < 10:
        base += 2
    if trend_confirmed:
        base += 5
    return min(base, 95)


def analyze(df: pd.DataFrame, trend_1h: str) -> dict:
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
    volume   = float(row["Volume"])
    vol_avg  = float(row["vol_avg"]) if row["vol_avg"] > 0 else 1
    adx      = float(row["adx"]) if not pd.isna(row["adx"]) else 0

    sl_dist = round(atr * ATR_SL_MULT, 2)
    tp_dist = round(atr * ATR_TP_MULT, 2)

    # Pattern candele
    candle_pattern, candle_dir = detect_candle_pattern(df)

    # Volume alto = sopra la media
    high_volume = volume > vol_avg * 1.2

    # Supporto e resistenza
    support, resistance, pivot = get_support_resistance(df)
    near_support    = abs(price - support) <= atr * 0.5
    near_resistance = abs(price - resistance) <= atr * 0.5

    # ADX — ignora segnali in mercato laterale
    trend_strong = adx >= 20

    # Punteggio BUY (0-15)
    buy_score = 0
    if ema20 > ema50:              buy_score += 1
    if macd > sig:                 buy_score += 1
    if rsi < 50:                   buy_score += 1
    if rsi < 40:                   buy_score += 1
    if rsi < 30:                   buy_score += 1
    if price <= bb_lower:          buy_score += 1
    if price < bb_lower:           buy_score += 1
    if stoch_k < 30:               buy_score += 1
    if stoch_k > stoch_d:          buy_score += 1
    if candle_dir == "BUY":        buy_score += 2
    if high_volume:                buy_score += 1
    if near_support:               buy_score += 1
    if trend_strong:               buy_score += 1

    # Punteggio SELL (0-15)
    sell_score = 0
    if ema20 < ema50:              sell_score += 1
    if macd < sig:                 sell_score += 1
    if rsi > 50:                   sell_score += 1
    if rsi > 60:                   sell_score += 1
    if rsi > 70:                   sell_score += 1
    if price >= bb_upper:          sell_score += 1
    if price > bb_upper:           sell_score += 1
    if stoch_k > 70:               sell_score += 1
    if stoch_k < stoch_d:          sell_score += 1
    if candle_dir == "SELL":       sell_score += 2
    if high_volume:                sell_score += 1
    if near_resistance:            sell_score += 1
    if trend_strong:               sell_score += 1

    if price <= bb_lower:
        bb_txt = f"📉 Prezzo sotto banda inferiore BB (${round(bb_lower, 2)})"
    elif price >= bb_upper:
        bb_txt = f"📈 Prezzo sopra banda superiore BB (${round(bb_upper, 2)})"
    else:
        bb_txt = f"📊 BB: {round(bb_lower, 2)} — {round(bb_upper, 2)}"

    if stoch_k < 30:
        stoch_txt = f"📉 Stocastico ipervenduto ({round(stoch_k, 1)})"
    elif stoch_k > 70:
        stoch_txt = f"📈 Stocastico ipercomprato ({round(stoch_k, 1)})"
    else:
        stoch_txt = f"📊 Stocastico: {round(stoch_k, 1)}"

    vol_txt   = "📊 Volume: 🔥 Alto" if high_volume else "📊 Volume: normale"
    trend_emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(trend_1h, "⚪")
    trend_txt   = f"{trend_emoji} Trend 1H: *{trend_1h}*"

    if buy_score >= 3:
        signal          = "BUY"
        trend_confirmed = trend_1h == "BUY"
        sl              = round(price - sl_dist, 2)
        tp              = round(price + tp_dist, 2)
        strength        = stars(buy_score)
        prob            = estimate_probability(buy_score, rsi, atr, trend_confirmed)
        confirm_txt     = "✅ Confermato dal trend 1H!" if trend_confirmed else "⚠️ Non confermato dal trend 1H"
        candle_txt      = candle_pattern if candle_pattern else ""
        reason = (
            f"EMA20 > EMA50 (trend rialzista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {'sopra' if macd > sig else 'sotto'} la signal line\n"
            f"{bb_txt}\n"
            f"{stoch_txt}\n"
            f"{vol_txt}\n"
            + (f"{candle_txt}\n" if candle_txt else "") +
            f"📍 Supporto: ${support} | Resistenza: ${resistance}\n"
            f"📐 ADX: {round(adx, 1)} ({'trend forte' if trend_strong else 'mercato laterale'})\n"
            f"{trend_txt}\n"
            f"{confirm_txt}\n"
            f"Punteggio: {buy_score}/15"
        )
    elif sell_score >= 3:
        signal          = "SELL"
        trend_confirmed = trend_1h == "SELL"
        sl              = round(price + sl_dist, 2)
        tp              = round(price - tp_dist, 2)
        strength        = stars(sell_score)
        prob            = estimate_probability(sell_score, rsi, atr, trend_confirmed)
        confirm_txt     = "✅ Confermato dal trend 1H!" if trend_confirmed else "⚠️ Non confermato dal trend 1H"
        candle_txt      = candle_pattern if candle_pattern else ""
        reason = (
            f"EMA20 < EMA50 (trend ribassista)\n"
            f"RSI: {round(rsi, 1)}\n"
            f"MACD {'sotto' if macd < sig else 'sopra'} la signal line\n"
            f"{bb_txt}\n"
            f"{stoch_txt}\n"
            f"{vol_txt}\n"
            + (f"{candle_txt}\n" if candle_txt else "") +
            f"📍 Supporto: ${support} | Resistenza: ${resistance}\n"
            f"📐 ADX: {round(adx, 1)} ({'trend forte' if trend_strong else 'mercato laterale'})\n"
            f"{trend_txt}\n"
            f"{confirm_txt}\n"
            f"Punteggio: {sell_score}/15"
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
            f"{stoch_txt}\n"
            f"{vol_txt}\n"
            f"{trend_txt}"
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
        f"_Fonte: Twelve Data | Timeframe: 5min + 1H_"
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
        df       = get_gold_data()
        df       = compute_indicators(df)
        trend_1h = get_trend_1h()
        data     = analyze(df, trend_1h)
        msg      = format_message(data)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Errore /signal: {e}")
        await update.message.reply_text(f"❌ Errore nell'analisi: {e}")


async def cmd_news(update, context: ContextTypes.DEFAULT_TYPE):
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
    try:
        stats  = compute_stats()
        recent = get_recent_signals(5)

        recent_txt = ""
        for h in recent:
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
            f"*Ultimi 15 segnali:*\n"
            f"{recent_txt}"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Totale segnali completati: {stats['total']}_"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore stats: {e}")


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
        f"⏱ Timeframe: *5min + 1H*\n"
        f"🗄 Database: *PostgreSQL persistente*\n"
        f"🤖 Stato: *Attivo*",
        parse_mode="Markdown"
    )


# ─────────────────────────────────────────────
# REPORT E NOTIZIE MATTINO
# ─────────────────────────────────────────────

async def send_morning_news(bot: Bot):
    news = get_gold_news()
    if not news:
        return
    msg = (
        f"🌅 *BUONGIORNO — NOTIZIE ORO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        + "\n\n".join(news) +
        f"\n\n━━━━━━━━━━━━━━━━━━━━\n"
        f"_⚠️ Notizie importanti possono invalidare i segnali tecnici_"
    )
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    logger.info("Notizie mattutine inviate")


async def send_daily_report(bot: Bot):
    try:
        stats   = compute_daily_stats()
        overall = compute_stats()
        today   = datetime.now(TIMEZONE).strftime("%d/%m/%Y")
        news    = get_gold_news()
        news_txt = "\n\n".join(news[:2]) if news else "Nessuna notizia disponibile."

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
    except Exception as e:
        logger.error(f"Errore report giornaliero: {e}")


# ─────────────────────────────────────────────
# JOB AUTOMATICO
# ─────────────────────────────────────────────

async def auto_check(bot: Bot):
    global last_signal

    if not is_market_open():
        logger.info("Mercato chiuso — segnale saltato")
        return

    try:
        df       = get_gold_data()
        df       = compute_indicators(df)
        trend_1h = get_trend_1h()
        data     = analyze(df, trend_1h)

        update_db_results(data["price"])

        is_new = data["signal"] != "NEUTRAL" and data["signal"] != last_signal and data["prob"] >= 60
        if data["signal"] != "NEUTRAL" and data["prob"] >= 60:
            last_signal = data["signal"]
            if is_new:
                add_signal_to_db(data["signal"], data["price"], data["tp"], data["sl"])

        if data["signal"] != "NEUTRAL" and data["prob"] >= 60:
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
    init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("status", cmd_status))

    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    scheduler.add_job(auto_check, "interval", minutes=CHECK_INTERVAL, args=[app.bot])
    scheduler.add_job(send_daily_report, "cron", hour=22, minute=0, args=[app.bot])
    scheduler.add_job(send_morning_news, "cron", hour=8, minute=0, args=[app.bot])
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
