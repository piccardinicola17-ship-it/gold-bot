"""
analyzer.py — Motore di analisi COMPLETO XAU/USD
Livelli: Dati multi-timeframe, SMC, Indicatori avanzati, Sentiment,
Calendario economico, Regime detection, Order type automatico,
Indicazioni operative, Risk management, Score system avanzato
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "85f2bac59bb24b3a8e55551a3337f844")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "d929b1d0334e4160872bbb1bef9fbb15")
TIMEZONE       = pytz.timezone("Europe/Rome")

# ═══════════════════════════════════════════════
# LIVELLO 01 — DATA COLLECTION
# ═══════════════════════════════════════════════

def get_data(interval="5min", outputsize=500) -> pd.DataFrame:
    """Scarica dati OHLCV da Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     "XAU/USD",
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVE_API_KEY
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Nessun dato: {data}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df["Volume"] = 0
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def get_current_price() -> float:
    """Prezzo live XAU/USD."""
    try:
        url = "https://api.twelvedata.com/price"
        r   = requests.get(url, params={"symbol": "XAU/USD", "apikey": TWELVE_API_KEY}, timeout=5)
        return float(r.json()["price"])
    except:
        return 0.0


def get_multi_timeframe_data() -> dict:
    """
    Raccoglie dati su tutti i timeframe chiave.
    Restituisce un dizionario {timeframe: DataFrame}.
    """
    timeframes = {
        "1min":  50,
        "5min":  500,
        "15min": 200,
        "1h":    200,
        "4h":    100,
        "1day":  100,
    }
    data = {}
    for tf, size in timeframes.items():
        try:
            data[tf] = get_data(interval=tf, outputsize=size)
            logger.info(f"Dati {tf} scaricati: {len(data[tf])} candele")
        except Exception as e:
            logger.warning(f"Errore dati {tf}: {e}")
    return data


# ═══════════════════════════════════════════════
# LIVELLO 02 — INDICATORI TECNICI COMPLETI
# ═══════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola tutti gli indicatori tecnici."""

    # ── Trend ──
    df["ema9"]   = ta.trend.ema_indicator(df["Close"], window=9)
    df["ema20"]  = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema50"]  = ta.trend.ema_indicator(df["Close"], window=50)
    df["ema100"] = ta.trend.ema_indicator(df["Close"], window=100)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], window=200)
    df["sma20"]  = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma50"]  = ta.trend.sma_indicator(df["Close"], window=50)

    # ── MACD ──
    macd             = ta.trend.MACD(df["Close"])
    df["macd"]       = macd.macd()
    df["macd_sig"]   = macd.macd_signal()
    df["macd_hist"]  = macd.macd_diff()

    # ── RSI ──
    df["rsi"]    = ta.momentum.rsi(df["Close"], window=14)
    df["rsi_fast"] = ta.momentum.rsi(df["Close"], window=7)

    # ── Bollinger Bands ──
    bb              = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]    = bb.bollinger_pband()

    # ── ATR ──
    df["atr"]    = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # ── Stocastico ──
    stoch         = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── ADX ──
    adx           = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ── CCI ──
    df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)

    # ── Williams %R ──
    df["willr"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=14)

    # ── ROC / Momentum ──
    df["roc"] = ta.momentum.roc(df["Close"], window=10)
    df["mom"] = df["Close"].diff(10)

    # ── Volume ──
    df["vol_avg"]  = df["Volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_avg"] + 1e-9)

    # ── Ichimoku ──
    try:
        ich            = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        df["ich_a"]    = ich.ichimoku_a()
        df["ich_b"]    = ich.ichimoku_b()
        df["ich_base"] = ich.ichimoku_base_line()
        df["ich_conv"] = ich.ichimoku_conversion_line()
    except:
        df["ich_a"] = df["ich_b"] = df["ich_base"] = df["ich_conv"] = np.nan

    # ── VWAP approssimato ──
    try:
        typical  = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap"] = (typical * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
    except:
        df["vwap"] = df["Close"]

    # ── Parabolic SAR ──
    try:
        psar        = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
        df["psar"]  = psar.psar()
        df["psar_up"]   = psar.psar_up()
        df["psar_down"] = psar.psar_down()
    except:
        df["psar"] = df["Close"]

    return df


# ═══════════════════════════════════════════════
# LIVELLO 03 — SMC COMPLETO
# ═══════════════════════════════════════════════

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Rileva swing high e swing low."""
    df["swing_high"] = False
    df["swing_low"]  = False
    for i in range(lookback, len(df) - lookback):
        if df["High"].iloc[i] == df["High"].iloc[i - lookback:i + lookback + 1].max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if df["Low"].iloc[i] == df["Low"].iloc[i - lookback:i + lookback + 1].min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
    return df


def detect_bos_choch(df: pd.DataFrame) -> dict:
    """Rileva BOS e CHoCH."""
    result = {"bos": None, "choch": None, "structure": "NEUTRAL",
              "last_high": None, "last_low": None}
    sh = df[df["swing_high"]]["High"].tail(5).values
    sl = df[df["swing_low"]]["Low"].tail(5).values
    price = float(df["Close"].iloc[-1])
    if len(sh) >= 2 and len(sl) >= 2:
        lh, ph = sh[-1], sh[-2]
        ll, pl = sl[-1], sl[-2]
        result["last_high"] = round(float(lh), 2)
        result["last_low"]  = round(float(ll), 2)
        if lh > ph and ll > pl:
            result["structure"] = "BULLISH"
            if price > lh:
                result["bos"] = "BOS_BULLISH"
        elif lh < ph and ll < pl:
            result["structure"] = "BEARISH"
            if price < ll:
                result["bos"] = "BOS_BEARISH"
        if result["structure"] == "BULLISH" and price < ll:
            result["choch"] = "CHOCH_BEARISH"
        elif result["structure"] == "BEARISH" and price > lh:
            result["choch"] = "CHOCH_BULLISH"
    return result


def detect_order_blocks(df: pd.DataFrame) -> dict:
    """Rileva Order Block bullish e bearish."""
    obs = {"bullish_ob": None, "bearish_ob": None}
    for i in range(max(0, len(df) - 20), len(df) - 1):
        c  = df.iloc[i]
        nc = df.iloc[i + 1]
        move_pct = abs(nc["Close"] - nc["Open"]) / nc["Open"] * 100
        if c["Close"] < c["Open"] and nc["Close"] > nc["Open"] and move_pct > 0.03:
            obs["bullish_ob"] = {"high": round(float(c["High"]), 2), "low": round(float(c["Low"]), 2)}
        if c["Close"] > c["Open"] and nc["Close"] < nc["Open"] and move_pct > 0.03:
            obs["bearish_ob"] = {"high": round(float(c["High"]), 2), "low": round(float(c["Low"]), 2)}
    return obs


def detect_fvg(df: pd.DataFrame) -> dict:
    """Rileva Fair Value Gap (FVG)."""
    fvg = {"bullish_fvg": None, "bearish_fvg": None}
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if c3["Low"] > c1["High"]:
            fvg["bullish_fvg"] = {"top": round(float(c3["Low"]), 2), "bottom": round(float(c1["High"]), 2)}
        if c3["High"] < c1["Low"]:
            fvg["bearish_fvg"] = {"top": round(float(c1["Low"]), 2), "bottom": round(float(c3["High"]), 2)}
    return fvg


def detect_liquidity(df: pd.DataFrame) -> dict:
    """Rileva EQH (Equal Highs) e EQL (Equal Lows) — zone di liquidità."""
    tol    = 0.0005
    recent = df.tail(100)
    highs  = recent["High"].values
    lows   = recent["Low"].values
    eqh, eql = [], []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / (highs[i] + 1e-9) < tol:
                eqh.append((highs[i] + highs[j]) / 2)
            if abs(lows[i] - lows[j]) / (lows[i] + 1e-9) < tol:
                eql.append((lows[i] + lows[j]) / 2)
    return {
        "eqh": round(float(np.mean(eqh)), 2) if eqh else None,
        "eql": round(float(np.mean(eql)), 2) if eql else None
    }


def detect_breaker_blocks(df: pd.DataFrame) -> dict:
    """
    Rileva Breaker Block: un Order Block che è stato violato.
    Dopo la violazione diventa zona di inversione.
    """
    result = {"bullish_bb": None, "bearish_bb": None}
    price  = float(df["Close"].iloc[-1])
    ob     = detect_order_blocks(df)
    if ob["bullish_ob"] and price < ob["bullish_ob"]["low"]:
        result["bearish_bb"] = ob["bullish_ob"]
    if ob["bearish_ob"] and price > ob["bearish_ob"]["high"]:
        result["bullish_bb"] = ob["bearish_ob"]
    return result


def detect_mitigation_blocks(df: pd.DataFrame) -> dict:
    """
    Rileva Mitigation Block: candela che ha mitigato (toccato) un OB precedente.
    """
    ob    = detect_order_blocks(df)
    price = float(df["Close"].iloc[-1])
    mit   = {"bullish_mit": False, "bearish_mit": False}
    if ob["bullish_ob"]:
        zone_high = ob["bullish_ob"]["high"]
        zone_low  = ob["bullish_ob"]["low"]
        if zone_low <= price <= zone_high:
            mit["bullish_mit"] = True
    if ob["bearish_ob"]:
        zone_high = ob["bearish_ob"]["high"]
        zone_low  = ob["bearish_ob"]["low"]
        if zone_low <= price <= zone_high:
            mit["bearish_mit"] = True
    return mit


def detect_premium_discount(df: pd.DataFrame, structure: dict) -> str:
    """
    Determina se il prezzo è in zona Premium (sopra equilibrio) o Discount (sotto).
    Equilibrio = 50% del range dell'ultimo swing.
    """
    last_high = structure.get("last_high")
    last_low  = structure.get("last_low")
    price     = float(df["Close"].iloc[-1])
    if last_high and last_low:
        midpoint = (last_high + last_low) / 2
        if price > midpoint:
            return "PREMIUM"
        else:
            return "DISCOUNT"
    return "EQUILIBRIUM"


# ═══════════════════════════════════════════════
# LIVELLO 04 — PATTERN CANDELE GIAPPONESI
# ═══════════════════════════════════════════════

def detect_candle_pattern(df: pd.DataFrame) -> tuple:
    """Rileva pattern candele giapponesi estesi."""
    if len(df) < 3:
        return "", "NEUTRAL"
    curr  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]
    o, h, l, c    = float(curr["Open"]), float(curr["High"]), float(curr["Low"]), float(curr["Close"])
    po, ph, pl, pc = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    p2o, p2c       = float(prev2["Open"]), float(prev2["Close"])

    body         = abs(c - o)
    rng          = h - l
    upper_wick   = h - max(o, c)
    lower_wick   = min(o, c) - l

    if rng == 0:
        return "", "NEUTRAL"

    if body <= rng * 0.1:
        return "🕯 Doji", "NEUTRAL"
    if lower_wick >= body * 2 and upper_wick <= body * 0.3 and c > o:
        return "🔨 Hammer", "BUY"
    if upper_wick >= body * 2 and lower_wick <= body * 0.3 and c < o:
        return "⭐ Shooting Star", "SELL"
    if c > o and pc < po and c > po and o < pc:
        return "📈 Engulfing Bullish", "BUY"
    if c < o and pc > po and c < po and o > pc:
        return "📉 Engulfing Bearish", "SELL"
    if p2c < p2o and abs(pc - po) <= (ph - pl) * 0.3 and c > o and c > (p2o + p2c) / 2:
        return "🌅 Morning Star", "BUY"
    if p2c > p2o and abs(pc - po) <= (ph - pl) * 0.3 and c < o and c < (p2o + p2c) / 2:
        return "🌆 Evening Star", "SELL"
    if lower_wick >= rng * 0.6 and body <= rng * 0.3:
        return "📌 Pinbar Bullish", "BUY"
    if upper_wick >= rng * 0.6 and body <= rng * 0.3:
        return "📌 Pinbar Bearish", "SELL"
    if c < o and abs(c - o) >= rng * 0.7:
        return "🕯 Marubozu Bearish", "SELL"
    if c > o and abs(c - o) >= rng * 0.7:
        return "🕯 Marubozu Bullish", "BUY"
    if c > o and pc > po and c > pc and o > po:
        return "📈 Tre Soldati Bianchi", "BUY"
    if c < o and pc < po and c < pc and o < po:
        return "📉 Tre Corvi Neri", "SELL"

    return "", "NEUTRAL"


# ═══════════════════════════════════════════════
# LIVELLO 05 — SUPPORTO E RESISTENZA
# ═══════════════════════════════════════════════

def get_support_resistance(df: pd.DataFrame) -> dict:
    """
    Calcola livelli S/R su più periodi e pivot points.
    """
    recent = df.tail(50)
    h      = df.tail(100)

    support_50    = round(float(recent["Low"].min()), 2)
    resistance_50 = round(float(recent["High"].max()), 2)
    support_100   = round(float(h["Low"].min()), 2)
    resistance_100 = round(float(h["High"].max()), 2)

    # Pivot classico
    last_high  = float(df["High"].iloc[-1])
    last_low   = float(df["Low"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    pivot  = round((last_high + last_low + last_close) / 3, 2)
    r1     = round(2 * pivot - last_low, 2)
    s1     = round(2 * pivot - last_high, 2)
    r2     = round(pivot + (last_high - last_low), 2)
    s2     = round(pivot - (last_high - last_low), 2)

    return {
        "support":      support_50,
        "resistance":   resistance_50,
        "support_100":  support_100,
        "resistance_100": resistance_100,
        "pivot":        pivot,
        "r1": r1, "r2": r2,
        "s1": s1, "s2": s2,
    }


# ═══════════════════════════════════════════════
# LIVELLO 06 — SENTIMENT E NOTIZIE
# ═══════════════════════════════════════════════

def get_news_sentiment() -> dict:
    """
    Scarica le ultime notizie sull'oro e calcola un sentiment score
    basato su keyword bullish/bearish.
    """
    try:
        url    = "https://newsapi.org/v2/everything"
        params = {
            "q":        "gold XAU price OR gold market OR fed rates OR inflation",
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 10,
            "apiKey":   NEWS_API_KEY
        }
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return {"score": 0, "label": "NEUTRAL", "articles": []}

        bullish_kw = ["surge", "rise", "rally", "gain", "bullish", "buy", "higher",
                      "record", "strong", "demand", "safe haven", "inflation", "dovish"]
        bearish_kw = ["fall", "drop", "decline", "bearish", "sell", "lower", "weak",
                      "pressure", "hawkish", "rate hike", "dollar strong", "risk on"]

        score    = 0
        articles = []
        for a in data.get("articles", [])[:5]:
            title = (a.get("title") or "").lower()
            desc  = (a.get("description") or "").lower()
            text  = title + " " + desc
            for kw in bullish_kw:
                if kw in text:
                    score += 1
            for kw in bearish_kw:
                if kw in text:
                    score -= 1
            articles.append({
                "title":  a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "date":   a.get("publishedAt", "")[:10]
            })

        label = "BULLISH" if score > 2 else "BEARISH" if score < -2 else "NEUTRAL"
        return {"score": score, "label": label, "articles": articles}

    except Exception as e:
        logger.warning(f"Errore sentiment notizie: {e}")
        return {"score": 0, "label": "NEUTRAL", "articles": []}


# ═══════════════════════════════════════════════
# LIVELLO 07 — CALENDARIO ECONOMICO
# ═══════════════════════════════════════════════

def get_economic_events() -> dict:
    """
    Controlla eventi macro importanti oggi tramite ForexFactory RSS.
    Filtra eventi ad alto impatto per USD e XAU.
    """
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r   = requests.get(url, timeout=10)
        events_raw = r.json()

        today    = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        high_imp = []
        keywords = ["fed", "fomc", "cpi", "ppi", "nfp", "gdp", "rate", "powell",
                    "inflation", "employment", "payroll", "interest"]

        for ev in events_raw:
            ev_date  = ev.get("date", "")[:10]
            impact   = ev.get("impact", "").lower()
            title    = ev.get("title", "").lower()
            currency = ev.get("country", "").lower()

            if ev_date == today and impact == "high" and currency in ["usd", "us"]:
                if any(kw in title for kw in keywords):
                    high_imp.append({
                        "title":    ev.get("title", ""),
                        "time":     ev.get("date", "")[11:16],
                        "forecast": ev.get("forecast", "N/A"),
                        "previous": ev.get("previous", "N/A"),
                    })

        has_high_impact = len(high_imp) > 0
        return {"events": high_imp, "high_impact_today": has_high_impact}

    except Exception as e:
        logger.warning(f"Errore calendario: {e}")
        return {"events": [], "high_impact_today": False}


# ═══════════════════════════════════════════════
# LIVELLO 08 — REGIME DI MERCATO
# ═══════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> dict:
    """
    Rileva il regime di mercato: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE.
    Usa ADX, Bollinger Width, ATR e slope EMA.
    """
    row     = df.iloc[-1]
    adx     = float(row["adx"])    if not pd.isna(row["adx"])    else 0
    bb_w    = float(row["bb_width"]) if not pd.isna(row["bb_width"]) else 0
    atr     = float(row["atr"])
    avg_atr = float(df["atr"].tail(20).mean())
    ema20   = float(row["ema20"])  if not pd.isna(row["ema20"])  else 0
    ema50   = float(row["ema50"])  if not pd.isna(row["ema50"])  else 0
    roc     = float(row["roc"])    if not pd.isna(row["roc"])    else 0

    if adx >= 25 and ema20 > ema50 and roc > 0:
        regime = "TRENDING_UP"
    elif adx >= 25 and ema20 < ema50 and roc < 0:
        regime = "TRENDING_DOWN"
    elif adx < 20 and bb_w < 0.015:
        regime = "RANGING"
    elif atr > avg_atr * 1.8:
        regime = "VOLATILE"
    else:
        regime = "NORMAL"

    return {
        "regime": regime,
        "adx":    round(adx, 1),
        "bb_w":   round(bb_w * 100, 2),
        "atr_vs_avg": round(atr / (avg_atr + 1e-9), 2)
    }


# ═══════════════════════════════════════════════
# LIVELLO 09 — TREND MULTI-TIMEFRAME
# ═══════════════════════════════════════════════

def get_mtf_trend(mtf_data: dict) -> dict:
    """
    Analizza il trend su tutti i timeframe disponibili.
    Restituisce un dizionario con la direzione per ogni TF.
    """
    trends = {}
    for tf, df in mtf_data.items():
        try:
            df  = compute_indicators(df)
            row = df.iloc[-1]
            ema20 = float(row["ema20"]) if not pd.isna(row["ema20"]) else 0
            ema50 = float(row["ema50"]) if not pd.isna(row["ema50"]) else 0
            macd  = float(row["macd"])  if not pd.isna(row["macd"])  else 0
            sig   = float(row["macd_sig"]) if not pd.isna(row["macd_sig"]) else 0
            rsi   = float(row["rsi"])   if not pd.isna(row["rsi"])   else 50

            if ema20 > ema50 and macd > sig and rsi > 50:
                trends[tf] = "BUY"
            elif ema20 < ema50 and macd < sig and rsi < 50:
                trends[tf] = "SELL"
            else:
                trends[tf] = "NEUTRAL"
        except:
            trends[tf] = "NEUTRAL"
    return trends


def get_mtf_score(mtf_trends: dict, signal: str) -> int:
    """
    Calcola quanti timeframe confermano il segnale.
    Più timeframe allineati = segnale più forte.
    """
    score = 0
    weights = {"1min": 1, "5min": 2, "15min": 2, "1h": 3, "4h": 3, "1day": 3}
    for tf, trend in mtf_trends.items():
        if trend == signal:
            score += weights.get(tf, 1)
    return score


# ═══════════════════════════════════════════════
# LIVELLO 10 — SISTEMA DI PUNTEGGIO AVANZATO
# ═══════════════════════════════════════════════

def compute_score(row, smc, ob, fvg, liquidity, mtf_trends,
                  candle_dir, sentiment, regime_data, sr) -> tuple:
    """
    Sistema di punteggio completo: 0-50 punti per BUY e SELL.
    Ogni categoria ha un peso diverso.
    """
    price  = float(row["Close"])
    ema20  = float(row["ema20"])  if not pd.isna(row["ema20"])  else price
    ema50  = float(row["ema50"])  if not pd.isna(row["ema50"])  else price
    ema200 = float(row["ema200"]) if not pd.isna(row["ema200"]) else price
    macd   = float(row["macd"])   if not pd.isna(row["macd"])   else 0
    sig    = float(row["macd_sig"]) if not pd.isna(row["macd_sig"]) else 0
    hist   = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
    rsi    = float(row["rsi"])    if not pd.isna(row["rsi"])    else 50
    rsi_f  = float(row["rsi_fast"]) if not pd.isna(row["rsi_fast"]) else 50
    bb_u   = float(row["bb_upper"]) if not pd.isna(row["bb_upper"]) else price
    bb_l   = float(row["bb_lower"]) if not pd.isna(row["bb_lower"]) else price
    sk     = float(row["stoch_k"]) if not pd.isna(row["stoch_k"]) else 50
    sd     = float(row["stoch_d"]) if not pd.isna(row["stoch_d"]) else 50
    adx    = float(row["adx"])    if not pd.isna(row["adx"])    else 0
    adxp   = float(row["adx_pos"]) if not pd.isna(row["adx_pos"]) else 0
    adxn   = float(row["adx_neg"]) if not pd.isna(row["adx_neg"]) else 0
    cci    = float(row["cci"])    if not pd.isna(row["cci"])    else 0
    willr  = float(row["willr"])  if not pd.isna(row["willr"])  else -50
    roc    = float(row["roc"])    if not pd.isna(row["roc"])    else 0
    atr    = float(row["atr"])

    support    = sr["support"]
    resistance = sr["resistance"]

    buy = 0
    sell = 0

    # ── Trend EMA (peso 8) ──
    if ema20 > ema50:  buy += 2
    else:              sell += 2
    if ema20 > ema200: buy += 2
    else:              sell += 2
    if price > ema200: buy += 2
    else:              sell += 2
    if ema50 > ema200: buy += 2
    else:              sell += 2

    # ── MACD (peso 6) ──
    if macd > sig:     buy += 2
    else:              sell += 2
    if hist > 0:       buy += 2
    else:              sell += 2
    if hist > 0 and float(row["macd_hist"]) > 0: buy += 2
    else:              sell += 2

    # ── RSI (peso 6) ──
    if rsi < 30:       buy += 3
    elif rsi < 45:     buy += 2
    elif rsi < 50:     buy += 1
    if rsi > 70:       sell += 3
    elif rsi > 55:     sell += 2
    elif rsi > 50:     sell += 1
    if rsi_f < 30:     buy += 1
    if rsi_f > 70:     sell += 1
    if rsi_f < rsi:    buy += 1
    else:              sell += 1

    # ── Bollinger (peso 4) ──
    if price <= bb_l:  buy += 2
    if price >= bb_u:  sell += 2
    if price < bb_l:   buy += 2
    if price > bb_u:   sell += 2

    # ── Stocastico (peso 4) ──
    if sk < 20 and sk > sd: buy += 2
    if sk > 80 and sk < sd: sell += 2
    if sk < 30:        buy += 1
    if sk > 70:        sell += 1
    if sk > sd:        buy += 1
    else:              sell += 1

    # ── ADX (peso 3) ──
    if adx >= 25:
        if adxp > adxn: buy += 2
        else:           sell += 2
    if adx >= 35:
        if adxp > adxn: buy += 1
        else:           sell += 1

    # ── CCI (peso 2) ──
    if cci < -100: buy += 1
    if cci > 100:  sell += 1
    if cci < -150: buy += 1
    if cci > 150:  sell += 1

    # ── Williams %R (peso 2) ──
    if willr < -80: buy += 1
    if willr > -20: sell += 1
    if willr < -90: buy += 1
    if willr > -10: sell += 1

    # ── ROC (peso 2) ──
    if roc > 0: buy += 1
    if roc < 0: sell += 1
    if roc > 0.5: buy += 1
    if roc < -0.5: sell += 1

    # ── S/R (peso 3) ──
    near_s = abs(price - support) <= atr * 0.5
    near_r = abs(price - resistance) <= atr * 0.5
    if near_s: buy += 2
    if near_r: sell += 2
    if price > resistance: buy += 1
    if price < support:    sell += 1

    # ── Candele (peso 4) ──
    if candle_dir == "BUY":  buy += 4
    if candle_dir == "SELL": sell += 4

    # ── SMC Struttura (peso 6) ──
    struct = smc["structure"]
    if struct == "BULLISH": buy += 3
    if struct == "BEARISH": sell += 3
    if smc.get("bos") == "BOS_BULLISH":  buy += 2
    if smc.get("bos") == "BOS_BEARISH":  sell += 2
    if smc.get("choch") == "CHOCH_BULLISH": buy += 1
    if smc.get("choch") == "CHOCH_BEARISH": sell += 1

    # ── Order Block (peso 4) ──
    if ob.get("bullish_ob"):
        ob_h = ob["bullish_ob"]["high"]
        ob_l = ob["bullish_ob"]["low"]
        if ob_l <= price <= ob_h + atr: buy += 4
    if ob.get("bearish_ob"):
        ob_h = ob["bearish_ob"]["high"]
        ob_l = ob["bearish_ob"]["low"]
        if ob_l - atr <= price <= ob_h: sell += 4

    # ── FVG (peso 3) ──
    if fvg.get("bullish_fvg"):
        f = fvg["bullish_fvg"]
        if f["bottom"] <= price <= f["top"]: buy += 3
    if fvg.get("bearish_fvg"):
        f = fvg["bearish_fvg"]
        if f["bottom"] <= price <= f["top"]: sell += 3

    # ── Liquidità (peso 2) ──
    if liquidity.get("eql") and abs(price - liquidity["eql"]) <= atr: buy += 2
    if liquidity.get("eqh") and abs(price - liquidity["eqh"]) <= atr: sell += 2

    # ── MTF (peso 8) ──
    buy  += get_mtf_score(mtf_trends, "BUY")
    sell += get_mtf_score(mtf_trends, "SELL")

    # ── Sentiment (peso 3) ──
    s_label = sentiment.get("label", "NEUTRAL")
    if s_label == "BULLISH": buy += 3
    if s_label == "BEARISH": sell += 3

    return buy, sell


def estimate_probability(score: int, max_score: int, rsi: float, trend_confirmed: bool,
                         regime: str, structure: str, sentiment_label: str,
                         economic_risk: bool) -> int:
    """Stima la probabilità di successo del trade."""
    ratio = score / (max_score + 1e-9)
    base  = int(40 + ratio * 55)

    if trend_confirmed:         base += 5
    if regime in ["TRENDING_UP", "TRENDING_DOWN"]: base += 3
    if structure in ["BULLISH", "BEARISH"]:        base += 2
    if sentiment_label != "NEUTRAL":               base += 2
    if economic_risk:                              base -= 10
    if rsi < 25 or rsi > 75:                       base += 3

    return min(max(base, 40), 97)


# ═══════════════════════════════════════════════
# LIVELLO 11 — TIPO ORDINE AUTOMATICO
# ═══════════════════════════════════════════════

def determine_order_type(signal: str, price: float, sr: dict, atr: float,
                         adx: float, rsi: float, structure: str, ob: dict,
                         fvg: dict, regime: str, pd_zone: str) -> tuple:
    """
    Determina automaticamente il tipo di ordine ottimale.
    Considera struttura, regime, zone SMC e momentum.
    """
    support    = sr["support"]
    resistance = sr["resistance"]
    near_s     = abs(price - support) <= atr * 0.5
    near_r     = abs(price - resistance) <= atr * 0.5

    if signal == "BUY":
        # In zona di discount + OB + struttura bullish → BUY a mercato
        if pd_zone == "DISCOUNT" and ob.get("bullish_ob") and structure == "BULLISH" and rsi < 45:
            return "BUY", price

        # Prezzo in OB bullish → BUY a mercato
        if ob.get("bullish_ob"):
            ob_low = ob["bullish_ob"]["low"]
            if abs(price - ob_low) <= atr:
                return "BUY", price

        # Breakout della resistenza → BUY STOP
        if price > resistance * 0.9995 and adx >= 20 and regime in ["TRENDING_UP", "NORMAL"]:
            entry = round(resistance + atr * 0.1, 2)
            return "BUY STOP", entry

        # Attesa ritracciamento → BUY LIMIT
        if not near_s and rsi < 60 and structure in ["BULLISH", "NEUTRAL"]:
            if ob.get("bullish_ob"):
                entry = round(ob["bullish_ob"]["high"], 2)
            elif fvg.get("bullish_fvg"):
                entry = round(fvg["bullish_fvg"]["bottom"], 2)
            else:
                entry = round(support + atr * 0.3, 2)
            return "BUY LIMIT", entry

        # Breakout volatile → BUY STOP LIMIT
        if price > resistance and regime == "VOLATILE":
            entry = round(resistance + atr * 0.2, 2)
            return "BUY STOP LIMIT", entry

        return "BUY", price

    elif signal == "SELL":
        # In zona di premium + OB bearish + struttura bearish → SELL a mercato
        if pd_zone == "PREMIUM" and ob.get("bearish_ob") and structure == "BEARISH" and rsi > 55:
            return "SELL", price

        # Prezzo in OB bearish → SELL a mercato
        if ob.get("bearish_ob"):
            ob_high = ob["bearish_ob"]["high"]
            if abs(price - ob_high) <= atr:
                return "SELL", price

        # Breakdown del supporto → SELL STOP
        if price < support * 1.0005 and adx >= 20 and regime in ["TRENDING_DOWN", "NORMAL"]:
            entry = round(support - atr * 0.1, 2)
            return "SELL STOP", entry

        # Attesa rimbalzo → SELL LIMIT
        if not near_r and rsi > 40 and structure in ["BEARISH", "NEUTRAL"]:
            if ob.get("bearish_ob"):
                entry = round(ob["bearish_ob"]["low"], 2)
            elif fvg.get("bearish_fvg"):
                entry = round(fvg["bearish_fvg"]["top"], 2)
            else:
                entry = round(resistance - atr * 0.3, 2)
            return "SELL LIMIT", entry

        # Breakdown volatile → SELL STOP LIMIT
        if price < support and regime == "VOLATILE":
            entry = round(support - atr * 0.2, 2)
            return "SELL STOP LIMIT", entry

        return "SELL", price

    return "NEUTRAL", price


# ═══════════════════════════════════════════════
# LIVELLO 12 — RISK MANAGEMENT
# ═══════════════════════════════════════════════

def calculate_risk_levels(signal: str, entry: float, atr: float,
                          regime: str, score: int) -> dict:
    """
    Calcola SL, TP1, TP2, TP3 e BE in modo dinamico
    in base alla volatilità e alla forza del segnale.
    """
    # Moltiplicatori dinamici basati sul regime
    if regime == "VOLATILE":
        sl_mult  = 1.0
        tp1_mult = 0.8
        tp2_mult = 1.5
        tp3_mult = 2.5
    elif regime in ["TRENDING_UP", "TRENDING_DOWN"]:
        sl_mult  = 0.6
        tp1_mult = 0.6
        tp2_mult = 1.4
        tp3_mult = 2.5
    elif regime == "RANGING":
        sl_mult  = 0.5
        tp1_mult = 0.5
        tp2_mult = 0.9
        tp3_mult = 1.4
    else:
        sl_mult  = 0.6
        tp1_mult = 0.6
        tp2_mult = 1.2
        tp3_mult = 2.0

    sl_dist  = round(atr * sl_mult, 2)
    tp1_dist = round(atr * tp1_mult, 2)
    tp2_dist = round(atr * tp2_mult, 2)
    tp3_dist = round(atr * tp3_mult, 2)

    be_pips = 10  # Break even fisso a +10 pips

    if signal == "BUY":
        sl   = round(entry - sl_dist, 2)
        tp1  = round(entry + tp1_dist, 2)
        tp2  = round(entry + tp2_dist, 2)
        tp3  = round(entry + tp3_dist, 2)
        be   = round(entry + be_pips, 2)
    else:
        sl   = round(entry + sl_dist, 2)
        tp1  = round(entry - tp1_dist, 2)
        tp2  = round(entry - tp2_dist, 2)
        tp3  = round(entry - tp3_dist, 2)
        be   = round(entry - be_pips, 2)

    rr1 = round(tp1_dist / sl_dist, 2) if sl_dist > 0 else 0
    rr2 = round(tp2_dist / sl_dist, 2) if sl_dist > 0 else 0
    rr3 = round(tp3_dist / sl_dist, 2) if sl_dist > 0 else 0

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "be": be, "rr1": rr1, "rr2": rr2, "rr3": rr3,
        "sl_dist": sl_dist
    }


# ═══════════════════════════════════════════════
# LIVELLO 13 — INDICAZIONI OPERATIVE
# ═══════════════════════════════════════════════

def generate_trade_notes(signal: str, order_type: str, score: int, max_score: int,
                         prob: int, rr1: float, rr3: float, regime: str,
                         structure: str, adx: float, rsi: float, bos,
                         choch, candle: str, sentiment_label: str,
                         economic_risk: bool, pd_zone: str, mtf_trends: dict) -> list:
    """
    Genera indicazioni operative complete e dinamiche.
    """
    notes = []
    ratio = score / (max_score + 1e-9)

    # ── Qualità del trade ──
    if ratio >= 0.75:
        notes.append("💎 Trade di altissima qualità — tieni fino a TP3")
    elif ratio >= 0.55:
        notes.append("✅ Trade solido — punta a TP2, lascia 30% per TP3")
    elif ratio >= 0.40:
        notes.append("⚡ Trade moderato — chiudi 50% a TP1, muovi BE")
    else:
        notes.append("⚠️ Trade debole — chiudi tutto a TP1")

    # ── Gestione posizione ──
    notes.append(f"📊 Gestione: chiudi 33% a TP1, 33% a TP2, 34% a TP3")

    # ── Regime ──
    if regime == "VOLATILE":
        notes.append("🌪 Mercato volatile — riduci size del 50%, SL più largo")
    elif regime == "RANGING":
        notes.append("📦 Mercato laterale — TP1 è l'obiettivo principale, non forzare TP3")
    elif regime in ["TRENDING_UP", "TRENDING_DOWN"]:
        notes.append("📈 Trend forte — lascia correre, trail SL dopo TP2")

    # ── Premium/Discount ──
    if signal == "BUY" and pd_zone == "DISCOUNT":
        notes.append("✅ Prezzo in zona Discount — entry ottimale per BUY")
    elif signal == "BUY" and pd_zone == "PREMIUM":
        notes.append("⚠️ Prezzo in zona Premium — BUY contro struttura, attenzione")
    elif signal == "SELL" and pd_zone == "PREMIUM":
        notes.append("✅ Prezzo in zona Premium — entry ottimale per SELL")
    elif signal == "SELL" and pd_zone == "DISCOUNT":
        notes.append("⚠️ Prezzo in zona Discount — SELL contro struttura, attenzione")

    # ── SMC ──
    if bos:
        notes.append(f"🏗 BOS {bos} — struttura confermata, trade con il trend")
    if choch:
        notes.append(f"🔄 CHoCH {choch} — possibile inversione, sii pronto al'uscita rapida")

    # ── Candele ──
    if candle:
        notes.append(f"🕯 Conferma candela: {candle}")

    # ── MTF Alignment ──
    aligned = [tf for tf, t in mtf_trends.items() if t == signal]
    if len(aligned) >= 4:
        notes.append(f"🎯 MTF allineati: {', '.join(aligned)} — segnale molto forte")
    elif len(aligned) >= 2:
        notes.append(f"📐 MTF parzialmente allineati: {', '.join(aligned)}")
    else:
        notes.append("⚠️ Scarso allineamento MTF — trade più rischioso")

    # ── Sentiment ──
    if sentiment_label == "BULLISH" and signal == "BUY":
        notes.append("📰 Sentiment news: BULLISH — conferma il trade")
    elif sentiment_label == "BEARISH" and signal == "SELL":
        notes.append("📰 Sentiment news: BEARISH — conferma il trade")
    elif sentiment_label != "NEUTRAL":
        notes.append(f"📰 Sentiment news: {sentiment_label} — contro il trade, cautela")

    # ── Rischio economico ──
    if economic_risk:
        notes.append("⚠️ ATTENZIONE: News ad alto impatto oggi! Riduci size o evita il trade")

    # ── RR ──
    if rr1 < 1.0:
        notes.append(f"⚠️ R:R a TP1 sotto 1:1 ({rr1}) — valuta se vale il rischio")
    if rr3 >= 3.0:
        notes.append(f"🏆 R:R a TP3 eccellente ({rr3}:1) — trade ad alto potenziale")

    # ── ADX ──
    if adx < 15:
        notes.append("📉 ADX molto basso — mercato piatto, preferisci aspettare")

    return notes


# ═══════════════════════════════════════════════
# ANALISI COMPLETA — ENTRY POINT
# ═══════════════════════════════════════════════

def full_analyze() -> dict:
    """
    Esegue l'analisi completa su tutti i livelli.
    Restituisce un dizionario con tutti i dati del trade.
    """
    now = datetime.now(TIMEZONE)

    # ── Dati multi-timeframe ──
    mtf_data = get_multi_timeframe_data()
    df_5m    = mtf_data.get("5min")
    if df_5m is None or len(df_5m) < 50:
        raise ValueError("Dati 5min non disponibili")

    # ── Indicatori ──
    df_5m = compute_indicators(df_5m)
    df_5m = detect_swing_points(df_5m)

    # ── MTF trend ──
    mtf_trends = get_mtf_trend(mtf_data)

    # ── Valori correnti ──
    row   = df_5m.iloc[-1]
    price = round(float(row["Close"]), 2)
    atr   = float(row["atr"])
    rsi   = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50
    adx   = float(row["adx"]) if not pd.isna(row["adx"]) else 0

    # ── SMC ──
    smc       = detect_bos_choch(df_5m)
    ob        = detect_order_blocks(df_5m)
    fvg       = detect_fvg(df_5m)
    liquidity = detect_liquidity(df_5m)
    bb        = detect_breaker_blocks(df_5m)
    mit       = detect_mitigation_blocks(df_5m)
    pd_zone   = detect_premium_discount(df_5m, smc)

    # ── S/R ──
    sr = get_support_resistance(df_5m)

    # ── Candele ──
    candle_pattern, candle_dir = detect_candle_pattern(df_5m)

    # ── Regime ──
    regime_data = detect_market_regime(df_5m)
    regime      = regime_data["regime"]

    # ── Sentiment ──
    sentiment = get_news_sentiment()

    # ── Calendario ──
    calendar = get_economic_events()
    econ_risk = calendar["high_impact_today"]

    # ── Punteggio ──
    buy_score, sell_score = compute_score(
        row, smc, ob, fvg, liquidity, mtf_trends,
        candle_dir, sentiment, regime_data, sr
    )
    max_score = 80  # punteggio massimo teorico

    # ── Segnale ──
    min_score = 20  # soglia minima per generare segnale

    if buy_score >= min_score and buy_score > sell_score:
        signal = "BUY"
        score  = buy_score
        trend_confirmed = mtf_trends.get("1h") == "BUY"
    elif sell_score >= min_score and sell_score > buy_score:
        signal = "SELL"
        score  = sell_score
        trend_confirmed = mtf_trends.get("1h") == "SELL"
    else:
        return {
            "signal":  "NEUTRAL",
            "price":   price,
            "regime":  regime,
            "buy_score":  buy_score,
            "sell_score": sell_score,
            "time":    now.strftime("%d/%m/%Y %H:%M"),
        }

    # ── Tipo ordine ──
    order_type, entry = determine_order_type(
        signal, price, sr, atr, adx, rsi,
        smc["structure"], ob, fvg, regime, pd_zone
    )

    # ── Risk levels ──
    risk = calculate_risk_levels(signal, entry, atr, regime, score)

    # ── Probabilità ──
    prob = estimate_probability(
        score, max_score, rsi, trend_confirmed,
        regime, smc["structure"],
        sentiment["label"], econ_risk
    )

    # ── Indicazioni ──
    notes = generate_trade_notes(
        signal, order_type, score, max_score, prob,
        risk["rr1"], risk["rr3"], regime, smc["structure"],
        adx, rsi, smc.get("bos"), smc.get("choch"),
        candle_pattern, sentiment["label"], econ_risk,
        pd_zone, mtf_trends
    )

    return {
        # Segnale base
        "signal":      signal,
        "order_type":  order_type,
        "price":       price,
        "entry":       entry,

        # Livelli
        "sl":          risk["sl"],
        "tp1":         risk["tp1"],
        "tp2":         risk["tp2"],
        "tp3":         risk["tp3"],
        "be":          risk["be"],

        # R:R
        "rr1":         risk["rr1"],
        "rr2":         risk["rr2"],
        "rr3":         risk["rr3"],

        # Metriche
        "prob":        prob,
        "score":       score,
        "max_score":   max_score,
        "atr":         round(atr, 2),
        "rsi":         round(rsi, 1),
        "adx":         round(adx, 1),

        # Regime e struttura
        "regime":      regime,
        "structure":   smc["structure"],
        "pd_zone":     pd_zone,
        "bos":         smc.get("bos"),
        "choch":       smc.get("choch"),

        # SMC dettagli
        "ob":          ob,
        "fvg":         fvg,
        "liquidity":   liquidity,
        "breaker":     bb,
        "mitigation":  mit,

        # MTF
        "mtf_trends":  mtf_trends,
        "trend_confirmed": trend_confirmed,

        # Candele
        "candle":      candle_pattern,

        # S/R
        "sr":          sr,

        # Sentiment
        "sentiment":   sentiment,

        # Calendario
        "calendar":    calendar,
        "econ_risk":   econ_risk,

        # Indicazioni
        "notes":       notes,

        # Meta
        "time":        now.strftime("%d/%m/%Y %H:%M"),
        "buy_score":   buy_score,
        "sell_score":  sell_score,
    }
