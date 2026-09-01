"""
analyzer.py — SISTEMA COMPLETO XAU/USD
Strategie: SMC v3.0, Trend Following, Mean Reversion, Momentum,
Event-Driven, Statistical Arbitrage, ML Alpha, Candlestick, Order Flow
Multi-timeframe: 1min, 5min, 15min, 1h, 4h, 1day
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
TIMEZONE       = pytz.timezone("Europe/Rome")

# ═══════════════════════════════════════════════════════════════
# CACHE DATI — riduce le chiamate API (piano gratuito: 8 call/min)
# ═══════════════════════════════════════════════════════════════
_data_cache = {}  # {interval: (timestamp_unix, dataframe)}
_price_cache = {"timestamp": 0, "price": 0.0}
_data_fail_cache = {}  # {interval: timestamp_unix ultimo fallimento totale}
_FAIL_BACKOFF = 30  # secondi di pausa dopo un fallimento totale su un TF

# Timestamp dell'ultimo fetch candele riuscito, su QUALSIASI timeframe/fonte.
# Inizializzato a "ora" (non 0) per non generare un falso allarme "cieco" nei
# secondi subito dopo l'avvio del processo, prima ancora del primo fetch.
_last_data_success_ts = time.time()


def seconds_since_last_data_success() -> float:
    """Da quanto tempo NESSUN timeframe riesce a scaricare candele da nessuna fonte."""
    return time.time() - _last_data_success_ts

# Se Twelve Data segnala quota giornaliera esaurita, smettiamo di richiamarla
# fino a mezzanotte UTC invece di continuare a provarci a ogni fetch — trovato
# in produzione: quando yfinance+Stooq falliscono insieme, OGNI fetch di OGNI
# timeframe cascata su Twelve Data (moltiplicato ×6 da get_multi_timeframe_data
# dentro full_analyze), esaurendo 800 crediti/giorno in poche ore invece che mai.
_twelvedata_blocked_until = 0.0


def _twelvedata_quota_exceeded(exc: Exception) -> bool:
    return "api credits" in str(exc).lower()


def _seconds_until_utc_midnight() -> float:
    from datetime import timezone
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (tomorrow - now).total_seconds()

# Ogni quanti secondi è lecito riscaricare ciascun timeframe
CACHE_TTL = {
    "1min":  300,   # ogni 5 min (era 55s — troppo costoso)
    "5min":  300,   # ogni 5 min (allineato al ciclo bot)
    "15min": 900,   # ogni 15 min
    "1h":    3600,  # ogni 60 min
    "4h":    7200,  # ogni 2 ore
    "1day":  14400, # ogni 4 ore
}

# ═══════════════════════════════════════════════════════════════
# LIVELLO 01 — DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def _interval_to_yf(interval: str) -> tuple:
    mapping = {
        "1min":  ("1m",  "7d"),
        "5min":  ("5m",  "60d"),
        "15min": ("15m", "60d"),
        # Yahoo limita gli intervalli >=1h e <1d a 730 giorni di storico:
        # è il massimo ottenibile gratis, anche per backtest lunghi.
        "1h":    ("1h",  "730d"),
        "4h":    ("1h",  "730d"),
        # Daily non ha il limite dei 730gg: "max" serve i backtest 2/5/10/20 anni.
        "1day":  ("1d",  "max"),
    }
    return mapping.get(interval, ("5m", "60d"))


def _fetch_yfinance(interval: str, outputsize: int) -> pd.DataFrame:
    """
    Yahoo Finance — gratuito, nessuna API key, nessun rate limit.
    GC=F (futures oro COMEX): il simbolo forex-spot "XAUUSD=X" è stato
    rimosso da Yahoo (404 su tutte le richieste) — GC=F è l'unico proxy
    gold ancora attivo. Scostamento tipico dallo spot: pochi dollari di basis.
    """
    import yfinance as yf
    yf_interval, yf_period = _interval_to_yf(interval)
    df = yf.Ticker("GC=F").history(period=yf_period, interval=yf_interval, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance: nessun dato XAU/USD {interval}")
    if interval == "4h":
        df = df.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    df = df[["Open","High","Low","Close"]].copy()
    df["Volume"] = 0
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    if len(df) > outputsize:
        df = df.iloc[-outputsize:]
    return df


def _fetch_twelvedata(interval: str, outputsize: int) -> pd.DataFrame:
    """Twelve Data — fallback quando yfinance non è disponibile."""
    if not TWELVE_API_KEY:
        raise RuntimeError("TWELVE_API_KEY non configurata")
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":"XAU/USD","interval":interval,"outputsize":outputsize,"apikey":TWELVE_API_KEY}
    r    = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Nessun dato {interval}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    df = df[["Open","High","Low","Close"]].astype(float)
    df["Volume"] = 0
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


def _fetch_stooq(interval: str, outputsize: int) -> pd.DataFrame:
    """
    Stooq.com — fonte storica gratuita, funziona da Railway.
    Supporta dati daily e intraday per XAUUSD.
    """
    # Stooq usa periodi fissi, non outputsize — prendiamo il massimo e tronchiamo
    stooq_interval = {
        "1min": "1", "5min": "5", "15min": "15",
        "1h": "60", "4h": "60",  # 4h: scarica 1h e ricampiona
        "1day": "d",
    }.get(interval, "5")

    if stooq_interval == "d":
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    else:
        url = f"https://stooq.com/q/d/l/?s=xauusd&i={stooq_interval}"

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Stooq: nessun dato per {interval}")

    # Stooq ritorna colonne: Date, Time, Open, High, Low, Close, Volume
    if "Time" in df.columns:
        df.index = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    else:
        df.index = pd.to_datetime(df["Date"])

    df = df[["Open", "High", "Low", "Close"]].copy()
    df["Volume"] = 0
    df = df.astype(float)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    if interval == "4h":
        df = df.resample("4h").agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
        }).dropna()

    if len(df) > outputsize:
        df = df.iloc[-outputsize:]
    return df


def get_data(interval="5min", outputsize=500, bypass_cache=False) -> pd.DataFrame:
    """
    Scarica candele XAU/USD con 3 fonti in cascata.
    1. yfinance — GC=F futures (gratuito, se non bloccato dall'IP Railway)
    2. Stooq (gratuito, funziona da server)
    3. Twelve Data (a pagamento, fallback finale)

    Se TUTTE le fonti falliscono nello stesso giro (es. rate limit
    transitorio dopo diversi download pesanti di fila, come in
    /backtest tutti) si ritenta una seconda volta dopo una breve pausa
    prima di arrendersi — invece di rinunciare al primo intoppo.

    Durante un blackout totale (tutte le fonti giù insieme) un breve
    backoff evita di ripetere lo stesso tentativo fallito a ogni singola
    chiamata: con 5 timeframe controllati ogni 5 minuti e full_analyze()
    che li ri-scarica tutti per ciascuno, senza backoff un blackout
    diventa decine di tentativi falliti al minuto.
    """
    global _twelvedata_blocked_until, _last_data_success_ts
    now = time.time()
    cache_key = (interval, outputsize)
    cached = _data_cache.get(cache_key)
    ttl    = CACHE_TTL.get(interval, 120)
    if not bypass_cache and cached and (now - cached[0]) < ttl:
        return cached[1].copy()

    if not bypass_cache:
        last_fail = _data_fail_cache.get(interval, 0)
        if now - last_fail < _FAIL_BACKOFF:
            if cached:
                return cached[1].copy()
            raise ValueError(f"Nessun dato disponibile per {interval} (backoff dopo fallimento recente)")

    sources = [
        ("yfinance", lambda: _fetch_yfinance(interval, outputsize)),
        ("Stooq",    lambda: _fetch_stooq(interval, outputsize)),
    ]
    if now >= _twelvedata_blocked_until:
        sources.append(("Twelve Data", lambda: _fetch_twelvedata(interval, outputsize)))

    for attempt in (1, 2):
        for name, fetch_fn in sources:
            try:
                df = fetch_fn()
                if df is not None and not df.empty:
                    _data_cache[cache_key] = (now, df)
                    _data_fail_cache.pop(interval, None)
                    _last_data_success_ts = now
                    logger.debug(f"Candele {interval} da {name}: {len(df)} barre")
                    return df.copy()
            except Exception as e:
                if name == "Twelve Data" and _twelvedata_quota_exceeded(e):
                    _twelvedata_blocked_until = now + _seconds_until_utc_midnight()
                    logger.warning(
                        f"Twelve Data: quota esaurita, sospesa per "
                        f"{(_twelvedata_blocked_until - now) / 3600:.1f}h (fino a mezzanotte UTC)"
                    )
                logger.warning(f"{name} fallito per {interval} (tentativo {attempt}/2): {e}")
        if attempt == 1:
            time.sleep(3)

    _data_fail_cache[interval] = now

    if cached:
        logger.warning(f"Tutte le fonti fallite per {interval}, uso cache stale")
        return cached[1].copy()

    raise ValueError(f"Nessun dato disponibile per {interval}")


def get_current_price() -> float:
    """Prezzo live XAU/USD. yfinance primario, Twelve Data fallback."""
    import time
    now = time.time()
    if now - _price_cache["timestamp"] < 30 and _price_cache["price"] > 100:
        return _price_cache["price"]

    try:
        import yfinance as yf
        price = float(yf.Ticker("GC=F").fast_info.last_price or 0)
        if price > 0:
            _price_cache["price"] = price
            _price_cache["timestamp"] = now
            return price
    except Exception:
        pass

    try:
        r = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol":"XAU/USD","apikey":TWELVE_API_KEY},
            timeout=5
        )
        price = float(r.json()["price"])
        if price > 0:
            _price_cache["price"] = price
            _price_cache["timestamp"] = now
            return price
    except Exception:
        pass

    return _price_cache["price"]


def get_dxy_price() -> float:
    """DXY via yfinance (DX-Y.NYB)."""
    try:
        import yfinance as yf
        return float(yf.Ticker("DX-Y.NYB").fast_info.last_price or 0)
    except Exception:
        try:
            r = requests.get("https://api.twelvedata.com/price",
                params={"symbol":"DXY","apikey":TWELVE_API_KEY},timeout=5)
            return float(r.json().get("price",0))
        except Exception:
            return 0.0


def get_us10y_price() -> float:
    """TLT (proxy tassi) via yfinance."""
    try:
        import yfinance as yf
        return float(yf.Ticker("TLT").fast_info.last_price or 0)
    except Exception:
        try:
            r = requests.get("https://api.twelvedata.com/price",
                params={"symbol":"TLT","apikey":TWELVE_API_KEY},timeout=5)
            return float(r.json().get("price",0))
        except Exception:
            return 0.0


def get_multi_timeframe_data() -> dict:
    """Dati su tutti i timeframe chiave.
    M1 è necessario per SMC v3. La cache a cinque minuti evita richieste
    duplicate quando più pipeline usano gli stessi dati nello stesso ciclo.
    """
    timeframes = {
        "1min":  120,
        "5min":  300,
        "15min": 200,
        "1h":    150,
        "4h":    100,
        "1day":  100,
    }
    data = {}
    for tf, size in timeframes.items():
        try:
            data[tf] = get_data(interval=tf, outputsize=size)
        except Exception as e:
            logger.warning(f"Errore dati {tf}: {e}")
    return data


# ═══════════════════════════════════════════════════════════════
# LIVELLO 02 — INDICATORI TECNICI COMPLETI
# ═══════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA
    df["ema9"]   = ta.trend.ema_indicator(df["Close"], window=9)
    df["ema20"]  = ta.trend.ema_indicator(df["Close"], window=20)
    df["ema50"]  = ta.trend.ema_indicator(df["Close"], window=50)
    df["ema100"] = ta.trend.ema_indicator(df["Close"], window=100)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], window=200)
    df["sma20"]  = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma50"]  = ta.trend.sma_indicator(df["Close"], window=50)

    # MACD
    macd            = ta.trend.MACD(df["Close"])
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # RSI
    df["rsi"]      = ta.momentum.rsi(df["Close"], window=14)
    df["rsi_fast"] = ta.momentum.rsi(df["Close"], window=7)

    # Bollinger
    bb             = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
    df["bb_pct"]   = bb.bollinger_pband()

    # ATR
    df["atr"]     = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["atr_pct"] = df["atr"] / (df["Close"] + 1e-9) * 100

    # Stocastico
    stoch         = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ADX
    adx           = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Oscillatori aggiuntivi
    df["cci"]   = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)
    df["willr"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=14)
    df["roc"]   = ta.momentum.roc(df["Close"], window=10)
    df["mom"]   = df["Close"].diff(10)

    # Parabolic SAR
    try:
        psar       = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
        df["psar"] = psar.psar()
        df["psar_up"]   = psar.psar_up()
        df["psar_down"] = psar.psar_down()
    except:
        df["psar"] = df["psar_up"] = df["psar_down"] = np.nan

    # VWAP approssimato
    # NOTA: Twelve Data non fornisce volume reale per XAU/USD (Volume=0).
    # Usare (typical * Volume).cumsum() / Volume.cumsum() con Volume=0 produce
    # VWAP ≈ 0, causando vwap_dev = (price-0)/atr >> 0 → bias SELL artificiale costante.
    # FIX: usiamo la media mobile del typical price (20 periodi) come proxy VWAP.
    # Non è il VWAP reale ma è neutro e non introduce bias direzionale.
    try:
        typical    = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap"] = typical.rolling(window=20, min_periods=1).mean()
    except:
        df["vwap"] = df["Close"]

    # Volume medio
    df["vol_avg"]   = df["Volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_avg"] + 1e-9)

    # Donchian Channel (per Trend Following)
    df["don_high"] = df["High"].rolling(window=20).max()
    df["don_low"]  = df["Low"].rolling(window=20).min()
    df["don_mid"]  = (df["don_high"] + df["don_low"]) / 2

    # Keltner Channel (per Mean Reversion)
    df["kelt_mid"]   = ta.trend.ema_indicator(df["Close"], window=20)
    df["kelt_upper"] = df["kelt_mid"] + 2 * df["atr"]
    df["kelt_lower"] = df["kelt_mid"] - 2 * df["atr"]

    # Squeeze Momentum (Bollinger inside Keltner = squeeze)
    df["squeeze"] = (df["bb_upper"] < df["kelt_upper"]) & (df["bb_lower"] > df["kelt_lower"])

    # Z-score per Mean Reversion
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std  = df["Close"].rolling(window=20).std()
    df["zscore"] = (df["Close"] - rolling_mean) / (rolling_std + 1e-9)

    # Hurst Exponent approssimato (>0.5 = trend, <0.5 = mean reverting)
    try:
        lags    = range(2, 20)
        tau     = [df["Close"].diff(lag).std() for lag in lags]
        poly    = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        df["hurst"] = poly[0] * 2.0
    except:
        df["hurst"] = 0.5

    return df


# ═══════════════════════════════════════════════════════════════
# LIVELLO 03 — SMC COMPLETO (Smart Money Concepts)
# ═══════════════════════════════════════════════════════════════

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Swing High e Swing Low."""
    df["swing_high"] = False
    df["swing_low"]  = False
    for i in range(lookback, len(df) - lookback):
        if df["High"].iloc[i] == df["High"].iloc[i-lookback:i+lookback+1].max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
        if df["Low"].iloc[i] == df["Low"].iloc[i-lookback:i+lookback+1].min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
    return df


def detect_bos_choch(df: pd.DataFrame) -> dict:
    """Break of Structure e Change of Character."""
    result = {
        "bos": None, "choch": None, "structure": "NEUTRAL",
        "last_high": None, "last_low": None,
        "prev_high": None, "prev_low": None
    }
    sh    = df[df["swing_high"]]["High"].tail(6).values
    sl    = df[df["swing_low"]]["Low"].tail(6).values
    price = float(df["Close"].iloc[-1])

    if len(sh) >= 2 and len(sl) >= 2:
        lh, ph = float(sh[-1]), float(sh[-2])
        ll, pl = float(sl[-1]), float(sl[-2])
        result["last_high"] = round(lh, 2)
        result["last_low"]  = round(ll, 2)
        result["prev_high"] = round(ph, 2)
        result["prev_low"]  = round(pl, 2)

        # Struttura rialzista: HH + HL
        if lh > ph and ll > pl:
            result["structure"] = "BULLISH"
            if price > lh:
                result["bos"] = "BOS_BULLISH"
        # Struttura ribassista: LH + LL
        elif lh < ph and ll < pl:
            result["structure"] = "BEARISH"
            if price < ll:
                result["bos"] = "BOS_BEARISH"

        # CHoCH: rottura contro struttura
        if result["structure"] == "BULLISH" and price < ll:
            result["choch"] = "CHOCH_BEARISH"
        elif result["structure"] == "BEARISH" and price > lh:
            result["choch"] = "CHOCH_BULLISH"

    return result


def detect_order_blocks(df: pd.DataFrame) -> dict:
    """Order Block bullish e bearish."""
    obs = {"bullish_ob": None, "bearish_ob": None}
    for i in range(max(0, len(df)-30), len(df)-1):
        c  = df.iloc[i]
        nc = df.iloc[i+1]
        move = abs(float(nc["Close"]) - float(nc["Open"])) / (float(nc["Open"]) + 1e-9) * 100
        if float(c["Close"]) < float(c["Open"]) and float(nc["Close"]) > float(nc["Open"]) and move > 0.02:
            obs["bullish_ob"] = {
                "high": round(float(c["High"]), 2),
                "low":  round(float(c["Low"]), 2),
                "mid":  round((float(c["High"]) + float(c["Low"])) / 2, 2)
            }
        if float(c["Close"]) > float(c["Open"]) and float(nc["Close"]) < float(nc["Open"]) and move > 0.02:
            obs["bearish_ob"] = {
                "high": round(float(c["High"]), 2),
                "low":  round(float(c["Low"]), 2),
                "mid":  round((float(c["High"]) + float(c["Low"])) / 2, 2)
            }
    return obs


def detect_fvg(df: pd.DataFrame) -> dict:
    """Fair Value Gap — gap tra candela 1 e candela 3."""
    fvg = {"bullish_fvg": None, "bearish_fvg": None}
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i-2], df.iloc[i]
        if float(c3["Low"]) > float(c1["High"]):
            fvg["bullish_fvg"] = {
                "top":    round(float(c3["Low"]), 2),
                "bottom": round(float(c1["High"]), 2),
                "mid":    round((float(c3["Low"]) + float(c1["High"])) / 2, 2)
            }
        if float(c3["High"]) < float(c1["Low"]):
            fvg["bearish_fvg"] = {
                "top":    round(float(c1["Low"]), 2),
                "bottom": round(float(c3["High"]), 2),
                "mid":    round((float(c1["Low"]) + float(c3["High"])) / 2, 2)
            }
    return fvg


def detect_liquidity(df: pd.DataFrame) -> dict:
    """EQH e EQL — zone di liquidità (Equal Highs/Lows)."""
    tol    = 0.001
    recent = df.tail(100)
    highs  = recent["High"].values
    lows   = recent["Low"].values
    eqh, eql = [], []
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i] - highs[j]) / (highs[i] + 1e-9) < tol:
                eqh.append((highs[i] + highs[j]) / 2)
            if abs(lows[i] - lows[j]) / (lows[i] + 1e-9) < tol:
                eql.append((lows[i] + lows[j]) / 2)
    return {
        "eqh": round(float(np.mean(eqh)), 2) if eqh else None,
        "eql": round(float(np.mean(eql)), 2) if eql else None
    }


def detect_breaker_blocks(df: pd.DataFrame, ob: dict) -> dict:
    """Breaker Block: OB violato che diventa zona di inversione."""
    price = float(df["Close"].iloc[-1])
    bb    = {"bullish_bb": None, "bearish_bb": None}
    if ob.get("bullish_ob") and price < ob["bullish_ob"]["low"]:
        bb["bearish_bb"] = ob["bullish_ob"]
    if ob.get("bearish_ob") and price > ob["bearish_ob"]["high"]:
        bb["bullish_bb"] = ob["bearish_ob"]
    return bb


def detect_mitigation(df: pd.DataFrame, ob: dict) -> dict:
    """Mitigation Block: prezzo che ritorna su un OB per mitigarlo."""
    price = float(df["Close"].iloc[-1])
    mit   = {"bullish_mit": False, "bearish_mit": False}
    if ob.get("bullish_ob"):
        if ob["bullish_ob"]["low"] <= price <= ob["bullish_ob"]["high"]:
            mit["bullish_mit"] = True
    if ob.get("bearish_ob"):
        if ob["bearish_ob"]["low"] <= price <= ob["bearish_ob"]["high"]:
            mit["bearish_mit"] = True
    return mit


def detect_premium_discount(df: pd.DataFrame, smc: dict) -> str:
    """Premium/Discount zone rispetto all'equilibrio dell'ultimo swing."""
    lh    = smc.get("last_high")
    ll    = smc.get("last_low")
    price = float(df["Close"].iloc[-1])
    if lh and ll:
        mid = (lh + ll) / 2
        if price > mid * 1.002:  return "PREMIUM"
        if price < mid * 0.998:  return "DISCOUNT"
        return "EQUILIBRIUM"
    return "EQUILIBRIUM"


# ═══════════════════════════════════════════════════════════════
# LIVELLO 04 — STRATEGIA SMC v3.0 PERSONALIZZATA
# ═══════════════════════════════════════════════════════════════

def smc_v3_strategy(df_15m: pd.DataFrame, df_1m: pd.DataFrame,
                    smc: dict, ob: dict, fvg: dict) -> dict:
    """
    Strategia SMC v3.0 — 5 Setup su XAU/USD
    Timeframe contesto: 15min
    Timeframe entry: 1min (CHoCH su 1min come conferma finale)
    Sessione operativa: 14:00-19:00 IT (NY Kill Zone: 15:30-17:30)
    """
    result = {"signal": "NEUTRAL", "setup": None, "score": 0}
    now    = datetime.now(TIMEZONE)

    # Filtro orario — solo 14:00-19:00
    if not (14 <= now.hour < 19):
        return result

    if df_15m is None or df_1m is None:
        return result

    df_15m = compute_indicators(df_15m)
    df_1m  = compute_indicators(df_1m)

    # SMC su 15min (contesto)
    smc_15m   = detect_bos_choch(df_15m)
    ob_15m    = detect_order_blocks(df_15m)
    fvg_15m   = detect_fvg(df_15m)
    liq_15m   = detect_liquidity(df_15m)

    # CHoCH su 1min (conferma finale)
    smc_1m    = detect_bos_choch(df_1m)

    price     = float(df_1m["Close"].iloc[-1])
    row_15m   = df_15m.iloc[-1]
    row_1m    = df_1m.iloc[-1]

    rsi_15m   = float(row_15m["rsi"]) if not pd.isna(row_15m["rsi"]) else 50
    rsi_1m    = float(row_1m["rsi"])  if not pd.isna(row_1m["rsi"])  else 50
    atr_15m   = float(row_15m["atr"]) if not pd.isna(row_15m["atr"]) else 5

    # NY Kill Zone bonus
    ny_kz     = (15 <= now.hour < 17) or (now.hour == 17 and now.minute <= 30)

    # ── SETUP 1: CHoCH + OB ──
    # Struttura cambia direzione su 15min, prezzo ritorna su OB, CHoCH su 1min conferma
    if smc_15m["choch"] == "CHOCH_BULLISH" and ob_15m.get("bullish_ob"):
        ob_zone = ob_15m["bullish_ob"]
        if ob_zone["low"] <= price <= ob_zone["high"] + atr_15m * 0.5:
            if smc_1m["choch"] == "CHOCH_BULLISH" or smc_1m["bos"] == "BOS_BULLISH":
                score = 8 + (2 if ny_kz else 0) + (1 if rsi_1m < 50 else 0)
                result = {"signal": "BUY", "setup": "Setup 1: CHoCH + OB Bullish", "score": score}

    if smc_15m["choch"] == "CHOCH_BEARISH" and ob_15m.get("bearish_ob"):
        ob_zone = ob_15m["bearish_ob"]
        if ob_zone["low"] - atr_15m * 0.5 <= price <= ob_zone["high"]:
            if smc_1m["choch"] == "CHOCH_BEARISH" or smc_1m["bos"] == "BOS_BEARISH":
                score = 8 + (2 if ny_kz else 0) + (1 if rsi_1m > 50 else 0)
                result = {"signal": "SELL", "setup": "Setup 1: CHoCH + OB Bearish", "score": score}

    # ── SETUP 2: BOS + FVG ──
    # BOS conferma trend, prezzo ritorna su FVG, CHoCH 1min conferma
    if result["signal"] == "NEUTRAL":
        if smc_15m["bos"] == "BOS_BULLISH" and fvg_15m.get("bullish_fvg"):
            fvg_zone = fvg_15m["bullish_fvg"]
            if fvg_zone["bottom"] <= price <= fvg_zone["top"]:
                if smc_1m["choch"] == "CHOCH_BULLISH":
                    score = 7 + (2 if ny_kz else 0)
                    result = {"signal": "BUY", "setup": "Setup 2: BOS + FVG Bullish", "score": score}

        if smc_15m["bos"] == "BOS_BEARISH" and fvg_15m.get("bearish_fvg"):
            fvg_zone = fvg_15m["bearish_fvg"]
            if fvg_zone["bottom"] <= price <= fvg_zone["top"]:
                if smc_1m["choch"] == "CHOCH_BEARISH":
                    score = 7 + (2 if ny_kz else 0)
                    result = {"signal": "SELL", "setup": "Setup 2: BOS + FVG Bearish", "score": score}

    # ── SETUP 3: Liquidity Sweep + Reversal ──
    # Prezzo sweeppa EQH/EQL, poi CHoCH 1min
    if result["signal"] == "NEUTRAL":
        eqh = liq_15m.get("eqh")
        eql = liq_15m.get("eql")
        if eqh and abs(price - eqh) <= atr_15m * 0.3:
            if smc_1m["choch"] == "CHOCH_BEARISH":
                score = 8 + (2 if ny_kz else 0)
                result = {"signal": "SELL", "setup": "Setup 3: EQH Sweep + Reversal", "score": score}
        if eql and abs(price - eql) <= atr_15m * 0.3:
            if smc_1m["choch"] == "CHOCH_BULLISH":
                score = 8 + (2 if ny_kz else 0)
                result = {"signal": "BUY", "setup": "Setup 3: EQL Sweep + Reversal", "score": score}

    # ── SETUP 4: OB + FVG Confluence ──
    # OB e FVG nella stessa zona — massima confluenza
    if result["signal"] == "NEUTRAL":
        if ob_15m.get("bullish_ob") and fvg_15m.get("bullish_fvg"):
            ob_z  = ob_15m["bullish_ob"]
            fvg_z = fvg_15m["bullish_fvg"]
            # Overlap tra OB e FVG
            overlap_low  = max(ob_z["low"], fvg_z["bottom"])
            overlap_high = min(ob_z["high"], fvg_z["top"])
            if overlap_low <= overlap_high and overlap_low <= price <= overlap_high + atr_15m * 0.3:
                if smc_1m["choch"] == "CHOCH_BULLISH":
                    score = 10 + (2 if ny_kz else 0)
                    result = {"signal": "BUY", "setup": "Setup 4: OB+FVG Confluence Bullish", "score": score}

        if ob_15m.get("bearish_ob") and fvg_15m.get("bearish_fvg"):
            ob_z  = ob_15m["bearish_ob"]
            fvg_z = fvg_15m["bearish_fvg"]
            overlap_low  = max(ob_z["low"], fvg_z["bottom"])
            overlap_high = min(ob_z["high"], fvg_z["top"])
            if overlap_low <= overlap_high and overlap_low - atr_15m * 0.3 <= price <= overlap_high:
                if smc_1m["choch"] == "CHOCH_BEARISH":
                    score = 10 + (2 if ny_kz else 0)
                    result = {"signal": "SELL", "setup": "Setup 4: OB+FVG Confluence Bearish", "score": score}

    # ── SETUP 5: Premium/Discount + Struttura ──
    # Prezzo in zona Discount con struttura Bullish o Premium con struttura Bearish
    if result["signal"] == "NEUTRAL":
        pd_zone  = detect_premium_discount(df_15m, smc_15m)
        struct   = smc_15m["structure"]
        if pd_zone == "DISCOUNT" and struct == "BULLISH" and rsi_15m < 45:
            if smc_1m["choch"] == "CHOCH_BULLISH":
                score = 7 + (2 if ny_kz else 0)
                result = {"signal": "BUY", "setup": "Setup 5: Discount + Bullish Structure", "score": score}
        if pd_zone == "PREMIUM" and struct == "BEARISH" and rsi_15m > 55:
            if smc_1m["choch"] == "CHOCH_BEARISH":
                score = 7 + (2 if ny_kz else 0)
                result = {"signal": "SELL", "setup": "Setup 5: Premium + Bearish Structure", "score": score}

    return result


# ═══════════════════════════════════════════════════════════════
# LIVELLO 05 — TREND FOLLOWING
# ═══════════════════════════════════════════════════════════════

def trend_following_strategy(df: pd.DataFrame) -> dict:
    """
    Trend Following: EMA crossover + ADX filter + Donchian breakout.
    Funziona meglio in regime TRENDING.
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}
    if len(df) < 50: return result

    row   = df.iloc[-1]
    price = float(row["Close"])
    ema20 = float(row["ema20"]) if not pd.isna(row["ema20"]) else price
    ema50 = float(row["ema50"]) if not pd.isna(row["ema50"]) else price
    ema200 = float(row["ema200"]) if not pd.isna(row["ema200"]) else price
    adx   = float(row["adx"])   if not pd.isna(row["adx"])   else 0
    adxp  = float(row["adx_pos"]) if not pd.isna(row["adx_pos"]) else 0
    adxn  = float(row["adx_neg"]) if not pd.isna(row["adx_neg"]) else 0
    don_h = float(row["don_high"]) if not pd.isna(row["don_high"]) else price
    don_l = float(row["don_low"])  if not pd.isna(row["don_low"])  else price
    macd  = float(row["macd"])   if not pd.isna(row["macd"])   else 0
    sig   = float(row["macd_sig"]) if not pd.isna(row["macd_sig"]) else 0
    psar  = float(row["psar"])   if not pd.isna(row["psar"])   else price

    score = 0
    reason_parts = []

    # ADX filter: minimo 20 per trend valido
    if adx < 20:
        return {"signal": "NEUTRAL", "score": 0, "reason": "ADX basso — no trend"}

    # Segnali BUY
    if ema20 > ema50:       score += 2; reason_parts.append("EMA20>EMA50")
    if ema50 > ema200:      score += 2; reason_parts.append("EMA50>EMA200")
    if price > ema200:      score += 1; reason_parts.append("Price>EMA200")
    if macd > sig:          score += 2; reason_parts.append("MACD>Signal")
    if adxp > adxn:         score += 2; reason_parts.append("+DI>-DI")
    if price >= don_h:      score += 3; reason_parts.append("Donchian Breakout UP")
    if price > psar:        score += 1; reason_parts.append("Price>PSAR")

    if score >= 7:
        return {"signal": "BUY", "score": score, "reason": ", ".join(reason_parts)}

    # Segnali SELL
    score = 0
    reason_parts = []
    if ema20 < ema50:       score += 2; reason_parts.append("EMA20<EMA50")
    if ema50 < ema200:      score += 2; reason_parts.append("EMA50<EMA200")
    if price < ema200:      score += 1; reason_parts.append("Price<EMA200")
    if macd < sig:          score += 2; reason_parts.append("MACD<Signal")
    if adxn > adxp:         score += 2; reason_parts.append("-DI>+DI")
    if price <= don_l:      score += 3; reason_parts.append("Donchian Breakout DOWN")
    if price < psar:        score += 1; reason_parts.append("Price<PSAR")

    if score >= 7:
        return {"signal": "SELL", "score": score, "reason": ", ".join(reason_parts)}

    return {"signal": "NEUTRAL", "score": 0, "reason": "No trend signal"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 06 — MEAN REVERSION
# ═══════════════════════════════════════════════════════════════

def mean_reversion_strategy(df: pd.DataFrame) -> dict:
    """
    Mean Reversion: Bollinger + RSI estremi + Z-score + Keltner.
    Funziona meglio in regime RANGING.
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}
    if len(df) < 30: return result

    row    = df.iloc[-1]
    price  = float(row["Close"])
    bb_u   = float(row["bb_upper"]) if not pd.isna(row["bb_upper"]) else price
    bb_l   = float(row["bb_lower"]) if not pd.isna(row["bb_lower"]) else price
    bb_mid = float(row["bb_mid"])   if not pd.isna(row["bb_mid"])   else price
    rsi    = float(row["rsi"])      if not pd.isna(row["rsi"])      else 50
    zscore = float(row["zscore"])   if not pd.isna(row["zscore"])   else 0
    sk     = float(row["stoch_k"])  if not pd.isna(row["stoch_k"])  else 50
    sd     = float(row["stoch_d"])  if not pd.isna(row["stoch_d"])  else 50
    kelt_l = float(row["kelt_lower"]) if not pd.isna(row["kelt_lower"]) else price
    kelt_u = float(row["kelt_upper"]) if not pd.isna(row["kelt_upper"]) else price
    cci    = float(row["cci"])      if not pd.isna(row["cci"])      else 0
    willr  = float(row["willr"])    if not pd.isna(row["willr"])    else -50
    adx    = float(row["adx"])      if not pd.isna(row["adx"])      else 0

    # Mean reversion funziona quando ADX è basso (mercato laterale)
    adx_filter = adx < 30

    score_buy = 0
    reason_buy = []
    if price <= bb_l:       score_buy += 3; reason_buy.append("BB Lower")
    if rsi < 30:            score_buy += 3; reason_buy.append(f"RSI {rsi:.0f}")
    elif rsi < 40:          score_buy += 2
    if zscore < -2.0:       score_buy += 3; reason_buy.append(f"Z-score {zscore:.1f}")
    elif zscore < -1.5:     score_buy += 2
    if sk < 20 and sk > sd: score_buy += 2; reason_buy.append("Stoch oversold cross")
    if price <= kelt_l:     score_buy += 2; reason_buy.append("Keltner Lower")
    if cci < -150:          score_buy += 2; reason_buy.append(f"CCI {cci:.0f}")
    if willr < -85:         score_buy += 1

    if score_buy >= 6 and adx_filter:
        return {"signal": "BUY", "score": score_buy, "reason": "MR: " + ", ".join(reason_buy)}

    score_sell = 0
    reason_sell = []
    if price >= bb_u:       score_sell += 3; reason_sell.append("BB Upper")
    if rsi > 70:            score_sell += 3; reason_sell.append(f"RSI {rsi:.0f}")
    elif rsi > 60:          score_sell += 2
    if zscore > 2.0:        score_sell += 3; reason_sell.append(f"Z-score {zscore:.1f}")
    elif zscore > 1.5:      score_sell += 2
    if sk > 80 and sk < sd: score_sell += 2; reason_sell.append("Stoch overbought cross")
    if price >= kelt_u:     score_sell += 2; reason_sell.append("Keltner Upper")
    if cci > 150:           score_sell += 2; reason_sell.append(f"CCI {cci:.0f}")
    if willr > -15:         score_sell += 1

    if score_sell >= 6 and adx_filter:
        return {"signal": "SELL", "score": score_sell, "reason": "MR: " + ", ".join(reason_sell)}

    return {"signal": "NEUTRAL", "score": 0, "reason": "No MR signal"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 07 — MOMENTUM
# ═══════════════════════════════════════════════════════════════

def momentum_strategy(df: pd.DataFrame, mtf_trends: dict) -> dict:
    """
    Momentum: ROC + MTF alignment + Squeeze breakout + MACD histogram.
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}
    if len(df) < 30: return result

    row    = df.iloc[-1]
    roc    = float(row["roc"])      if not pd.isna(row["roc"])    else 0
    mom    = float(row["mom"])      if not pd.isna(row["mom"])    else 0
    hist   = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
    squeeze = bool(row["squeeze"])  if not pd.isna(row["squeeze"]) else False
    adx    = float(row["adx"])      if not pd.isna(row["adx"])    else 0
    rsi    = float(row["rsi"])      if not pd.isna(row["rsi"])    else 50

    # MTF alignment score
    mtf_buy  = sum(1 for v in mtf_trends.values() if v == "BUY")
    mtf_sell = sum(1 for v in mtf_trends.values() if v == "SELL")

    # Squeeze breakout
    prev_squeeze = bool(df["squeeze"].iloc[-3]) if len(df) > 3 else False
    squeeze_breakout_up   = prev_squeeze and not squeeze and mom > 0
    squeeze_breakout_down = prev_squeeze and not squeeze and mom < 0

    score_buy = 0
    reason_buy = []
    if roc > 0.3:              score_buy += 2; reason_buy.append(f"ROC {roc:.2f}%")
    if mom > 0:                score_buy += 1
    if hist > 0:               score_buy += 2; reason_buy.append("MACD hist positive")
    if squeeze_breakout_up:    score_buy += 4; reason_buy.append("Squeeze Breakout UP")
    if mtf_buy >= 4:           score_buy += 3; reason_buy.append(f"MTF {mtf_buy}/6 BUY")
    elif mtf_buy >= 3:         score_buy += 2
    if adx >= 25:              score_buy += 1
    if 45 < rsi < 70:          score_buy += 1

    if score_buy >= 6:
        return {"signal": "BUY", "score": score_buy, "reason": "MOM: " + ", ".join(reason_buy)}

    score_sell = 0
    reason_sell = []
    if roc < -0.3:             score_sell += 2; reason_sell.append(f"ROC {roc:.2f}%")
    if mom < 0:                score_sell += 1
    if hist < 0:               score_sell += 2; reason_sell.append("MACD hist negative")
    if squeeze_breakout_down:  score_sell += 4; reason_sell.append("Squeeze Breakout DOWN")
    if mtf_sell >= 4:          score_sell += 3; reason_sell.append(f"MTF {mtf_sell}/6 SELL")
    elif mtf_sell >= 3:        score_sell += 2
    if adx >= 25:              score_sell += 1
    if 30 < rsi < 55:          score_sell += 1

    if score_sell >= 6:
        return {"signal": "SELL", "score": score_sell, "reason": "MOM: " + ", ".join(reason_sell)}

    return {"signal": "NEUTRAL", "score": 0, "reason": "No momentum"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 08 — EVENT-DRIVEN
# ═══════════════════════════════════════════════════════════════

def _parse_calendar_datetime(raw_date: str) -> datetime:
    """
    Interpreta sia ISO-8601 con offset sia le vecchie date senza timezone.
    Le date naive del feed vengono considerate America/New_York.
    """
    if not raw_date:
        raise ValueError("Data calendario vuota")
    normalized = raw_date.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = datetime.strptime(raw_date[:16], "%Y-%m-%dT%H:%M")
    if parsed.tzinfo is None:
        parsed = pytz.timezone("America/New_York").localize(parsed)
    return parsed.astimezone(TIMEZONE)


def get_upcoming_events(days_ahead: int = 7, hours_lookback: float = 0.0) -> list:
    """
    Ritorna tutti gli eventi USD ad alto impatto per i prossimi N giorni.
    Usato dall'AI assistant per rispondere a domande sui prossimi eventi.

    hours_lookback: include anche eventi già avvenuti fino a questa distanza
    nel passato (default 0 = solo futuri, comportamento originale). Serve a
    gold_bot.check_macro_alerts() per il controllo POST-evento: con
    hours_lookback=0 un evento appena passato spariva dalla lista prima
    ancora che il codice potesse verificare la finestra "8-15 minuti dopo",
    rendendo il resoconto post-evento morto dal codice (mai potuto scattare
    per nessun evento) — scoperto in produzione il 1 settembre 2026.
    """
    import math
    try:
        events_raw = _fetch_calendar_raw()
        now   = datetime.now(TIMEZONE)
        today = now.strftime("%Y-%m-%d")
        upcoming = []

        for ev in events_raw:
            raw_date = ev.get("date", "")
            impact   = ev.get("impact", "").lower()
            currency = ev.get("country", "").lower()
            title    = ev.get("title", "")

            if impact != "high" or currency not in ["usd", "us"]:
                continue
            if len(raw_date) < 16:
                continue

            try:
                ev_it       = _parse_calendar_datetime(raw_date)
                diff_h      = (ev_it - now).total_seconds() / 3600
                if diff_h < -hours_lookback or diff_h > days_ahead * 24:
                    continue
            except (ValueError, TypeError):
                continue

            upcoming.append({
                "title":      title,
                "date":       ev_it.strftime("%Y-%m-%d"),
                "time":       ev_it.strftime("%H:%M"),   # orario IT corretto
                "datetime":   ev_it.isoformat(),
                "impact":     "HIGH",
                "forecast":   ev.get("forecast", "N/A"),
                "previous":   ev.get("previous", "N/A"),
                "hours_away": round(diff_h, 1) if diff_h < 999 else None,
            })

        # Ordina per data/ora
        upcoming.sort(key=lambda x: x["date"] + x["time"])
        return upcoming
    except Exception as e:
        logger.warning(f"Errore upcoming events: {e}")
        return []


# Cache calendario — aggiornato max 1 volta ogni 30 minuti
_calendar_cache = {"data": None, "ts": 0.0}

def _fetch_calendar_raw() -> list:
    """Scarica il calendario da FairEconomy con cache 30 minuti per evitare 429."""
    import time
    global _calendar_cache
    now = time.time()

    # Usa cache se fresca (30 minuti)
    if _calendar_cache["data"] is not None and now - _calendar_cache["ts"] < 1800:
        logger.debug("[CALENDARIO] Uso cache")
        return _calendar_cache["data"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.forexfactory.com/",
        "Origin": "https://www.forexfactory.com",
    }
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    r   = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200 and r.text.strip():
        data = r.json()
        _calendar_cache = {"data": data, "ts": now}
        logger.info(f"[CALENDARIO] Aggiornato — {len(data)} eventi")
        return data
    raise ValueError(f"Calendario status {r.status_code}")


def get_economic_events() -> dict:
    """
    Calendario economico — eventi ad alto impatto oggi.

    FIX: Gli orari di FairEconomy sono in ET (Eastern Time USA), NON in IT.
    La versione precedente confrontava ev_hour (ET) con now_hour (IT) direttamente,
    causando blackout nelle ore sbagliate (6h di offset in estate, 6h in inverno).
    Ora convertiamo correttamente ET → Europe/Rome con pytz.
    """
    try:
        events_raw = _fetch_calendar_raw()
    except Exception as e:
        logger.error(f"[CALENDARIO] Errore fetch: {e}")
        # FAIL-SAFE: ritorna errore esplicito → gold_bot bloccherà i trade
        return {
            "events": [], "upcoming": [], "high_impact_today": False,
            "imminent": False, "count": 0, "error": True
        }

    try:
        now_it   = datetime.now(TIMEZONE)
        today_it = now_it.strftime("%Y-%m-%d")
        high_imp = []
        upcoming = []

        for ev in events_raw:
            raw_date = ev.get("date", "")
            impact   = ev.get("impact", "").lower()
            currency = ev.get("country", "").lower()

            if impact != "high" or currency not in ["usd", "us"]:
                continue
            if len(raw_date) < 16:
                continue

            try:
                ev_it = _parse_calendar_datetime(raw_date)
            except (ValueError, TypeError):
                continue

            ev_date_it = ev_it.strftime("%Y-%m-%d")
            ev_time_it = ev_it.strftime("%H:%M")
            ev_hour_it = ev_it.hour

            if ev_date_it != today_it:
                continue

            ev_data = {
                "title":    ev.get("title", ""),
                "time":     ev_time_it,   # orario in IT (corretto)
                "forecast": ev.get("forecast", "N/A"),
                "previous": ev.get("previous", "N/A"),
                "hour":     ev_hour_it,   # ora IT (per confronti)
                "date":     ev_date_it,
                "datetime": ev_it.isoformat(),
                "impact":   "HIGH",
            }
            high_imp.append(ev_data)
            if ev_it > now_it:
                upcoming.append(ev_data)

        imminent = any(
            abs(
                (
                    datetime.fromisoformat(ev["datetime"]).astimezone(TIMEZONE)
                    - now_it
                ).total_seconds()
            ) <= 30 * 60
            for ev in high_imp
        )

        return {
            "events":            high_imp,
            "upcoming":          upcoming,
            "high_impact_today": len(high_imp) > 0,
            "imminent":          imminent,
            "count":             len(high_imp),
            "error":             False,
        }
    except Exception as e:
        logger.error(f"[CALENDARIO] Errore parsing: {e}")
        return {
            "events": [], "upcoming": [], "high_impact_today": False,
            "imminent": False, "count": 0, "error": True
        }


def event_driven_strategy(calendar: dict, sentiment: dict) -> dict:
    """
    Event-Driven: posizionamento pre/post eventi macro.
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}

    if calendar.get("imminent"):
        return {"signal": "NEUTRAL", "score": 0, "reason": "Evento imminente — no trade"}

    s_label = sentiment.get("label", "NEUTRAL")
    s_score = sentiment.get("score", 0)

    if s_label == "BULLISH" and s_score >= 4:
        return {"signal": "BUY", "score": 5, "reason": f"ED: Sentiment BULLISH ({s_score})"}
    if s_label == "BEARISH" and s_score <= -4:
        return {"signal": "SELL", "score": 5, "reason": f"ED: Sentiment BEARISH ({s_score})"}

    return result


# ═══════════════════════════════════════════════════════════════
# LIVELLO 09 — STATISTICAL ARBITRAGE (XAU vs DXY, XAU vs US10Y)
# ═══════════════════════════════════════════════════════════════

def get_dxy_history(outputsize: int = 30) -> pd.DataFrame:
    """Storico DXY per calcolare la media mobile di riferimento."""
    try:
        return get_data_generic("DXY", interval="1day", outputsize=outputsize)
    except Exception as e:
        logger.warning(f"Errore storico DXY: {e}")
        return pd.DataFrame()


def get_tlt_history(outputsize: int = 30) -> pd.DataFrame:
    """Storico TLT (proxy tassi) per calcolare la media mobile di riferimento."""
    try:
        return get_data_generic("TLT", interval="1day", outputsize=outputsize)
    except Exception as e:
        logger.warning(f"Errore storico TLT: {e}")
        return pd.DataFrame()


def get_data_generic(symbol: str, interval: str = "1day", outputsize: int = 30) -> pd.DataFrame:
    """Scarica dati generici per un simbolo qualsiasi (per correlazioni)."""
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_API_KEY}
    r    = requests.get(url, params=params, timeout=10)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df["close"] = df["close"].astype(float)
    df.sort_index(inplace=True)
    return df


def statistical_arbitrage_strategy(price_xau: float, dxy: float, us10y: float) -> dict:
    """
    Statistical Arbitrage: XAU/USD vs DXY e TLT (proxy tassi).
    Usa la deviazione dalla MEDIA MOBILE RECENTE (20 giorni) invece di soglie
    assolute fisse — più robusto perché si adatta ai livelli di mercato attuali
    invece di restare ancorato a valori storici che diventano obsoleti.
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}

    if dxy == 0 or us10y == 0:
        return result

    score_buy  = 0
    score_sell = 0
    reasons    = []

    try:
        dxy_hist = get_dxy_history(20)
        if not dxy_hist.empty and len(dxy_hist) >= 10:
            dxy_ma  = float(dxy_hist["close"].mean())
            dxy_dev = (dxy - dxy_ma) / dxy_ma * 100  # % deviazione dalla media

            if dxy_dev > 1.0:      # DXY sopra la sua media di oltre 1%
                score_sell += 2
                reasons.append(f"DXY +{dxy_dev:.1f}% vs MA20")
            elif dxy_dev < -1.0:   # DXY sotto la sua media di oltre 1%
                score_buy += 2
                reasons.append(f"DXY {dxy_dev:.1f}% vs MA20")
    except Exception as e:
        logger.warning(f"Errore calcolo DXY MA: {e}")

    try:
        tlt_hist = get_tlt_history(20)
        if not tlt_hist.empty and len(tlt_hist) >= 10:
            tlt_ma  = float(tlt_hist["close"].mean())
            tlt_dev = (us10y - tlt_ma) / tlt_ma * 100

            # TLT alto = yields bassi = positivo per oro
            if tlt_dev > 1.0:
                score_buy += 2
                reasons.append(f"TLT +{tlt_dev:.1f}% vs MA20 (yields bassi)")
            elif tlt_dev < -1.0:
                score_sell += 2
                reasons.append(f"TLT {tlt_dev:.1f}% vs MA20 (yields alti)")
    except Exception as e:
        logger.warning(f"Errore calcolo TLT MA: {e}")

    if score_buy >= 2 and score_buy > score_sell:
        return {"signal": "BUY", "score": score_buy, "reason": "StatArb: " + ", ".join(reasons)}
    if score_sell >= 2 and score_sell > score_buy:
        return {"signal": "SELL", "score": score_sell, "reason": "StatArb: " + ", ".join(reasons)}

    return result


# ═══════════════════════════════════════════════════════════════
# LIVELLO 10 — ML ALPHA (Feature-based scoring)
# ═══════════════════════════════════════════════════════════════

def ml_alpha_strategy(df: pd.DataFrame, mtf_trends: dict, smc: dict) -> dict:
    """
    ML Alpha v2 — Regressione Logistica Reale su Trade Storici.

    VERSIONE PRECEDENTE: formula pesata manualmente (score composito).
    VERSIONE ATTUALE: regressione logistica che si allena sui trade
    storici del DB goldbot.db. Le feature sono le stesse variabili
    tecniche, ma i pesi vengono ottimizzati su dati reali.

    Fallback automatico alla versione rule-based se:
    - Meno di MIN_SAMPLES trade nel DB
    - Errore nel training
    - sklearn non disponibile
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}
    if len(df) < 50:
        return result

    row   = df.iloc[-1]
    price = float(row["Close"])

    # ── Estrazione Feature ────────────────────────────────────────────────────
    ema9   = float(row.get("ema9",   price) or price)
    ema20  = float(row.get("ema20",  price) or price)
    ema50  = float(row.get("ema50",  price) or price)
    ema200 = float(row.get("ema200", price) or price)
    rsi    = float(row.get("rsi",    50)    or 50)
    rsi_f  = float(row.get("rsi_fast", 50) or 50)
    macd_h = float(row.get("macd_hist", 0) or 0)
    macd_h2= float(df["macd_hist"].iloc[-2] if len(df) > 2 else 0) or 0
    atr    = max(float(row.get("atr", 5) or 5), 0.1)
    avg_atr= float(df["atr"].tail(20).mean()) or atr
    adx    = float(row.get("adx", 20) or 20)
    bb_up  = float(row.get("bb_upper", price) or price)
    bb_lo  = float(row.get("bb_lower", price) or price)
    bb_mid = (bb_up + bb_lo) / 2

    emas_above = sum([price > ema9, price > ema20, price > ema50, price > ema200])
    mtf_buy  = sum(1 for v in mtf_trends.values() if v == "BUY")
    mtf_sell = sum(1 for v in mtf_trends.values() if v == "SELL")
    struct   = smc.get("structure", "NEUTRAL")
    vol_ratio = atr / (avg_atr + 1e-9)

    # Feature vector (14 feature, normalizzate)
    features = [
        emas_above / 4,                          # 0-1: EMA alignment
        (rsi - 50) / 50,                         # -1 to +1: RSI centrato
        (rsi_f - rsi) / 10,                      # divergenza RSI fast/slow
        1 if macd_h > 0 else -1,                 # MACD histogram sign
        1 if macd_h > macd_h2 else -1,           # MACD momentum
        min(max(vol_ratio - 1, -1), 1),          # volatilità relativa
        (price - ema20) / atr,                   # distanza da EMA20 in ATR
        (price - bb_mid) / (bb_up - bb_lo + 1), # posizione nelle BB
        (mtf_buy - mtf_sell) / 6,                # bias MTF normalizzato
        adx / 50,                                # forza trend
        1 if struct == "BULLISH" else (-1 if struct == "BEARISH" else 0),
        (price - ema200) / atr,                  # distanza da EMA200
        min(max((price - ema9) / atr, -3), 3),  # distanza da EMA9
        1 if macd_h > 0 and rsi < 60 else (-1 if macd_h < 0 and rsi > 40 else 0),
    ]

    # ── Tenta ML reale dal DB ─────────────────────────────────────────────────
    MIN_SAMPLES = 15  # trade minimi per allenare il modello

    try:
        import os, sqlite3
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        db_path = os.environ.get("DB_PATH", os.path.join(
            os.environ.get("BOT_DIR", "/tmp"), "goldbot.db"
        ))

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT signal, result, prob, regime,
                          tp1_hit, tp2_hit, be_hit, pnl_r
                   FROM trades
                   WHERE status='CLOSED'
                     AND result IN ('WIN_TP1','WIN_TP2','WIN_TP3','LOSS')
                   ORDER BY id DESC LIMIT 200"""
            ).fetchall()
            conn.close()

            if len(rows) >= MIN_SAMPLES:
                # Costruisci dataset semplice dai metadati del trade
                # (non ricostruiamo le feature originali — usiamo proxy dai metadati)
                X_hist, y_hist = [], []
                for r in rows:
                    # Proxy feature dai metadati del trade
                    sig     = 1 if r["signal"] == "BUY" else -1
                    prob_n  = (float(r["prob"] or 50) - 50) / 50
                    tp1h    = float(r["tp1_hit"] or 0)
                    beh     = float(r["be_hit"] or 0)
                    pnl     = float(r["pnl_r"] or 0)
                    X_hist.append([sig, prob_n, tp1h, beh])
                    y_hist.append(1 if "WIN" in str(r["result"]) else 0)

                X_arr = np.array(X_hist)
                y_arr = np.array(y_hist)

                # Allena un classificatore leggero
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X_arr)
                clf    = LogisticRegression(C=0.5, max_iter=200, random_state=42)
                clf.fit(X_sc, y_arr)

                # Predici sulla situazione attuale usando proxy feature
                sig_now  = 1 if sum([price > ema20, rsi < 50, macd_h > 0]) >= 2 else -1
                prob_now = (rsi - 50) / 50
                tp1_est  = 1 if emas_above >= 3 and rsi < 65 else 0
                be_est   = 1 if vol_ratio < 1.3 else 0
                x_now    = scaler.transform([[sig_now, prob_now, tp1_est, be_est]])
                win_prob = clf.predict_proba(x_now)[0][1]  # probabilità WIN

                # Converti in segnale
                if win_prob > 0.62:
                    # Il modello dice WIN → segui la direzione tecnica
                    if sig_now == 1 and emas_above >= 2:
                        score = int(win_prob * 10)
                        return {
                            "signal": "BUY",
                            "score":  min(score, 10),
                            "reason": f"ML v2: win_prob={win_prob:.0%} ({len(rows)} trade storici)"
                        }
                    elif sig_now == -1 and emas_above <= 2:
                        score = int(win_prob * 10)
                        return {
                            "signal": "SELL",
                            "score":  min(score, 10),
                            "reason": f"ML v2: win_prob={win_prob:.0%} ({len(rows)} trade storici)"
                        }
                elif win_prob < 0.38:
                    # Il modello dice LOSS → invia il segnale inverso (contrarian)
                    if sig_now == 1:
                        return {
                            "signal": "SELL",
                            "score":  6,
                            "reason": f"ML v2 contrarian: loss_prob={1-win_prob:.0%}"
                        }
                    else:
                        return {
                            "signal": "BUY",
                            "score":  6,
                            "reason": f"ML v2 contrarian: loss_prob={1-win_prob:.0%}"
                        }

                return {"signal": "NEUTRAL", "score": 0,
                        "reason": f"ML v2: incerto win_prob={win_prob:.0%}"}

    except ImportError:
        pass  # sklearn non installato → fallback rule-based
    except Exception as e:
        logger.debug(f"ML training fallito: {e}")

    # ── Fallback: versione rule-based (come prima ma come backup) ─────────────
    ml_buy = (
        emas_above * 1.5
        + max(0, (50 - rsi) / 10)
        + (2 if rsi_f < rsi else 0)
        + (3 if macd_h > macd_h2 and macd_h > 0 else 0)
        + mtf_buy * 1.5
        + (2 if struct == "BULLISH" else 0)
        + (1 if 0.8 < vol_ratio < 1.5 else 0)
    )
    ml_sell = (
        (4 - emas_above) * 1.5
        + max(0, (rsi - 50) / 10)
        + (2 if rsi_f > rsi else 0)
        + (3 if macd_h < macd_h2 and macd_h < 0 else 0)
        + mtf_sell * 1.5
        + (2 if struct == "BEARISH" else 0)
        + (1 if 0.8 < vol_ratio < 1.5 else 0)
    )

    if ml_buy >= 12 and ml_buy > ml_sell:
        return {"signal": "BUY",  "score": min(int(ml_buy), 10),
                "reason": f"ML rule-based: {ml_buy:.1f}, MTF {mtf_buy}/6, EMA {emas_above}/4"}
    if ml_sell >= 12 and ml_sell > ml_buy:
        return {"signal": "SELL", "score": min(int(ml_sell), 10),
                "reason": f"ML rule-based: {ml_sell:.1f}, MTF {mtf_sell}/6, EMA {emas_above}/4"}

    return {"signal": "NEUTRAL", "score": 0, "reason": "No ML signal"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 11 — CANDLESTICK PATTERNS AVANZATI
# ═══════════════════════════════════════════════════════════════

def candlestick_strategy(df: pd.DataFrame) -> dict:
    """Pattern candele giapponesi completi."""
    result = {"signal": "NEUTRAL", "score": 0, "pattern": "", "reason": ""}
    if len(df) < 5: return result

    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    c4 = df.iloc[-5]

    o0, h0, l0, c_0 = float(c0["Open"]), float(c0["High"]), float(c0["Low"]), float(c0["Close"])
    o1, h1, l1, c_1 = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
    o2, h2, l2, c_2 = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])
    o3, c_3         = float(c3["Open"]), float(c3["Close"])

    body0 = abs(c_0 - o0)
    rng0  = h0 - l0
    uw0   = h0 - max(o0, c_0)
    lw0   = min(o0, c_0) - l0

    body1 = abs(c_1 - o1)
    rng1  = h1 - l1

    if rng0 == 0: return result

    patterns_found = []
    signal = "NEUTRAL"
    score  = 0

    # ── Singola candela ──
    # Doji
    if body0 <= rng0 * 0.08:
        patterns_found.append("🕯 Doji")

    # Hammer (rialzista)
    if lw0 >= body0 * 2.5 and uw0 <= body0 * 0.4 and c_0 > o0:
        patterns_found.append("🔨 Hammer")
        signal = "BUY"; score = 6

    # Inverted Hammer
    if uw0 >= body0 * 2.5 and lw0 <= body0 * 0.4 and c_0 > o0:
        patterns_found.append("🔨 Inverted Hammer")
        signal = "BUY"; score = 5

    # Shooting Star (ribassista)
    if uw0 >= body0 * 2.5 and lw0 <= body0 * 0.4 and c_0 < o0:
        patterns_found.append("⭐ Shooting Star")
        signal = "SELL"; score = 6

    # Hanging Man
    if lw0 >= body0 * 2.5 and uw0 <= body0 * 0.4 and c_0 < o0:
        patterns_found.append("🪢 Hanging Man")
        signal = "SELL"; score = 5

    # Marubozu Bullish
    if c_0 > o0 and body0 >= rng0 * 0.85:
        patterns_found.append("🕯 Marubozu Bullish")
        signal = "BUY"; score = 5

    # Marubozu Bearish
    if c_0 < o0 and body0 >= rng0 * 0.85:
        patterns_found.append("🕯 Marubozu Bearish")
        signal = "SELL"; score = 5

    # Pinbar Bullish
    if lw0 >= rng0 * 0.65 and body0 <= rng0 * 0.25:
        patterns_found.append("📌 Pinbar Bullish")
        signal = "BUY"; score = 7

    # Pinbar Bearish
    if uw0 >= rng0 * 0.65 and body0 <= rng0 * 0.25:
        patterns_found.append("📌 Pinbar Bearish")
        signal = "SELL"; score = 7

    # ── Due candele ──
    # Engulfing Bullish
    if c_0 > o0 and c_1 < o1 and c_0 > o1 and o0 < c_1:
        patterns_found.append("📈 Engulfing Bullish")
        signal = "BUY"; score = 8

    # Engulfing Bearish
    if c_0 < o0 and c_1 > o1 and c_0 < o1 and o0 > c_1:
        patterns_found.append("📉 Engulfing Bearish")
        signal = "SELL"; score = 8

    # Tweezer Bottom
    if abs(l0 - l1) <= rng0 * 0.02 and c_0 > o0 and c_1 < o1:
        patterns_found.append("🔽 Tweezer Bottom")
        signal = "BUY"; score = 6

    # Tweezer Top
    if abs(h0 - h1) <= rng0 * 0.02 and c_0 < o0 and c_1 > o1:
        patterns_found.append("🔼 Tweezer Top")
        signal = "SELL"; score = 6

    # Harami Bullish
    if c_1 < o1 and c_0 > o0 and o0 > c_1 and c_0 < o1 and body0 < body1 * 0.6:
        patterns_found.append("📊 Harami Bullish")
        signal = "BUY"; score = 5

    # Harami Bearish
    if c_1 > o1 and c_0 < o0 and o0 < c_1 and c_0 > o1 and body0 < body1 * 0.6:
        patterns_found.append("📊 Harami Bearish")
        signal = "SELL"; score = 5

    # ── Tre candele ──
    # Morning Star
    if (c_2 < o2 and abs(c_1 - o1) <= rng1 * 0.35 and
            c_0 > o0 and c_0 > (o2 + c_2) / 2):
        patterns_found.append("🌅 Morning Star")
        signal = "BUY"; score = 9

    # Evening Star
    if (c_2 > o2 and abs(c_1 - o1) <= rng1 * 0.35 and
            c_0 < o0 and c_0 < (o2 + c_2) / 2):
        patterns_found.append("🌆 Evening Star")
        signal = "SELL"; score = 9

    # Three White Soldiers
    if all([
        float(df.iloc[-i]["Close"]) > float(df.iloc[-i]["Open"]) for i in [1, 2, 3]
    ]) and c_0 > c_1 > c_2:
        patterns_found.append("⚔️ Tre Soldati Bianchi")
        signal = "BUY"; score = 8

    # Three Black Crows
    if all([
        float(df.iloc[-i]["Close"]) < float(df.iloc[-i]["Open"]) for i in [1, 2, 3]
    ]) and c_0 < c_1 < c_2:
        patterns_found.append("🐦 Tre Corvi Neri")
        signal = "SELL"; score = 8

    if patterns_found and signal != "NEUTRAL":
        return {
            "signal":  signal,
            "score":   score,
            "pattern": patterns_found[-1],
            "reason":  "Candle: " + ", ".join(patterns_found)
        }

    return {"signal": "NEUTRAL", "score": 0, "pattern": "", "reason": "No pattern"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 12 — ORDER FLOW ANALYSIS
# ═══════════════════════════════════════════════════════════════

def order_flow_strategy(df: pd.DataFrame) -> dict:
    """
    Order Flow v2 — Price Action Pura per XAU/USD.

    Il volume reale non è disponibile su XAU/USD da Twelve Data.
    La vecchia versione ritornava sempre NEUTRAL per questo motivo.

    Questa versione usa proxy di price action affidabili:
    1. Tick Direction Momentum — direzione delle ultime N candele
    2. Wick Rejection Analysis — pressione implicita nelle ombre
    3. Body/Range Ratio — forza del movimento
    4. Price Acceleration — velocità di variazione del prezzo
    5. VWAP Proxy Deviation — prezzo vs media mobile pesata sul range
    """
    result = {"signal": "NEUTRAL", "score": 0, "reason": ""}
    if len(df) < 30:
        return result

    row   = df.iloc[-1]
    price = float(row["Close"])
    atr   = max(float(row["atr"]) if not pd.isna(row["atr"]) else 5, 0.5)

    # ── Feature 1: Tick Direction Momentum ───────────────────────────────────
    # Conta quante candele delle ultime 10 chiudono nella stessa direzione
    recent10 = df.tail(10)
    bull10 = sum(1 for _, r in recent10.iterrows() if float(r["Close"]) > float(r["Open"]))
    bear10 = 10 - bull10

    # ── Feature 2: Wick Rejection Analysis (ultime 5 candele) ────────────────
    # Ombre superiori grandi → pressione di vendita; inferiori → pressione di acquisto
    recent5 = df.tail(5)
    upper_wick_total = 0.0
    lower_wick_total = 0.0
    for _, r in recent5.iterrows():
        o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
        upper_wick_total += h - max(o, c)
        lower_wick_total += min(o, c) - l
    total_wick = upper_wick_total + lower_wick_total + 1e-9
    upper_ratio = upper_wick_total / total_wick  # alto = pressione SELL
    lower_ratio = lower_wick_total / total_wick  # alto = pressione BUY

    # ── Feature 3: Body Strength (ultima candela) ─────────────────────────────
    o0 = float(row["Open"])
    h0 = float(row["High"])
    l0 = float(row["Low"])
    body0  = abs(price - o0)
    range0 = h0 - l0 + 1e-9
    body_ratio = body0 / range0     # > 0.6 = candela forte
    bull_body  = price > o0         # True = candela rialzista

    # ── Feature 4: Price Acceleration ─────────────────────────────────────────
    # Variazione percentuale del Close nelle ultime 3 candele vs le precedenti 3
    if len(df) >= 6:
        closes = df["Close"].values
        accel  = (closes[-1] - closes[-4]) - (closes[-4] - closes[-7]) if len(df) >= 7 else 0
        accel_norm = accel / atr  # normalizzato per ATR
    else:
        accel_norm = 0.0

    # ── Feature 5: VWAP Proxy Deviation ──────────────────────────────────────
    # Usa la media del prezzo tipico su 20 periodi (non usa volume)
    vwap = float(row["vwap"]) if not pd.isna(row.get("vwap", float("nan"))) else price
    vwap_dev = (price - vwap) / atr  # positivo = sopra VWAP → pressione SELL

    # ── Calcolo Score BUY ─────────────────────────────────────────────────────
    score_buy  = 0
    reason_buy = []

    if bull10 >= 7:
        score_buy += 3; reason_buy.append(f"Momentum {bull10}/10 bullish")
    elif bull10 >= 6:
        score_buy += 2; reason_buy.append(f"Momentum {bull10}/10 bullish")

    if lower_ratio > 0.60:
        score_buy += 3; reason_buy.append(f"Lower wick rejection {lower_ratio:.0%}")
    elif lower_ratio > 0.50:
        score_buy += 2; reason_buy.append(f"Lower wick {lower_ratio:.0%}")

    if bull_body and body_ratio > 0.65:
        score_buy += 2; reason_buy.append(f"Strong bull body {body_ratio:.0%}")

    if accel_norm > 0.8:
        score_buy += 2; reason_buy.append(f"Price acceleration +{accel_norm:.1f}σ")

    if vwap_dev < -0.8:
        score_buy += 2; reason_buy.append("Price below VWAP proxy")

    # ── Calcolo Score SELL ────────────────────────────────────────────────────
    score_sell  = 0
    reason_sell = []

    if bear10 >= 7:
        score_sell += 3; reason_sell.append(f"Momentum {bear10}/10 bearish")
    elif bear10 >= 6:
        score_sell += 2; reason_sell.append(f"Momentum {bear10}/10 bearish")

    if upper_ratio > 0.60:
        score_sell += 3; reason_sell.append(f"Upper wick rejection {upper_ratio:.0%}")
    elif upper_ratio > 0.50:
        score_sell += 2; reason_sell.append(f"Upper wick {upper_ratio:.0%}")

    if not bull_body and body_ratio > 0.65:
        score_sell += 2; reason_sell.append(f"Strong bear body {body_ratio:.0%}")

    if accel_norm < -0.8:
        score_sell += 2; reason_sell.append(f"Price deceleration {accel_norm:.1f}σ")

    if vwap_dev > 0.8:
        score_sell += 2; reason_sell.append("Price above VWAP proxy")

    # ── Decisione ─────────────────────────────────────────────────────────────
    threshold = 6
    if score_buy >= threshold and score_buy > score_sell:
        return {"signal": "BUY", "score": min(score_buy, 10),
                "reason": "OF: " + ", ".join(reason_buy)}
    if score_sell >= threshold and score_sell > score_buy:
        return {"signal": "SELL", "score": min(score_sell, 10),
                "reason": "OF: " + ", ".join(reason_sell)}

    return {"signal": "NEUTRAL", "score": 0, "reason": "No OF signal"}


# ═══════════════════════════════════════════════════════════════
# LIVELLO 13 — REGIME DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_market_regime(df: pd.DataFrame) -> dict:
    """
    Regime detection per la pipeline dei segnali (BLOCKED_COMBOS,
    aggregate_strategies, ecc.) — schema esteso con regime_compat,
    best_strategies, probability.

    NOTA: regime_detector.py espone un sistema separato e più semplice
    (regime/strength/details, usato solo dal comando /regime). In passato
    questa funzione provava a chiamare regime_detector.detect_regime_v2()
    e ne rimappava l'output, ma quella funzione è solo un alias verso lo
    schema semplice: mancano le chiavi attese qui (indicators/best_strategies/
    probability/...), quindi il tentativo falliva SEMPRE con KeyError e si
    finiva sempre nel fallback sotto — su ogni singola candela, in tutta la
    pipeline. Innocuo su una chiamata isolata, ma su un backtest di migliaia
    di candele l'overhead delle eccezioni si somma inutilmente. Rimosso il
    tentativo morto: si usa direttamente la logica che era comunque l'unica
    ad essere eseguita.
    """
    return _detect_market_regime_fallback(df)


def _detect_market_regime_fallback(df: pd.DataFrame) -> dict:
    """Logica di regime detection usata da detect_market_regime()."""
    row     = df.iloc[-1]
    adx     = float(row["adx"])      if not pd.isna(row["adx"])      else 0
    bb_w    = float(row["bb_width"]) if not pd.isna(row["bb_width"]) else 0
    atr     = float(row["atr"])
    avg_atr = float(df["atr"].tail(20).mean())
    ema20   = float(row["ema20"])    if not pd.isna(row["ema20"])    else 0
    ema50   = float(row["ema50"])    if not pd.isna(row["ema50"])    else 0
    roc     = float(row["roc"])      if not pd.isna(row["roc"])      else 0

    if adx >= 25 and ema20 > ema50 and roc > 0.1:
        regime = "TRENDING_UP"
    elif adx >= 25 and ema20 < ema50 and roc < -0.1:
        regime = "TRENDING_DOWN"
    elif adx < 18 and bb_w < 0.012:
        regime = "RANGING"
    elif atr > avg_atr * 1.8:
        regime = "VOLATILE"
    else:
        regime = "NORMAL"

    return {
        "regime":          regime,
        "regime_compat":   regime,
        "adx":             round(adx, 1),
        "atr":             round(atr, 2),
        "hurst":           0.5,
        "best_strategies": [],
        "probability":     70.0,
        "reason":          "Fallback regime",
        "actions":         {},
        "history":         [],
        "raw_scores":      {},
    }


# ═══════════════════════════════════════════════════════════════
# LIVELLO 14 — MTF TREND
# ═══════════════════════════════════════════════════════════════

def get_mtf_trend(mtf_data: dict) -> dict:
    trends = {}
    for tf, df in mtf_data.items():
        try:
            df    = compute_indicators(df)
            row   = df.iloc[-1]
            ema20 = float(row["ema20"]) if not pd.isna(row["ema20"]) else 0
            ema50 = float(row["ema50"]) if not pd.isna(row["ema50"]) else 0
            macd  = float(row["macd"])  if not pd.isna(row["macd"])  else 0
            sig   = float(row["macd_sig"]) if not pd.isna(row["macd_sig"]) else 0
            rsi   = float(row["rsi"])   if not pd.isna(row["rsi"])   else 50
            adx   = float(row["adx"])   if not pd.isna(row["adx"])   else 0
            if ema20 > ema50 and macd > sig and rsi > 48:
                trends[tf] = "BUY"
            elif ema20 < ema50 and macd < sig and rsi < 52:
                trends[tf] = "SELL"
            else:
                trends[tf] = "NEUTRAL"
        except:
            trends[tf] = "NEUTRAL"
    return trends


# ═══════════════════════════════════════════════════════════════
# LIVELLO 15 — SENTIMENT E NOTIZIE
# ═══════════════════════════════════════════════════════════════

def get_news_sentiment() -> dict:
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
                      "record", "strong", "demand", "safe haven", "inflation", "dovish",
                      "rate cut", "weak dollar", "geopolitical"]
        bearish_kw = ["fall", "drop", "decline", "bearish", "sell", "lower", "weak",
                      "pressure", "hawkish", "rate hike", "dollar strong", "risk on",
                      "tightening", "sell off"]

        score    = 0
        articles = []
        for a in data.get("articles", [])[:5]:
            title = (a.get("title") or "").lower()
            desc  = (a.get("description") or "").lower()
            text  = title + " " + desc
            for kw in bullish_kw:
                if kw in text: score += 1
            for kw in bearish_kw:
                if kw in text: score -= 1
            articles.append({
                "title":  a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "date":   a.get("publishedAt", "")[:10]
            })

        label = "BULLISH" if score > 2 else "BEARISH" if score < -2 else "NEUTRAL"
        return {"score": score, "label": label, "articles": articles}
    except Exception as e:
        logger.warning(f"Errore sentiment: {e}")
        return {"score": 0, "label": "NEUTRAL", "articles": []}


def get_extended_news() -> list:
    """Notizie estese per il report mattutino."""
    try:
        queries = [
            "gold XAU price forecast today",
            "federal reserve interest rates decision",
            "dollar index DXY today",
            "inflation CPI data",
            "geopolitical risk safe haven gold"
        ]
        all_articles = []
        for q in queries:
            url    = "https://newsapi.org/v2/everything"
            params = {"q": q, "language": "en", "sortBy": "publishedAt",
                      "pageSize": 3, "apiKey": NEWS_API_KEY}
            r    = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("status") == "ok":
                for a in data.get("articles", []):
                    title  = a.get("title", "")
                    source = a.get("source", {}).get("name", "")
                    date   = a.get("publishedAt", "")[:10]
                    if title and source:
                        all_articles.append(f"📰 *{source}* ({date})\n_{title}_")
        seen, unique = set(), []
        for a in all_articles:
            if a not in seen:
                seen.add(a); unique.append(a)
        return unique[:10]
    except Exception as e:
        logger.warning(f"Errore notizie estese: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# LIVELLO 16 — S/R E LIVELLI CHIAVE
# ═══════════════════════════════════════════════════════════════

def get_support_resistance(df: pd.DataFrame) -> dict:
    recent_20  = df.tail(20)
    recent_50  = df.tail(50)
    recent_100 = df.tail(100)

    support    = round(float(recent_100["Low"].min()), 2)
    resistance = round(float(recent_100["High"].max()), 2)
    s_near     = round(float(recent_20["Low"].min()), 2)
    r_near     = round(float(recent_20["High"].max()), 2)

    prev = df.iloc[-2]
    pivot = round((float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3, 2)
    r1    = round(2 * pivot - float(prev["Low"]), 2)
    s1    = round(2 * pivot - float(prev["High"]), 2)
    r2    = round(pivot + (float(prev["High"]) - float(prev["Low"])), 2)
    s2    = round(pivot - (float(prev["High"]) - float(prev["Low"])), 2)

    return {
        "support":    support,
        "resistance": resistance,
        "s_near":     s_near,
        "r_near":     r_near,
        "pivot":      pivot,
        "r1": r1, "r2": r2,
        "s1": s1, "s2": s2,
    }


# ═══════════════════════════════════════════════════════════════
# LIVELLO 17 — TIPO ORDINE AUTOMATICO
# ═══════════════════════════════════════════════════════════════

def determine_order_type(signal: str, price: float, sr: dict, atr: float,
                         adx: float, rsi: float, structure: str,
                         ob: dict, fvg: dict, regime: str, pd_zone: str) -> tuple:
    s_near = sr.get("s_near", sr["support"])
    r_near = sr.get("r_near", sr["resistance"])
    safe_atr = max(atr, 2.0)

    if signal == "BUY":
        # A mercato: prezzo già in zona ottimale
        if pd_zone == "DISCOUNT" and structure == "BULLISH" and rsi < 50:
            return "BUY", price
        if ob.get("bullish_ob") and ob["bullish_ob"]["low"] <= price <= ob["bullish_ob"]["high"] + safe_atr * 0.5:
            return "BUY", price
        if fvg.get("bullish_fvg") and fvg["bullish_fvg"]["bottom"] <= price <= fvg["bullish_fvg"]["top"]:
            return "BUY", price
        # Breakout resistenza
        if price >= r_near * 0.9998 and adx >= 20 and regime in ["TRENDING_UP", "NORMAL"]:
            return "BUY STOP", round(r_near + safe_atr * 0.15, 2)
        # Ritracciamento a OB
        if ob.get("bullish_ob"):
            return "BUY LIMIT", round(ob["bullish_ob"]["high"], 2)
        # Ritracciamento a FVG
        if fvg.get("bullish_fvg"):
            return "BUY LIMIT", round(fvg["bullish_fvg"]["bottom"], 2)
        # Ritracciamento a supporto
        return "BUY LIMIT", round(s_near + safe_atr * 0.3, 2)

    elif signal == "SELL":
        if pd_zone == "PREMIUM" and structure == "BEARISH" and rsi > 50:
            return "SELL", price
        if ob.get("bearish_ob") and ob["bearish_ob"]["low"] - safe_atr * 0.5 <= price <= ob["bearish_ob"]["high"]:
            return "SELL", price
        if fvg.get("bearish_fvg") and fvg["bearish_fvg"]["bottom"] <= price <= fvg["bearish_fvg"]["top"]:
            return "SELL", price
        if price <= s_near * 1.0002 and adx >= 20 and regime in ["TRENDING_DOWN", "NORMAL"]:
            return "SELL STOP", round(s_near - safe_atr * 0.15, 2)
        if ob.get("bearish_ob"):
            return "SELL LIMIT", round(ob["bearish_ob"]["low"], 2)
        if fvg.get("bearish_fvg"):
            return "SELL LIMIT", round(fvg["bearish_fvg"]["top"], 2)
        return "SELL LIMIT", round(r_near - safe_atr * 0.3, 2)

    return "NEUTRAL", price


# ═══════════════════════════════════════════════════════════════
# LIVELLO 18 — RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def calculate_risk_levels(signal: str, entry: float, atr: float, regime: str) -> dict:
    safe_atr = max(atr, 2.0)

    mult = {
        "TRENDING_UP":   (1.0, 1.2, 2.2, 3.5),
        "TRENDING_DOWN": (1.0, 1.2, 2.2, 3.5),
        "RANGING":       (0.8, 0.8, 1.5, 2.2),
        "VOLATILE":      (1.5, 1.2, 2.5, 4.0),
        "NORMAL":        (1.0, 1.0, 2.0, 3.2),
    }
    sl_m, tp1_m, tp2_m, tp3_m = mult.get(regime, (1.0, 1.0, 2.0, 3.2))

    sl_d  = round(safe_atr * sl_m,  2)
    tp1_d = round(safe_atr * tp1_m, 2)
    tp2_d = round(safe_atr * tp2_m, 2)
    tp3_d = round(safe_atr * tp3_m, 2)
    be_d  = 10  # $10 = 10 pips fissi

    if signal == "BUY":
        sl  = round(entry - sl_d,  2)
        tp1 = round(entry + tp1_d, 2)
        tp2 = round(entry + tp2_d, 2)
        tp3 = round(entry + tp3_d, 2)
        be  = round(entry + be_d,  2)
    else:
        sl  = round(entry + sl_d,  2)
        tp1 = round(entry - tp1_d, 2)
        tp2 = round(entry - tp2_d, 2)
        tp3 = round(entry - tp3_d, 2)
        be  = round(entry - be_d,  2)

    rr1 = round(tp1_d / sl_d, 2) if sl_d > 0 else 0
    rr2 = round(tp2_d / sl_d, 2) if sl_d > 0 else 0
    rr3 = round(tp3_d / sl_d, 2) if sl_d > 0 else 0

    return {"sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "be": be, "rr1": rr1, "rr2": rr2, "rr3": rr3}


def calculate_position_size(account_balance: float, risk_percent: float,
                            entry: float, sl: float, pip_value: float = 1.0) -> dict:
    """Compatibilità: usa l'unico motore di sizing del risk manager."""
    del pip_value
    from risk_manager import calculate_lot_size

    return calculate_lot_size(account_balance, risk_percent, entry, sl)


# ═══════════════════════════════════════════════════════════════
# LIVELLO 19 — META-AI: AGGREGATORE STRATEGIE
# ═══════════════════════════════════════════════════════════════

def aggregate_strategies(strategies: dict, regime: dict, timeframe: str = "5min") -> dict:
    """
    Aggrega i segnali di tutte le strategie con pesi dinamici per regime.
    Ogni strategia vota con il suo score. Il Meta-AI decide il segnale finale.
    """
    # Usa regime_compat per BLOCKED_COMBOS (compatibilità con 5 nuovi stati)
    regime_name = regime.get("regime_compat") or regime.get("regime", "NORMAL")

    # Pesi per regime — aggiornati sulla base del backtest reale su 3000 candele:
    # Mean Reversion: 19% win rate (n=21) -> azzerata ovunque, pattern negativo confermato
    # Trend Following: 48-50% win rate anche nei suoi regimi ideali (n=192-305) -> peso dimezzato
    weights = {
        "TRENDING_UP": {
            "smc":         1.2,
            "trend":       1.0,
            "mean_rev":    0.0,
            "momentum":    1.8,
            "event":       0.8,
            "stat_arb":    1.0,
            "ml":          1.5,
            "candle":      1.0,
            "order_flow":  1.2,
        },
        "TRENDING_DOWN": {
            "smc":         1.2,
            "trend":       1.0,
            "mean_rev":    0.0,
            "momentum":    1.8,
            "event":       0.8,
            "stat_arb":    1.0,
            "ml":          1.5,
            "candle":      1.0,
            "order_flow":  1.2,
        },
        "RANGING": {
            "smc":         1.0,
            "trend":       0.4,
            "mean_rev":    0.0,
            "momentum":    0.5,
            "event":       1.0,
            "stat_arb":    1.2,
            "ml":          1.2,
            "candle":      1.8,
            "order_flow":  1.5,
        },
        "VOLATILE": {
            "smc":         1.5,
            "trend":       0.5,
            "mean_rev":    0.0,
            "momentum":    0.8,
            "event":       1.5,
            "stat_arb":    0.8,
            "ml":          1.0,
            "candle":      1.2,
            "order_flow":  1.3,
        },
        "NORMAL": {
            "smc":         1.5,
            "trend":       0.7,
            "mean_rev":    0.0,
            "momentum":    1.2,
            "event":       1.0,
            "stat_arb":    1.0,
            "ml":          1.5,
            "candle":      1.2,
            "order_flow":  1.2,
        }
    }

    w = dict(weights.get(regime_name, weights["NORMAL"]))

    # Applica soltanto moltiplicatori deterministici e conservativi calcolati
    # dai trade realmente chiusi. Se il file non esiste, i default restano invariati.
    learned = {}
    try:
        from self_learning import load_learned_weights

        learned = load_learned_weights()
    except Exception as exc:
        logger.warning("Pesi appresi non disponibili: %s", exc)

    if regime_name in set(learned.get("blocked_regimes", [])):
        return {
            "signal": "NEUTRAL",
            "total_score": 0,
            "buy_count": 0,
            "sell_count": 0,
            "active": [],
            "filter_note": f"Regime {regime_name} bloccato dai dati reali",
        }

    strategy_multipliers = learned.get("strategy_multipliers", {})
    for name in w:
        multiplier = float(strategy_multipliers.get(name, 1.0))
        w[name] *= min(1.20, max(0.80, multiplier))

    regime_multiplier = float(
        learned.get("regime_multipliers", {}).get(regime_name, 1.0)
    )
    regime_multiplier = min(1.15, max(0.85, regime_multiplier))
    direction_bias = learned.get("direction_bias", {})

    buy_vote  = 0
    sell_vote = 0
    active_strategies = []

    for name, result in strategies.items():
        sig   = result.get("signal", "NEUTRAL")
        score = result.get("score", 0)
        wt    = w.get(name, 1.0)

        if sig == "BUY":
            buy_vote += score * wt * regime_multiplier
            active_strategies.append(f"{name}:BUY({score})")
        elif sig == "SELL":
            sell_vote += score * wt * regime_multiplier
            active_strategies.append(f"{name}:SELL({score})")

    # Conta quante strategie concordano
    buy_count  = sum(1 for r in strategies.values() if r.get("signal") == "BUY")
    sell_count = sum(1 for r in strategies.values() if r.get("signal") == "SELL")
    buy_vote *= min(1.10, max(0.90, float(direction_bias.get("BUY", 1.0))))
    sell_vote *= min(1.10, max(0.90, float(direction_bias.get("SELL", 1.0))))

    # Score massimo di una singola strategia per ogni lato
    buy_max_single  = max([r.get("score", 0) for r in strategies.values() if r.get("signal") == "BUY"], default=0)
    sell_max_single = max([r.get("score", 0) for r in strategies.values() if r.get("signal") == "SELL"], default=0)

    # Regola A: almeno 2 strategie concordi + vote score >= 7 (era 10 — troppo restrittivo)
    # Regola B: anche 1 sola strategia se il suo score singolo è >= 8 (segnale forte isolato)
    buy_qualifies  = (buy_count >= 2 and buy_vote >= 7) or (buy_count >= 1 and buy_max_single >= 8)
    sell_qualifies = (sell_count >= 2 and sell_vote >= 7) or (sell_count >= 1 and sell_max_single >= 8)

    # ── FILTRO REGIME + DIREZIONE ──────────────────────────────────────────
    # NOTA: i BLOCKED_COMBOS precedenti bloccavano BUY in TRENDING_UP/DOWN e
    # VOLATILE su entrambe le direzioni, basandosi su backtest con n=21-192 trade
    # — campione troppo piccolo per bloccare intere categorie.
    # Con quei filtri il bot produceva quasi sempre NEUTRAL (confermato dai log).
    # Manteniamo solo blocchi con evidenza forte da paper trading reale (min 30 trade).
    # Per ora: nessun combo bloccato — il risk manager e il MIN_PROB filtrano i segnali deboli.
    BLOCKED_COMBOS: set = set()

    # NORMAL su M5 bloccato se abbiamo evidenza negativa (allentiamo da H1+ a tutti i TF
    # solo se supportato da dati reali futuri — per ora commentiamo)
    # TIMEFRAMES_BLOCK_NORMAL = {"1h", "4h", "1day"}
    # if regime_name == "NORMAL" and timeframe in TIMEFRAMES_BLOCK_NORMAL: ...

    def _is_blocked(signal: str) -> bool:
        return (regime_name, signal) in BLOCKED_COMBOS

    if buy_qualifies and buy_vote >= sell_vote:
        if _is_blocked("BUY"):
            logger.debug(f"[FILTRO] BUY bloccato in regime {regime_name}")
            if sell_qualifies and not _is_blocked("SELL"):
                return {"signal": "SELL", "total_score": round(sell_vote, 1),
                        "buy_count": buy_count, "sell_count": sell_count,
                        "active": active_strategies, "filter_note": f"BUY bloccato ({regime_name})"}
            return {"signal": "NEUTRAL", "total_score": 0, "buy_count": buy_count,
                    "sell_count": sell_count, "active": active_strategies,
                    "filter_note": f"BUY bloccato ({regime_name})"}
        return {"signal": "BUY", "total_score": round(buy_vote, 1),
                "buy_count": buy_count, "sell_count": sell_count, "active": active_strategies}

    elif sell_qualifies and sell_vote >= buy_vote:
        if _is_blocked("SELL"):
            logger.debug(f"[FILTRO] SELL bloccato in regime {regime_name}")
            return {"signal": "NEUTRAL", "total_score": 0, "buy_count": buy_count,
                    "sell_count": sell_count, "active": active_strategies,
                    "filter_note": f"SELL bloccato ({regime_name})"}
        return {"signal": "SELL", "total_score": round(sell_vote, 1),
                "buy_count": buy_count, "sell_count": sell_count, "active": active_strategies}

    return {"signal": "NEUTRAL", "total_score": 0, "buy_count": buy_count,
            "sell_count": sell_count, "active": active_strategies}


def estimate_probability(total_score: float, buy_count: int, sell_count: int,
                         trend_confirmed: bool, regime: str, structure: str,
                         economic_risk: bool) -> int:
    # NOTA IMPORTANTE: questo NON è una probabilità statistica reale.
    # È uno score di confidenza normalizzato nell'intervallo [40, 97].
    # Un valore di 75 NON significa "75% di probabilità di successo".
    # Significa "il segnale ha una confidenza media-alta basata sui fattori tecnici".
    # Per avere probabilità reali servirebbero migliaia di trade backtestati.
    # Usare il valore solo come filtro qualitativo (soglia MIN_PROB=55).
    base = min(40 + int(total_score * 0.8), 90)
    strategies_agree = max(buy_count, sell_count)
    if strategies_agree >= 5:  base += 8
    elif strategies_agree >= 4: base += 5
    elif strategies_agree >= 3: base += 3
    if trend_confirmed:           base += 5
    if regime in ["TRENDING_UP", "TRENDING_DOWN"]: base += 3
    if structure in ["BULLISH", "BEARISH"]:        base += 2
    if economic_risk:                              base -= 10
    return min(max(base, 40), 97)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT PRINCIPALE — M5 e H1/H4
# ═══════════════════════════════════════════════════════════════

def full_analyze(timeframe_focus: str = "5min") -> dict:
    """
    Analisi completa su tutti i livelli.
    timeframe_focus: '5min' per segnali M5, '1h' per H1, '4h' per H4
    """
    now = datetime.now(TIMEZONE)

    # Dati multi-timeframe
    mtf_data = get_multi_timeframe_data()
    df_main  = mtf_data.get(timeframe_focus)
    df_5m    = mtf_data.get("5min")
    df_15m   = mtf_data.get("15min")
    df_1m    = mtf_data.get("1min")
    df_1h    = mtf_data.get("1h")
    df_4h    = mtf_data.get("4h")

    if df_main is None or len(df_main) < 50:
        raise ValueError(f"Dati {timeframe_focus} non disponibili")

    # Calcola indicatori sul TF principale
    df_main = compute_indicators(df_main)
    df_main = detect_swing_points(df_main)

    # MTF trend
    mtf_trends = get_mtf_trend(mtf_data)

    # Valori correnti
    row   = df_main.iloc[-1]
    data_timestamp = pd.Timestamp(df_main.index[-1]).isoformat()
    price = round(float(row["Close"]), 2)
    atr   = max(float(row["atr"]) if not pd.isna(row["atr"]) else 5, 2.0)
    rsi   = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50
    adx   = float(row["adx"]) if not pd.isna(row["adx"]) else 0

    # SMC sul TF principale
    smc       = detect_bos_choch(df_main)
    ob        = detect_order_blocks(df_main)
    fvg       = detect_fvg(df_main)
    liquidity = detect_liquidity(df_main)
    breaker   = detect_breaker_blocks(df_main, ob)
    mit       = detect_mitigation(df_main, ob)
    pd_zone   = detect_premium_discount(df_main, smc)
    sr        = get_support_resistance(df_main)

    # Regime
    regime_data = detect_market_regime(df_main)
    regime      = regime_data["regime"]

    # Sentiment e calendario
    sentiment  = get_news_sentiment()
    calendar   = get_economic_events()
    econ_risk  = calendar["high_impact_today"]
    # Se il calendario ha dato errore → tratta come rischio economico elevato (fail-safe)
    if calendar.get("error"):
        econ_risk = True
        logger.warning("[ANALYZER] Errore calendario → econ_risk=True (fail-safe)")

    # Dati correlati
    dxy   = get_dxy_price()
    us10y = get_us10y_price()

    # ── ESEGUI TUTTE LE STRATEGIE ──
    strategies = {}

    # 1. SMC v3.0 (solo su 5min/15min con 1min entry)
    if df_15m is not None and df_1m is not None and len(df_15m) > 30 and len(df_1m) > 30:
        df_15m_ind = compute_indicators(detect_swing_points(df_15m))
        df_1m_ind  = compute_indicators(detect_swing_points(df_1m))
        smc_result = smc_v3_strategy(df_15m_ind, df_1m_ind, smc, ob, fvg)
        strategies["smc"] = smc_result
    else:
        strategies["smc"] = {"signal": "NEUTRAL", "score": 0}

    # 2. Trend Following
    strategies["trend"] = trend_following_strategy(df_main)

    # 3. Mean Reversion
    strategies["mean_rev"] = mean_reversion_strategy(df_main)

    # 4. Momentum
    strategies["momentum"] = momentum_strategy(df_main, mtf_trends)

    # 5. Event-Driven
    strategies["event"] = event_driven_strategy(calendar, sentiment)

    # 6. Statistical Arbitrage
    strategies["stat_arb"] = statistical_arbitrage_strategy(price, dxy, us10y)

    # 7. ML Alpha
    try:
        strategies["ml"] = ml_alpha_strategy(df_main, mtf_trends, smc)
    except Exception as _e:
        logger.warning(f"ml_alpha_strategy errore: {_e}")
        strategies["ml"] = {"signal": "NEUTRAL", "score": 0, "confidence": 0}

    # 8. Candlestick
    strategies["candle"] = candlestick_strategy(df_main)

    # 9. Order Flow
    strategies["order_flow"] = order_flow_strategy(df_main)

    # ── AGGREGAZIONE META-AI ──
    aggregated = aggregate_strategies(strategies, regime_data, timeframe=timeframe_focus)

    if aggregated["signal"] == "NEUTRAL":
        return {
            "signal":      "NEUTRAL",
            "price":       price,
            "regime":      regime,
            "buy_count":   aggregated["buy_count"],
            "sell_count":  aggregated["sell_count"],
            "active":      aggregated["active"],
            "timeframe":   timeframe_focus,
            "data_timestamp": data_timestamp,
            "time":        now.strftime("%d/%m/%Y %H:%M"),
        }

    signal = aggregated["signal"]

    # Tipo ordine
    order_type, entry = determine_order_type(
        signal, price, sr, atr, adx, rsi,
        smc["structure"], ob, fvg, regime, pd_zone
    )

    # Livelli di rischio
    risk = calculate_risk_levels(signal, entry, atr, regime)

    # Probabilità
    trend_confirmed = mtf_trends.get("1h") == signal
    prob = estimate_probability(
        aggregated["total_score"],
        aggregated["buy_count"],
        aggregated["sell_count"],
        trend_confirmed, regime,
        smc["structure"], econ_risk
    )

    # Setup SMC attivo
    smc_setup = strategies["smc"].get("setup", "")

    return {
        # Core
        "signal":      signal,
        "order_type":  order_type,
        "price":       price,
        "entry":       entry,
        "timeframe":   timeframe_focus,
        "data_timestamp": data_timestamp,

        # Livelli
        "sl":          risk["sl"],
        "tp1":         risk["tp1"],
        "tp2":         risk["tp2"],
        "tp3":         risk["tp3"],
        "be":          risk["be"],
        "rr1":         risk["rr1"],
        "rr2":         risk["rr2"],
        "rr3":         risk["rr3"],

        # Metriche
        "prob":        prob,
        "total_score": aggregated["total_score"],
        "buy_count":   aggregated["buy_count"],
        "sell_count":  aggregated["sell_count"],
        "active":      aggregated["active"],
        "atr":         round(atr, 2),
        "rsi":         round(rsi, 1),
        "adx":         round(adx, 1),

        # Contesto
        "regime":      regime,
        "structure":   smc["structure"],
        "pd_zone":     pd_zone,
        "bos":         smc.get("bos"),
        "choch":       smc.get("choch"),
        "last_high":   smc.get("last_high"),
        "last_low":    smc.get("last_low"),
        "smc_setup":   smc_setup,
        "ob":          ob,
        "fvg":         fvg,
        "liquidity":   liquidity,
        "breaker":     breaker,
        "mitigation":  mit,
        "mtf_trends":  mtf_trends,
        "trend_confirmed": trend_confirmed,
        "candle":      strategies["candle"].get("pattern", ""),
        "sr":          sr,

        # Correlazioni
        "dxy":         round(dxy, 2) if dxy > 0 else None,
        "us10y":       round(us10y, 2) if us10y > 0 else None,

        # Sentiment
        "sentiment":   sentiment,
        "calendar":    calendar,
        "econ_risk":   econ_risk,

        # Strategie individuali
        "strategies":  {k: {"signal": v.get("signal"), "score": v.get("score", 0)}
                        for k, v in strategies.items()},

        "time":        now.strftime("%d/%m/%Y %H:%M"),
    }
