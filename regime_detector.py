"""
regime_detector.py — Rilevamento regime di mercato per GoldMind.
FIX: aggiunta detect_regime_v2 (alias di detect_regime) per compatibilità
     con agent_orchestrator.py che la importa.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import pytz

logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")


def detect_market_regime(df) -> dict:
    """
    Rileva il regime di mercato corrente dalle candele.
    Ritorna un dict con 'regime', 'strength', 'details'.
    """
    try:
        if df is None or len(df) < 20:
            return {"regime": "UNKNOWN", "strength": 0, "details": "Dati insufficienti"}

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # EMA trend
        ema20  = close.ewm(span=20).mean().iloc[-1]
        ema50  = close.ewm(span=50).mean().iloc[-1]
        ema200 = close.ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema50
        last   = float(close.iloc[-1])

        # ATR per volatilità
        tr_list = []
        for i in range(1, min(15, len(df))):
            h = float(high.iloc[-i])
            l = float(low.iloc[-i])
            pc = float(close.iloc[-i-1])
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(tr_list) / len(tr_list) if tr_list else 1.0
        atr_pct = (atr / last) * 100 if last > 0 else 0

        # Range recente (ultime 20 candele)
        recent_high = float(high.iloc[-20:].max())
        recent_low  = float(low.iloc[-20:].min())
        range_pct   = ((recent_high - recent_low) / last * 100) if last > 0 else 0

        # Logica regime
        trending_up   = last > ema20 > ema50 and ema20 > ema200
        trending_down = last < ema20 < ema50 and ema20 < ema200
        volatile      = atr_pct > 0.8
        ranging       = range_pct < 1.5 and not trending_up and not trending_down

        if volatile and trending_up:
            regime   = "TRENDING_UP"
            strength = 80
        elif volatile and trending_down:
            regime   = "TRENDING_DOWN"
            strength = 80
        elif trending_up:
            regime   = "TRENDING_UP"
            strength = 65
        elif trending_down:
            regime   = "TRENDING_DOWN"
            strength = 65
        elif volatile:
            regime   = "VOLATILE"
            strength = 70
        elif ranging:
            regime   = "RANGING"
            strength = 60
        else:
            regime   = "NORMAL"
            strength = 50

        return {
            "regime":   regime,
            "strength": strength,
            "details": {
                "last":     round(last, 2),
                "ema20":    round(float(ema20), 2),
                "ema50":    round(float(ema50), 2),
                "atr_pct":  round(atr_pct, 3),
                "range_pct":round(range_pct, 3),
            }
        }

    except Exception as e:
        logger.error(f"detect_market_regime errore: {e}")
        return {"regime": "NORMAL", "strength": 50, "details": str(e)}


# ── ALIAS richiesto da agent_orchestrator.py ──────────────────────────────────
# La versione precedente esportava detect_regime_v2; la manteniamo come alias
# per non rompere nessun import esistente.
def detect_regime_v2(df) -> dict:
    """Alias di detect_market_regime per compatibilità backward."""
    return detect_market_regime(df)


def format_regime_message(regime_data: dict) -> str:
    """Formatta il regime in un messaggio Telegram leggibile."""
    regime   = regime_data.get("regime", "UNKNOWN")
    strength = regime_data.get("strength", 0)
    details  = regime_data.get("details", {})

    regime_map = {
        "TRENDING_UP":   "📈 Trending Up",
        "TRENDING_DOWN": "📉 Trending Down",
        "RANGING":       "📦 Ranging",
        "VOLATILE":      "🌪 Volatile",
        "NORMAL":        "➡️ Normale",
        "ACCUMULATION":  "🔄 Accumulation",
        "MANIPULATION":  "⚠️ Manipulation",
        "DISTRIBUTION":  "📤 Distribution",
        "REVERSAL":      "🔁 Reversal",
        "UNKNOWN":       "❓ Sconosciuto",
    }
    label = regime_map.get(regime, regime)

    msg = (
        f"🌍 *REGIME DI MERCATO*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Regime: *{label}*\n"
        f"Forza: *{strength}%*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )

    if isinstance(details, dict):
        if details.get("last"):
            msg += f"Prezzo: ${details['last']}\n"
        if details.get("ema20"):
            msg += f"EMA20: ${details['ema20']} | EMA50: ${details.get('ema50','?')}\n"
        if details.get("atr_pct"):
            msg += f"ATR%: {details['atr_pct']}% | Range%: {details.get('range_pct','?')}%\n"
    else:
        msg += f"_Dettagli: {details}_\n"

    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"_Aggiornato: {datetime.now(TIMEZONE).strftime('%H:%M IT')}_"
    return msg
