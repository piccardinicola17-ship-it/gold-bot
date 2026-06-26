"""
trade_manager.py — Gestione trade attiva completa
Monitora posizioni aperte, manda notifiche BE/TP/SL in tempo reale
"""

import json
import os
import logging
import requests
import pytz

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")
TWELVE_API_KEY    = os.environ.get("TWELVE_API_KEY", "85f2bac59bb24b3a8e55551a3337f844")
ACTIVE_TRADE_FILE = "/tmp/active_trade.json"


def save_active_trade(data: dict):
    with open(ACTIVE_TRADE_FILE, "w") as f:
        json.dump(data, f, default=str)


def load_active_trade() -> dict:
    if not os.path.exists(ACTIVE_TRADE_FILE):
        return {}
    try:
        with open(ACTIVE_TRADE_FILE) as f:
            return json.load(f)
    except:
        return {}


def clear_active_trade():
    if os.path.exists(ACTIVE_TRADE_FILE):
        os.remove(ACTIVE_TRADE_FILE)


def get_current_price() -> float:
    try:
        url = "https://api.twelvedata.com/price"
        r   = requests.get(url, params={"symbol": "XAU/USD", "apikey": TWELVE_API_KEY}, timeout=5)
        return float(r.json()["price"])
    except:
        return 0.0


def be_message(trade: dict, price: float) -> str:
    return (
        f"⚖️ *BREAK EVEN — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Prezzo attuale: *${price}*\n"
        f"📍 Entry: ${trade['entry']}\n"
        f"🔒 *Sposta SL a ${trade['entry']} (break even)*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_+10 pips raggiunti — proteggi il capitale_"
    )


def tp1_message(trade: dict, price: float) -> str:
    return (
        f"🎯 *TP1 RAGGIUNTO — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Prezzo: *${price}*\n"
        f"✅ *Chiudi il 33% della posizione*\n"
        f"🔒 Sposta SL a break even (${trade['entry']})\n"
        f"🎯 Prossimo: TP2 @ ${trade['tp2']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Profitto parziale assicurato — lascia correre il resto_"
    )


def tp2_message(trade: dict, price: float) -> str:
    return (
        f"🎯🎯 *TP2 RAGGIUNTO — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Prezzo: *${price}*\n"
        f"✅ *Chiudi un altro 33%*\n"
        f"🔒 Sposta SL a TP1 (${trade['tp1']})\n"
        f"🎯 Prossimo: TP3 @ ${trade['tp3']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Ottimo trade — lascia correre il 34% restante_"
    )


def tp3_message(trade: dict, price: float) -> str:
    return (
        f"🏆 *TP3 RAGGIUNTO — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Prezzo: *${price}*\n"
        f"✅ *Chiudi tutto — obiettivo massimo raggiunto!*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Trade perfetto completato_ 🎉\n"
        f"R:R realizzato: 1:{trade.get('rr3', 'N/A')}"
    )


def sl_message(trade: dict, price: float) -> str:
    return (
        f"🛑 *STOP LOSS COLPITO — XAUUSD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Prezzo: *${price}*\n"
        f"❌ *Chiudi tutta la posizione*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Perdita controllata — il rischio era gestito_"
    )


async def monitor_active_trade(bot, chat_id: str):
    """
    Controlla se il trade attivo ha raggiunto BE, TP1, TP2, TP3 o SL.
    Manda un messaggio separato per ogni evento.
    """
    trade = load_active_trade()
    if not trade:
        return

    signal   = trade.get("signal")
    entry    = float(trade.get("entry", 0))
    sl       = float(trade.get("sl", 0))
    tp1      = float(trade.get("tp1", 0))
    tp2      = float(trade.get("tp2", 0))
    tp3      = float(trade.get("tp3", 0))
    be       = float(trade.get("be", entry + 10 if signal == "BUY" else entry - 10))
    notified = trade.get("notified", {})
    price    = get_current_price()

    if price == 0:
        return

    msgs = []

    if signal == "BUY":
        if not notified.get("be")  and price >= be:
            msgs.append(be_message(trade, price));  notified["be"]  = True
        if not notified.get("tp1") and price >= tp1:
            msgs.append(tp1_message(trade, price)); notified["tp1"] = True
        if not notified.get("tp2") and price >= tp2:
            msgs.append(tp2_message(trade, price)); notified["tp2"] = True
        if not notified.get("tp3") and price >= tp3:
            msgs.append(tp3_message(trade, price)); notified["tp3"] = True
            clear_active_trade()
        if not notified.get("sl")  and price <= sl:
            msgs.append(sl_message(trade, price));  notified["sl"]  = True
            clear_active_trade()

    elif signal == "SELL":
        if not notified.get("be")  and price <= be:
            msgs.append(be_message(trade, price));  notified["be"]  = True
        if not notified.get("tp1") and price <= tp1:
            msgs.append(tp1_message(trade, price)); notified["tp1"] = True
        if not notified.get("tp2") and price <= tp2:
            msgs.append(tp2_message(trade, price)); notified["tp2"] = True
        if not notified.get("tp3") and price <= tp3:
            msgs.append(tp3_message(trade, price)); notified["tp3"] = True
            clear_active_trade()
        if not notified.get("sl")  and price >= sl:
            msgs.append(sl_message(trade, price));  notified["sl"]  = True
            clear_active_trade()

    if msgs:
        trade["notified"] = notified
        save_active_trade(trade)
        for msg in msgs:
            await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
