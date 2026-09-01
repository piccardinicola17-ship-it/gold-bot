"""
ai_assistant.py — Assistente conversazionale GoldMind.
Usa Groq per rispondere a domande libere sul mercato con contesto live.
"""

import os
import logging
import requests
import asyncio
from datetime import datetime
import pytz

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "groq/compound-mini"


def build_context_snapshot() -> str:
    from analyzer import full_analyze, get_news_sentiment, get_economic_events, get_extended_news
    from trade_manager import load_active_trade

    now   = datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M")
    parts = [f"Data e ora attuali: {now} (Europe/Rome)."]

    try:
        data = full_analyze(timeframe_focus="5min")
        parts.append(f"\nPrezzo XAU/USD attuale: ${data.get('price')}")
        parts.append(f"Regime di mercato M5: {data.get('regime')}")
        if data.get("signal") != "NEUTRAL":
            parts.append(f"Segnale M5 attivo: {data['signal']} {data.get('order_type')} @ {data.get('entry')} — prob {data.get('prob')}%")
        else:
            parts.append(f"Nessun segnale M5. BUY: {data.get('buy_count',0)}, SELL: {data.get('sell_count',0)}.")
        mtf = data.get("mtf_trends", {})
        if mtf:
            parts.append(f"Trend MTF: {', '.join(f'{tf}:{t}' for tf,t in mtf.items())}")
    except Exception as e:
        parts.append(f"(Analisi mercato non disponibile: {e})")

    try:
        sentiment = get_news_sentiment()
        parts.append(f"\nSentiment notizie oro: {sentiment.get('label')} (score {sentiment.get('score',0):+d})")
        news = get_extended_news()
        if news:
            parts.append(f"Ultime notizie:\n" + "\n".join(n.replace("*","").replace("_","") for n in news[:5]))
    except Exception as e:
        parts.append(f"(Notizie non disponibili: {e})")

    try:
        from analyzer import get_upcoming_events
        cal = get_economic_events()
        if cal.get("high_impact_today"):
            events_txt = "; ".join(
                f"{ev['title']} alle {ev['time']} IT (prev: {ev['forecast']}, prec: {ev['previous']})"
                for ev in cal.get("events", [])
            )
            parts.append(f"\nEventi OGGI ad alto impatto: {events_txt}")
        else:
            parts.append("\nNessun evento macro ad alto impatto oggi.")

        upcoming = get_upcoming_events(days_ahead=7)
        today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        future = [e for e in upcoming if e["date"] > today_str or (e["date"] == today_str and e.get("hours_away",0) > 0)]
        if future:
            future_txt = "; ".join(
                f"{ev['title']} il {ev['date']} alle {ev['time']} IT"
                for ev in future[:6]
            )
            parts.append(f"\nProssimi eventi macro settimana: {future_txt}")
    except Exception as e:
        parts.append(f"(Calendario non disponibile: {e})")

    try:
        trade = load_active_trade()
        if trade:
            parts.append(
                f"\nTrade attivo: {trade.get('signal')} {trade.get('order_type')} "
                f"@ {trade.get('entry')} [{trade.get('timeframe','').upper()}] — "
                f"SL {trade.get('sl')}, TP1 {trade.get('tp1')}, TP2 {trade.get('tp2')}, TP3 {trade.get('tp3')}"
            )
        else:
            parts.append("\nNessun trade attivo al momento.")
    except Exception as e:
        parts.append(f"(Stato trade non disponibile: {e})")

    return "\n".join(parts)


SYSTEM_PROMPT = """Sei l'assistente AI integrato in un bot di trading Telegram specializzato su XAU/USD (Oro).
Rispondi in italiano, in modo diretto, colloquiale e competente, come un trader esperto.

REGOLE:
- Usa SEMPRE i dati di contesto forniti (prezzo, regime, strategie, notizie, calendario, trade attivo).
- Se ti chiedono dei prossimi eventi macro, usa i dati nel contesto.
- Spiega sempre l'impatto atteso di ogni evento su XAU/USD.
- Sii sintetico: 6-8 righe per risposta.
- Non inventare dati — se mancano davvero, dillo.
- Non dare consigli assoluti: presenta dati e lascia la decisione all'utente."""


async def ask_ai(question: str) -> str:
    if not GROQ_API_KEY:
        return "Assistente AI non configurato — manca GROQ_API_KEY."

    try:
        context = await asyncio.to_thread(build_context_snapshot)
    except Exception as e:
        logger.error(f"Errore contesto: {e}")
        context = "(Contesto di mercato non disponibile)"

    def _request():
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"CONTESTO:\n{context}\n\nDOMANDA:\n{question}"}
                ],
                "temperature": 0.4,
                "max_tokens":  500,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    try:
        data   = await asyncio.to_thread(_request)
        answer = data["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        logger.error(f"Errore Groq: {e}")
        return f"Errore nel generare la risposta: {e}"
