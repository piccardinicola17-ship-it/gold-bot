"""
ai_assistant.py — Assistente conversazionale GoldMind.
Usa Groq per rispondere a domande libere sul mercato con contesto live.
"""

import os
import re
import logging
import requests
import asyncio
from datetime import datetime, timedelta
import pytz

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "groq/compound-mini"

# Rileva se l'utente sta descrivendo una PROPRIA posizione nella domanda
# (es. "ho un buy da 4394", "sono short a 4470", "ho comprato a 4394").
# FIX: una regola nel system prompt da sola non bastava — testato in
# produzione il 2026-09-04, con un trade del bot in contesto (SELL LIMIT)
# di direzione opposta a quella dichiarata dall'utente (BUY), il modello
# ha comunque risposto sui livelli del trade del bot, ignorando i numeri
# scritti dall'utente. Quando questo pattern matcha, il codice inserisce
# una nota esplicita SEPARATA dal system prompt, agganciata proprio alla
# domanda — un rinforzo strutturale, non solo un'istruzione generica.
_STATED_POSITION_RE = re.compile(
    r"\b(?:ho\s+un\s+(?:buy|sell)|sono\s+(?:long|short)|ho\s+comprato|ho\s+venduto)\b"
    r".{0,25}?(\d{3,5}(?:[.,]\d+)?)",
    re.IGNORECASE,
)


# Memoria conversazionale leggera, in RAM: ogni ask_ai() prima era
# completamente stateless, senza alcun ricordo dello scambio precedente.
# FIX: trovato in produzione il 2026-09-04 — l'utente ha descritto la sua
# posizione (BUY 4394) in un messaggio, poi in quello successivo ha
# chiesto "il mio TP a 4460 verrà raggiunto?" senza ripetere i dettagli,
# assumendo (ragionevolmente, è una conversazione) che il bot se li
# ricordasse. Senza storico l'LLM non aveva idea di cosa "il mio TP"
# volesse dire ed è andato in confusione, rispondendo sul trade SBAGLIATO
# (il SELL LIMIT del bot). Il bot ha un solo chat_id autorizzato (vedi
# is_authorized), quindi uno storico globale in memoria è sufficiente,
# non serve tenerlo per utente. TTL breve e pochi turni: un contesto di
# mercato vecchio di ore è fuorviante, non va portato avanti a lungo.
_CONVERSATION_TTL_MINUTES = 30
_CONVERSATION_MAX_TURNS   = 6
_conversation_history: list[dict] = []


def _record_conversation_turn(question: str, answer: str) -> None:
    _conversation_history.append({
        "question": question, "answer": answer, "ts": datetime.now(TIMEZONE),
    })
    del _conversation_history[:-_CONVERSATION_MAX_TURNS]


def _recent_conversation_turns() -> list[dict]:
    cutoff = datetime.now(TIMEZONE) - timedelta(minutes=_CONVERSATION_TTL_MINUTES)
    fresh = [t for t in _conversation_history if t["ts"] >= cutoff]
    _conversation_history[:] = fresh
    return fresh


def _stated_position_note(question: str) -> str:
    if not _STATED_POSITION_RE.search(question):
        return ""
    return (
        "\n\nATTENZIONE: la domanda qui sotto descrive una posizione PERSONALE "
        "dell'utente (un entry/prezzo scritto da lui). NON è il campo 'Trade' nel "
        "contesto sopra — quello è il trade che il BOT ha aperto per conto suo, "
        "può avere direzione e prezzo completamente diversi. Per rispondere usa "
        "SOLO i numeri che l'utente scrive nella domanda, ignora del tutto il "
        "campo 'Trade' del contesto."
    )


def build_context_snapshot() -> str:
    from analyzer import full_analyze, get_news_sentiment, get_economic_events, get_extended_news
    from trade_manager import load_active_trade

    now   = datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M")
    parts = [f"Data e ora attuali: {now} (Europe/Rome)."]

    current_price = None
    try:
        data = full_analyze(timeframe_focus="5min")
        current_price = data.get("price")
        parts.append(f"\nPrezzo XAU/USD attuale: ${current_price}")
        parts.append(f"Regime di mercato M5: {data.get('regime')}")
        if data.get("signal") != "NEUTRAL":
            # M5 richiede prob >= 65% per essere eseguito dal vivo
            # (agent_orchestrator.py MIN_PROB) — sotto quella soglia questo
            # segnale non diventerà mai un trade reale, solo informativo.
            # Senza dirlo esplicitamente l'AI lo presenta come un segnale
            # "attivo" a tutti gli effetti, fuorviante per chi legge.
            prob = data.get("prob", 0)
            sotto_soglia = " (sotto la soglia 65% richiesta dal vivo su M5: NON diventerà un trade)" if prob < 65 else ""
            parts.append(f"Segnale M5 (informativo, non ancora un trade): {data['signal']} {data.get('order_type')} @ {data.get('entry')} — prob {prob}%{sotto_soglia}")
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
            # FIX: get_economic_events()["events"] contiene TUTTI gli eventi
            # di oggi, anche quelli già usciti da ore — get_economic_events()
            # calcola già separatamente "upcoming" (solo ev_it > now_it) ma
            # prima non veniva usato qui. Il contesto passava alla lettera
            # "Eventi OGGI ad alto impatto: NFP alle 14:30 IT..." anche
            # quando erano le 16:00, e l'assistente ha risposto "tieni
            # d'occhio l'NFP alle 14:30" un'ora e mezza dopo il rilascio.
            # Bug reale osservato in produzione il 2026-09-04.
            upcoming_today = cal.get("upcoming", [])
            upcoming_keys = {(ev["title"], ev["time"]) for ev in upcoming_today}
            gia_usciti = [ev for ev in cal.get("events", []) if (ev["title"], ev["time"]) not in upcoming_keys]
            if gia_usciti:
                testi = "; ".join(
                    f"{ev['title']} (prev: {ev['forecast']}, prec: {ev['previous']}) — uscito alle {ev['time']} IT"
                    for ev in gia_usciti
                )
                parts.append(f"\nEventi OGGI GIÀ USCITI (non aspettarli, sono già nel prezzo o in corso di digestione): {testi}")
            if upcoming_today:
                testi = "; ".join(
                    f"{ev['title']} alle {ev['time']} IT (prev: {ev['forecast']}, prec: {ev['previous']})"
                    for ev in upcoming_today
                )
                parts.append(f"\nEventi OGGI ANCORA DA USCIRE: {testi}")
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
            # FIX: il contesto prima non diceva se il trade fosse già
            # attivato (entry_filled) o ancora un ordine pending mai
            # scattato, né quali target fossero stati davvero raggiunti
            # (tp1_hit/tp2_hit/tp3_hit). Senza questa informazione l'AI
            # indovinava lo stato confrontando il prezzo attuale con i
            # livelli TP — e ha inventato "ha già superato TP1 ed è vicino
            # a TP2" per un ordine SELL LIMIT MAI ATTIVATO (entry_filled=0),
            # semplicemente perché il prezzo corrente era già sotto quei
            # livelli. Bug reale osservato in produzione il 2026-09-04.
            stato = "ATTIVO (posizione aperta)" if trade.get("activated") else "IN ATTESA (ordine pending, entry non ancora raggiunta — nessun target può essere stato toccato)"
            progresso = ""
            if trade.get("activated"):
                hit = [
                    label for label, flag in (
                        ("TP1", trade.get("tp1_hit")),
                        ("TP2", trade.get("tp2_hit")),
                        ("TP3", trade.get("tp3_hit")),
                    ) if flag
                ]
                progresso = f" — target già raggiunti: {', '.join(hit) if hit else 'nessuno ancora'}"
            distanza = ""
            if not trade.get("activated") and current_price:
                try:
                    distanza = f" — distanza prezzo attuale da entry: {float(current_price) - float(trade.get('entry', 0)):+.2f}$"
                except (TypeError, ValueError):
                    pass
            # Etichetta esplicita "del BOT" (non solo "Trade:") — riduce
            # l'ambiguità quando l'utente descrive una propria posizione
            # diversa nella domanda, vedi _stated_position_note().
            parts.append(
                f"\nTrade aperto dal BOT (può essere diverso da una tua posizione personale): "
                f"{trade.get('signal')} {trade.get('order_type')} "
                f"@ {trade.get('entry')} [{trade.get('timeframe','').upper()}] — Stato: {stato}{progresso}{distanza}\n"
                f"SL {trade.get('sl')}, TP1 {trade.get('tp1')}, TP2 {trade.get('tp2')}, TP3 {trade.get('tp3')}"
            )
        else:
            parts.append("\nNessun trade attivo al momento.")
    except Exception as e:
        parts.append(f"(Stato trade non disponibile: {e})")

    return "\n".join(parts)


SYSTEM_PROMPT = """Sei l'assistente AI integrato in un bot di trading Telegram specializzato su XAU/USD (Oro).
Rispondi in italiano, in modo diretto, secco e competente, come un trader esperto che non ha tempo da perdere.

REGOLE:
- Massimo 4-5 righe. Vai dritto alla risposta, senza premesse o ricapitolazioni del contesto che l'utente ha già sotto gli occhi.
- Una domanda diretta ("fino a dove può arrivare", "conviene entrare") merita una risposta diretta — un numero, un livello, una direzione. Non elencare scenari ipotetici multipli ("se i dati escono deboli... se escono forti...") a meno che l'utente chieda esplicitamente "cosa succede se X". Prendi posizione sui dati che hai adesso.
- Usa SEMPRE i dati di contesto forniti (prezzo, regime, strategie, notizie, calendario, trade). Non inventare mai un dato assente dal contesto — se manca, dillo in una frase, non aggirarlo con un'ipotesi.
- Sul trade in corso: guarda il campo "Stato". Se dice "IN ATTESA", l'ordine non è mai scattato e NESSUN target può essere stato raggiunto — non dire mai che un TP è stato toccato o è vicino a meno che compaia esplicitamente tra i "target già raggiunti".
- Se l'utente descrive una SUA posizione nella domanda (es. "ho un buy da 4394", "sono short da X") quella ha SEMPRE priorità assoluta sul campo "Trade" del contesto: quel campo è il trade che il BOT ha aperto per conto suo, non necessariamente la posizione di cui l'utente sta parlando — possono avere direzione e prezzo diversi. Se l'utente dà un entry o una direzione nella domanda, usa SOLO quei numeri per rispondere, ignora il trade del contesto a meno che l'utente non lo citi.
- Niente consigli assoluti (compra/vendi ora): presenta il dato e lascia la decisione all'utente, senza girarci troppo intorno."""


async def ask_ai(question: str) -> str:
    if not GROQ_API_KEY:
        return "Assistente AI non configurato — manca GROQ_API_KEY."

    try:
        context = await asyncio.to_thread(build_context_snapshot)
    except Exception as e:
        logger.error(f"Errore contesto: {e}")
        context = "(Contesto di mercato non disponibile)"

    question_note = _stated_position_note(question)

    # Turni recenti come veri messaggi user/assistant alternati — più
    # naturale per un modello di chat che infilare tutto come testo in un
    # unico messaggio, e il modello segue meglio il filo del discorso.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in _recent_conversation_turns():
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({
        "role": "user",
        "content": (
            f"CONTESTO AGGIORNATO — usa SEMPRE questi dati (prezzo, livelli, "
            f"stato del trade), non quelli citati nei tuoi messaggi precedenti "
            f"qui sopra: potrebbero essere cambiati nel frattempo.\n{context}\n\n"
            f"DOMANDA:\n{question}{question_note}"
        ),
    })

    def _request():
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.4,
                # Ridotto da 500: la regola "massimo 4-5 righe" nel prompt
                # non bastava da sola a tenere le risposte brevi — un limite
                # più stretto fa da argine anche quando il modello non la
                # rispetta.
                "max_tokens":  250,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    try:
        data   = await asyncio.to_thread(_request)
        answer = data["choices"][0]["message"]["content"].strip()
        _record_conversation_turn(question, answer)
        return answer
    except Exception as e:
        logger.error(f"Errore Groq: {e}")
        return f"Errore nel generare la risposta: {e}"
