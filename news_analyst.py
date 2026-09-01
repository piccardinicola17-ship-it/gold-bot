"""
news_analyst.py — Analisi notizie e calendario macro per GoldMind.
FIX: escape Markdown nelle headlines per evitare "Can't parse entities"
"""

import os
import logging
import requests
from datetime import datetime
import pytz

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "groq/compound-mini"

MACRO_DB = {
    "NFP": {"nome":"Non-Farm Payrolls","impatto":"MOLTO ALTO","logica":"NFP forte → dollaro su → oro giù. NFP debole → oro su.","soglia":"Sorpresa > ±50k","ora_tipica":"15:30 IT"},
    "CPI": {"nome":"Consumer Price Index","impatto":"MOLTO ALTO","logica":"CPI alto → Fed hawkish → oro giù. CPI basso → oro su.","soglia":"Sorpresa > ±0.2%","ora_tipica":"15:30 IT"},
    "FOMC": {"nome":"Federal Open Market Committee","impatto":"MOLTO ALTO","logica":"Rialzo tassi → oro giù. Taglio tassi → oro su.","soglia":"Qualsiasi decisione inattesa","ora_tipica":"21:00 IT"},
    "PPI": {"nome":"Producer Price Index","impatto":"ALTO","logica":"PPI alto → inflazione futura → oro su lungo termine.","soglia":"Sorpresa > ±0.3%","ora_tipica":"15:30 IT"},
    "GDP": {"nome":"Gross Domestic Product","impatto":"ALTO","logica":"GDP forte → dollaro su → oro giù.","soglia":"Sorpresa > ±0.5%","ora_tipica":"15:30 IT"},
    "ISM": {"nome":"ISM PMI","impatto":"MEDIO-ALTO","logica":"ISM > 50 espansione → dollaro su. ISM < 50 → oro su.","soglia":"Sorpresa > ±2 punti","ora_tipica":"17:00 IT"},
    "POWELL": {"nome":"Discorso Powell/Fed","impatto":"MOLTO ALTO","logica":"Hawkish → oro giù. Dovish → oro su.","soglia":"Qualsiasi cambiamento guidance","ora_tipica":"Variabile"},
    "PCE": {"nome":"Personal Consumption Expenditures","impatto":"ALTO","logica":"PCE alto → hawkish Fed → oro giù.","soglia":"Sorpresa > ±0.2%","ora_tipica":"15:30 IT"},
    "JOLTS": {"nome":"Job Openings","impatto":"MEDIO","logica":"Posti vacanti alti → Fed hawkish → oro giù.","soglia":"Sorpresa > ±200k","ora_tipica":"17:00 IT"},
    "RETAIL": {"nome":"Retail Sales","impatto":"MEDIO-ALTO","logica":"Vendite forti → dollaro su → oro giù.","soglia":"Sorpresa > ±0.5%","ora_tipica":"15:30 IT"},
}


def _find_macro_db_info(event_title: str) -> dict:
    title_upper = event_title.upper()
    for key, info in MACRO_DB.items():
        if key in title_upper:
            return info
    return {}


def _call_groq(system: str, user: str, max_tokens: int = 500) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY non configurata."
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "temperature":0.3, "max_tokens":max_tokens},
            timeout=25,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Errore Groq: {e}")
        return f"Analisi AI non disponibile: {e}"


def _escape_md(text: str) -> str:
    """
    Escapa i caratteri speciali Markdown v1 di Telegram nei testi raw
    (titoli di notizie, nomi evento) per evitare 'Can't parse entities'.
    NON usare su testo che contiene già Markdown legittimo (bold, italic).
    """
    for ch in ("_", "*", "`", "[", "]"):
        text = text.replace(ch, "\\" + ch)
    return text


def format_news_message(news: list, current_price: float = 0) -> str:
    """Messaggio notizie breve e diretto. Aggiorna il prezzo se stale."""
    if not news:
        return "Nessuna notizia disponibile al momento."

    # Aggiorna prezzo se zero/stale — usa fxratesapi sempre disponibile
    price = current_price
    if price <= 100:
        try:
            import requests as _req
            r = _req.get(
                "https://api.fxratesapi.com/latest?currencies=XAU&base=USD",
                timeout=6
            )
            d = r.json()
            if d.get("success") and d.get("rates", {}).get("XAU"):
                price = round(1.0 / float(d["rates"]["XAU"]), 2)
        except Exception:
            price = current_price

    price_txt = f"*${price:,.2f}*" if price > 0 else "*N/D*"

    # Analisi AI: solo 3 righe — bias, motivo, livello chiave
    news_plain = "\n".join(
        str(n).replace("*","").replace("_","").replace("`","")[:120]
        for n in news[:5]
    )
    analysis = _call_groq(
        system=(
            "Analista XAU/USD. Rispondi SOLO con:\n"
            "Bias: BULLISH / BEARISH / NEUTRALE\n"
            "Motivo: [max 10 parole]\n"
            "Livello: [supporto o resistenza principale in $]"
        ),
        user=f"Prezzo: ${price}\nNotizie:\n{news_plain}",
        max_tokens=60,
    )

    # Max 4 titoli, plain text, escape caratteri Markdown
    headlines = []
    for n in news[:4]:
        raw = str(n).replace("*","").replace("_"," ").replace("`","")
        first = raw.split("\n")[0].strip()[:90]
        headlines.append(f"• {first}")

    msg = (
        f"📰 *NEWS XAU/USD* — {price_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(headlines) +
        f"\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{analysis}"
    )
    return msg[:2000] if len(msg) > 2000 else msg


def analyze_macro_event(event_title: str, forecast: str = "N/A", previous: str = "N/A", actual: str = "N/A", current_price: float = 0) -> str:
    """
    Bias direzionale corto pre-evento — niente pip/livelli/TP/SL inventati.

    FIX: prima chiedeva a Groq un'analisi in 5 punti con "movimento tipico
    in pips" e "consiglio operativo" con entry/SL/TP specifici. L'LLM non ha
    alcun modello statistico dietro quei numeri — li genera plausibili ma
    senza fondamento, e infatti si sono rivelati sbagliati di un ordine di
    grandezza rispetto al movimento reale. Ora si chiede solo un bias
    direzionale (BUY/SELL/NEUTRO) con una riga di motivazione qualitativa,
    esplicitamente senza cifre precise.
    """
    db_info   = _find_macro_db_info(event_title)
    price_txt = f"${current_price}" if current_price > 0 else "N/D"
    context   = [f"Evento: {event_title}", f"Previsione: {forecast} | Precedente: {previous}"]
    if actual and actual not in ("N/A", "uscito — vedi notizie"):
        context.append(f"Uscito: {actual}")
    if db_info:
        context.append(f"Logica: {db_info.get('logica','')}")
    context.append(f"Prezzo XAU/USD: {price_txt}")
    analysis = _call_groq(
        system=(
            "Sei un analista macro XAU/USD. Dai solo un bias direzionale sintetico, "
            "MAI cifre precise (niente pip, niente livelli di prezzo, niente entry/SL/TP): "
            "non hai un modello statistico per generarle in modo affidabile e inventarle è "
            "fuorviante. Rispondi in italiano con ESATTAMENTE questo formato, 2 righe:\n"
            "Bias: BUY|SELL|NEUTRO\n"
            "Motivo: <una frase, massimo 20 parole, solo logica qualitativa>"
        ),
        user="\n".join(context),
        max_tokens=80,
    )
    return analysis


def analyze_breaking_news(source_label: str, title: str, summary: str = "",
                           xau_bias: str = "N/D", current_price: float = 0) -> str:
    """
    Spiegazione breve di un breaking alert Fed (comunicato o discorso) per chi
    non ha tempo di leggere la fonte: cos'è, di cosa parla, cosa implica per
    XAU/USD accanto al prezzo attuale. Stesso principio di analyze_macro_event:
    MAI cifre/pip/livelli inventati — solo lettura qualitativa, e onestà
    quando il contenuto non ha nulla a che fare con politica monetaria
    (es. discorsi Fed su temi non di mercato, come inclusione finanziaria).
    """
    price_txt = f"${current_price:,.2f}" if current_price > 0 else "N/D"
    context = [f"Tipo: {source_label}", f"Titolo: {title}"]
    if summary:
        context.append(f"Estratto: {summary[:400]}")
    if xau_bias and xau_bias != "N/D":
        context.append(f"Tono da screening a parole chiave: {xau_bias}")
    context.append(f"Prezzo XAU/USD attuale: {price_txt}")

    return _call_groq(
        system=(
            "Sei un analista che spiega in italiano, in modo brevissimo, una "
            "comunicazione ufficiale della Fed a un trader XAU/USD che non ha "
            "tempo di leggerla. Rispondi in ESATTAMENTE questo formato, 3 righe:\n"
            "Cos'è: <tipo di comunicazione in poche parole>\n"
            "Di cosa parla: <una frase, max 20 parole, il succo reale del contenuto>\n"
            "Per l'oro: BUY|SELL|NEUTRO — <motivo, max 15 parole, MAI cifre precise, "
            "pip o livelli di prezzo: se il contenuto non riguarda politica "
            "monetaria/inflazione/tassi, dillo onestamente e usa NEUTRO>"
        ),
        user="\n".join(context),
        max_tokens=140,
    )


def get_macro_briefing(events: list, current_price: float = 0) -> str:
    if not events:
        return "Nessun evento macro ad alto impatto oggi."
    price_txt  = f"XAU/USD: ${current_price}" if current_price > 0 else ""
    events_txt = "\n".join(
        f"- {ev.get('title','?')} alle {ev.get('time','?')} IT (prev: {ev.get('forecast','N/A')}, prec: {ev.get('previous','N/A')})"
        for ev in events[:5]
    )
    briefing = _call_groq(
        system="Sei un analista macro XAU/USD. Briefing mattutino per un trader. Italiano, max 6 righe, tono diretto.",
        user=f"{price_txt}\n\nEventi oggi:\n{events_txt}\n\nImpatto su XAU/USD per ogni evento e bias complessivo giornata.",
        max_tokens=400,
    )
    header = "*EVENTI MACRO OGGI*\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
    for ev in events[:5]:
        db_info = _find_macro_db_info(ev.get("title",""))
        impatto = db_info.get("impatto","MEDIO") if db_info else "MEDIO"
        # Escape il titolo evento (può contenere caratteri speciali)
        safe_title = _escape_md(ev.get("title","?"))
        header += f"\u2022 {safe_title} \u2014 {ev.get('time','?')} IT [{impatto}]\n  Prev: {ev.get('forecast','N/A')} | Prec: {ev.get('previous','N/A')}\n"
    result = header + f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\nAnalisi AI:\n{briefing}"
    return result[:4000] if len(result) > 4000 else result
