"""
news_analyst.py — Analisi notizie e calendario macro per GoldMind.
FIX: escape Markdown nelle headlines per evitare "Can't parse entities"
"""

import os
import logging
from datetime import datetime
import pytz

import groq_client

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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


def _find_macro_db_info(event_title: str, currency: str = "") -> dict:
    """
    MACRO_DB contiene solo logiche USD-specifiche (Fed, NFP, ecc.) — il
    match era una semplice substring sul titolo, senza controllare la
    valuta. Fino a quando gli alert erano filtrati a solo USD (vedi
    analyzer._passes_news_filter) andava bene per costruzione, ma dal
    2026-09-16 il bot include anche eventi di altre valute major (EUR,
    GBP, JPY, ecc.) — un titolo come "GBP CPI y/y" contiene comunque la
    substring "CPI" e avrebbe agganciato la logica "CPI alto -> Fed
    hawkish -> oro giù", sbagliata (quella è la Bank of England, non la
    Fed). currency vuota = comportamento invariato (retrocompatibile, i
    chiamanti esistenti non passano valuta); currency non-USD = nessun
    lookup, l'LLM ragiona senza l'indizio precompilato invece di riceverne
    uno sbagliato.
    """
    if currency and currency.upper() not in ("USD", "US"):
        return {}
    title_upper = event_title.upper()
    for key, info in MACRO_DB.items():
        if key in title_upper:
            return info
    return {}


def _call_groq(system: str, user: str, max_tokens: int = 500) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY non configurata."
    try:
        return groq_client.chat(
            GROQ_API_KEY,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.3, timeout=25,
        )
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

    # Max 4 titoli. FIX: la sanificazione manuale (.replace ripetuti)
    # copriva solo 3 caratteri su 5 e CANCELLAVA quelli speciali invece di
    # escaparli (perdendo pezzi reali del titolo, es. "Fed [Update]" ->
    # "Fed Update"). Usa _escape_md() come ovunque altrove in questo file —
    # stessi 5 caratteri gestiti, e il titolo resta fedele (Telegram mostra
    # il carattere escapato, non lo nasconde).
    #
    # FIX 2026-09-13: get_extended_news() produce ogni voce su DUE righe
    # ("fonte (data)\ntitolo") — prendere raw.split("\n")[0] mostrava SOLO
    # fonte+data e scartava il titolo vero, il contenuto informativo. Il
    # messaggio in produzione mostrava bullet come "Yahoo Entertainment
    # (2026-09-10)" senza alcun titolo, inutile per capire la notizia (e
    # scollegato dal Bias/Motivo sotto, che l'LLM deriva invece dal testo
    # completo). Ora si prende l'ULTIMA riga non vuota (il titolo, se
    # presente) — compatibile anche con un'eventuale voce a riga singola.
    headlines = []
    for n in news[:4]:
        lines = [l.strip() for l in str(n).split("\n") if l.strip()]
        text = _escape_md(lines[-1])[:90] if lines else ""
        headlines.append(f"• {text}")

    msg = (
        f"📰 *NEWS XAU/USD* — {price_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(headlines) +
        f"\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{analysis}"
    )
    return msg[:2000] if len(msg) > 2000 else msg


def get_bias_briefing(news: list, current_price: float = 0) -> str:
    """Bias giornaliero sintetico (2 righe: Bias + Motivo) per il report
    mattutino unico — niente titoli di notizie, niente livello di prezzo
    (l'utente non li vuole nel report combinato, vedi format_news_message
    per la versione con notizie e livello usata altrove)."""
    if not news:
        return "Bias: NEUTRALE\nMotivo: notizie non disponibili."

    news_plain = "\n".join(
        str(n).replace("*","").replace("_","").replace("`","")[:120]
        for n in news[:5]
    )
    return _call_groq(
        system=(
            "Analista XAU/USD. Rispondi SOLO con:\n"
            "Bias: BULLISH / BEARISH / NEUTRALE\n"
            "Motivo: [max 10 parole]"
        ),
        user=f"Prezzo: ${current_price}\nNotizie:\n{news_plain}",
        max_tokens=40,
    )


def analyze_macro_event(event_title: str, forecast: str = "N/A", previous: str = "N/A", actual: str = "N/A", current_price: float = 0, currency: str = "", related_context: str = "") -> str:
    """
    Bias direzionale corto pre-evento — niente pip/livelli/TP/SL inventati.

    FIX: prima chiedeva a Groq un'analisi in 5 punti con "movimento tipico
    in pips" e "consiglio operativo" con entry/SL/TP specifici. L'LLM non ha
    alcun modello statistico dietro quei numeri — li genera plausibili ma
    senza fondamento, e infatti si sono rivelati sbagliati di un ordine di
    grandezza rispetto al movimento reale. Ora si chiede solo un bias
    direzionale (BUY/SELL/NEUTRO) con una riga di motivazione qualitativa,
    esplicitamente senza cifre precise.

    currency: valuta dell'evento (USD/EUR/GBP/JPY/...) — dal 2026-09-16 il
    bot copre anche eventi non-USD (vedi analyzer._passes_news_filter),
    quindi va detto esplicitamente all'LLM di quale banca centrale/valuta
    si tratta invece di lasciargli assumere che sia sempre la Fed/USD
    (l'evento non lo indica da solo: il titolo grezzo del calendario è
    "CPI y/y", non "GBP CPI y/y").

    related_context: bug reale trovato il 2026-09-16 — la FOMC Press
    Conference delle 20:30 ha ricevuto bias NEUTRO con motivazione "in
    attesa della decisione della Fed", nonostante Federal Funds Rate/FOMC
    Statement fossero già usciti 30 minuti prima (stessa riunione, stessa
    valuta, stesso giorno) con bias SELL confermato. L'LLM non aveva alcun
    modo di saperlo: ogni gruppo (raggruppato per data+ora+valuta) viene
    analizzato in isolamento, senza memoria di eventi correlati usciti
    poco prima. Questo parametro (costruito da
    gold_bot._find_related_macro_context) porta quel contesto quando
    esiste, invece di lasciare che l'LLM ragioni alla cieca su una
    conferenza stampa che segue un annuncio già noto.
    """
    db_info   = _find_macro_db_info(event_title, currency)
    price_txt = f"${current_price}" if current_price > 0 else "N/D"
    context   = [f"Evento: {event_title}"]
    if currency and currency.upper() not in ("USD", "US"):
        context.append(f"Valuta: {currency.upper()} (non USD — ragiona sulla banca centrale/economia di questa valuta, non sulla Fed, e su come si riflette sul dollaro/DXY e quindi sull'oro)")
    context.append(f"Previsione: {forecast} | Precedente: {previous}")
    if actual and actual not in ("N/A", "uscito — vedi notizie"):
        context.append(f"Uscito: {actual}")
    if db_info:
        context.append(f"Logica: {db_info.get('logica','')}")
    if related_context:
        context.append(related_context)
    context.append(f"Prezzo XAU/USD: {price_txt}")
    analysis = _call_groq(
        system=(
            "Sei un analista macro XAU/USD. Dai solo un bias direzionale sintetico, "
            "MAI cifre precise (niente pip, niente livelli di prezzo, niente entry/SL/TP): "
            "non hai un modello statistico per generarle in modo affidabile e inventarle è "
            "fuorviante. Se è presente una riga 'Contesto correlato', questo evento è la "
            "continuazione/spiegazione di una decisione GIÀ NOTA (es. una conferenza stampa "
            "dopo l'annuncio scritto della stessa riunione) — non trattarlo come un esito "
            "ancora incerto, ragiona a partire da quella decisione già uscita. Rispondi in "
            "italiano con ESATTAMENTE questo formato, 2 righe:\n"
            "Bias: BUY|SELL|NEUTRO\n"
            "Motivo: <una frase, massimo 20 parole, solo logica qualitativa>"
        ),
        user="\n".join(context),
        max_tokens=80,
    )
    return analysis


def analyze_combined_macro_event(events: list, current_price: float = 0, related_context: str = "") -> str:
    """
    Bias UNICO per più indicatori macro che escono ALLA STESSA ORA (es. CPI
    m/m + CPI y/y + Core CPI m/m + Core CPI y/y, tutti alle 14:30 — stesso
    rilascio, quattro angolazioni dello stesso dato). Prima, gold_bot.py
    chiamava analyze_macro_event() una volta per titolo: con temperature>0
    ogni chiamata è indipendente e può produrre bias diversi o persino
    opposti per lo stesso identico momento (bug reale in produzione,
    screenshot utente 2026-09-11 — 4 alert consecutivi con bias NEUTRO/BUY/
    SELL/SELL tutti per le 14:30). Ora una SOLA chiamata ragiona su tutti
    gli indicatori insieme e produce un solo bias coerente.

    events: lista di dict con almeno "title", "forecast", "previous",
    "currency" — il chiamante (gold_bot.check_macro_alerts) raggruppa per
    data+ora+valuta, quindi tutti gli eventi qui dentro condividono la
    stessa valuta per costruzione (un evento USD e uno GBP alla stessa ora
    non sono lo stesso rilascio, non vanno mai nello stesso gruppo).
    Con un solo elemento delega a analyze_macro_event() — stesso output di
    prima, nessun cambio di comportamento nel caso comune (1 evento).
    """
    if len(events) == 1:
        ev = events[0]
        return analyze_macro_event(
            ev["title"], ev.get("forecast", "N/A"), ev.get("previous", "N/A"), "N/A", current_price,
            currency=ev.get("currency", ""), related_context=related_context,
        )

    currency  = events[0].get("currency", "")
    price_txt = f"${current_price}" if current_price > 0 else "N/D"
    righe = [
        f"- {e['title']}: Previsione {e.get('forecast','N/A')} | Precedente {e.get('previous','N/A')}"
        for e in events
    ]
    context_lines = [f"Prezzo XAU/USD: {price_txt}"]
    if currency and currency.upper() not in ("USD", "US"):
        context_lines.append(f"Valuta: {currency.upper()} (non USD — ragiona sulla banca centrale/economia di questa valuta e su come si riflette sul dollaro/DXY e quindi sull'oro)")
    if related_context:
        context_lines.append(related_context)
    context_lines.append("Indicatori in uscita insieme (stesso orario, stesso rilascio):")
    context = "\n".join(context_lines) + "\n" + "\n".join(righe)
    return _call_groq(
        system=(
            "Sei un analista macro XAU/USD. Più indicatori escono ALLA STESSA ORA, fanno parte dello "
            "stesso rilascio (es. dato mensile+annuale, headline+core dello stesso report) — dai UN "
            "SOLO bias direzionale complessivo che li consideri TUTTI insieme, mai un bias per "
            "indicatore preso isolatamente. MAI cifre precise (niente pip, livelli, entry/SL/TP). "
            "Se è presente una riga 'Contesto correlato', questi indicatori sono la continuazione/"
            "spiegazione di una decisione GIÀ NOTA — non trattarli come un esito ancora incerto. "
            "Rispondi in italiano con ESATTAMENTE questo formato, 2 righe:\n"
            "Bias: BUY|SELL|NEUTRO\n"
            "Motivo: <una frase, massimo 25 parole, che spieghi il ragionamento complessivo>"
        ),
        user=context,
        max_tokens=90,
    )


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
    # L'RSS dei discorsi Fed quasi mai include un estratto reale del testo,
    # solo il titolo (spesso generico, es. "The Economic Outlook and Some
    # Comments on My Policy Communication" — zero contenuto hawkish/dovish
    # leggibile). Senza questo segnale esplicito l'LLM, forzato a compilare
    # comunque le 3 righe, inventava un "Per l'oro: BUY/SELL" plausibile ma
    # non fondato su nulla di reale — bug osservato in produzione il
    # 2026-09-03 (discorso Waller, bias SELL dato dal solo titolo mentre
    # l'oro saliva). Ora l'assenza di contenuto è dichiarata esplicitamente,
    # non semplicemente omessa.
    has_content = bool(summary and len(summary.strip()) >= 40)
    if has_content:
        context.append(f"Estratto: {summary[:400]}")
    else:
        context.append(
            "Estratto: NON DISPONIBILE — hai SOLO il titolo, nessun testo del "
            "discorso/comunicato. Non indovinare il contenuto dal titolo."
        )
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
            "monetaria/inflazione/tassi, dillo onestamente e usa NEUTRO>\n"
            "REGOLA FONDAMENTALE: se l'estratto è marcato NON DISPONIBILE, NON "
            "hai contenuto reale su cui basarti — un titolo da solo non dice "
            "nulla sul tono del discorso. In quel caso rispondi 'Di cosa parla: "
            "contenuto non disponibile, solo titolo' e 'Per l'oro: NEUTRO — dati "
            "insufficienti, leggi la fonte'. Non inventare un bias plausibile "
            "dal solo titolo."
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
        currency = ev.get("currency", "")
        db_info = _find_macro_db_info(ev.get("title",""), currency)
        impatto = db_info.get("impatto","MEDIO") if db_info else "MEDIO"
        # Escape il titolo evento (può contenere caratteri speciali)
        safe_title = _escape_md(ev.get("title","?"))
        cur_tag = f" [{currency}]" if currency and currency.upper() not in ("USD", "US") else ""
        header += f"\u2022 {safe_title}{cur_tag} \u2014 {ev.get('time','?')} IT [{impatto}]\n  Prev: {ev.get('forecast','N/A')} | Prec: {ev.get('previous','N/A')}\n"
    result = header + f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\nAnalisi AI:\n{briefing}"
    return result[:4000] if len(result) > 4000 else result


def get_weekly_smc_narrative(ctx: dict, current_price: float) -> str:
    """
    Spiegazione discorsiva di cosa potrebbe fare il prezzo (rimbalzo su una
    zona, liquidazione di un livello, imbalance da colmare, possibile swing
    e in che direzione) per l'analisi weekend.

    Stesso principio di analyze_macro_event: all'LLM vengono dati SOLO fatti
    gi\u00e0 calcolati da analyzer.py (struttura BOS/CHoCH, order block, fair
    value gap, liquidit\u00e0 EQH/EQL, zona premium/discount, regime) \u2014 il suo
    compito \u00e8 interpretarli in prosa, MAI inventare nuovi prezzi o livelli
    che non gli sono stati passati nel contesto. `ctx` \u00e8 il dict prodotto da
    weekly_chart.compute_smc_context(), calcolato sulla stessa serie 4h
    usata per il grafico e le "zone chiave" nel testo, quindi non pu\u00f2 mai
    raccontare una storia diversa dai numeri gi\u00e0 mostrati.
    """
    structure  = ctx["structure"]
    ob         = ctx["order_blocks"]
    fvg        = ctx["fvg"]
    liq        = ctx["liquidity"]
    mitigation = ctx["mitigation"]

    lines = [
        f"Prezzo attuale: {current_price:,.2f}",
        f"Struttura di mercato (4H): {structure.get('structure', 'NEUTRAL')}"
        + (f" | BOS: {structure['bos']}" if structure.get("bos") else "")
        + (f" | CHoCH: {structure['choch']}" if structure.get("choch") else ""),
        f"Ultimo swing high: {structure.get('last_high', 'N/D')} (precedente: {structure.get('prev_high', 'N/D')})",
        f"Ultimo swing low: {structure.get('last_low', 'N/D')} (precedente: {structure.get('prev_low', 'N/D')})",
        f"Zona Premium/Discount: {ctx['premium_discount']}",
        f"Regime: {ctx['regime']} (ADX {ctx['adx']})",
    ]
    if ob.get("bullish_ob"):
        tag = "gi\u00e0 mitigato" if mitigation.get("bullish_mit") else "non ancora mitigato"
        lines.append(f"Order Block rialzista ({tag}): {ob['bullish_ob']['low']}-{ob['bullish_ob']['high']}")
    if ob.get("bearish_ob"):
        tag = "gi\u00e0 mitigato" if mitigation.get("bearish_mit") else "non ancora mitigato"
        lines.append(f"Order Block ribassista ({tag}): {ob['bearish_ob']['low']}-{ob['bearish_ob']['high']}")
    if fvg.get("bullish_fvg"):
        lines.append(f"Fair Value Gap rialzista da colmare: {fvg['bullish_fvg']['bottom']}-{fvg['bullish_fvg']['top']}")
    if fvg.get("bearish_fvg"):
        lines.append(f"Fair Value Gap ribassista da colmare: {fvg['bearish_fvg']['bottom']}-{fvg['bearish_fvg']['top']}")
    if liq.get("eqh"):
        lines.append(f"Liquidit\u00e0 sopra il prezzo (Equal Highs): {liq['eqh']}")
    if liq.get("eql"):
        lines.append(f"Liquidit\u00e0 sotto il prezzo (Equal Lows): {liq['eql']}")

    return _call_groq(
        system=(
            "Sei un analista tecnico Smart Money Concepts su XAU/USD. Ricevi SOLO fatti gi\u00e0 "
            "calcolati (struttura BOS/CHoCH, order block, fair value gap, liquidit\u00e0 EQH/EQL, zona "
            "premium/discount, regime/ADX) \u2014 il tuo compito \u00e8 SOLO spiegarli in un paragrafo "
            "discorsivo e chiaro per un trader retail: cosa potrebbe fare il prezzo (rimbalzare su "
            "una zona, andare a liquidare un livello, colmare un imbalance/gap, partire in un "
            "movimento pi\u00f9 ampio e in quale direzione), citando SEMPRE e SOLO i livelli che ti "
            "vengono dati nel contesto. VIETATO inventare prezzi, pip, percentuali, probabilit\u00e0 o "
            "livelli non presenti nel contesto. Se un elemento non \u00e8 nel contesto (es. nessun FVG "
            "rilevato) non parlarne affatto, non inventarlo. Rispondi in italiano, un paragrafo "
            "unico discorsivo di massimo 130 parole, senza elenchi puntati, tono diretto e concreto "
            "come se stessi spiegando la situazione a voce a un trader."
        ),
        user="\n".join(lines),
        max_tokens=260,
    )
