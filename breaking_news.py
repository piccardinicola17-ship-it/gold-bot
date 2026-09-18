"""
breaking_news.py — Rilevamento notizie NON programmate (banche centrali, fiscali, geopolitica).

Le notizie SCHEDULATE (CPI, NFP, FOMC, ecc.) sono già coperte da
analyzer.get_upcoming_events() (calendario FairEconomy) e da
check_macro_alerts() in gold_bot.py. Questo modulo copre invece un buco
reale: un comunicato/discorso non programmato di una banca centrale, o una
notizia ad alto impatto imprevista — oggi il bot non se ne accorge finché
non si riflette già nel prezzo.

Origine: adattato dal progetto "XAU News Intelligence Master" (29 moduli)
condiviso dall'utente. La maggior parte di quel progetto richiede dati
storici per l'addestramento e feed a pagamento (Databento, OANDA) che non
esistono qui — vedi la valutazione fatta in chat. Questo file integra SOLO
la parte verificata realistica: fonti gratuite reali (testate dal vivo),
classificazione a parole chiave (non è "AI", è uno screening grezzo — va
trattato come tale, non come segnale definitivo).

Fonti usate (Fed verificata il 1 settembre 2026, BCE/BoJ/BoE aggiunte e
verificate dal vivo il 2026-09-18 su richiesta esplicita dell'utente —
"voglio essere aggiornato a 360 gradi"):
- Fed press releases RSS + Fed speeches RSS (ufficiali, gratis)
- BCE: un solo feed che copre già comunicati/discorsi/interviste insieme
  (rss/press.html — la description del feed stesso lo dichiara)
- BoJ: whatsnew.xml versione inglese (decisioni di politica monetaria,
  modifiche agli strumenti operativi)
- BoE: rss/news (comunicati, minute, trascrizioni) + rss/speeches separato

NON disponibile da nessuna di queste fonti ufficiali: la foto del
governatore/membro citato nello screenshot di esempio dell'utente. Quella
foto è aggiunta manualmente dal servizio aggregatore (es. Volatility
Now/@financialjuice) — nessun feed RSS ufficiale di banca centrale la
include (verificato: nessun tag <enclosure>/<media:content> in nessuna
delle fonti sopra). Per non rischiare di allegare la foto sbagliata,
questo modulo non ne invia nessuna.

Fonti scartate dopo verifica dal vivo (non per pigrizia):
- BLS ICS calendar (bls.gov) — risponde 403 anche con user-agent da browser
  vero, bloccata lato server. Il calendario BLS (date NFP/CPI) è comunque
  già coperto dal mirror FairEconomy usato altrove nel bot.
- Treasury press-releases (home.treasury.gov) — la pagina è renderizzata
  via JavaScript: l'HTML statico non contiene i comunicati, solo il menu
  di navigazione. Niente feed RSS alternativo trovato. Serirebbe un
  browser headless (Selenium/Playwright) per leggerla — dipendenza pesante
  e fragile su Railway, non ne vale la pena per questo scopo. I
  classificatori fiscale/geopolitico restano comunque disponibili qui
  sotto, pronti per essere applicati ad altre fonti testuali in futuro
  (es. titoli NewsAPI già scaricati altrove nel bot).
- BCE rss/speeches.html separato — 404, i discorsi sono già dentro
  rss/press.html (vedi sopra), niente da aggiungere.

Nessuna nuova dipendenza: XML delle RSS con xml.etree (stdlib) — niente
beautifulsoup4/feedparser.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime
from xml.etree import ElementTree

import requests

from news_analyst import _escape_md

logger = logging.getLogger(__name__)

FED_PRESS_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"
FED_SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"
ECB_PRESS_RSS = "https://www.ecb.europa.eu/rss/press.html"
BOJ_WHATSNEW_RSS = "https://www.boj.or.jp/en/rss/whatsnew.xml"
BOE_NEWS_RSS = "https://www.bankofengland.co.uk/rss/news"
BOE_SPEECHES_RSS = "https://www.bankofengland.co.uk/rss/speeches"

_HTTP_TIMEOUT = 10
_USER_AGENT = "Mozilla/5.0 (compatible; GoldMindBot/1.0)"

# ─────────────────────────────────────────────────────────────
# Classificatori a parole chiave — screening grezzo, non "AI".
# Punteggio positivo = hawkish/rischio-safe-haven UP per l'oro;
# negativo = dovish/de-escalation, tipicamente ribassista per l'oro.
#
# Vocabolario generico di politica monetaria in inglese (non specifico
# Fed): usato anche per BCE/BoJ/BoE, il cui linguaggio ufficiale in
# inglese ricorre alle stesse espressioni ("rate cuts", "restrictive
# stance", "inflation has eased", ...).
# ─────────────────────────────────────────────────────────────

HAWKISH_PHRASES = {
    "higher for longer": 1.0,
    "further tightening": 1.0,
    "additional rate increases": 0.9,
    "not appropriate to cut": 0.9,
    "upside risks to inflation": 0.8,
    "inflation remains elevated": 0.6,
    "restrictive stance": 0.5,
    "vigilant": 0.4,
}
DOVISH_PHRASES = {
    "appropriate to reduce": -1.0,
    "rate cuts": -0.7,
    "policy is restrictive": -0.4,
    "downside risks to employment": -0.8,
    "inflation has eased": -0.5,
    "labor market has cooled": -0.5,
    "further easing": -0.7,
}

FISCAL_SHOCK_KEYWORDS = (
    "debt ceiling", "government shutdown", "credit rating", "downgrade",
    "quarterly refunding", "buyback program", "auction tail", "bid-to-cover",
    "deficit", "emergency funding", "default risk", "debt limit",
)

GEO_CRITICAL = (
    "nuclear", "missile launch", "airstrike", "invasion", "strait of hormuz",
    "state of emergency", "military strike", "declares war", "attack on",
)
GEO_DEESCALATION = (
    "ceasefire agreed", "peace agreement", "truce signed", "sanctions lifted",
    "de-escalation", "withdraws troops",
)


def classify_monetary_policy_text(text: str) -> dict:
    """Screening hawkish/dovish grezzo a parole chiave. Non sostituisce una lettura umana."""
    t = (text or "").lower()
    hits = []
    score = 0.0
    for phrase, weight in {**HAWKISH_PHRASES, **DOVISH_PHRASES}.items():
        if phrase in t:
            score += weight
            hits.append(phrase)
    score = max(-1.0, min(1.0, score))
    label = "HAWKISH" if score > 0.15 else ("DOVISH" if score < -0.15 else "NEUTRO")
    return {
        "score": round(score, 2),
        "label": label,
        "matched": hits,
        "xau_bias": "SELL" if score > 0.15 else ("BUY" if score < -0.15 else "N/D"),
    }


def classify_fiscal_text(text: str) -> dict:
    t = (text or "").lower()
    hits = [k for k in FISCAL_SHOCK_KEYWORDS if k in t]
    return {"shock_detected": bool(hits), "matched": hits}


def classify_geopolitical_text(text: str) -> dict:
    t = (text or "").lower()
    crit = [k for k in GEO_CRITICAL if k in t]
    de = [k for k in GEO_DEESCALATION if k in t]
    if crit and not de:
        return {"risk_off": True, "xau_bias": "BUY", "severity": "ALTA", "matched": crit}
    if de and not crit:
        return {"risk_off": False, "xau_bias": "SELL", "severity": "media", "matched": de}
    return {"risk_off": False, "xau_bias": "N/D", "severity": "normale", "matched": []}


# ─────────────────────────────────────────────────────────────
# Fetch — RSS (stdlib xml.etree) e HTML Treasury (stdlib html.parser)
# ─────────────────────────────────────────────────────────────

def _fetch_rss(url: str) -> list[dict]:
    """Parsa un feed RSS 2.0 standard. Ritorna [{title, link, pub_date, summary}]."""
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        summary = (item.findtext("description") or "").strip()
        if not title:
            continue
        items.append({
            "title": html.unescape(title),
            "link": link,
            "pub_date": pub_date,
            "summary": html.unescape(re.sub("<[^>]+>", " ", summary)).strip(),
        })
    return items


def _item_id(source: str, item: dict) -> str:
    basis = f"{source}|{item.get('link') or item.get('title', '')}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


# ─────────────────────────────────────────────────────────────
# Entry point — chiamato dallo scheduler
# ─────────────────────────────────────────────────────────────

def check_breaking_news(seen_ids: dict) -> tuple[list[dict], dict]:
    """
    Controlla le fonti Fed (press releases + discorsi), filtra ciò che è già
    stato notificato (seen_ids), classifica ogni item nuovo. Ritorna
    (alert_da_mandare, seen_ids_aggiornato). Il chiamante è responsabile di
    persistere seen_ids (vedi gold_bot.py).

    seen_ids è un dict usato come "set ordinato" (chiave = id già visto,
    valore ignorato) — vedi trade_manager.load_breaking_news_seen per il
    perché: un set vero non preserva l'ordine di inserimento, e il tetto di
    sicurezza qui sotto ("tieni solo gli ultimi 300") deve poter contare
    sull'ordine per scartare davvero i più vecchi, non un sottoinsieme
    arbitrario (che potrebbe far sparire un id appena visto e reinviare la
    stessa breaking news).
    """
    alerts: list[dict] = []
    new_seen = dict.fromkeys(seen_ids)  # accetta anche un set (es. nei test)

    sources = [
        ("fed_press", FED_PRESS_RSS, _fetch_rss, classify_monetary_policy_text),
        ("fed_speech", FED_SPEECHES_RSS, _fetch_rss, classify_monetary_policy_text),
        ("ecb_press", ECB_PRESS_RSS, _fetch_rss, classify_monetary_policy_text),
        ("boj_press", BOJ_WHATSNEW_RSS, _fetch_rss, classify_monetary_policy_text),
        ("boe_news", BOE_NEWS_RSS, _fetch_rss, classify_monetary_policy_text),
        ("boe_speech", BOE_SPEECHES_RSS, _fetch_rss, classify_monetary_policy_text),
    ]

    for name, url, fetch_fn, classify_fn in sources:
        try:
            items = fetch_fn(url)
        except Exception as e:
            logger.debug(f"breaking_news: fetch fallito per {name}: {e}")
            continue

        # BUG REALE (2026-09-18): quando BCE/BoJ/BoE sono state aggiunte a
        # questa lista, is_first_run in gold_bot.py era già False (Fed
        # girava da mesi) — il loro storico recente (fino a 20 item ciascuna)
        # è stato quindi trattato come "breaking" invece che come baseline,
        # inviando 20 messaggi Telegram in sequenza al primo giro dopo il
        # deploy. is_first_run è un flag GLOBALE, non per-fonte: non
        # protegge una fonte aggiunta dopo che le altre hanno già uno
        # storico. Fix: baseline esplicita per singola fonte, marcata dentro
        # seen_ids stesso (stesso dict, nessuno storage nuovo) — la prima
        # volta che una fonte compare, i suoi item si salvano come visti ma
        # non generano alert.
        baseline_marker = f"__baseline_{name}__"
        source_is_new = baseline_marker not in seen_ids
        if source_is_new:
            new_seen[baseline_marker] = True

        for item in items[:20]:  # solo i più recenti, evita di rimasticare tutto lo storico
            item_id = _item_id(name, item)
            if item_id in seen_ids:
                continue
            new_seen[item_id] = True
            if source_is_new:
                continue

            text = f"{item.get('title','')} {item.get('summary','')}"
            classification = classify_fn(text)
            # FIX (audit 2026-09-04): classify_fiscal_text() era definito,
            # pesato e persino letto da format_breaking_alert() (c["shock_detected"])
            # ma non veniva mai chiamato qui — nessun alert su debt ceiling/
            # shutdown/downgrade/deficit poteva mai scattare, nonostante siano
            # esattamente il tipo di notizia non programmata che questo modulo
            # dovrebbe coprire (vedi docstring del file).
            fiscal = classify_fiscal_text(text)
            if fiscal["shock_detected"]:
                classification["shock_detected"] = True
                classification["matched"] = classification.get("matched", []) + fiscal["matched"]
            geo = classify_geopolitical_text(text)

            alerts.append({
                "source": name,
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "link": item.get("link", ""),
                "classification": classification,
                "geopolitical": geo if geo["risk_off"] or geo["xau_bias"] != "N/D" else None,
            })

    # Tetto di sicurezza: non lasciare crescere seen_ids all'infinito.
    if len(new_seen) > 500:
        new_seen = dict(list(new_seen.items())[-300:])

    return alerts, new_seen


def format_breaking_alert(alert: dict, current_price: float = 0, ai_analysis: str | None = None) -> str:
    source_label = {
        "fed_press": "🏛 FED — Comunicato",
        "fed_speech": "🎙 FED — Discorso",
        "ecb_press": "🏦 BCE — Comunicato/Discorso",
        "boj_press": "🗾 BoJ — Comunicato",
        "boe_news": "🏛 BoE — Comunicato",
        "boe_speech": "🎙 BoE — Discorso",
        "treasury": "💵 TREASURY — Comunicato",
    }.get(alert["source"], alert["source"])

    # Titolo grezzo da feed RSS esterno: va escapato come ogni altro titolo
    # in questo bot (vedi news_analyst._escape_md) — senza, un titolo con
    # _ * ` [ ] fa fallire il parsing Markdown di Telegram e il messaggio
    # (già inviato una volta sola, deduplicato per item_id) sparisce.
    lines = [f"🔴 *BREAKING — {source_label}*", f"_{_escape_md(alert['title'])}_"]

    price_txt = f"${current_price:,.2f}" if current_price > 0 else "N/D"
    lines.append(f"XAU/USD: *{price_txt}*")

    if ai_analysis:
        # Spiegazione breve (cos'è / di cosa parla / bias oro) — vedi
        # news_analyst.analyze_breaking_news(). Sostituisce la riga "Tono"
        # a parole chiave quando disponibile: più utile e sempre presente,
        # anche quando lo screening rileva NEUTRO (prima la riga spariva
        # del tutto e il messaggio restava senza alcun contesto).
        lines.append(ai_analysis)
    else:
        c = alert.get("classification", {})
        if "label" in c and c["label"] != "NEUTRO":
            lines.append(f"Tono: *{c['label']}* (bias oro: {c['xau_bias']}) — {', '.join(c['matched'][:3])}")
        if c.get("shock_detected"):
            lines.append(f"⚠️ Possibile shock fiscale — parole chiave: {', '.join(c['matched'][:3])}")

    geo = alert.get("geopolitical")
    if geo:
        lines.append(f"🌍 Rischio geopolitico: severità {geo['severity']}, bias oro {geo['xau_bias']}")

    if alert.get("link"):
        lines.append(alert["link"])

    lines.append(
        "_Screening automatico a parole chiave — non è un segnale di trading, "
        "verifica sempre la fonte prima di agire._"
    )
    return "\n".join(lines)
