"""
macro_predictor.py — Fase 5 del progetto dati storici: previsione statistica
live per gli eventi macro validati nella Fase 4 e con una fonte dati live
gratuita affidabile.

Due livelli di affidabilità della fonte ACTUAL, dichiarati esplicitamente
per ciascun evento (mai mescolati in silenzio):

1. FRED (Federal Reserve Bank di St. Louis) — governativa, gratuita per
   sempre, nessuna API key, aggiorna in modo affidabile. Usata per "Core
   CPI m/m", "Unemployment Claims", "ADP Non-Farm Employment Change".
   Financial Modeling Prep (candidato iniziale) si è rivelato a pagamento
   per l'endpoint economic-calendar nonostante la documentazione lasciasse
   intendere il contrario (verificato con una chiave reale).

2. RSS comunicati stampa (PR Newswire) — per le 3 serie validate in Fase 4
   con edge forte (ISM Manufacturing PMI n=196 5/5 orizzonti, ISM Services
   PMI n=163, CB Consumer Confidence n=180) ma SENZA alcuna serie FRED
   nativa (verificato il 2026-09-08: ISM rimossa da FRED nel 2016 - dati
   diventati proprietari - zero risultati per "ISM" su una ricerca FRED
   diretta; Conference Board idem, i soli risultati per "consumer
   confidence conference board" su FRED sono l'indice DIVERSO di
   Michigan e falsi positivi "CB Insights Metaverse"). Fonti alternative
   scartate dopo verifica dal vivo, non per pigrizia: il sito ismworld.org
   blocca il fetch automatico (redirect loop / 400, probabile protezione
   anti-bot); DBnomics espone "ISM/pmi" ma con valori implausibili
   (10-11 invece di ~50, probabile sotto-indice o errore di mapping, non
   verificabile con fiducia). ISM e Conference Board distribuiscono
   entrambi i comunicati anche via PR Newswire (verificato dal vivo,
   titoli identici e prevedibili su 9 mesi consecutivi 2026 per ISM:
   "Manufacturing PMI® at X%; <mese> <anno> ISM® Manufacturing PMI®
   Report" / "Services PMI® at X%; ..."), il cui feed RSS per categoria è
   pubblico, gratuito, senza chiave (vedi RSS_SERIES sotto). Rischio
   REALE e diverso da FRED: si rompe silenziosamente se PR Newswire
   cambia formato feed o se ISM/Conference Board cambiano il testo del
   titolo — per questo ogni estrazione fallita ritorna None invece di
   indovinare, esattamente come per un ritardo FRED.

Il modello (pendenza Theil-Sen + statistiche storiche della sorpresa) è
pre-calcolato offline da historical_model.fit_final_model() e salvato in
macro_models.json — questo modulo si limita a caricarlo e applicarlo, mai
a ri-addestrarlo.

Onestà sui limiti: FRED aggiorna in genere entro alcune ore dal rilascio
ufficiale, non sempre entro i 10 minuti in cui check_macro_alerts controlla
la finestra post-evento — se il dato non è ancora disponibile, la funzione
ritorna None e il bot lo dice esplicitamente, non inventa nulla. Stesso
principio per l'RSS: se il titolo non combacia col pattern atteso, None.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from breaking_news import _fetch_rss
from historical_events import _parse_number
from news_analyst import _escape_md
from trade_manager import (
    load_fred_last_seen, save_fred_last_seen,
    load_rss_macro_last_seen, save_rss_macro_last_seen,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "macro_models.json"

# Mappa evento (come appare nel calendario FairEconomy/nel dataset storico
# di Fase 1) -> serie FRED da cui leggere l'actual una volta rilasciato, e
# come interpretarne il valore:
# - "mom_pct": la serie FRED è un indice-livello, il modello è stato
#   allenato sulla variazione % mese su mese (es. Core CPI m/m).
# - "level": la serie FRED è già nella stessa unità del dataset di
#   training — un conteggio grezzo, MAI una % (es. Unemployment Claims,
#   "219K" nel calendario = 219000 sia in FRED sia nel dataset storico).
# - "mom_diff": la serie FRED è un livello, ma il dataset di training è la
#   DIFFERENZA assoluta mese su mese (non %) nella stessa unità grezza
#   (es. ADP Non-Farm Employment Change, "155K" = 155000 persone in più
#   rispetto al mese precedente, non il 155000° livello).
#   Confondere questi tre tipi darebbe una sorpresa completamente diversa
#   da quella su cui il modello è stato calibrato.
FRED_SERIES = {
    "Core CPI m/m":                    {"series_id": "CPILFESL",     "value_type": "mom_pct"},
    "Unemployment Claims":             {"series_id": "ICSA",         "value_type": "level"},
    "ADP Non-Farm Employment Change":  {"series_id": "ADPMNUSNERSA", "value_type": "mom_diff"},
    # Aggiunte 2026-09-14: le 2 serie (su 3 mai trovate in nessuna fonte,
    # vedi historical_events.py) validate con edge genuino dopo lo sblocco
    # dati Kaggle — vedi historical_model.DEPLOYED_EVENTS. "Core CPI y/y"
    # legge la STESSA serie livello di "Core CPI m/m" (confronto sui 12
    # mesi invece che sul mese precedente, vedi value_type "yoy_pct") — il
    # terzo elemento del calendario CPI oggi coperto (m/m, y/y, entrambi
    # core). PPI y/y usa PPIFIS ("Producer Price Index by Final Demand",
    # la definizione moderna BLS usata dal 2009 in poi - non PPIACO, serie
    # storica diversa/discontinuata per questo scopo, verificato
    # confrontando i valori ricalcolati contro gli actual storici reali).
    "Core CPI y/y":                    {"series_id": "CPILFESL",     "value_type": "yoy_pct"},
    "PPI y/y":                         {"series_id": "PPIFIS",       "value_type": "yoy_pct"},
}

# Serie senza fonte FRED, lette dai comunicati stampa via RSS (vedi nota
# in cima al file). "feeds": quali categorie PR Newswire interrogare (più
# di una per sicurezza, non è garantito che ISM/Conference Board usino
# sempre la stessa categoria). "title_pattern": deve trovare il valore
# SOLO nel titolo (affidabile: verificato identico su 9 mesi consecutivi).
# "body_pattern": serve quando il valore non è nel titolo (CB Consumer
# Confidence varia: "US Consumer Confidence Edged Down..." senza numero) -
# cerca nel testo unito titolo+riassunto RSS, meno affidabile perché il
# riassunto RSS a volte è troppo corto per contenere la frase col numero;
# se non lo trova ritorna None invece di indovinare (mai un fallback sul
# titolo per questi).
_PRNEWSWIRE_FEEDS = {
    "manufacturing": "https://www.prnewswire.com/rss/heavy-industry-manufacturing-latest-news/heavy-industry-manufacturing-latest-news-list.rss",
    "general":       "https://www.prnewswire.com/rss/general-business-latest-news/general-business-latest-news-list.rss",
}

RSS_SERIES = {
    "ISM Manufacturing PMI": {
        "feeds": ("manufacturing", "general"),
        "title_pattern": re.compile(r"Manufacturing PMI.{0,3}\s+at\s+([\d.]+)%.*\bISM\b.*Manufacturing PMI.{0,3}\s+Report", re.IGNORECASE),
        "value_type": "index",
    },
    "ISM Services PMI": {
        "feeds": ("manufacturing", "general"),
        "title_pattern": re.compile(r"Services PMI.{0,3}\s+at\s+([\d.]+)%.*\bISM\b.*Services PMI.{0,3}\s+Report", re.IGNORECASE),
        "value_type": "index",
    },
    "CB Consumer Confidence": {
        "feeds": ("general",),
        "title_pattern": None,
        "body_pattern": re.compile(r"Consumer Confidence Index.{0,40}?\bto\s+([\d.]+)\s*\(1985", re.IGNORECASE),
        "title_must_contain": ("consumer confidence",),
        "value_type": "index",
    },
}


def _load_model(event_name: str) -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        models = json.loads(MODEL_PATH.read_text())
    except Exception as e:
        logger.warning(f"macro_models.json illeggibile: {e}")
        return None
    return models.get(event_name)


_FRED_SERIES_CACHE: dict[str, tuple[float, pd.Series]] = {}
# TTL più corto del ciclo dello scheduler (5 min, vedi check_macro_alerts in
# gold_bot.py): serve solo a deduplicare più chiamate per la STESSA serie
# nello stesso giro (es. più eventi pending che condividono una serie FRED),
# non a ritardare la rilevazione di un nuovo dato tra un giro e l'altro —
# a quel punto la cache è già scaduta comunque, stesso comportamento di
# prima. Senza questo, un evento macro monitorato per fino a 3 ore
# (-180<=mins_away<=0) ri-scaricava l'intera serie storica ogni 5 minuti,
# 35 volte su 36 solo per scoprire "nessun dato nuovo".
_FRED_CACHE_TTL_SECONDS = 240


def _fetch_fred_series(series_id: str) -> pd.Series:
    cached = _FRED_SERIES_CACHE.get(series_id)
    if cached and (time.time() - cached[0]) < _FRED_CACHE_TTL_SECONDS:
        return cached[1]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.dropna().set_index("date")["value"].sort_index()
    _FRED_SERIES_CACHE[series_id] = (time.time(), series)
    return series


def _new_release_mom_pct(series_id: str, last_seen: dict, seen_key: str | None = None) -> tuple[str, float] | None:
    """
    Ritorna (data_iso, variazione_percentuale_mese_su_mese) se su FRED è
    comparso un punto dati più recente dell'ultimo già processato per
    questa serie, altrimenti None (nessuna novità, o FRED non ancora
    aggiornato).

    seen_key: chiave con cui tracciare "ultimo visto" in last_seen — di
    default series_id, ma va passato ESPLICITO (l'event_name) quando due
    eventi diversi condividono la stessa serie FRED sottostante (es. "Core
    CPI m/m" e "Core CPI y/y" leggono entrambi CPILFESL) — altrimenti il
    primo dei due che gira segna la serie come "vista" e l'altro non
    scatterebbe mai più per lo stesso rilascio. Bug evitato qui prima che
    esistesse, introducendo Core CPI y/y (2026-09-14).
    """
    seen_key = seen_key or series_id
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(seen_key)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    pos = s.index.get_loc(latest_date)
    if pos == 0:
        return None
    previous_value = s.iloc[pos - 1]
    current_value = s.iloc[pos]
    if pd.isna(previous_value) or previous_value == 0:
        return None

    # Arrotondato a 1 decimale come il BLS/calendario (0.2%, 0.3%...), non 3
    # decimali: dare più precisione di quella che la fonte può davvero
    # garantire è falsa precisione. Verificato empiricamente il 2026-09-03
    # confrontando 10 mesi FRED-ricalcolati con l'actual ufficiale salvato
    # nel dataset di training (stessa fonte/metodo usato per allenare il
    # modello): 8/10 coincidono, ma 2/10 hanno uno scarto reale di 0.1pp
    # (es. FRED ricalcola 0.255% → arrotonda a 0.3%, l'ufficiale era 0.2%) —
    # dovuto a revisioni dell'indice FRED successive alla pubblicazione
    # originale. Nessun fix pulito possibile senza cambiare fonte (servirebbe
    # ALFRED, dati vintage point-in-time, non ancora integrato): il margine
    # d'errore va solo dichiarato onestamente, vedi format_prediction().
    mom_pct = round((current_value - previous_value) / previous_value * 100, 1)
    return latest_date.isoformat(), float(mom_pct)


def _new_release_yoy_pct(series_id: str, last_seen: dict, seen_key: str | None = None) -> tuple[str, float] | None:
    """
    Come _new_release_mom_pct ma variazione percentuale ANNO su anno
    (confronto col valore di 12 mesi prima, non del mese precedente) —
    es. Core CPI y/y e PPI y/y leggono lo stesso indice-livello FRED già
    usato per la variante m/m (CPILFESL per Core CPI), ma il training si
    aspetta il confronto sui 12 mesi.

    Scarto vs actual ufficiale più ampio del mom_pct (verificato
    empiricamente il 2026-09-14 su 6 mesi 2024 di PPI y/y: fino a ~0.5pp,
    contro ~0.1pp di mom_pct) — atteso: un confronto a 12 mesi accumula le
    revisioni di TUTTI gli 11 mesi intermedi, non solo dell'ultimo. Vedi
    disclaimer dedicato in format_prediction().
    """
    seen_key = seen_key or series_id
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(seen_key)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    pos = s.index.get_loc(latest_date)
    if pos < 12:
        return None
    year_ago_value = s.iloc[pos - 12]
    current_value = s.iloc[pos]
    if pd.isna(year_ago_value) or year_ago_value == 0:
        return None

    yoy_pct = round((current_value - year_ago_value) / year_ago_value * 100, 1)
    return latest_date.isoformat(), float(yoy_pct)


def _new_release_mom_diff(series_id: str, last_seen: dict, seen_key: str | None = None) -> tuple[str, float] | None:
    """
    Come _new_release_mom_pct ma per serie dove il dataset di training è
    la DIFFERENZA ASSOLUTA mese su mese (non %) - es. ADP Non-Farm
    Employment Change: FRED (ADPMNUSNERSA) pubblica il LIVELLO totale
    degli occupati, il calendario/training riportano invece quante
    persone in più/meno rispetto al mese precedente ("155K").
    """
    seen_key = seen_key or series_id
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(seen_key)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    pos = s.index.get_loc(latest_date)
    if pos == 0:
        return None
    previous_value = s.iloc[pos - 1]
    current_value = s.iloc[pos]
    if pd.isna(previous_value) or pd.isna(current_value):
        return None

    # Arrotondato al migliaio, come il comunicato ADP ("155K", non
    # "154823") - stessa logica di falsa precisione evitata per mom_pct.
    diff = round((current_value - previous_value) / 1000) * 1000
    return latest_date.isoformat(), float(diff)


def _new_release_level(series_id: str, last_seen: dict, seen_key: str | None = None) -> tuple[str, float] | None:
    """
    Come _new_release_mom_pct ma per serie il cui valore è già un livello
    grezzo nel dataset di training (es. Unemployment Claims, "219K") —
    nessuna trasformazione: solo l'ultimo valore pubblicato su FRED, così
    com'è, comparabile direttamente al forecast (stessa unità).
    """
    seen_key = seen_key or series_id
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(seen_key)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    current_value = s.iloc[-1]
    if pd.isna(current_value):
        return None
    return latest_date.isoformat(), float(current_value)


def _new_release_rss(event_name: str, rss_cfg: dict, last_seen: dict) -> tuple[str, float] | None:
    """
    Cerca nei feed RSS PR Newswire configurati (vedi RSS_SERIES) il
    comunicato più recente non ancora processato per questo evento.
    Ritorna (pub_date_iso, valore) o None — mai un valore indovinato: se
    il titolo/riassunto non combacia col pattern atteso, è come se FRED
    non avesse ancora pubblicato.
    """
    title_pattern = rss_cfg.get("title_pattern")
    body_pattern = rss_cfg.get("body_pattern")
    must_contain = rss_cfg.get("title_must_contain", ())
    prev_seen = last_seen.get(event_name)

    candidates = []
    for feed_key in rss_cfg["feeds"]:
        url = _PRNEWSWIRE_FEEDS[feed_key]
        try:
            items = _fetch_rss(url)
        except Exception as e:
            logger.warning(f"[{event_name}] RSS non raggiungibile ({feed_key}): {e}")
            continue
        for item in items:
            title = item.get("title", "")
            if must_contain and not any(kw.lower() in title.lower() for kw in must_contain):
                continue
            value = None
            if title_pattern:
                m = title_pattern.search(title)
                if m:
                    value = float(m.group(1))
            if value is None and body_pattern:
                m = body_pattern.search(f"{title} {item.get('summary', '')}")
                if m:
                    value = float(m.group(1))
            if value is None:
                continue
            candidates.append((item.get("pub_date", ""), value, item.get("link", "")))

    if not candidates:
        return None

    # Il più recente per pub_date RSS (formato RFC 822, es. "Mon, 08 Sep
    # 2026 19:03:00 GMT") - pandas lo parsa senza bisogno di un formato
    # esplicito. Se non è nuovo rispetto all'ultimo già processato, None.
    parsed = [(pd.to_datetime(pd_, utc=True, errors="coerce"), v, link) for pd_, v, link in candidates]
    parsed = [p for p in parsed if p[0] is not None and not pd.isna(p[0])]
    if not parsed:
        return None
    parsed.sort(key=lambda p: p[0])
    latest_dt, latest_value, latest_link = parsed[-1]

    if prev_seen and latest_dt <= pd.Timestamp(prev_seen):
        return None

    logger.info(f"[{event_name}] Trovato via RSS: {latest_value} ({latest_link})")
    return latest_dt.isoformat(), latest_value


def _fetch_new_actual(event_name: str) -> tuple[str, float, str] | None:
    """
    Cerca un actual nuovo per event_name, provando prima FRED (fonte
    affidabile) poi RSS PR Newswire (fonte più fragile, solo per le serie
    senza equivalente FRED - vedi note in cima al file). Ritorna
    (release_date_iso, actual_value, source_tier) o None.
    """
    fred_cfg = FRED_SERIES.get(event_name)
    if fred_cfg:
        series_id = fred_cfg["series_id"]
        value_type = fred_cfg["value_type"]
        # seen_key=event_name, MAI series_id: da quando Core CPI y/y legge
        # la stessa serie CPILFESL di Core CPI m/m, tracciare "ultimo
        # visto" per series_id farebbe si' che il primo dei due eventi
        # controllato segni la serie come vista e l'altro non scatti mai
        # piu' per lo stesso rilascio (vedi commento in _new_release_mom_pct).
        last_seen = load_fred_last_seen()
        # Migrazione one-time dal vecchio schema (chiave = series_id): le 3
        # serie già live prima di questo cambio avevano il loro "ultimo
        # visto" salvato sotto series_id, non event_name. Senza questo
        # seed, al primo giro dopo il deploy risulterebbero tutte "mai
        # viste" e potrebbero far scattare un alert per un rilascio già
        # notificato giorni/settimane fa sotto la chiave vecchia.
        if event_name not in last_seen and series_id in last_seen:
            last_seen[event_name] = last_seen[series_id]
        try:
            if value_type == "level":
                release = _new_release_level(series_id, last_seen, seen_key=event_name)
            elif value_type == "mom_diff":
                release = _new_release_mom_diff(series_id, last_seen, seen_key=event_name)
            elif value_type == "yoy_pct":
                release = _new_release_yoy_pct(series_id, last_seen, seen_key=event_name)
            else:
                release = _new_release_mom_pct(series_id, last_seen, seen_key=event_name)
        except Exception as e:
            logger.warning(f"FRED non raggiungibile per {series_id}: {e}")
            return None
        if release is None:
            return None
        release_date, actual_value = release
        last_seen[event_name] = release_date
        save_fred_last_seen(last_seen)
        return release_date, actual_value, "fred"

    rss_cfg = RSS_SERIES.get(event_name)
    if rss_cfg:
        last_seen = load_rss_macro_last_seen()
        try:
            release = _new_release_rss(event_name, rss_cfg, last_seen)
        except Exception as e:
            logger.warning(f"RSS non raggiungibile per {event_name}: {e}")
            return None
        if release is None:
            return None
        release_date, actual_value = release
        last_seen[event_name] = release_date
        save_rss_macro_last_seen(last_seen)
        return release_date, actual_value, "rss"

    return None


def predict_reaction(event_name: str, forecast_raw: str) -> dict | None:
    """
    Se per event_name è appena comparso un nuovo dato reale (FRED o, per le
    serie senza fonte FRED, un comunicato RSS riconosciuto) mai processato
    prima, e c'è un modello calibrato per quella serie, ritorna la
    previsione statistica; altrimenti None — nessuna previsione inventata
    quando manca un pezzo (modello assente, fonte non ancora aggiornata,
    forecast non numerico, titolo RSS non riconosciuto).

    Side effect: se trova un nuovo rilascio lo segna come processato
    (persistito), anche se poi la previsione non può essere completata
    per altri motivi — non ha senso ritentare all'infinito lo stesso
    numero già visto.
    """
    model = _load_model(event_name)
    if not model:
        return None

    value_type = (FRED_SERIES.get(event_name) or RSS_SERIES.get(event_name, {})).get("value_type")
    if not value_type:
        return None

    release = _fetch_new_actual(event_name)
    if release is None:
        return None
    release_date, actual_value, source_tier = release

    forecast_num, _ = _parse_number(forecast_raw)
    if forecast_num is None:
        logger.info(f"[{event_name}] Actual disponibile ({actual_value}) ma forecast non numerico: {forecast_raw!r}")
        return None

    surprise_raw = actual_value - forecast_num
    z = (surprise_raw - model["surprise_mean"]) / model["surprise_std"] if model["surprise_std"] else 0.0
    z = max(-model["surprise_zscore_clip"], min(model["surprise_zscore_clip"], z))
    predicted = model["slope"] * z + model["intercept"]

    return {
        "event_name": event_name,
        "value_type": value_type,
        "source_tier": source_tier,
        "actual_value": actual_value,
        "forecast_value": forecast_num,
        "surprise_raw": round(surprise_raw, 3),
        "surprise_zscore": round(z, 2),
        "predicted_reaction_usd": round(float(predicted), 2),
        "horizon": model["horizon"],
        "n_historical": model["n"],
    }


# Sotto questa soglia la sorpresa è talmente vicina a zero che "reazione
# attesa" nel messaggio è quasi solo l'intercetta del modello (rumore di
# fondo), non un vero segnale legato a QUESTO rilascio — l'utente ha
# segnalato confusione l'11/09/2026 vedendo "z=+0.0" insieme a una reazione
# attesa diversa da zero, senza capire perché. Solo per chiarezza del
# messaggio, non cambia il modello/training.
SURPRISE_INSIGNIFICANT_Z = 0.3


def format_prediction(pred: dict) -> str:
    direction = "ribassista per l'oro" if pred["predicted_reaction_usd"] < 0 else "rialzista per l'oro"
    horizon_label = pred["horizon"].replace("reaction_", "").replace("m", " min")
    value_type = pred.get("value_type")

    if value_type == "level":
        actual_fmt   = f"{pred['actual_value']:,.0f}"
        forecast_fmt = f"{pred['forecast_value']:,.0f}"
    elif value_type == "mom_diff":
        actual_fmt   = f"{pred['actual_value']:+,.0f}"
        forecast_fmt = f"{pred['forecast_value']:+,.0f}"
    elif value_type == "index":
        actual_fmt   = f"{pred['actual_value']:.1f}"
        forecast_fmt = f"{pred['forecast_value']:.1f}"
    else:
        actual_fmt   = f"{pred['actual_value']:+.2f}%"
        forecast_fmt = f"{pred['forecast_value']:+.2f}%"

    if pred.get("source_tier") == "rss":
        note_source = (
            "_L'actual è letto dal comunicato stampa via feed RSS (ISM/Conference Board non hanno "
            "una fonte governativa gratuita come FRED) — fonte meno consolidata: se il formato del "
            "comunicato cambia, questa previsione può semplicemente non comparire, mai un numero "
            "sbagliato._"
        )
    elif value_type == "mom_diff":
        note_source = (
            "_L'actual è ricalcolato da FRED come differenza dal mese precedente, non letto dal "
            "comunicato ufficiale — in rari casi FRED può revisionare un dato dopo la pubblicazione "
            "originale._"
        )
    elif value_type == "level" or value_type == "index":
        note_source = (
            "_L'actual è l'ultimo valore pubblicato da FRED (aggiornato entro circa un'ora "
            "dal rilascio ufficiale), non letto dal comunicato — in rari casi FRED può "
            "revisionare un dato dopo la pubblicazione originale._"
        )
    elif value_type == "yoy_pct":
        note_source = (
            "_L'actual è ricalcolato da FRED confrontando col dato di 12 mesi fa, non letto dal "
            "comunicato ufficiale — lo scarto da revisioni può essere più ampio che per un dato "
            "mese su mese (fino a ~0.5pp, verificato empiricamente) perché si accumulano le "
            "revisioni di un anno intero, non di un solo mese._"
        )
    else:
        note_source = (
            "_L'actual è ricalcolato da FRED, non letto dal comunicato ufficiale: può differire di "
            "±0.1pp in rari casi di revisione dell'indice — verificato empiricamente, ~20% dei mesi._"
        )

    # FIX 2026-09-13: mancava il nome dell'evento — con più rilasci CPI
    # simultanei (CPI m/m, CPI y/y, Core CPI m/m, Core CPI y/y, tutti alle
    # 14:30) l'utente non aveva modo di capire a quale dei quattro si
    # riferisse questo messaggio (oggi solo Core CPI m/m ha un modello
    # deployato, ma il messaggio non lo diceva). "Prev" rinominato
    # esplicitamente "Forecast" per non confondersi con "Prec." (precedente)
    # usato nell'alert pre-evento — stessa parola, significati diversi.
    warning = ""
    if abs(pred["surprise_zscore"]) < SURPRISE_INSIGNIFICANT_Z:
        warning = (
            "\n⚠️ _Sorpresa quasi nulla (actual in linea con le attese) — la reazione attesa qui sopra "
            "riflette soprattutto il rumore di fondo del modello, non un segnale legato a questo dato._"
        )

    return (
        f"📐 *Previsione statistica — {_escape_md(pred.get('event_name', 'evento'))}* "
        f"(n={pred['n_historical']} storici, {horizon_label})\n"
        f"Actual {actual_fmt} vs Forecast {forecast_fmt} "
        f"(sorpresa z={pred['surprise_zscore']:+.1f})\n"
        f"Reazione attesa: *{pred['predicted_reaction_usd']:+.2f}$* — {direction}"
        f"{warning}\n"
        f"_Stima statistica su dati storici, non una garanzia — margine d'errore reale, vedi Fase 4._\n"
        f"{note_source}"
    )
