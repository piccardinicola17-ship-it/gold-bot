"""
historical_fed_speeches.py — Progetto separato: hawkish/dovish scoring dei
discorsi e delle testimonianze dei Fed Chair (Powell/Yellen/Bernanke) fuori
dalle riunioni FOMC programmate.

Stessa idea di historical_fomc_text.py (Statement/Minutes/Press Conference)
e historical_cb_text.py (BCE/BOJ): un comunicato TESTUALE senza forecast
numerico da confrontare con un actual, quindi fuori dal modello a sorpresa
di historical_model.py per qualunque quantità di dati. Riusa
save_text_scores()/validate_fomc_scores() da historical_fomc_text.py invece
di duplicare la logica di INSERT/validazione (vedi
feedback_dual_mechanism_drift_pattern in memoria).

SCOPE — solo i 3 Fed Chair, non gli altri membri FOMC (Bullard/Dudley/
Evans/ecc.): quelle serie hanno 1-6 eventi ciascuna in macro_events, non
potranno mai superare la soglia n>=100 del progetto individualmente, e non
sono la voce che decide davvero la policy — pooling con i Chair sarebbe
metodologicamente discutibile (peso diverso). I 3 Chair insieme (Speaks +
Testifies, 6 event_name distinti) sommano 357 eventi 2007-2025, l'unico
sottoinsieme che può realisticamente raggiungere n>=100.

Testo scaricato da fetch_fed_chair_texts() — stesso host/principio di
fetch_fomc_statements() (federalreserve.gov), pattern
/newsevents/{speech|testimony}/{lastname}{YYYYMMDD}a.htm. A differenza
delle pressreleases (id="article" ... class="lastUpdate" con doppi apici),
qui il markup usa apici singoli su class — regex adattata per accettare
entrambi.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

import requests

from historical_events import HIST_DB_PATH, _connect
from historical_fomc_text import save_text_scores, validate_fomc_scores as validate_text_scores  # noqa: F401

FED_SPEECH_URL = "https://www.federalreserve.gov/newsevents/{kind}/{lastname}{ymd}{suffix}.htm"

# event_name in macro_events -> (cognome per l'URL, "speech"|"testimony")
FED_CHAIR_EVENTS: dict[str, tuple[str, str]] = {
    "Fed Chair Powell Speaks": ("powell", "speech"),
    "Fed Chair Powell Testifies": ("powell", "testimony"),
    "Fed Chair Yellen Speaks": ("yellen", "speech"),
    "Fed Chair Yellen Testifies": ("yellen", "testimony"),
    "Fed Chairman Bernanke Speaks": ("bernanke", "speech"),
    "Fed Chairman Bernanke Testifies": ("bernanke", "testimony"),
}

FED_CHAIR_SPEECHES_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "fed_chair_speeches_hawkish_scores.json"
)

_ARTICLE_RE = re.compile(r'<div id="article">(.*?)<div class=[\'"]lastUpdate[\'"]', re.S)


def fetch_fed_chair_texts(db_path: str = HIST_DB_PATH) -> dict:
    """Scarica il testo di ogni discorso/testimonianza dei 3 Fed Chair già
    presente in macro_events sotto uno dei 6 event_name di FED_CHAIR_EVENTS.
    Alcune date hanno più di un intervento pubblico lo stesso giorno — il
    sito Fed distingue con suffisso 'a'/'b'/'c': si prova 'a' e, solo se
    risponde 404, si tenta 'b' (raro, copre il caso senza appesantire ogni
    fetch con richieste extra inutili). Ritorna {event_name: {date_utc:
    {"url","text"} o {"error"}}}."""
    results: dict[str, dict] = {}
    with _connect(db_path) as conn:
        for event_name, (lastname, kind) in FED_CHAIR_EVENTS.items():
            dates = [r[0] for r in conn.execute(
                "SELECT date_utc FROM macro_events WHERE event_name=? ORDER BY date_utc", (event_name,)
            ).fetchall()]
            per_event: dict = {}
            for d in dates:
                ymd = d.replace("-", "")
                text, url, err = None, None, None
                for suffix in ("a", "b"):
                    url = FED_SPEECH_URL.format(kind=kind, lastname=lastname, ymd=ymd, suffix=suffix)
                    try:
                        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                        if r.status_code != 200:
                            err = f"HTTP {r.status_code}"
                            time.sleep(0.5)
                            continue
                        m = _ARTICLE_RE.search(r.text)
                        if not m:
                            err = "no article div"
                            time.sleep(0.5)
                            continue
                        raw = re.sub(r'<[^>]+>', ' ', m.group(1))
                        text = re.sub(r'\s+', ' ', raw).strip()
                        err = None
                        break
                    except Exception as e:
                        err = str(e)
                    time.sleep(0.5)
                per_event[d] = {"url": url, "text": text} if text else {"url": url, "error": err}
                time.sleep(0.5)
            results[event_name] = per_event
    return results


if __name__ == "__main__":
    texts = fetch_fed_chair_texts()
    scratch_path = os.path.join(os.path.dirname(__file__), "data", "fed_chair_speeches_raw.json")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    for event_name, per_event in texts.items():
        ok = sum(1 for v in per_event.values() if "text" in v)
        print(f"{event_name}: {ok}/{len(per_event)} testi scaricati")
    print(f"\nSalvato in {scratch_path}")
