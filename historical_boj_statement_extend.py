"""
historical_boj_statement_extend.py — Estende "Monetary Policy Statement"
(JPY) oltre il 2023-12-19, dove si fermava la fonte primaria
hf_forexfactory_cache (congelata al 2025-04-07 e comunque priva di questo
event_name per il periodo 2024+, verificato). BOJ tiene 7-8 riunioni MPM
l'anno: da fine 2023 a oggi (2026-09-16) ne mancavano 20, quasi il doppio
del campione scorato finora (99) — il gap piu' vistoso tra le fonti
testuali di banche centrali di questo progetto (2.75 anni).

Stesso pattern di historical_boj_outlook.py (URL diretti boj.or.jp,
gestione mista pdf/htm) — non riusa quel file perche' l'evento e' diverso
("Monetary Policy Statement", non "BOJ Outlook Report") e le pubblicazioni
"Statement" sono brevi (1-3 pagine, comunicato + tabella voti), a
differenza dell'Outlook Report (lungo, richiede sotto-agenti).

Orario di rilascio reale non disponibile per queste date (fuori dalla
finestra della fonte primaria) - usato un fallback fisso 02:30 UTC
(11:30 JST), calibrato sulle 99 righe reali gia' in macro_events per lo
stesso event_name: orari osservati concentrati 01:00-03:13 UTC
(10:00-12:13 JST), mediana vicina a 02:30 UTC — piu' stretto del fallback
gia' accettato in historical_boj_outlook.py (13:00 JST su un range
osservato di 3+ ore).
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pdfplumber
import requests

from historical_events import HIST_DB_PATH, _connect, _event_uid
from historical_fomc_text import save_text_scores, validate_fomc_scores as validate_text_scores  # noqa: F401

EVENT_NAME = "Monetary Policy Statement"
CURRENCY = "JPY"
SOURCE_NAME_NEW = "boj_direct_fetch"

BOJ_STATEMENT_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "boj_statement_hawkish_scores.json"
)

_FALLBACK_HOUR_UTC = 2
_FALLBACK_MINUTE_UTC = 30

_HTML_RE = re.compile(r'id="contents">(.*?)</main>', re.S)

# (data ISO, formato "pdf"|"htm", url) — le 20 riunioni MPM dal 2024-01-23
# (prima non coperta da nessuna fonte) al 2026-07-31 (ultima gia' conclusa
# e pubblicata al 2026-09-16; la riunione del 17-18 settembre 2026 non si
# e' ancora tenuta). Verificate una per una con richiesta HTTP diretta
# (200 su tutte) prima di essere inserite qui.
BOJ_STATEMENT_URLS: list[tuple[str, str, str]] = [
    ("2024-01-23", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240123a.htm"),
    # 19 marzo 2024: fine di NIRP e Yield Curve Control (QQE) - mancava dalla
    # prima lista costruita a partire da una pagina indice del sito BOJ che
    # per questa data non elencava lo statement separatamente (probabilmente
    # per il titolo diverso, "Changes in the Monetary Policy Framework"
    # invece di "Statement on Monetary Policy") - trovata verificando lo
    # schema riunioni MPM ufficiale (8/anno) contro la lista fetchata (ne
    # aveva solo 7 per il 2024) e testando l'URL prevedibile a mano.
    ("2024-03-19", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240319a.htm"),
    ("2024-04-26", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240426a.htm"),
    ("2024-06-14", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240614a.htm"),
    ("2024-07-31", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240731a.htm"),
    ("2024-09-20", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240920a.htm"),
    ("2024-10-31", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k241031a.htm"),
    ("2024-12-19", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k241219a.htm"),
    ("2025-01-24", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250124a.htm"),
    ("2025-03-19", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250319a.htm"),
    ("2025-05-01", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250501a.htm"),
    ("2025-06-17", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250617a.htm"),
    ("2025-07-31", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250731a.htm"),
    ("2025-09-19", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k250919a.htm"),
    ("2025-10-30", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k251030a.htm"),
    ("2025-12-19", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/k251219a.htm"),
    ("2026-01-23", "htm", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2026/k260123a.htm"),
    ("2026-03-19", "pdf", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260319a.pdf"),
    ("2026-04-28", "pdf", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260428a.pdf"),
    ("2026-06-16", "pdf", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260616a.pdf"),
    ("2026-07-31", "pdf", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2026/k260731a.pdf"),
]


def _publish_date_to_datetime_utc(date_iso: str) -> str:
    y, m, d = map(int, date_iso.split("-"))
    naive_utc = datetime(y, m, d, _FALLBACK_HOUR_UTC, _FALLBACK_MINUTE_UTC, tzinfo=ZoneInfo("UTC"))
    return naive_utc.isoformat()


def ensure_boj_statement_events(db_path: str = HIST_DB_PATH) -> int:
    added = 0
    with _connect(db_path) as conn:
        now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
        for date_iso, fmt, url in BOJ_STATEMENT_URLS:
            existing = conn.execute(
                "SELECT 1 FROM macro_events WHERE event_name=? AND currency=? AND date_utc=?",
                (EVENT_NAME, CURRENCY, date_iso),
            ).fetchone()
            if existing:
                continue
            dt_utc = _publish_date_to_datetime_utc(date_iso)
            uid = _event_uid(CURRENCY, EVENT_NAME, dt_utc, SOURCE_NAME_NEW)
            conn.execute(
                "INSERT OR IGNORE INTO macro_events "
                "(event_uid, datetime_utc, date_utc, currency, impact, event_name, "
                "macro_category, source, source_detail, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uid, dt_utc, date_iso, CURRENCY, "HIGH", EVENT_NAME,
                 "BOJ", SOURCE_NAME_NEW, url, now_iso),
            )
            added += 1
    return added


def fetch_boj_statement_texts() -> dict:
    results: dict = {}
    for date_iso, fmt, url in BOJ_STATEMENT_URLS:
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                results[date_iso] = {"url": url, "error": f"HTTP {r.status_code}"}
                time.sleep(0.5)
                continue
            if fmt == "pdf":
                with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                text = re.sub(r"\s+", " ", text).strip()
            else:
                m = _HTML_RE.search(r.text)
                if not m:
                    results[date_iso] = {"url": url, "error": "no id=contents match"}
                    time.sleep(0.5)
                    continue
                raw = re.sub(r"<[^>]+>", " ", m.group(1))
                text = re.sub(r"\s+", " ", raw).strip()
            if not text:
                results[date_iso] = {"url": url, "error": "testo vuoto dopo estrazione"}
                time.sleep(0.5)
                continue
            results[date_iso] = {"url": url, "text": text}
        except Exception as e:
            results[date_iso] = {"url": url, "error": str(e)}
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    n_events = ensure_boj_statement_events()
    print(f"macro_events: {n_events} nuove righe inserite (su {len(BOJ_STATEMENT_URLS)} totali note)")

    texts = fetch_boj_statement_texts()
    scratch_path = os.path.join(os.path.dirname(__file__), "data", "boj_statement_extend_raw.json")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    ok = sum(1 for v in texts.values() if "text" in v)
    print(f"Testi scaricati: {ok}/{len(texts)}")
    print(f"Salvato in {scratch_path}")
