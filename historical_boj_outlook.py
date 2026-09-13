"""
historical_boj_outlook.py — Estende "BOJ Outlook Report" da n=38 (2015-10/
2025-01, mai scorato finora) a n=59 (2008-04/2026-07), per abilitare un
pooling combinato con historical_boj_opinions.py ("BOJ Summary of
Opinions") — nessuna delle due fonti da sola raggiunge n>=100 (tetto
teorico BOJ Outlook Report da solo: ~76 anche a copertura storica
completa dal 2000, vedi memoria progetto), ma insieme potrebbero.

Recupera la versione "The Bank's View" (suffisso "a", NON "b" che è il
full text pubblicato alcuni giorni dopo) per ciascuna uscita — stessa
versione già presente per le 38 righe esistenti in macro_events (fonte
hf_forexfactory_cache, verificato: le date combaciano esattamente).
Formato misto: PDF quasi sempre, HTML per un breve periodo 2018-2020
(gor1801a.htm ... gor2001a.htm) — gestito per-URL in base all'estensione.

Orario di pubblicazione: NON fisso come per Summary of Opinions (8:50
JST) — dipende da quando termina la MPM che lo decide, osservato
02:55-06:00 UTC (11:55-15:00 JST) sulle 38 righe già presenti. Per le 21
nuove righe (senza un orario osservato reale) si usa una stima
rappresentativa (13:00 JST = 04:00 UTC, punto medio del range osservato)
— un'approssimazione dichiarata, accettabile per uno studio esplorativo
locale che non tocca il bot live, non per un deploy.
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

EVENT_NAME = "BOJ Outlook Report"
CURRENCY = "JPY"
SOURCE_NAME_NEW = "boj_direct_fetch"

BOJ_OUTLOOK_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "boj_outlook_hawkish_scores.json"
)

_HTML_RE = re.compile(r'id="contents">(.*?)</main>', re.S)

# Stima rappresentativa per le pubblicazioni senza orario osservato reale
# (vedi docstring del modulo) — 13:00 JST, punto medio del range 11:55-15:00
# JST osservato sulle 38 righe già presenti da hf_forexfactory_cache.
_FALLBACK_HOUR_JST = 13

# (data ISO, formato "pdf"|"htm", url) — le 59 pubblicazioni "The Bank's
# View" note (2008-2026), dalla pagina indice
# boj.or.jp/en/mopo/outlook/index.htm (2026-09-13).
BOJ_OUTLOOK_URLS: list[tuple[str, str, str]] = [
    ("2008-04-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor0804a.pdf"),
    ("2008-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor0810a.pdf"),
    ("2009-04-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor0904a.pdf"),
    ("2009-10-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor0910a.pdf"),
    ("2010-04-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1004a.pdf"),
    ("2010-10-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1010a.pdf"),
    ("2011-04-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1104a.pdf"),
    ("2011-10-27", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1110a.pdf"),
    ("2012-04-27", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1204a.pdf"),
    ("2012-10-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1210a.pdf"),
    ("2013-04-26", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1304a.pdf"),
    ("2013-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1310a.pdf"),
    ("2014-04-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1404a.pdf"),
    ("2014-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1410a.pdf"),
    ("2015-04-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1504a.pdf"),
    ("2015-10-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1510a.pdf"),
    ("2016-01-29", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1601a.pdf"),
    ("2016-04-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1604a.pdf"),
    ("2016-07-29", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1607a.pdf"),
    ("2016-11-01", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1610a.pdf"),
    ("2017-01-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1701a.pdf"),
    ("2017-04-27", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1704a.pdf"),
    ("2017-07-20", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1707a.pdf"),
    ("2017-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor1710a.pdf"),
    ("2018-01-23", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1801a.htm"),
    ("2018-04-27", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1804a.htm"),
    ("2018-07-31", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1807a.htm"),
    ("2018-10-31", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1810a.htm"),
    ("2019-01-23", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1901a.htm"),
    ("2019-04-25", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1904a.htm"),
    ("2019-07-30", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1907a.htm"),
    ("2019-10-31", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor1910a.htm"),
    ("2020-01-21", "htm", "https://www.boj.or.jp/en/mopo/outlook/gor2001a.htm"),
    ("2020-04-27", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2004a.pdf"),
    ("2020-07-15", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2007a.pdf"),
    ("2020-10-29", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2010a.pdf"),
    ("2021-01-21", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2101a.pdf"),
    ("2021-04-27", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2104a.pdf"),
    ("2021-07-16", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2107a.pdf"),
    ("2021-10-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2110a.pdf"),
    ("2022-01-18", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2201a.pdf"),
    ("2022-04-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2204a.pdf"),
    ("2022-07-21", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2207a.pdf"),
    ("2022-10-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2210a.pdf"),
    ("2023-01-18", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2301a.pdf"),
    ("2023-04-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2304a.pdf"),
    ("2023-07-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2307a.pdf"),
    ("2023-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2310a.pdf"),
    ("2024-01-23", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2401a.pdf"),
    ("2024-04-26", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2404a.pdf"),
    ("2024-07-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2407a.pdf"),
    ("2024-10-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2410a.pdf"),
    ("2025-01-24", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2501a.pdf"),
    ("2025-05-01", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2504a.pdf"),
    ("2025-07-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2507a.pdf"),
    ("2025-10-30", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2510a.pdf"),
    ("2026-01-23", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2601a.pdf"),
    ("2026-04-28", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2604a.pdf"),
    ("2026-07-31", "pdf", "https://www.boj.or.jp/en/mopo/outlook/gor2607a.pdf"),
]


def _publish_date_to_datetime_utc(date_iso: str) -> str:
    y, m, d = map(int, date_iso.split("-"))
    local = datetime(y, m, d, _FALLBACK_HOUR_JST, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    return local.astimezone(ZoneInfo("UTC")).isoformat()


def ensure_boj_outlook_events(db_path: str = HIST_DB_PATH) -> int:
    """Inserisce le righe mancanti (le 38 già presenti da
    hf_forexfactory_cache, con orario osservato reale, non vengono
    toccate — combaciano già esattamente su date_utc)."""
    added = 0
    with _connect(db_path) as conn:
        now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
        for date_iso, fmt, url in BOJ_OUTLOOK_URLS:
            existing = conn.execute(
                "SELECT 1 FROM macro_events WHERE event_name=? AND date_utc=?",
                (EVENT_NAME, date_iso),
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


def fetch_boj_outlook_texts() -> dict:
    results: dict = {}
    for date_iso, fmt, url in BOJ_OUTLOOK_URLS:
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
    n_events = ensure_boj_outlook_events()
    print(f"macro_events: {n_events} nuove righe inserite (su {len(BOJ_OUTLOOK_URLS)} totali note)")

    texts = fetch_boj_outlook_texts()
    scratch_path = os.path.join(os.path.dirname(__file__), "data", "boj_outlook_raw.json")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    ok = sum(1 for v in texts.values() if "text" in v)
    print(f"Testi scaricati: {ok}/{len(texts)}")
    print(f"Salvato in {scratch_path}")
