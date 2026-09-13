"""
historical_ecb_accounts.py — Progetto separato: estende la copertura di
"ECB Monetary Policy Meeting Accounts" da n=12 (2015-02/2020-05, unica
copertura disponibile in hf_forexfactory_cache) a n=95 (2015-02/2026-08),
scoperto il 2026-09-13 durante una ricerca approfondita richiesta
dall'utente sui buchi rimasti del progetto testuale.

SCOPERTA CHIAVE: la fonte hf_forexfactory_cache non copre affatto "poche
uscite per limite strutturale" (come inizialmente concluso per errore) —
la BCE ha continuato a pubblicare questi account ogni ~6 settimane senza
interruzione dal 2015 ad oggi (confermato: pagina indice ECB, ricerca
mirata "site:ecb.europa.eu/press/accounts 2024"). Il gap è un buco di
COPERTURA della fonte calendario usata finora (si ferma al 2020-05), non
un limite dell'evento stesso. Recuperate tutte le 95 date reali 2015-2026
via Wayback Machine CDX API (le pagine ecb.europa.eu sono poi scaricabili
DIRETTAMENTE dal sito live, verificato — Wayback è servito solo per
scoprire gli URL/hash, non per il fetch del testo).

n=95 è appena sotto la soglia n>=100 del progetto (vedi
feedback_conservative_validation_standard) — ma cresce di ~8-9/anno, quindi
supererà 100 nel giro di qualche mese/anno. Prerequisito comunque
soddisfatto per iniziare lo scoring: n>=20 per split (vedi
historical_fomc_text.validate_fomc_scores, richiede n>=100 solo per il
VERDETTO finale "EDGE VALIDATO", non per salvare gli score).

URL — 3 ere della BCE (stesso pattern già scoperto per ECB Press
Conference in una sessione precedente): 2015-2016 "mg{YYMMDD}.en.html"
(nessun prefisso "ecb.", nessun hash), 2017-2018 "ecb.mg{YYMMDD}.en.html"
(prefisso ma nessun hash), 2019+ "ecb.mg{YYMMDD}~{hash}.en.html" (con
hash). Contenuto estratto da <main>...</main> (diverso dal pattern
id="article" usato per federalreserve.gov — verificato con fetch diretto
su un esempio per ciascuna era).

data_utc = data di PUBBLICAZIONE (non della riunione, che avviene ~4
settimane prima) — stessa convenzione delle 12 righe già presenti da
hf_forexfactory_cache, confermata riga per riga. Orario: la BCE pubblica
sempre alle 13:30 ora di Francoforte (11:30 UTC in CEST, 12:30 UTC in
CET) — dedotto dalle 12 righe esistenti (pattern esatto: mesi CEST
Apr-Ott -> 11:30 UTC, mesi CET Nov-Mar -> 12:30 UTC) e replicato qui via
conversione timezone-aware invece di un offset fisso.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from historical_events import HIST_DB_PATH, _connect, _event_uid
from historical_fomc_text import save_text_scores, validate_fomc_scores as validate_text_scores  # noqa: F401

FRANKFURT_TZ = ZoneInfo("Europe/Berlin")
EVENT_NAME = "ECB Monetary Policy Meeting Accounts"
CURRENCY = "EUR"
SOURCE_NAME = "ecb_direct_fetch"

ECB_ACCOUNTS_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "ecb_accounts_hawkish_scores.json"
)

_ARTICLE_RE = re.compile(r'<main[^>]*>(.*?)</main>', re.S)

# (datecode YYMMDD, url) — le 95 pubblicazioni trovate via Wayback Machine
# CDX API (2026-09-13), URL live su ecb.europa.eu verificati direttamente
# fetchabili (non serve passare da Wayback per il testo). Un duplicato con
# hash identico (220406/220407) è stato rimosso tenendo solo una copia.
ECB_ACCOUNTS_URLS: list[tuple[str, str]] = [
    ("150219", "http://www.ecb.europa.eu/press/accounts/2015/html/mg150219.en.html"),
    ("150402", "http://www.ecb.europa.eu/press/accounts/2015/html/mg150402.en.html"),
    ("150521", "http://www.ecb.europa.eu/press/accounts/2015/html/mg150521.en.html"),
    ("150702", "http://www.ecb.europa.eu/press/accounts/2015/html/mg150702.en.html"),
    ("150813", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150813.en.html"),
    ("151008", "http://www.ecb.europa.eu/press/accounts/2015/html/mg151008.en.html"),
    ("151119", "http://www.ecb.europa.eu/press/accounts/2015/html/mg151119.en.html"),
    ("160114", "http://www.ecb.europa.eu/press/accounts/2016/html/mg160114.en.html"),
    ("160218", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160218.en.html"),
    ("160407", "http://www.ecb.europa.eu/press/accounts/2016/html/mg160407.en.html"),
    ("160519", "http://www.ecb.europa.eu/press/accounts/2016/html/mg160519.en.html"),
    ("160707", "http://www.ecb.europa.eu/press/accounts/2016/html/mg160707.en.html"),
    ("160818", "http://www.ecb.europa.eu/press/accounts/2016/html/mg160818.en.html"),
    ("161006", "http://www.ecb.europa.eu/press/accounts/2016/html/mg161006.en.html"),
    ("161117", "http://www.ecb.europa.eu/press/accounts/2016/html/mg161117.en.html"),
    ("170112", "https://www.ecb.europa.eu/press/accounts/2017/html/mg170112.en.html"),
    ("170216", "http://www.ecb.europa.eu/press/accounts/2017/html/mg170216.en.html"),
    ("170406", "http://www.ecb.europa.eu/press/accounts/2017/html/mg170406.en.html"),
    ("170518", "http://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170518.en.html"),
    ("170706", "http://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170706.en.html"),
    ("170817", "http://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170817.en.html"),
    ("171005", "http://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg171005.en.html"),
    ("171123", "http://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg171123.en.html"),
    ("180111", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180111.en.html"),
    ("180222", "http://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180222.en.html"),
    ("180412", "http://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180412.en.html"),
    ("180524", "http://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180524.en.html"),
    ("180712", "http://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180712.en.html"),
    ("180823", "http://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180823.en.html"),
    ("181011", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg181011.en.html"),
    ("181122", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg181122.en.html"),
    ("190110", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190110.en.html"),
    ("190221", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190221~0f3dd919fa.en.html"),
    ("190404", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190404~edc605830b.en.html"),
    ("190523", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190523~3e19e27fb7.en.html"),
    ("190711", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190711~16eb146254.en.html"),
    ("190822", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190822~63660ecd81.en.html"),
    ("191010", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg191010~d8086505d0.en.html"),
    ("191121", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg191121~b1d36734d7.en.html"),
    ("200116", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200116~973b558e59.en.html"),
    ("200220", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200220~c4d71ec138.en.html"),
    ("200409", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200409_1~baf4b2ad06.en.html"),
    ("200522", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200522~f0355619ae.en.html"),
    ("200625", "http://www.ecb.europa.eu/press/accounts/2020/html/ecb%E2%80%8B.mg200625~fd97330d5f.en.html"),
    ("200820", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200820~c30e2e26b9.en.html"),
    ("201008", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg201008~49aeff32e1.en.html"),
    ("201126", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg201126~20e838e857.en.html"),
    ("210114", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210114~14ef04b8bd.en.html"),
    ("210218", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210218~9dab5cb5f7.en.html"),
    ("210408", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210408~46b9deaa4a.en.html"),
    ("210514", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210514~4b2606bff9.en.html"),
    ("210709", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210709~8d7a056036.en.html"),
    ("210729", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210729~b83737e3b5.en.html"),
    ("210826", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210826~16a0691c87.en.html"),
    ("211007", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg211007~1c2f4db595.en.html"),
    ("211125", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg211125~ca9833f9a9.en.html"),
    ("220120", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220120~7ed187b5b1.en.html"),
    ("220303", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220303~7ac13bacbe.en.html"),
    ("220406", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220406~8e7069ffa0.en.html"),
    ("220519", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220519~c9200dba08.en.html"),
    ("220707", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220707~d5c3246061.en.html"),
    ("220825", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220825~162cfabae9.en.html"),
    ("221006", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg221006~a5f7fb03f3.en.html"),
    ("221124", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg221124~3527764024.en.html"),
    ("230119", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230119~e522ad4e37.en.html"),
    ("230302", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230302~009d06dd5a.en.html"),
    ("230420", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230420~e8043d2d3d.en.html"),
    ("230601", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230601~9d35f80dee.en.html"),
    ("230713", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230713~f7e54fdb87.en.html"),
    ("230831", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230831~b04764f45f.en.html"),
    ("231012", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg231012~2f3d803d32.en.html"),
    ("231123", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg231123~40c9631bc7.en.html"),
    ("240118", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240118~57d24ff18f.en.html"),
    ("240222", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240222~1af5fcd5f9.en.html"),
    ("240404", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240404~b79e424115.en.html"),
    ("240510", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240510~6505e9dac3.en.html"),
    ("240704", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240704~fbde4f46aa.en.html"),
    ("240822", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240822~d49b920824.en.html"),
    ("241010", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg241010~1036884a9a.en.html"),
    ("241114", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg241114~c0e6f53cf7.en.html"),
    ("250116", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250116~2f8f2a2ad3.en.html"),
    ("250227", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250227~5a2b6faa14.en.html"),
    ("250403", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250403~e578b5dbea.en.html"),
    ("250522", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250522~31b2c664d4.en.html"),
    ("250703", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250703~07feaceb60.en.html"),
    ("250828", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250828_1~e9f77119ce.en.html"),
    ("251009", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg251009~eec3e95eb5.en.html"),
    ("251127", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg251127~dc88fc4bec.en.html"),
    ("260122", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260122~5ca84e0f51.en.html"),
    ("260305", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260305~4a9b7afe1c.en.html"),
    ("260416", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260416~6a27b0c258.en.html"),
    ("260528", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260528~a93230dc4b.en.html"),
    ("260709", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260709~0e7f8241c9.en.html"),
    ("260827", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260827~f06c21fd54.en.html"),
]


def _datecode_to_datetime_utc(datecode: str) -> str:
    """'260827' -> datetime UTC ISO, assumendo pubblicazione alle 13:30 ora
    di Francoforte (dedotto dalle 12 righe già presenti in macro_events:
    11:30 UTC nei mesi CEST, 12:30 UTC nei mesi CET — coerente con 13:30
    locale tutto l'anno). zoneinfo gestisce l'ora legale automaticamente."""
    yy, mm, dd = int(datecode[:2]), int(datecode[2:4]), int(datecode[4:6])
    local = datetime(2000 + yy, mm, dd, 13, 30, tzinfo=FRANKFURT_TZ)
    return local.astimezone(ZoneInfo("UTC")).isoformat()


def ensure_ecb_accounts_events(db_path: str = HIST_DB_PATH) -> int:
    """Inserisce in macro_events le righe mancanti per le 95 pubblicazioni
    note (le 12 già presenti da hf_forexfactory_cache vengono lasciate
    intatte, non duplicate — stesso event_uid scheme, stessa data_utc,
    quindi l'INSERT OR IGNORE le salta automaticamente)."""
    added = 0
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uid TEXT NOT NULL UNIQUE,
                datetime_utc TEXT NOT NULL,
                date_utc TEXT NOT NULL,
                currency TEXT NOT NULL DEFAULT 'USD',
                impact TEXT NOT NULL,
                event_name TEXT NOT NULL,
                macro_category TEXT,
                actual_raw TEXT, forecast_raw TEXT, previous_raw TEXT,
                actual_num REAL, forecast_num REAL, previous_num REAL,
                unit TEXT,
                source TEXT NOT NULL,
                source_detail TEXT,
                ingested_at TEXT NOT NULL
            )
        """)
        now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
        for datecode, url in ECB_ACCOUNTS_URLS:
            dt_utc = _datecode_to_datetime_utc(datecode)
            date_utc = dt_utc[:10]
            uid = _event_uid(CURRENCY, EVENT_NAME, dt_utc, SOURCE_NAME)
            # Se esiste già una riga per questa data da hf_forexfactory_cache
            # (le 12 storiche), non serve un'altra riga duplicata per lo
            # stesso evento reale — salta.
            existing = conn.execute(
                "SELECT 1 FROM macro_events WHERE event_name=? AND date_utc=?",
                (EVENT_NAME, date_utc),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO macro_events "
                "(event_uid, datetime_utc, date_utc, currency, impact, event_name, "
                "macro_category, source, source_detail, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uid, dt_utc, date_utc, CURRENCY, "HIGH", EVENT_NAME,
                 "ECB", SOURCE_NAME, url, now_iso),
            )
            added += 1
    return added


def fetch_ecb_accounts_texts() -> dict:
    """Scarica il testo di tutte le 95 pubblicazioni note, direttamente da
    ecb.europa.eu (non serve Wayback per il testo, solo per aver scoperto
    gli URL). Ritorna {date_utc: {"url","text"} o {"error"}}."""
    results: dict = {}
    for datecode, url in ECB_ACCOUNTS_URLS:
        date_utc = _datecode_to_datetime_utc(datecode)[:10]
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                results[date_utc] = {"url": url, "error": f"HTTP {r.status_code}"}
                time.sleep(0.5)
                continue
            m = _ARTICLE_RE.search(r.text)
            if not m:
                results[date_utc] = {"url": url, "error": "no <main> match"}
                time.sleep(0.5)
                continue
            raw = re.sub(r'<[^>]+>', ' ', m.group(1))
            text = re.sub(r'\s+', ' ', raw).strip()
            results[date_utc] = {"url": url, "text": text}
        except Exception as e:
            results[date_utc] = {"url": url, "error": str(e)}
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    n_events = ensure_ecb_accounts_events()
    print(f"macro_events: {n_events} nuove righe inserite (su {len(ECB_ACCOUNTS_URLS)} totali note)")

    texts = fetch_ecb_accounts_texts()
    scratch_path = os.path.join(os.path.dirname(__file__), "data", "ecb_accounts_raw.json")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    ok = sum(1 for v in texts.values() if "text" in v)
    print(f"Testi scaricati: {ok}/{len(texts)}")
    print(f"Salvato in {scratch_path}")
