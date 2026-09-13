"""
historical_boj_opinions.py — Progetto separato: "Summary of Opinions at the
Monetary Policy Meeting" della BOJ, mai considerato nelle sessioni
precedenti (scoperto il 2026-09-13 durante la ricerca approfondita sui
buchi rimasti del progetto testuale). Pubblicato in inglese dall'MPM di
dicembre 2015 (introdotto nel 2016 come iniziativa di trasparenza — non
esisteva prima), 8 volte l'anno.

n=86 (2015-12/2026-08) — da solo SOTTO la soglia n>=100 del progetto anche
con copertura storica completa (tetto teorico ~89-90 contando fino a fine
2026), stesso limite di frequenza di BOJ Outlook Report. Non deployabile
da solo, ma è il pezzo mancante per un eventuale "BOJ combinato" (Outlook
Report + Summary of Opinions) — stessa logica già usata per il "Combinato"
FOMC (Statement+Minutes+PressConf) e per il pooling Fed Chair
(historical_fed_speeches.py).

Formato misto: PDF per 2015-2017 e per una parte del 2026, HTML per
2018-2025 (BOJ è passata a HTML poi tornata a PDF per motivi non noti —
gestito per-URL in base all'estensione, non per anno). Estrazione HTML da
`id="contents">...</main>`; estrazione PDF via pdfplumber (dipendenza già
presente nell'ambiente).

Orario di pubblicazione: "Not to be released until 8:50 a.m. Japan
Standard Time" (dichiarato nel testo stesso dei PDF) — JST è UTC+9 tutto
l'anno (nessuna ora legale in Giappone), quindi 8:50 JST = 23:50 UTC del
giorno PRECEDENTE alla data di pubblicazione indicata sul sito.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pdfplumber
import requests

from historical_events import HIST_DB_PATH, _connect, _event_uid
from historical_fomc_text import save_text_scores, validate_fomc_scores as validate_text_scores  # noqa: F401

EVENT_NAME = "BOJ Summary of Opinions"
CURRENCY = "JPY"
SOURCE_NAME = "boj_direct_fetch"

BOJ_OPINIONS_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "boj_opinions_hawkish_scores.json"
)

_HTML_RE = re.compile(r'id="contents">(.*?)</main>', re.S)

# (data_pubblicazione ISO, formato "pdf"|"htm", url) — le 86 pubblicazioni
# note (2015-12/2026-08), estratte dalle pagine indice annuali
# boj.or.jp/en/mopo/mpmsche_minu/opinion_{anno}/index.htm (2026-09-13).
BOJ_OPINIONS_URLS: list[tuple[str, str, str]] = [
    ("2016-01-08", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi151218.pdf"),
    ("2016-02-08", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160129.pdf"),
    ("2016-03-24", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160315.pdf"),
    ("2016-05-12", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160428.pdf"),
    ("2016-06-24", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160616.pdf"),
    ("2016-08-08", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160729.pdf"),
    ("2016-09-30", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi160921.pdf"),
    ("2016-11-10", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi161101.pdf"),
    ("2016-12-29", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2016/opi161220.pdf"),
    ("2017-02-08", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170131.pdf"),
    ("2017-03-27", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170316.pdf"),
    ("2017-05-10", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170427.pdf"),
    ("2017-06-26", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170616.pdf"),
    ("2017-07-28", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170720.pdf"),
    ("2017-09-29", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi170921.pdf"),
    ("2017-11-09", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi171031.pdf"),
    ("2017-12-28", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2017/opi171221.pdf"),
    ("2018-01-31", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180123.htm"),
    ("2018-03-19", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180309.htm"),
    ("2018-05-10", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180427.htm"),
    ("2018-06-25", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180615.htm"),
    ("2018-08-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180731.htm"),
    ("2018-09-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi180919.htm"),
    ("2018-11-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi181031.htm"),
    ("2018-12-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2018/opi181220.htm"),
    ("2019-01-31", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190123.htm"),
    ("2019-03-26", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190315.htm"),
    ("2019-05-10", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190425.htm"),
    ("2019-06-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190620.htm"),
    ("2019-08-07", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190730.htm"),
    ("2019-09-30", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi190919.htm"),
    ("2019-11-11", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi191031.htm"),
    ("2019-12-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2019/opi191219.htm"),
    ("2020-01-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200121.htm"),
    ("2020-03-25", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200316.htm"),
    ("2020-05-11", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200427.htm"),
    ("2020-06-24", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200616.htm"),
    ("2020-07-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200715.htm"),
    ("2020-09-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi200917.htm"),
    ("2020-11-09", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi201029.htm"),
    ("2020-12-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2020/opi201218.htm"),
    ("2021-01-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210121.htm"),
    ("2021-03-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210319.htm"),
    ("2021-05-11", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210427.htm"),
    ("2021-06-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210618.htm"),
    ("2021-07-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210716.htm"),
    ("2021-10-01", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi210922.htm"),
    ("2021-11-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi211028.htm"),
    ("2021-12-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2021/opi211217.htm"),
    ("2022-01-26", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220118.htm"),
    ("2022-03-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220318.htm"),
    ("2022-05-12", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220428.htm"),
    ("2022-06-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220617.htm"),
    ("2022-07-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220721.htm"),
    ("2022-10-03", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi220922.htm"),
    ("2022-11-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi221028.htm"),
    ("2022-12-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2022/opi221220.htm"),
    ("2023-01-26", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230118.htm"),
    ("2023-03-20", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230310.htm"),
    ("2023-05-11", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230428.htm"),
    ("2023-06-26", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230616.htm"),
    ("2023-08-07", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230728.htm"),
    ("2023-10-02", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi230922.htm"),
    ("2023-11-09", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi231031.htm"),
    ("2023-12-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2023/opi231219.htm"),
    ("2024-01-31", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240123.htm"),
    ("2024-03-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240319.htm"),
    ("2024-05-09", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240426.htm"),
    ("2024-06-24", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240614.htm"),
    ("2024-08-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240731.htm"),
    ("2024-10-01", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi240920.htm"),
    ("2024-11-11", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi241031.htm"),
    ("2024-12-27", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2024/opi241219.htm"),
    ("2025-02-03", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250124.htm"),
    ("2025-03-28", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250319.htm"),
    ("2025-05-13", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250501.htm"),
    ("2025-06-25", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250617.htm"),
    ("2025-08-08", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250731.htm"),
    ("2025-09-30", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi250919.htm"),
    ("2025-11-10", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi251030.htm"),
    ("2025-12-29", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/opi251219.htm"),
    ("2026-02-02", "htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260123.htm"),
    ("2026-03-30", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260319.pdf"),
    ("2026-05-12", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260428.pdf"),
    ("2026-06-24", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260616.pdf"),
    ("2026-08-10", "pdf", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2026/opi260731.pdf"),
]


def _publish_date_to_datetime_utc(date_iso: str) -> str:
    """'2026-08-10' -> datetime UTC ISO, pubblicazione alle 8:50 JST (UTC+9,
    nessuna ora legale) = 23:50 UTC del giorno PRECEDENTE."""
    y, m, d = map(int, date_iso.split("-"))
    local = datetime(y, m, d, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))
    return local.astimezone(ZoneInfo("UTC")).isoformat()


def ensure_boj_opinions_events(db_path: str = HIST_DB_PATH) -> int:
    """Inserisce in macro_events le righe per le 86 pubblicazioni note
    (evento mai presente prima in nessuna fonte usata da questo progetto)."""
    added = 0
    with _connect(db_path) as conn:
        now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
        for date_iso, fmt, url in BOJ_OPINIONS_URLS:
            dt_utc = _publish_date_to_datetime_utc(date_iso)
            uid = _event_uid(CURRENCY, EVENT_NAME, dt_utc, SOURCE_NAME)
            existing = conn.execute(
                "SELECT 1 FROM macro_events WHERE event_name=? AND date_utc=?",
                (EVENT_NAME, dt_utc[:10]),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO macro_events "
                "(event_uid, datetime_utc, date_utc, currency, impact, event_name, "
                "macro_category, source, source_detail, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uid, dt_utc, dt_utc[:10], CURRENCY, "HIGH", EVENT_NAME,
                 "BOJ", SOURCE_NAME, url, now_iso),
            )
            added += 1
    return added


def fetch_boj_opinions_texts() -> dict:
    """Scarica il testo di tutte le pubblicazioni note, gestendo sia il
    formato PDF (pdfplumber) sia HTML (id="contents">...</main>)."""
    results: dict = {}
    for date_iso, fmt, url in BOJ_OPINIONS_URLS:
        dt_utc = _publish_date_to_datetime_utc(date_iso)
        date_key = dt_utc[:10]
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                results[date_key] = {"url": url, "error": f"HTTP {r.status_code}"}
                time.sleep(0.5)
                continue
            if fmt == "pdf":
                with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                text = re.sub(r"\s+", " ", text).strip()
            else:
                m = _HTML_RE.search(r.text)
                if not m:
                    results[date_key] = {"url": url, "error": "no id=contents match"}
                    time.sleep(0.5)
                    continue
                raw = re.sub(r"<[^>]+>", " ", m.group(1))
                text = re.sub(r"\s+", " ", raw).strip()
            if not text:
                results[date_key] = {"url": url, "error": "testo vuoto dopo estrazione"}
                time.sleep(0.5)
                continue
            results[date_key] = {"url": url, "text": text}
        except Exception as e:
            results[date_key] = {"url": url, "error": str(e)}
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    n_events = ensure_boj_opinions_events()
    print(f"macro_events: {n_events} nuove righe inserite (su {len(BOJ_OPINIONS_URLS)} totali note)")

    texts = fetch_boj_opinions_texts()
    scratch_path = os.path.join(os.path.dirname(__file__), "data", "boj_opinions_raw.json")
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    ok = sum(1 for v in texts.values() if "text" in v)
    print(f"Testi scaricati: {ok}/{len(texts)}")
    print(f"Salvato in {scratch_path}")
