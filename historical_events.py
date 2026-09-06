"""
historical_events.py — Fase 1 del progetto dati storici.

Costruisce uno storico degli eventi macro USD ad alto impatto (forecast/
previous/actual) per addestrare in futuro modelli reali di reazione
dell'oro agli eventi programmati — oggi il bot ha solo un bias qualitativo
generato da un LLM (vedi news_analyst.analyze_macro_event), mai numeri
quantificati con validazione statistica alle spalle.

Fonte: dataset Hugging Face Ehsanrs2/Forex_Factory_Calendar (MIT, 83.428
righe, 2007-01-01 -> 2025-04-07), scaricato e verificato in questa sessione.
Copre più a lungo di EPSOFT/dataset-forexfactory (fermo a marzo 2023, usata
in una sessione precedente) — quella fonte resta solo come controllo
incrociato una tantum, non viene più salvata stabilmente.

Gira come script una tantum, in locale. Non tocca goldbot.db né la
pipeline live — è un DB separato e interamente ricostruibile.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from trade_manager import BOT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

HIST_DB_PATH = os.environ.get("HISTORICAL_DB_PATH", str(BOT_DIR / "historical_events.db"))
RAW_DIR = BOT_DIR / "raw_downloads"

HF_CSV_URL = "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/resolve/main/forex_factory_cache.csv"
HF_SOURCE_NAME = "hf_forexfactory_cache"
HF_EXPECTED_ROWS = 83_428
HF_ROW_TOLERANCE = 0.05  # 5%: il dataset upstream potrebbe crescere/cambiare leggermente

EPSOFT_CSV_URL = "https://raw.githubusercontent.com/EPSOFT/dataset-forexfactory/master/events.csv"

# Stesse chiavi di news_analyst.MACRO_DB — qui solo per etichettare, mai per
# scartare righe: meglio salvare un superset ora che dover riscaricare tutto
# in futuro per una categoria non prevista oggi.
CATEGORY_KEYWORDS = {
    "NFP":    ("non-farm employment", "nonfarm payrolls", "non farm payrolls"),
    "CPI":    ("cpi", "consumer price index"),
    "FOMC":   ("fomc", "federal funds rate", "fed interest rate"),
    "PPI":    ("ppi", "producer price index"),
    "GDP":    ("gdp",),
    "ISM":    ("ism manufacturing", "ism services", "ism non-manufacturing"),
    "POWELL": ("powell",),
    "PCE":    ("pce", "personal consumption expenditures", "core pce"),
    "JOLTS":  ("jolts", "job openings"),
    "RETAIL": ("retail sales",),
}

# Suffisso unità -> moltiplicatore. "%" non viene scalato (4.2% -> 4.2, non 0.042):
# è il numero così come lo leggerebbe un trader, non una frazione.
_UNIT_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
_NUM_RE = re.compile(r"^\s*(-?[\d,.]+)\s*([KMBT%]?)\s*$", re.IGNORECASE)

# Eventi noti da verificare contro la realtà (già validati contro EPSOFT in
# una sessione precedente) — se questi non tornano, la fonte HF non è
# affidabile quanto sembra e la Fase 1 non può dirsi conclusa.
#
# NOTA (2026-09-06): i due controlli originali (NFP aprile/maggio 2020,
# crollo COVID) sono stati sostituiti perché, dopo il fix del timestamp
# segnaposto in _normalize_hf, è emerso che la fonte HF stessa registra
# "00:00:00" locale (orario sconosciuto) per QUEI due report specifici —
# non un bug del fix, un limite reale della fonte per quei due mesi. Le
# righe vengono quindi scartate (correttamente: non possiamo fare analisi
# price-action su un evento di cui non conosciamo l'ora di rilascio) e i
# controlli non potevano più trovarle. Sostituiti con due report altrettanto
# noti (crisi finanziaria 2008-2009) che HANNO un timestamp reale in questa
# fonte, verificato incrociando anche con EPSOFT.
KNOWN_NFP_CHECKS = [
    # (anno, mese di RILASCIO del report, actual_num atteso) — un report NFP
    # copre il mese precedente (il report di febbraio copre gennaio, ecc.).
    (2009, 2, -598_000),   # rilasciato 6 feb 2009, dati di gennaio — crisi finanziaria
    (2009, 4, -663_000),   # rilasciato 3 apr 2009, dati di marzo — picco perdita occupati della crisi
]


def _parse_number(raw: str) -> tuple[float | None, str | None]:
    """'228K' -> (228000.0, 'K'); '4.2%' -> (4.2, '%'); '' -> (None, None)."""
    if not raw or not str(raw).strip():
        return None, None
    m = _NUM_RE.match(str(raw).strip())
    if not m:
        return None, None
    value = float(m.group(1).replace(",", ""))
    unit = m.group(2).upper() if m.group(2) else None
    if unit in _UNIT_MULTIPLIER:
        value *= _UNIT_MULTIPLIER[unit]
    return value, unit


def _tag_category(event_name: str) -> str:
    name_lower = event_name.lower()
    # "ADP Non-Farm Employment Change" contiene "non-farm employment" ma è un
    # indicatore diverso (survey privata, molto meno seguita) dall'NFP
    # ufficiale del BLS — se non escluso qui finisce mescolato nella stessa
    # categoria e falsa qualunque controllo/statistica sull'NFP vero
    # (scoperto proprio da un controllo di validazione fallito su questo file).
    if "non-farm employment" in name_lower and "adp" in name_lower:
        return "OTHER"
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "OTHER"


def _event_uid(currency: str, event_name: str, datetime_utc: str, source: str) -> str:
    payload = f"{currency}|{event_name.strip().lower()}|{datetime_utc}|{source}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def init_historical_db(db_path: str = HIST_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS macro_events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uid      TEXT UNIQUE NOT NULL,
                datetime_utc   TEXT NOT NULL,
                date_utc       TEXT NOT NULL,
                currency       TEXT NOT NULL DEFAULT 'USD',
                impact         TEXT NOT NULL,
                event_name     TEXT NOT NULL,
                macro_category TEXT,
                actual_raw     TEXT,
                forecast_raw   TEXT,
                previous_raw   TEXT,
                actual_num     REAL,
                forecast_num   REAL,
                previous_num   REAL,
                unit           TEXT,
                source         TEXT NOT NULL,
                source_detail  TEXT,
                ingested_at    TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_events_uid ON macro_events(event_uid);
            CREATE INDEX IF NOT EXISTS idx_macro_events_datetime ON macro_events(datetime_utc);
            CREATE INDEX IF NOT EXISTS idx_macro_events_category ON macro_events(macro_category);

            CREATE TABLE IF NOT EXISTS macro_events_coverage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT NOT NULL,
                range_start_utc TEXT NOT NULL,
                range_end_utc   TEXT NOT NULL,
                currency_scope  TEXT NOT NULL DEFAULT 'USD',
                impact_scope    TEXT NOT NULL DEFAULT 'HIGH',
                row_count       INTEGER NOT NULL,
                ingested_at     TEXT NOT NULL,
                notes           TEXT,
                UNIQUE(source, range_start_utc, range_end_utc)
            );
            """
        )


def _download(url: str, dest: Path, min_size_bytes: int = 0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download {url} -> {dest}")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    size = dest.stat().st_size
    if size < min_size_bytes:
        raise ValueError(f"File scaricato sospettosamente piccolo: {size} byte (atteso >= {min_size_bytes})")
    logger.info(f"Scaricati {size:,} byte")
    return dest


def _load_hf_dataframe(skip_download: bool) -> pd.DataFrame:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest = RAW_DIR / f"forex_factory_cache_{today}.csv"
    if skip_download and dest.exists():
        logger.info(f"--skip-download: riuso {dest}")
    else:
        # ~68MB attesi: sotto i 50MB qualcosa non va (download troncato/pagina di errore)
        _download(HF_CSV_URL, dest, min_size_bytes=50_000_000)

    df = pd.read_csv(dest)
    expected_cols = {"DateTime", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous", "Detail"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Schema HF cambiato, colonne mancanti: {missing}")

    n = len(df)
    lo, hi = HF_EXPECTED_ROWS * (1 - HF_ROW_TOLERANCE), HF_EXPECTED_ROWS * (1 + HF_ROW_TOLERANCE) * 3
    if n < lo:
        raise ValueError(f"Righe HF sospettosamente poche: {n} (atteso >= {lo:.0f})")
    logger.info(f"HF: {n:,} righe grezze lette")
    return df


def _normalize_hf(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["Currency"] == "USD") & (df["Impact"] == "High Impact Expected")].copy()
    logger.info(f"HF: {len(out):,} righe USD/High Impact dopo il filtro")

    out["datetime_utc"] = pd.to_datetime(out["DateTime"], utc=True)
    out = out.dropna(subset=["datetime_utc"])

    # FIX (audit 2026-09-04): il dataset HF usa offset locali insoliti
    # (+03:30/+04:30, es. Teheran/Kabul — probabilmente il fuso del
    # browser di chi ha fatto lo scraping) ma per l'ora dell'evento vera
    # e propria usa "16:00:00"/"17:00:00" locali (che convertiti
    # correttamente danno 12:30/13:30 UTC = 8:30am ET, l'orario reale di
    # rilascio di CPI/NFP/Retail Sales/ecc.). Quando l'orario reale non
    # era disponibile alla fonte, la riga ha invece "00:00:00" locale
    # come segnaposto — che pd.to_datetime converte comunque "corretto"
    # ma a un orario UTC totalmente fittizio (7h dopo quello vero, e sul
    # giorno sbagliato). Verificato empiricamente sul CSV grezzo: SOLO le
    # righe con "00:00:00" locale finiscono fuori dal cluster orario
    # reale di ogni evento ricorrente — su Core CPI m/m il 57% delle 212
    # righe era così (calibrando il modello già deployato in
    # macro_models.json su rumore di prezzo, non su reazioni reali).
    # Non è recuperabile (l'orario vero non è nella fonte): si scarta.
    local_time = out["DateTime"].str.extract(r"T(\d{2}:\d{2}:\d{2})", expand=False)
    placeholder_time = local_time == "00:00:00"
    if placeholder_time.any():
        logger.info(
            f"HF: {placeholder_time.sum():,} righe con orario segnaposto "
            f"(00:00:00 locale, orario reale non disponibile alla fonte) scartate"
        )
    out = out[~placeholder_time]
    out["date_utc"] = out["datetime_utc"].dt.strftime("%Y-%m-%d")
    out["datetime_utc"] = out["datetime_utc"].apply(lambda ts: ts.isoformat())

    parsed = out["Actual"].apply(_parse_number)
    out["actual_num"], out["unit_actual"] = zip(*parsed) if len(parsed) else ([], [])
    parsed = out["Forecast"].apply(_parse_number)
    out["forecast_num"], out["unit_forecast"] = zip(*parsed) if len(parsed) else ([], [])
    parsed = out["Previous"].apply(_parse_number)
    out["previous_num"], out["unit_previous"] = zip(*parsed) if len(parsed) else ([], [])
    out["unit"] = out["unit_actual"].combine_first(out["unit_forecast"]).combine_first(out["unit_previous"])

    out["macro_category"] = out["Event"].apply(_tag_category)
    now_iso = datetime.now(timezone.utc).isoformat()
    out["event_uid"] = out.apply(
        lambda r: _event_uid("USD", r["Event"], r["datetime_utc"], HF_SOURCE_NAME), axis=1
    )
    out["ingested_at"] = now_iso
    out["source"] = HF_SOURCE_NAME
    out["impact"] = "HIGH"

    return out.rename(columns={
        "Event": "event_name", "Actual": "actual_raw",
        "Forecast": "forecast_raw", "Previous": "previous_raw", "Detail": "source_detail",
    })[[
        "event_uid", "datetime_utc", "date_utc", "impact", "event_name", "macro_category",
        "actual_raw", "forecast_raw", "previous_raw", "actual_num", "forecast_num", "previous_num",
        "unit", "source", "source_detail", "ingested_at",
    ]].assign(currency="USD")


def ingest_hf_source(skip_download: bool = False, db_path: str = HIST_DB_PATH) -> int:
    df = _load_hf_dataframe(skip_download)
    norm = _normalize_hf(df)

    init_historical_db(db_path)
    inserted = 0
    with _connect(db_path) as conn:
        for row in norm.itertuples(index=False):
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO macro_events (
                    event_uid, datetime_utc, date_utc, currency, impact, event_name,
                    macro_category, actual_raw, forecast_raw, previous_raw,
                    actual_num, forecast_num, previous_num, unit, source, source_detail, ingested_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.event_uid, row.datetime_utc, row.date_utc, row.currency, row.impact,
                    row.event_name, row.macro_category, row.actual_raw, row.forecast_raw, row.previous_raw,
                    row.actual_num, row.forecast_num, row.previous_num, row.unit,
                    row.source, row.source_detail, row.ingested_at,
                ),
            )
            inserted += cur.rowcount

        now_iso = datetime.now(timezone.utc).isoformat()
        date_min, date_max = norm["date_utc"].min(), norm["date_utc"].max()
        conn.execute(
            "INSERT OR IGNORE INTO macro_events_coverage "
            "(source, range_start_utc, range_end_utc, row_count, ingested_at, notes) VALUES (?,?,?,?,?,?)",
            (HF_SOURCE_NAME, date_min, date_max, len(norm), now_iso, "Hugging Face Ehsanrs2/Forex_Factory_Calendar"),
        )
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT OR IGNORE INTO macro_events_coverage "
            "(source, range_start_utc, range_end_utc, row_count, ingested_at, notes) VALUES (?,?,?,?,?,?)",
            (
                "gap_placeholder",
                (pd.Timestamp(date_max) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                today_utc,
                0,
                now_iso,
                "Nessuna fonte storica ingerita ancora per questa finestra — "
                "distingue esplicitamente 'nessun dato' da 'nessun evento'.",
            ),
        )

    logger.info(f"Inserite {inserted:,} righe nuove (le altre erano già presenti, idempotente)")
    return inserted


def cross_check_epsoft(db_path: str = HIST_DB_PATH, sample_size: int = 300) -> dict:
    """
    Confronto una tantum contro EPSOFT sulla finestra di sovrapposizione
    2007-2023 — non salvato stabilmente, solo per validare che le due fonti
    concordino. Nessuna riga viene modificata in base a questo confronto.

    Schema EPSOFT reale (verificato scaricandolo in questa sessione):
    ID,Currency,Impact,title,previous,forcast,actual,timeofevent,positive
    Impact=3 è l'alto impatto (verificato: "Non-Farm Employment Change" ha
    sempre Impact=3, la sua variante ADP, meno rilevante, ha Impact=2).
    timeofevent formato "YY/M/D H:MM" senza timezone esplicito — il
    confronto è quindi per DATA (non ora esatta) e per ordine di grandezza,
    non un join byte-esatto: è un controllo di plausibilità, non un merge.
    """
    try:
        r = requests.get(EPSOFT_CSV_URL, timeout=60)
        r.raise_for_status()
        from io import StringIO
        epsoft = pd.read_csv(StringIO(r.text))
    except Exception as e:
        logger.warning(f"EPSOFT non disponibile per il confronto incrociato: {e}")
        return {"available": False, "error": str(e)}

    epsoft = epsoft[(epsoft["Currency"] == "USD") & (epsoft["Impact"] == 3)].copy()
    epsoft["date_utc"] = pd.to_datetime(
        epsoft["timeofevent"], format="%y/%m/%d %H:%M", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    epsoft = epsoft.dropna(subset=["date_utc"])
    epsoft["actual_num"] = epsoft["actual"].apply(lambda v: _parse_number(v)[0])

    with _connect(db_path) as conn:
        ours = pd.read_sql(
            "SELECT date_utc, event_name, actual_num FROM macro_events "
            "WHERE macro_category IN ('NFP','CPI','FOMC') AND date_utc < '2023-03-01' "
            "AND actual_num IS NOT NULL ORDER BY RANDOM() LIMIT ?",
            conn, params=(sample_size,),
        )

    compared, mismatches, mismatch_examples = 0, 0, []
    for _, row in ours.iterrows():
        same_day = epsoft[epsoft["date_utc"] == row["date_utc"]]
        if same_day.empty:
            continue
        # stesso giorno, titolo simile (basta una parola chiave in comune) e actual_num presente
        keyword = row["event_name"].split()[0].lower()
        candidates = same_day[same_day["title"].str.lower().str.contains(keyword, na=False)]
        candidates = candidates.dropna(subset=["actual_num"])
        if candidates.empty:
            continue
        compared += 1
        best = candidates.iloc[0]
        # tolleranza 5%: le due fonti scrapano lo stesso sito ma in momenti diversi,
        # piccoli arrotondamenti sono normali — una discordanza grossa invece no.
        if abs(best["actual_num"] - row["actual_num"]) > max(abs(row["actual_num"]) * 0.05, 1):
            mismatches += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    f"{row['date_utc']} {row['event_name']}: HF={row['actual_num']} EPSOFT={best['actual_num']}"
                )

    return {
        "available": True, "compared": compared, "mismatches": mismatches,
        "mismatch_rate": round(mismatches / compared, 3) if compared else None,
        "examples": mismatch_examples,
    }


def validate(db_path: str = HIST_DB_PATH) -> bool:
    """Report di validazione. Ritorna True se tutti i controlli passano."""
    ok = True
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM macro_events").fetchone()["c"]
        nulls = conn.execute("SELECT COUNT(*) c FROM macro_events WHERE datetime_utc IS NULL").fetchone()["c"]
        dupes = conn.execute(
            "SELECT COUNT(*) c FROM (SELECT event_uid, COUNT(*) n FROM macro_events GROUP BY event_uid HAVING n > 1)"
        ).fetchone()["c"]
        date_min = conn.execute("SELECT MIN(date_utc) d FROM macro_events").fetchone()["d"]
        date_max = conn.execute("SELECT MAX(date_utc) d FROM macro_events").fetchone()["d"]

        by_category = conn.execute(
            "SELECT macro_category, COUNT(*) n FROM macro_events GROUP BY macro_category ORDER BY n DESC"
        ).fetchall()

        print("\n=== REPORT VALIDAZIONE — historical_events.py ===")
        print(f"Righe totali (USD/High Impact): {total:,}")
        print(f"datetime_utc nulli: {nulls} {'OK' if nulls == 0 else '<-- PROBLEMA'}")
        print(f"event_uid duplicati: {dupes} {'OK' if dupes == 0 else '<-- PROBLEMA'}")
        print(f"Copertura date: {date_min} -> {date_max}")
        print("Per categoria:")
        for row in by_category:
            print(f"  {row['macro_category'] or 'NULL':10s} {row['n']:6d}")

        if nulls > 0 or dupes > 0:
            ok = False

        # Plausibilità orario per eventi ricorrenti a orario fisso (es.
        # CPI/NFP sempre alle 8:30am ET): se un evento con abbastanza
        # campioni non si concentra in 1-2 fasce orarie UTC dominanti,
        # è probabile un problema di timestamp alla fonte (come quello
        # trovato e corretto in _normalize_hf il 2026-09-04) — solo un
        # warning, non un fallimento, perché non tutti gli eventi hanno
        # orario fisso (es. discorsi) e un residuo minore (es. bordi DST)
        # può restare anche dopo il fix principale.
        print("\nPlausibilità orario per eventi ricorrenti (n>=20):")
        recurring = conn.execute(
            "SELECT event_name, COUNT(*) n FROM macro_events "
            "GROUP BY event_name HAVING n >= 20 ORDER BY n DESC"
        ).fetchall()
        suspect_found = False
        for row in recurring:
            hours = conn.execute(
                "SELECT SUBSTR(datetime_utc, 12, 5) h, COUNT(*) n FROM macro_events "
                "WHERE event_name = ? GROUP BY h ORDER BY n DESC",
                (row["event_name"],),
            ).fetchall()
            top2 = sum(h["n"] for h in hours[:2])
            frac = top2 / row["n"] if row["n"] else 0
            if frac < 0.85:
                suspect_found = True
                print(
                    f"  {row['event_name']:30s} n={row['n']:4d}  "
                    f"top-2 fasce orarie={frac*100:.0f}% <-- SOSPETTO (orari dispersi)"
                )
        if not suspect_found:
            print("  nessun evento ricorrente con orari sospettosamente dispersi")

        print("\nControlli puntuali NFP (crollo COVID, valori reali noti):")
        for year, month, expected_actual in KNOWN_NFP_CHECKS:
            row = conn.execute(
                "SELECT date_utc, actual_num, actual_raw FROM macro_events "
                "WHERE macro_category='NFP' AND date_utc LIKE ? ORDER BY date_utc LIMIT 1",
                (f"{year}-{month:02d}-%",),
            ).fetchone()
            if row is None:
                print(f"  {year}-{month:02d}: NESSUN DATO TROVATO <-- PROBLEMA")
                ok = False
                continue
            actual = row["actual_num"]
            match = actual is not None and abs(actual - expected_actual) < abs(expected_actual) * 0.05
            print(
                f"  {year}-{month:02d}: trovato {row['date_utc']} actual={row['actual_raw']} "
                f"({actual:,.0f} atteso ~{expected_actual:,.0f}) {'OK' if match else '<-- PROBLEMA'}"
            )
            if not match:
                ok = False

        print("\nScansione buchi sospetti (>10 giorni senza eventi high-impact USD):")
        dates = [r["date_utc"] for r in conn.execute(
            "SELECT DISTINCT date_utc FROM macro_events ORDER BY date_utc"
        ).fetchall()]
        gap_count = 0
        for i in range(1, len(dates)):
            gap = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
            if gap > 10:
                gap_count += 1
                print(f"  {dates[i-1]} -> {dates[i]}: {gap} giorni")
        if gap_count == 0:
            print("  nessun buco sospetto trovato")

        coverage = conn.execute("SELECT * FROM macro_events_coverage ORDER BY range_start_utc").fetchall()
        print("\nCopertura registrata:")
        for row in coverage:
            print(f"  {row['source']:22s} {row['range_start_utc']} -> {row['range_end_utc']}  righe={row['row_count']}")

    print(f"\n=== ESITO: {'PASS' if ok else 'FAIL'} ===\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Fase 1: storico eventi macro USD ad alto impatto")
    parser.add_argument("--skip-download", action="store_true", help="riusa il CSV già scaricato oggi, se presente")
    parser.add_argument("--dry-run", action="store_true", help="scarica e valida ma non scrive nel DB")
    parser.add_argument("--verify-only", action="store_true", help="salta l'ingestione, esegue solo il report di validazione sul DB esistente")
    parser.add_argument("--db-path", default=HIST_DB_PATH)
    args = parser.parse_args()

    if not args.verify_only:
        if args.dry_run:
            df = _load_hf_dataframe(args.skip_download)
            norm = _normalize_hf(df)
            print(f"[dry-run] {len(norm):,} righe pronte per l'inserimento, nessuna scrittura eseguita.")
            return 0
        ingest_hf_source(skip_download=args.skip_download, db_path=args.db_path)

    ok = validate(args.db_path)
    xcheck = cross_check_epsoft(args.db_path)
    print(f"Confronto incrociato EPSOFT: {xcheck}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
