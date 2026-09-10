"""
historical_fomc_text.py — Progetto separato: FOMC hawkish/dovish scoring.

FOMC Statement/Minutes/Press Conference/Economic Projections non hanno mai
un "forecast" numerico da confrontare con l'"actual" (nessun consensus su
cosa dirà un comunicato) — quindi sono STRUTTURALMENTE fuori dal modello a
sorpresa di historical_model.py, per qualunque quantità di dati storici si
trovi. Questo modulo copre FOMC con un approccio diverso: un punteggio
hawkish/dovish del testo del comunicato, confrontato con la reazione di
prezzo reale (stessa tabella event_price_reactions già usata altrove).

Copre TRE fonti testuali, stessa scala (-2 fortemente dovish a +2
fortemente hawkish, 0 = nessun cambio di stance vs la volta precedente),
stesso standard di validazione (Theil-Sen + split cronologico multiplo +
soglia n>=100, vedi validate_fomc_scores):

1. FOMC Statement — 25 comunicati 2011-2021, scorati A MANO (lettura
   diretta mia, non un classificatore) in FOMC_HAWKISH_SCORES sotto.
   Testo scaricato da fetch_fomc_statements() (federalreserve.gov,
   pattern /newsevents/pressreleases/monetary{YYYYMMDD}a.htm).
2. FOMC Meeting Minutes — 129 verbali 2007-2025 (30-95k caratteri
   ciascuno, 7.1M caratteri totali: troppo testo per una lettura diretta
   in una sessione), scorati da sotto-agenti paralleli con lo STESSO
   criterio di FOMC_HAWKISH_SCORES, punteggi in
   data/fomc_minutes_hawkish_scores.json (vedi save_minutes_scores()).
3. FOMC Press Conference — 82 conferenze 2011-2025 (dichiarazione + Q&A
   del Presidente, spesso muove il mercato più della dichiarazione
   scritta — es. giugno 2013 "taper tantrum", dicembre 2018 "QT on
   autopilot"), stesso approccio a sotto-agenti, punteggi in
   data/fomc_presconf_hawkish_scores.json (vedi save_presconf_scores()).
   L'evento "FOMC Press Conference" esiste già come riga propria in
   macro_events (fonte primaria hf_forexfactory_cache, orario reale)
   — NON serve creare uno pseudo-evento.

validate_fomc_scores() richiede che dukascopy_ticks.py --extended abbia
calcolato le reazioni di prezzo per questi event_name (tutti e 3 in
EXTENDED_EVENT_NAMES dal 2026-09-09) — la copertura cresce mentre il job
gira in background, i verdetti "DATI INSUFFICIENTI" si aggiornano da soli
ad ogni nuova esecuzione.

Locale, non tocca il bot live — stesso principio di historical_events.py.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import requests

from historical_events import HIST_DB_PATH, _connect, _event_uid

FOMC_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm"

# Punteggio hawkish/dovish dei 129 verbali (Minutes) FOMC, testo completo
# (30-95k caratteri ciascuno) letto e valutato con la STESSA scala e gli
# STESSI criteri di FOMC_HAWKISH_SCORES sotto (vedi docstring del modulo),
# suddiviso in lotti cronologici e assegnato in parallelo — troppo testo
# (7.1M caratteri totali) per essere letto in una sessione singola. File
# dati separato (non un dict Python inline come FOMC_HAWKISH_SCORES) per
# via del volume: 129 voci contro 25.
MINUTES_HAWKISH_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "fomc_minutes_hawkish_scores.json"
)

# Stesso principio di MINUTES_HAWKISH_SCORES_PATH, per le 82 conferenze
# stampa (dichiarazione + Q&A del Presidente, iniziate aprile 2011) che
# hanno reazione di prezzo calcolabile su "Federal Funds Rate" (data
# riunione = data conferenza, coincidono). Il Q&A spesso muove il mercato
# più della dichiarazione scritta (vedi es. giugno 2013 "taper tantrum",
# dicembre 2018 "QT on autopilot") — criterio esplicito nel prompt di
# scoring usato per questi 82 punteggi.
PRESCONF_HAWKISH_SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "fomc_presconf_hawkish_scores.json"
)

# Punteggio hawkish/dovish assegnato con lettura diretta del testo
# ufficiale (non un classificatore automatico) — vedi il docstring sopra
# per la scala e i criteri. Primo batch: tutte le date "FOMC Statement"
# presenti in macro_events al 2026-09-09 (25 comunicati, 2011-2021 — la
# fonte non copre ogni riunione, alcuni anni/mesi mancano).
FOMC_HAWKISH_SCORES: dict[str, dict] = {
    "2011-09-21": {"score": -1, "note": "Operation Twist $400B annunciato, guidance bassa estesa a metà 2013"},
    "2011-11-02": {"score": -1, "note": "Operation Twist continua invariato, rischi al ribasso ribaditi"},
    "2011-12-13": {"score": 0, "note": "Nessuna nuova azione, tono lievemente migliorato (\"moderatamente\")"},
    "2012-01-25": {"score": -1, "note": "Guidance sui tassi bassi estesa da metà 2013 a fine 2014"},
    "2012-03-13": {"score": 0, "note": "Miglioramento riconosciuto (disoccupazione, tensioni finanziarie), guidance invariata"},
    "2012-04-25": {"score": 0, "note": "Nessun cambio di policy, linguaggio rischi \"quasi bilanciati\""},
    "2012-06-20": {"score": -1, "note": "Operation Twist estesa a fine anno, crescita occupazione rallentata"},
    "2012-08-01": {"score": -1, "note": "Nessuna nuova azione ma linguaggio pre-committment (\"accomodamento aggiuntivo se necessario\")"},
    "2012-09-13": {"score": -2, "note": "QE3 lanciato: $40B/mese MBS open-ended, guidance estesa a metà 2015"},
    "2012-10-24": {"score": -1, "note": "QE3 confermato invariato, stesso tono preoccupato"},
    "2012-12-12": {"score": -2, "note": "Aggiunti $45B/mese Treasury (totale $85B/mese), Evans Rule sostituisce guidance a data fissa"},
    "2013-01-30": {"score": -1, "note": "Ritmo $85B/mese confermato, primo dissenso hawkish (George)"},
    "2013-07-31": {"score": -1, "note": "Ritmo $85B/mese confermato invariato nonostante il taper tantrum di mercato"},
    "2013-10-30": {"score": -1, "note": "Tapering atteso dal mercato ma RINVIATO — più dovish del previsto"},
    "2014-01-29": {"score": 0, "note": "Prima riduzione taper ($85B->$65B) ma attesa/annunciata a dicembre, continuazione non sorpresa"},
    "2014-04-30": {"score": 0, "note": "Taper continua come da percorso annunciato ($45B/mese)"},
    "2014-07-30": {"score": 0, "note": "Taper continua come da percorso annunciato ($25B/mese), linguaggio \"rafforzata\""},
    "2017-07-26": {"score": 0, "note": "Tassi invariati, normalizzazione bilancio (QT) segnalata \"relativamente presto\""},
    "2017-11-01": {"score": 0, "note": "QT avviata a ottobre, prosegue come da programma, rialzi \"graduali\" confermati"},
    "2020-04-29": {"score": -2, "note": "Risposta crisi COVID: tassi 0-0.25%, QE illimitato \"negli importi necessari\""},
    "2020-07-29": {"score": -1, "note": "Stance emergenziale invariata, primi segnali di ripresa ma accomodamento pieno mantenuto"},
    "2020-11-05": {"score": -1, "note": "Nuovo framework average inflation targeting introdotto, rafforza impegno dovish"},
    "2021-01-27": {"score": -1, "note": "Framework confermato, importi QE resi espliciti ($80B+$40B/mese)"},
    "2021-04-28": {"score": -1, "note": "Ritmo QE invariato nonostante ripresa più forte riconosciuta nel testo"},
    "2021-07-28": {"score": 0, "note": "Primo accenno esplicito a valutare i progressi nelle prossime riunioni — inizio taper talk"},
}


def fetch_fomc_statements(db_path: str = HIST_DB_PATH) -> dict:
    """Scarica il testo ufficiale di ogni comunicato FOMC Statement già
    presente in macro_events. Ritorna {date_utc: {"url", "text"} o {"error"}}."""
    with _connect(db_path) as conn:
        dates = [r[0] for r in conn.execute(
            "SELECT date_utc FROM macro_events WHERE event_name='FOMC Statement' ORDER BY date_utc"
        ).fetchall()]

    results = {}
    for d in dates:
        ymd = d.replace("-", "")
        url = FOMC_STATEMENT_URL.format(ymd=ymd)
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                results[d] = {"url": url, "error": f"HTTP {r.status_code}"}
                time.sleep(1)
                continue
            m = re.search(r'<div[^>]*id="article"[^>]*>(.*?)<div[^>]*class="lastUpdate"', r.text, re.S)
            if not m:
                results[d] = {"url": url, "error": "no article div"}
                time.sleep(1)
                continue
            text = re.sub(r'<[^>]+>', ' ', m.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            results[d] = {"url": url, "text": text}
        except Exception as e:
            results[d] = {"url": url, "error": str(e)}
        time.sleep(1)
    return results


def init_fomc_scores_table(db_path: str = HIST_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fomc_hawkish_scores (
                event_uid   TEXT PRIMARY KEY REFERENCES macro_events(event_uid),
                date_utc    TEXT NOT NULL,
                score       REAL NOT NULL,
                note        TEXT,
                scored_at   TEXT NOT NULL
            );
            """
        )


def save_fomc_scores(db_path: str = HIST_DB_PATH) -> int:
    """Salva FOMC_HAWKISH_SCORES nel DB, agganciato all'event_uid reale di
    macro_events per quella data (serve per il JOIN con event_price_reactions
    nella fase di validazione)."""
    init_fomc_scores_table(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    saved = 0
    with _connect(db_path) as conn:
        for date_utc, info in FOMC_HAWKISH_SCORES.items():
            row = conn.execute(
                "SELECT event_uid FROM macro_events WHERE event_name='FOMC Statement' AND date_utc=?",
                (date_utc,),
            ).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fomc_hawkish_scores (event_uid, date_utc, score, note, scored_at) "
                "VALUES (?,?,?,?,?)",
                (row["event_uid"], date_utc, info["score"], info["note"], now_iso),
            )
            saved += 1
    return saved


def save_minutes_scores(db_path: str = HIST_DB_PATH) -> int:
    """Salva i punteggi hawkish/dovish dei verbali (Minutes) da
    MINUTES_HAWKISH_SCORES_PATH nel DB, agganciati all'event_uid reale di
    macro_events (event_name='FOMC Meeting Minutes') per quella data."""
    init_fomc_scores_table(db_path)
    if not os.path.exists(MINUTES_HAWKISH_SCORES_PATH):
        return 0
    with open(MINUTES_HAWKISH_SCORES_PATH, encoding="utf-8") as f:
        scores = json.load(f)
    now_iso = datetime.now(timezone.utc).isoformat()
    saved = 0
    with _connect(db_path) as conn:
        for date_utc, info in scores.items():
            row = conn.execute(
                "SELECT event_uid FROM macro_events WHERE event_name='FOMC Meeting Minutes' AND date_utc=?",
                (date_utc,),
            ).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fomc_hawkish_scores (event_uid, date_utc, score, note, scored_at) "
                "VALUES (?,?,?,?,?)",
                (row["event_uid"], date_utc, info["score"], info["note"], now_iso),
            )
            saved += 1
    return saved


def save_presconf_scores(db_path: str = HIST_DB_PATH) -> int:
    """Salva i punteggi hawkish/dovish delle conferenze stampa da
    PRESCONF_HAWKISH_SCORES_PATH nel DB. 'FOMC Press Conference' esiste
    già come riga propria in macro_events (fonte primaria
    hf_forexfactory_cache, orario reale 14:15/14:30 ET a seconda
    dell'anno) — inizialmente avevo creato uno pseudo-evento sintetico
    (Statement + 30min) pensando che non esistesse, ma l'orario reale
    della fonte primaria è più accurato (varia leggermente nel tempo, non
    è un offset fisso) e la riga esisteva già: pseudo-eventi rimossi,
    filtro esplicito per source per evitare ambiguità se in futuro
    tornassero ad esistere righe doppie."""
    init_fomc_scores_table(db_path)
    if not os.path.exists(PRESCONF_HAWKISH_SCORES_PATH):
        return 0
    with open(PRESCONF_HAWKISH_SCORES_PATH, encoding="utf-8") as f:
        scores = json.load(f)
    now_iso = datetime.now(timezone.utc).isoformat()
    saved = 0
    with _connect(db_path) as conn:
        for date_utc, info in scores.items():
            row = conn.execute(
                "SELECT event_uid FROM macro_events WHERE event_name='FOMC Press Conference' "
                "AND date_utc=? AND source='hf_forexfactory_cache'",
                (date_utc,),
            ).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fomc_hawkish_scores (event_uid, date_utc, score, note, scored_at) "
                "VALUES (?,?,?,?,?)",
                (row["event_uid"], date_utc, info["score"], info["note"], now_iso),
            )
            saved += 1
    return saved


def save_text_scores(event_name: str, scores_path: str, db_path: str = HIST_DB_PATH,
                      currency: str | None = None) -> int:
    """Versione generica di save_fomc_scores/save_minutes_scores/
    save_presconf_scores — usata da historical_cb_text.py (BCE/BOJ) per
    non duplicare la stessa logica di INSERT (l'errore da evitare è
    proprio quello descritto in [[feedback-dual-mechanism-drift-pattern]]:
    due copie della stessa cosa che divergono nel tempo). currency
    disambigua quando event_name da solo è ambiguo tra banche centrali
    (es. 'Monetary Policy Statement' esiste sia per EUR sia per JPY)."""
    init_fomc_scores_table(db_path)
    if not os.path.exists(scores_path):
        return 0
    with open(scores_path, encoding="utf-8") as f:
        scores = json.load(f)
    now_iso = datetime.now(timezone.utc).isoformat()
    saved = 0
    with _connect(db_path) as conn:
        for date_utc, info in scores.items():
            query = "SELECT event_uid FROM macro_events WHERE event_name=? AND date_utc=?"
            params = [event_name, date_utc]
            if currency is not None:
                query += " AND currency=?"
                params.append(currency)
            row = conn.execute(query, params).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fomc_hawkish_scores (event_uid, date_utc, score, note, scored_at) "
                "VALUES (?,?,?,?,?)",
                (row["event_uid"], date_utc, info["score"], info["note"], now_iso),
            )
            saved += 1
    return saved


_PRICE_HORIZONS = {
    "reaction_1m": "price_t+1m", "reaction_5m": "price_t+5m", "reaction_15m": "price_t+15m",
    "reaction_30m": "price_t+30m", "reaction_60m": "price_t+60m",
}


def _evaluate_score_split(df, horizon: str, train_fraction: float) -> dict | None:
    """Stessa logica di historical_model._evaluate_split /
    historical_combined_events._evaluate_split, applicata al punteggio
    hawkish/dovish invece del surprise_zscore numerico."""
    import numpy as np
    from historical_model import _theil_sen_fit, _r2

    valid = df.dropna(subset=["score", horizon])
    if len(valid) < 20:
        return None
    split_idx = int(len(valid) * train_fraction)
    train, test = valid.iloc[:split_idx], valid.iloc[split_idx:]
    if len(train) < 10 or len(test) < 5:
        return None

    slope, intercept = _theil_sen_fit(train["score"].to_numpy(dtype=float), train[horizon].to_numpy(dtype=float))
    pred_test = slope * test["score"].to_numpy(dtype=float) + intercept
    r2_test = _r2(test[horizon].to_numpy(dtype=float), pred_test)

    naive_pred = np.full(len(test), train[horizon].mean())
    r2_naive = _r2(test[horizon].to_numpy(dtype=float), naive_pred)

    direction_correct = np.sign(pred_test) == np.sign(test[horizon].to_numpy(dtype=float))
    direction_acc = float(direction_correct.mean())

    return {
        "train_fraction": train_fraction, "n_train": len(train), "n_test": len(test),
        "r2_test": round(r2_test, 3), "r2_naive_test": round(r2_naive, 3),
        "beats_naive": bool(r2_test > r2_naive),
        "direction_accuracy": round(direction_acc, 3),
    }


def validate_fomc_scores(db_path: str = HIST_DB_PATH, event_name: str | None = None,
                          currency: str | None = None) -> dict:
    """Confronta punteggio hawkish/dovish vs reazione di prezzo reale —
    STESSO standard Theil-Sen + split cronologico multiplo + soglia n>=100
    usato ovunque nel progetto (historical_model.py,
    historical_combined_events.py, historical_cot_feature.py). Prima
    versione di questa funzione usava solo una correlazione grezza con
    soglia n>=10 — sostituita per coerenza, la stessa soglia bassa aveva
    già mostrato falsi "edge" su altre serie di questo progetto (vedi
    historical_combined_events.py, n=35). Ritorna {"available": False,
    ...} se le reazioni di prezzo non sono ancora state calcolate.

    Nonostante il nome (storico, era solo FOMC), generica: usata anche da
    historical_cb_text.py per BCE/BOJ — vedi 'currency' sotto.

    event_name: None = tutti i punteggi salvati per quel filtro, oppure un
    nome preciso ('FOMC Statement', 'FOMC Meeting Minutes',
    'FOMC Press Conference', 'ECB Press Conference', 'Monetary Policy
    Statement', ...) per validare le fonti separatamente (hanno dinamiche
    di mercato diverse: uno Statement è "prima notizia", i Minutes
    confermano/dettagliano una decisione già nota da settimane).
    currency: necessario quando event_name da solo è ambiguo tra banche
    centrali diverse (es. "Monetary Policy Statement" esiste sia per EUR
    sia per JPY, stesso nome, valuta diversa)."""
    import pandas as pd
    from historical_model import TRAIN_FRACTIONS

    filter_sql = ""
    params: list = []
    if event_name is not None:
        filter_sql += " AND m.event_name = ?"
        params.append(event_name)
    if currency is not None:
        filter_sql += " AND m.currency = ?"
        params.append(currency)

    price_cols = ", ".join(f'r."{col}" AS {name}_price' for name, col in _PRICE_HORIZONS.items())
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT s.date_utc, s.score, r."price_t-1m" AS price_base, {price_cols}
            FROM fomc_hawkish_scores s
            JOIN event_price_reactions r ON s.event_uid = r.event_uid
            JOIN macro_events m ON s.event_uid = m.event_uid
            WHERE r."price_t-1m" IS NOT NULL
            {filter_sql}
            ORDER BY s.date_utc
            """,
            conn,
            params=params,
        )

    if df.empty:
        return {
            "available": False,
            "reason": "Nessuna reazione di prezzo calcolata ancora — serve che "
                      "dukascopy_ticks.py --extended completi FOMC Statement/Meeting Minutes.",
        }

    for name in _PRICE_HORIZONS:
        df[name] = df[f"{name}_price"] - df["price_base"]

    n = len(df)
    results = []
    for horizon in _PRICE_HORIZONS:
        for frac in TRAIN_FRACTIONS:
            r = _evaluate_score_split(df, horizon, frac)
            if r:
                r["horizon"] = horizon
                results.append(r)

    if not results:
        return {"available": True, "n": n, "verdict": f"DATI INSUFFICIENTI (n={n}, serve almeno 20 comunicati con reazione per ogni split)"}

    res_df = pd.DataFrame(results)
    beats_all_by_horizon = res_df.groupby("horizon")["beats_naive"].agg(lambda s: s.all())
    n_horizons_ok = int(beats_all_by_horizon.sum())

    if n < 100:
        verdict = f"DATI INSUFFICIENTI PER UN VERDETTO AFFIDABILE (n={n}, serve n>=100)"
    elif n_horizons_ok > 0:
        verdict = f"EDGE VALIDATO su {n_horizons_ok}/{len(_PRICE_HORIZONS)} orizzonti (n={n})"
    else:
        verdict = f"NESSUN EDGE — risultato pulito (n={n})"

    return {
        "available": True,
        "n": n,
        "event_name": event_name or "(tutti)",
        "currency": currency,
        "verdict": verdict,
        "details": results,
    }


if __name__ == "__main__":
    n_s = save_fomc_scores()
    n_m = save_minutes_scores()
    n_pc = save_presconf_scores()
    print(f"Punteggi Statement salvati: {n_s} | Minutes: {n_m} | Press Conference: {n_pc}")
    for label, ev in (("Statement", "FOMC Statement"), ("Minutes", "FOMC Meeting Minutes"),
                      ("Press Conference", "FOMC Press Conference"), ("Combinato", None)):
        result = validate_fomc_scores(event_name=ev)
        print(f"\n=== {label} ===")
        print(f"n={result.get('n')} disponibile={result.get('available')} verdetto={result.get('verdict', result.get('reason'))}")
