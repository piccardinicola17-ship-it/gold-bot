"""
historical_fomc_text.py — Progetto separato: FOMC hawkish/dovish scoring.

FOMC Statement/Minutes/Press Conference/Economic Projections non hanno mai
un "forecast" numerico da confrontare con l'"actual" (nessun consensus su
cosa dirà un comunicato) — quindi sono STRUTTURALMENTE fuori dal modello a
sorpresa di historical_model.py, per qualunque quantità di dati storici si
trovi. Questo modulo copre FOMC con un approccio diverso: un punteggio
hawkish/dovish del testo del comunicato, confrontato con la reazione di
prezzo reale (stessa tabella event_price_reactions già usata altrove).

Fasi:
1. fetch_fomc_statements() — scarica il testo ufficiale da
   federalreserve.gov per ogni data "FOMC Statement" già in macro_events
   (URL pattern verificato: /newsevents/pressreleases/monetary{YYYYMMDD}a.htm).
2. Punteggio hawkish/dovish — fatto a mano (lettura diretta, non un
   classificatore automatico) per il primo batch di 25 comunicati
   2011-2021, salvato in FOMC_HAWKISH_SCORES sotto. Scala -2 (fortemente
   dovish/allentamento) a +2 (fortemente hawkish/restrizione), 0 = nessun
   cambiamento di stance rispetto alla riunione precedente. Criteri:
   direzione della decisione sui tassi, variazioni nel ritmo di
   acquisto/riduzione titoli (QE/QT), cambi nella forward guidance,
   variazioni nel linguaggio sulla valutazione economica rispetto al
   comunicato PRECEDENTE (non in assoluto — il FOMC comunica per
   incrementi, un comunicato "identico" al precedente è neutro anche se il
   contenuto resta accomodante).
3. validate_fomc_scores() — Theil-Sen + split cronologico, stesso standard
   di historical_model.py, tra punteggio e reazione di prezzo reale — PUò
   GIRARE SOLO DOPO che dukascopy_ticks.py ha calcolato le reazioni per
   "FOMC Statement" (aggiunto a EXTENDED_EVENT_NAMES il 2026-09-09, ma il
   job in corso in quel momento non lo includeva ancora nella sua coda —
   serve un secondo giro di dukascopy_ticks.py --extended dopo che il
   primo (2.237 eventi) finisce).

Locale, non tocca il bot live — stesso principio di historical_events.py.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime, timezone

import requests

from historical_events import HIST_DB_PATH, _connect

FOMC_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm"

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


def validate_fomc_scores(db_path: str = HIST_DB_PATH) -> dict:
    """Confronta punteggio hawkish/dovish vs reazione di prezzo reale a 30
    minuti — stesso standard Theil-Sen + split cronologico di
    historical_model.py. Ritorna {"available": False, ...} se le reazioni
    di prezzo per FOMC Statement non sono ancora state calcolate."""
    import numpy as np
    import pandas as pd

    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT s.date_utc, s.score, r."price_t-1m" AS price_base, r."price_t+30m" AS price_30m
            FROM fomc_hawkish_scores s
            JOIN event_price_reactions r ON s.event_uid = r.event_uid
            WHERE r."price_t-1m" IS NOT NULL AND r."price_t+30m" IS NOT NULL
            ORDER BY s.date_utc
            """,
            conn,
        )

    if df.empty:
        return {
            "available": False,
            "reason": "Nessuna reazione di prezzo calcolata ancora per FOMC Statement — "
                      "serve rilanciare dukascopy_ticks.py --extended (aggiunto a "
                      "EXTENDED_EVENT_NAMES dopo l'avvio dell'ultimo giro in corso).",
        }

    df["reaction_30m"] = df["price_30m"] - df["price_base"]
    n = len(df)
    if n < 10:
        return {"available": True, "n": n, "verdict": f"DATI INSUFFICIENTI (n={n}, servono almeno 10 comunicati con reazione)"}

    x = df["score"].to_numpy(dtype=float)
    y = df["reaction_30m"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else None

    return {
        "available": True,
        "n": n,
        "correlation": round(corr, 3) if corr is not None else None,
        "note": "Punteggio negativo = dovish atteso -> reazione positiva per l'oro (correlazione negativa attesa se il segnale funziona)",
        "rows": df[["date_utc", "score", "reaction_30m"]].to_dict("records"),
    }


if __name__ == "__main__":
    n = save_fomc_scores()
    print(f"Punteggi salvati: {n}")
    result = validate_fomc_scores()
    print(result)
