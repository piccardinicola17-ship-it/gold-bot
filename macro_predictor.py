"""
macro_predictor.py — Fase 5 del progetto dati storici: previsione statistica
live per gli eventi macro validati nella Fase 4 (oggi: solo "Core CPI m/m",
l'unico con edge confermato su tutti gli split cronologici testati — vedi
historical_model.py e MEMORY di sessione del 2-3 settembre 2026).

Fonti dati separate per ruolo, non per comodità:
- FairEconomy (già in uso da check_macro_alerts) resta la fonte per
  orario/previsione dell'evento — funziona, nessun motivo di cambiarla.
- FRED (Federal Reserve Bank di St. Louis) è la fonte per il valore ACTUAL
  una volta rilasciato — governativa, gratuita per sempre, nessuna API key
  richiesta. Scelta dopo che Financial Modeling Prep (candidato iniziale)
  si è rivelato a pagamento per l'endpoint economic-calendar nonostante la
  documentazione lasciasse intendere il contrario (verificato con una
  chiave reale, non solo sulla carta).

Il modello (pendenza Theil-Sen + statistiche storiche della sorpresa) è
pre-calcolato offline da historical_model.fit_final_model() e salvato in
data/macro_models.json — questo modulo si limita a caricarlo e applicarlo,
mai a ri-addestrarlo.

Onestà sui limiti: FRED aggiorna in genere entro alcune ore dal rilascio
ufficiale, non sempre entro i 10 minuti in cui check_macro_alerts controlla
la finestra post-evento — se il dato non è ancora disponibile, la funzione
ritorna None e il bot lo dice esplicitamente, non inventa nulla.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from historical_events import _parse_number
from trade_manager import load_fred_last_seen, save_fred_last_seen

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "macro_models.json"

# Mappa evento (come appare nel calendario FairEconomy/nel dataset storico
# di Fase 1) -> serie FRED da cui leggere l'actual una volta rilasciato.
FRED_SERIES = {
    "Core CPI m/m": "CPILFESL",
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


def _fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index()


def _new_release_mom_pct(series_id: str, last_seen: dict) -> tuple[str, float] | None:
    """
    Ritorna (data_iso, variazione_percentuale_mese_su_mese) se su FRED è
    comparso un punto dati più recente dell'ultimo già processato per
    questa serie, altrimenti None (nessuna novità, o FRED non ancora
    aggiornato).
    """
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(series_id)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    pos = s.index.get_loc(latest_date)
    if pos == 0:
        return None
    previous_value = s.iloc[pos - 1]
    current_value = s.iloc[pos]
    if pd.isna(previous_value) or previous_value == 0:
        return None

    mom_pct = (current_value - previous_value) / previous_value * 100
    return latest_date.isoformat(), round(float(mom_pct), 3)


def predict_reaction(event_name: str, forecast_raw: str) -> dict | None:
    """
    Se per event_name è appena comparso un nuovo dato reale su FRED (mai
    processato prima) e c'è un modello calibrato per quella serie, ritorna
    la previsione statistica; altrimenti None — nessuna previsione
    inventata quando manca un pezzo (modello assente, FRED non ancora
    aggiornato, forecast non numerico).

    Side effect: se trova un nuovo rilascio lo segna come processato
    (persistito), anche se poi la previsione non può essere completata
    per altri motivi — non ha senso ritentare all'infinito lo stesso
    numero già visto.
    """
    series_id = FRED_SERIES.get(event_name)
    model = _load_model(event_name)
    if not series_id or not model:
        return None

    last_seen = load_fred_last_seen()
    try:
        release = _new_release_mom_pct(series_id, last_seen)
    except Exception as e:
        logger.warning(f"FRED non raggiungibile per {series_id}: {e}")
        return None
    if release is None:
        return None

    release_date, actual_pct = release
    last_seen[series_id] = release_date
    save_fred_last_seen(last_seen)

    forecast_num, _ = _parse_number(forecast_raw)
    if forecast_num is None:
        logger.info(f"[{event_name}] Actual FRED disponibile ({actual_pct}%) ma forecast non numerico: {forecast_raw!r}")
        return None

    surprise_raw = actual_pct - forecast_num
    z = (surprise_raw - model["surprise_mean"]) / model["surprise_std"] if model["surprise_std"] else 0.0
    z = max(-model["surprise_zscore_clip"], min(model["surprise_zscore_clip"], z))
    predicted = model["slope"] * z + model["intercept"]

    return {
        "event_name": event_name,
        "actual_pct": actual_pct,
        "forecast_pct": forecast_num,
        "surprise_raw": round(surprise_raw, 3),
        "surprise_zscore": round(z, 2),
        "predicted_reaction_usd": round(float(predicted), 2),
        "horizon": model["horizon"],
        "n_historical": model["n"],
    }


def format_prediction(pred: dict) -> str:
    direction = "ribassista per l'oro" if pred["predicted_reaction_usd"] < 0 else "rialzista per l'oro"
    horizon_label = pred["horizon"].replace("reaction_", "").replace("m", " min")
    return (
        f"📐 *Previsione statistica* (n={pred['n_historical']} storici, {horizon_label})\n"
        f"Actual {pred['actual_pct']:+.2f}% vs Prev {pred['forecast_pct']:+.2f}% "
        f"(sorpresa z={pred['surprise_zscore']:+.1f})\n"
        f"Reazione attesa: *{pred['predicted_reaction_usd']:+.2f}$* — {direction}\n"
        f"_Stima statistica su dati storici, non una garanzia — margine d'errore reale, vedi Fase 4._"
    )
