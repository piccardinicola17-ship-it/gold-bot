"""
macro_predictor.py — Fase 5 del progetto dati storici: previsione statistica
live per gli eventi macro validati nella Fase 4 e con una fonte dati live
gratuita affidabile (oggi: "Core CPI m/m" e "Unemployment Claims" — vedi
historical_model.DEPLOYED_EVENTS e MEMORY di sessione del 2-3/6 settembre 2026).

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
# di Fase 1) -> serie FRED da cui leggere l'actual una volta rilasciato, e
# come interpretarne il valore:
# - "mom_pct": la serie FRED è un indice-livello, il modello è stato
#   allenato sulla variazione % mese su mese (es. Core CPI m/m).
# - "level": la serie FRED è già nella stessa unità del dataset di
#   training — un conteggio grezzo, MAI una % (es. Unemployment Claims,
#   "219K" nel calendario = 219000 sia in FRED sia nel dataset storico).
#   Confuso una volta con "mom_pct" darebbe una sorpresa completamente
#   diversa da quella su cui il modello è stato calibrato.
FRED_SERIES = {
    "Core CPI m/m":        {"series_id": "CPILFESL", "value_type": "mom_pct"},
    "Unemployment Claims": {"series_id": "ICSA",      "value_type": "level"},
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


def _new_release_level(series_id: str, last_seen: dict) -> tuple[str, float] | None:
    """
    Come _new_release_mom_pct ma per serie il cui valore è già un livello
    grezzo nel dataset di training (es. Unemployment Claims, "219K") —
    nessuna trasformazione: solo l'ultimo valore pubblicato su FRED, così
    com'è, comparabile direttamente al forecast (stessa unità).
    """
    s = _fetch_fred_series(series_id)
    if s.empty:
        return None
    latest_date = s.index.max()
    prev_seen = last_seen.get(series_id)
    if prev_seen and latest_date <= pd.Timestamp(prev_seen):
        return None

    current_value = s.iloc[-1]
    if pd.isna(current_value):
        return None
    return latest_date.isoformat(), float(current_value)


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
    series_cfg = FRED_SERIES.get(event_name)
    model = _load_model(event_name)
    if not series_cfg or not model:
        return None

    series_id = series_cfg["series_id"]
    value_type = series_cfg["value_type"]

    last_seen = load_fred_last_seen()
    try:
        if value_type == "level":
            release = _new_release_level(series_id, last_seen)
        else:
            release = _new_release_mom_pct(series_id, last_seen)
    except Exception as e:
        logger.warning(f"FRED non raggiungibile per {series_id}: {e}")
        return None
    if release is None:
        return None

    release_date, actual_value = release
    last_seen[series_id] = release_date
    save_fred_last_seen(last_seen)

    forecast_num, _ = _parse_number(forecast_raw)
    if forecast_num is None:
        logger.info(f"[{event_name}] Actual FRED disponibile ({actual_value}) ma forecast non numerico: {forecast_raw!r}")
        return None

    surprise_raw = actual_value - forecast_num
    z = (surprise_raw - model["surprise_mean"]) / model["surprise_std"] if model["surprise_std"] else 0.0
    z = max(-model["surprise_zscore_clip"], min(model["surprise_zscore_clip"], z))
    predicted = model["slope"] * z + model["intercept"]

    return {
        "event_name": event_name,
        "value_type": value_type,
        "actual_value": actual_value,
        "forecast_value": forecast_num,
        "surprise_raw": round(surprise_raw, 3),
        "surprise_zscore": round(z, 2),
        "predicted_reaction_usd": round(float(predicted), 2),
        "horizon": model["horizon"],
        "n_historical": model["n"],
    }


def format_prediction(pred: dict) -> str:
    direction = "ribassista per l'oro" if pred["predicted_reaction_usd"] < 0 else "rialzista per l'oro"
    horizon_label = pred["horizon"].replace("reaction_", "").replace("m", " min")

    if pred.get("value_type") == "level":
        actual_fmt   = f"{pred['actual_value']:,.0f}"
        forecast_fmt = f"{pred['forecast_value']:,.0f}"
        note_source = (
            "_L'actual è l'ultimo valore pubblicato da FRED (aggiornato entro circa un'ora "
            "dal rilascio ufficiale), non letto dal comunicato — in rari casi FRED può "
            "revisionare un dato dopo la pubblicazione originale._"
        )
    else:
        actual_fmt   = f"{pred['actual_value']:+.2f}%"
        forecast_fmt = f"{pred['forecast_value']:+.2f}%"
        note_source = (
            "_L'actual è ricalcolato da FRED, non letto dal comunicato ufficiale: può differire di "
            "±0.1pp in rari casi di revisione dell'indice — verificato empiricamente, ~20% dei mesi._"
        )

    return (
        f"📐 *Previsione statistica* (n={pred['n_historical']} storici, {horizon_label})\n"
        f"Actual {actual_fmt} vs Prev {forecast_fmt} "
        f"(sorpresa z={pred['surprise_zscore']:+.1f})\n"
        f"Reazione attesa: *{pred['predicted_reaction_usd']:+.2f}$* — {direction}\n"
        f"_Stima statistica su dati storici, non una garanzia — margine d'errore reale, vedi Fase 4._\n"
        f"{note_source}"
    )
