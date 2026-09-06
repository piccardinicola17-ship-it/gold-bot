"""
historical_model.py — Fase 4 del progetto dati storici.

Regressione lineare semplice (surprise_zscore -> reazione di prezzo) per
ciascuna serie macro della Fase 3, validata SEMPRE con uno split
cronologico train/test (mai mescolato: il test è sempre successivo al
training, per simulare una previsione genuina nel futuro, non un
indovinare il passato).

Perché lineare e non qualcosa di più sofisticato: i campioni disponibili
per serie sono poche centinaia (61-207) — un modello complesso (random
forest, rete neurale) su questi numeri farebbe overfitting quasi
garantito. La Fase 3 ha già mostrato che dove il segnale esiste (Core CPI
m/m) è lineare e persistente su più orizzonti; non c'è motivo di partire
con qualcosa di più complesso finché un modello semplice non ha
dimostrato di non bastare.

La pendenza si stima con Theil-Sen (mediana delle pendenze tra tutte le
coppie di punti), non con una least-squares classica: la prima versione
di questo script usava OLS e otteneva R² fuori scala (fino a -70) su
Non-Farm Employment Change e Federal Funds Rate. Causa trovata, non un
bug: eventi storici reali con sorprese enormi rispetto alla norma — NFP
giugno 2020 (previsti -7,75M posti persi, arrivati +2,5M: sorpresa di
10,26M, z-score 69, il crollo/rimbalzo Covid) e Fed Funds Rate giugno
2022 (sorpresa di 25 punti base in piena fase di rialzi aggressivi,
z-score 7). Sono outlier legittimi della storia economica, non errori
nei dati da correggere — ma una singola coppia del genere, essendo un
punto a leva enorme, può ribaltare da sola la pendenza di una
least-squares. Theil-Sen usa comunque tutti i punti (nessun dato scartato
o troncato), semplicemente non lascia che uno o due di essi dominino
il risultato.

Criterio di accettazione, deciso PRIMA di guardare i risultati (per non
raccontarsi la versione più comoda a posteriori): un modello si considera
utilizzabile solo se, fuori campione, batte SIA (a) un R² positivo SIA
(b) la baseline naive (prevedere sempre la media del training, l'unica
cosa nota senza guardare il futuro) — e se il risultato regge su più
split cronologici diversi, non solo su uno fortunato. Stesso standard di
rigore con cui M5/M15 sono stati tolti dal trading automatico dopo il
backtest dell'1 settembre 2026.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from historical_events import HIST_DB_PATH, _connect

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Più split cronologici diversi, non uno solo: un modello che sembra
# funzionare solo con un taglio specifico è probabilmente fortuna, non edge.
TRAIN_FRACTIONS = (0.5, 0.6, 0.7, 0.8)
MIN_TEST_SAMPLES = 10

# Theil-Sen rende robusta la stima della pendenza, ma un evento con uno
# z-score fuori scala (es. NFP giugno 2020, z=69: il rimbalzo Covid,
# previsti -7,75M posti persi, arrivati +2,5M) resta comunque un input
# enorme da cui estrapolare — se finisce nel test set, anche una pendenza
# ragionevole produce una previsione mostruosa su quel singolo punto,
# e l'errore quadratico di un solo punto del genere basta a far crollare
# l'R² sull'intero test set. Il winsorizing (pratica standard in
# econometria finanziaria) limita l'INPUT (lo z-score della sorpresa) a
# un intervallo plausibile: oltre ±4 deviazioni standard, la magnitudine
# esatta di una sorpresa non è comunque informativa in modo affidabile
# per un modello lineare — trattarla come "eccezionalmente positiva"
# invece che "esattamente 69 volte la norma" è più onesto, non un modo
# per far sparire il dato. La REAZIONE di prezzo osservata (y) non viene
# mai toccata: è il fatto reale che vogliamo spiegare/prevedere.
SURPRISE_ZSCORE_CLIP = 4.0

HORIZONS = ["reaction_1m", "reaction_5m", "reaction_15m", "reaction_30m", "reaction_60m"]


def _load(event_name: str, db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM event_features WHERE event_name=? AND surprise_zscore IS NOT NULL "
            "ORDER BY datetime_utc ASC",
            conn,
            params=(event_name,),
        )
    return df


def _r2(y_true: np.ndarray, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _theil_sen_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pendenza = mediana delle pendenze tra tutte le coppie di punti (i<j,
    x_i != x_j); intercetta = mediana di (y - pendenza*x). Robusta agli
    outlier per costruzione: un singolo punto a leva enorme (es. NFP
    giugno 2020) sposta la mediana molto meno di quanto sposterebbe una
    least-squares, che minimizza il quadrato degli errori e quindi pesa
    moltissimo proprio i punti più estremi.
    """
    n = len(x)
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        valid = dx != 0
        if valid.any():
            slopes.extend((dy[valid] / dx[valid]).tolist())
    if not slopes:
        return 0.0, float(np.median(y))
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _evaluate_split(df: pd.DataFrame, horizon: str, train_fraction: float) -> dict | None:
    d = df.dropna(subset=["surprise_zscore", horizon])
    n = len(d)
    split = int(n * train_fraction)
    train, test = d.iloc[:split], d.iloc[split:]
    if len(train) < 20 or len(test) < MIN_TEST_SAMPLES:
        return None

    x_train = np.clip(train["surprise_zscore"].to_numpy(), -SURPRISE_ZSCORE_CLIP, SURPRISE_ZSCORE_CLIP)
    x_test  = np.clip(test["surprise_zscore"].to_numpy(),  -SURPRISE_ZSCORE_CLIP, SURPRISE_ZSCORE_CLIP)
    y_train, y_test = train[horizon].to_numpy(), test[horizon].to_numpy()

    slope, intercept = _theil_sen_fit(x_train, y_train)
    y_pred = slope * x_test + intercept

    r2_model = _r2(y_test, y_pred)
    # Mediana, non media, per lo stesso motivo del fit: sulle serie con
    # outlier storici (NFP, Fed Funds Rate) la media del training è essa
    # stessa distorta da un singolo evento estremo — la mediana è la
    # baseline naive più onesta da battere.
    naive_pred = np.full_like(y_test, np.median(y_train))
    r2_naive = _r2(y_test, naive_pred)

    true_sign = np.sign(y_test)
    pred_sign = np.sign(y_pred)
    has_direction = true_sign != 0
    dir_acc = float((pred_sign[has_direction] == true_sign[has_direction]).mean()) if has_direction.sum() > 0 else float("nan")

    return {
        "train_fraction": train_fraction,
        "n_train": len(train),
        "n_test": len(test),
        "slope": float(slope),
        "r2_test": r2_model,
        "r2_naive_test": r2_naive,
        "beats_naive": bool(r2_model > r2_naive and r2_model > 0),
        "direction_accuracy": dir_acc,
    }


def evaluate_series(event_name: str, horizon: str, db_path: str = HIST_DB_PATH) -> list[dict]:
    df = _load(event_name, db_path)
    results = []
    for frac in TRAIN_FRACTIONS:
        r = _evaluate_split(df, horizon, frac)
        if r:
            r["event_name"] = event_name
            r["horizon"] = horizon
            results.append(r)
    return results


def run_all(db_path: str = HIST_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        series = [row[0] for row in conn.execute("SELECT DISTINCT event_name FROM event_features WHERE surprise_zscore IS NOT NULL")]

    rows = []
    for event_name in series:
        for horizon in HORIZONS:
            rows.extend(evaluate_series(event_name, horizon, db_path))
    return pd.DataFrame(rows)


def _total_n_for_series(group: pd.DataFrame) -> int:
    """Campione totale disponibile per una serie, sul suo orizzonte
    migliore. n_train+n_test è costante DENTRO uno stesso orizzonte
    (dipende solo da quante righe hanno dati validi per quell'orizzonte,
    non dal punto di split train/test) — prendere .max() indipendente
    sulle due colonne (bug corretto qui) le mescola tra orizzonte/split
    diversi e sovrastima il campione reale."""
    n_by_horizon = group.groupby("horizon").apply(
        lambda g: int(g["n_train"].iloc[0] + g["n_test"].iloc[0])
    )
    return int(n_by_horizon.max())


def summarize(results: pd.DataFrame, all_series: list[str] | None = None) -> None:
    if results.empty:
        print("Nessun risultato (dati insufficienti ovunque).")
        return

    print(f"{'Serie':32s} {'Orizzonte':14s} {'split':>6s} {'n_test':>7s} {'R2 modello':>11s} {'R2 naive':>9s} {'batte naive':>12s} {'dir.acc':>8s}")
    print("-" * 105)
    for _, r in results.sort_values(["event_name", "horizon", "train_fraction"]).iterrows():
        print(
            f"{r['event_name']:32s} {r['horizon']:14s} {r['train_fraction']:>6.1f} {r['n_test']:>7d} "
            f"{r['r2_test']:>+11.3f} {r['r2_naive_test']:>+9.3f} {str(r['beats_naive']):>12s} {r['direction_accuracy']:>8.1%}"
        )

    print()
    print("=== Combinazioni serie+orizzonte che battono la baseline naive su TUTTI gli split testati ===")
    consistent = (
        results.groupby(["event_name", "horizon"])["beats_naive"]
        .agg(["sum", "count"])
        .reset_index()
    )
    beats_all = consistent[consistent["sum"] == consistent["count"]]
    if beats_all.empty:
        print("Nessuna — nessuna combinazione regge su tutti gli split cronologici testati.")
    else:
        for _, row in beats_all.iterrows():
            print(f"  {row['event_name']} @ {row['horizon']} — regge su {row['count']}/{row['count']} split")

    print()
    print("=== Verdetto per serie ===")
    # Il campione minimo (n_train + n_test al split più piccolo, 0.5) decide se
    # un "no" è un verdetto affidabile o solo dati insufficienti per dirlo —
    # Federal Funds Rate (n totale 61) è troppo piccolo per un giudizio, a
    # differenza di NFP/CPI (111-220) dove un "nessun edge" pulito è un
    # risultato vero, non un limite di campionamento.
    series_with_results = set()
    for event_name, group in results.groupby("event_name"):
        series_with_results.add(event_name)
        total_n = _total_n_for_series(group)
        n_horizons_ok = (
            group.groupby("horizon")["beats_naive"].agg(lambda s: s.all()).sum()
        )
        if total_n < 100:
            verdict = f"DATI INSUFFICIENTI (n={total_n}, serve un campione più ampio prima di poter dire sì o no)"
        elif n_horizons_ok > 0:
            verdict = f"EDGE VALIDATO su {n_horizons_ok}/{len(HORIZONS)} orizzonti (n={total_n})"
        else:
            verdict = f"NESSUN EDGE — risultato pulito e affidabile (n={total_n})"
        print(f"  {event_name:32s} {verdict}")

    # FIX: una serie che non supera MAI la soglia minima per split
    # (n_train>=20/n_test>=10, vedi _evaluate_split) non compare in
    # `results` e prima spariva silenziosamente dal report — indistin-
    # guibile da "mai considerata". Ora, se il chiamante passa
    # all_series (l'universo completo da event_features), le serie
    # mancanti vengono elencate esplicitamente.
    if all_series is not None:
        missing = sorted(set(all_series) - series_with_results)
        for event_name in missing:
            print(f"  {event_name:32s} DATI INSUFFICIENTI (nessuno split valido, n troppo piccolo per ogni orizzonte)")


DEPLOY_HORIZON = "reaction_30m"


def fit_final_model(event_name: str, horizon: str = DEPLOY_HORIZON, db_path: str = HIST_DB_PATH) -> dict:
    """
    Calibra il modello da usare in produzione (Fase 5): pendenza Theil-Sen
    fittata su TUTTI i dati storici disponibili (non uno split di
    validazione — qui non stiamo più misurando quanto sia buono, quello
    l'ha già fatto run_all()/summarize(), stiamo preparando l'artefatto
    da usare davvero). Include anche media e deviazione standard storica
    della sorpresa grezza (actual-forecast): servono al bot live per
    calcolare lo z-score di un evento NUOVO con la stessa normalizzazione
    usata in Fase 3, dato che quell'evento non esiste ancora in
    event_features quando arriva in tempo reale.
    """
    with _connect(db_path) as conn:
        raw = pd.read_sql_query(
            "SELECT surprise_raw, surprise_zscore, " + horizon + " AS y "
            "FROM event_features WHERE event_name=? AND surprise_zscore IS NOT NULL",
            conn, params=(event_name,),
        )
    raw = raw.dropna(subset=["surprise_zscore", "y"])
    if len(raw) < 20:
        raise ValueError(f"Dati insufficienti per calibrare {event_name} (n={len(raw)})")

    x = np.clip(raw["surprise_zscore"].to_numpy(), -SURPRISE_ZSCORE_CLIP, SURPRISE_ZSCORE_CLIP)
    y = raw["y"].to_numpy()
    slope, intercept = _theil_sen_fit(x, y)

    return {
        "event_name": event_name,
        "horizon": horizon,
        "slope": slope,
        "intercept": intercept,
        "surprise_mean": float(raw["surprise_raw"].mean()),
        "surprise_std": float(raw["surprise_raw"].std()),
        "surprise_zscore_clip": SURPRISE_ZSCORE_CLIP,
        "n": int(len(raw)),
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
    }


# Nome serie -> orizzonte VALIDATO per QUELLA serie specifica (non più una
# costante globale unica): serie diverse possono validare a orizzonti
# diversi — Core CPI m/m regge su reaction_30m, Unemployment Claims solo
# su reaction_1m (vedi run_all()/summarize() del 2026-09-05). Solo le
# serie con edge validato in Fase 4 E una fonte dati live gratuita
# affidabile (vedi macro_predictor.FRED_SERIES) vanno qui.
DEPLOYED_EVENTS = {
    "Core CPI m/m": DEPLOY_HORIZON,
    "Unemployment Claims": "reaction_1m",
}


def regenerate_deployed_models(db_path: str = HIST_DB_PATH) -> None:
    """
    Rigenera macro_models.json (nella root del progetto, NON in data/ che è
    gitignored — questo file va invece committato e deployato col codice,
    lo legge macro_predictor.py in produzione). Da rilanciare se si
    aggiungono eventi alla Fase 2/3 o si estende DEPLOYED_EVENTS dopo aver
    validato una nuova serie in Fase 4.
    """
    import json
    from pathlib import Path

    # FIX: prima fittava sempre a DEPLOY_HORIZON alla cieca, senza
    # verificare che fosse davvero l'orizzonte che ha superato la
    # validazione (beats_naive su TUTTI gli split cronologici, vedi
    # summarize()) per quella specifica serie — una serie che valida solo
    # a un altro orizzonte (es. Unemployment Claims @ reaction_1m, non
    # reaction_30m) si sarebbe vista deployare un orizzonte mai validato,
    # senza alcun avviso. Ora ogni serie usa il proprio orizzonte
    # dichiarato in DEPLOYED_EVENTS e fallisce rumorosamente se non regge.
    results = run_all(db_path=db_path)
    models = {}
    for name, horizon in DEPLOYED_EVENTS.items():
        subset = results[(results["event_name"] == name) & (results["horizon"] == horizon)]
        if subset.empty or not subset["beats_naive"].all():
            raise ValueError(
                f"{name} @ {horizon}: non risulta validato su tutti gli split "
                f"cronologici testati (vedi summarize()) — non lo deploy alla cieca."
            )
        models[name] = fit_final_model(name, horizon=horizon, db_path=db_path)

    out_path = Path(__file__).resolve().parent / "macro_models.json"
    out_path.write_text(json.dumps(models, indent=2))
    logger.info(f"Modelli deployati rigenerati: {list(models)} -> {out_path}")


if __name__ == "__main__":
    import sys

    if "--regenerate-models" in sys.argv:
        regenerate_deployed_models()
    else:
        with _connect(HIST_DB_PATH) as _conn:
            all_series = [
                r[0] for r in _conn.execute(
                    "SELECT DISTINCT event_name FROM event_features WHERE surprise_zscore IS NOT NULL"
                )
            ]
        results = run_all()
        summarize(results, all_series=all_series)
