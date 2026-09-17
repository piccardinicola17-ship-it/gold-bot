"""
Struttura di mercato XAU/USD (liquidità, FVG/inefficienze, order block) —
calcolo interamente deterministico su dati OHLC reali (stessa cascata di
fonti di analyzer.get_data), NESSUNA interpretazione dell'AI.

Aggiunto il 2026-09-17 su richiesta esplicita dell'utente: arricchire i
messaggi macro con un'analisi tecnica a grafico ("come si è mosso, se ha
lasciato liquidità, inefficienza") per bias più affidabili. La ricerca
già fatta su FOMC/BCE/BOJ (vedi memoria progetto) non ha trovato nessun
edge genuino chiedendo all'AI di "ragionare a parole" su un evento — lo
stesso rischio esisterebbe chiedendole di descrivere un grafico che non
vede davvero. Per questo qui i concetti SMC (liquidità = swing high/low
reali, FVG = gap a 3 candele reale, order block = ultima candela opposta
prima di un impulso forte) sono numeri calcolati da pandas, non un
racconto generato — e per ora sono SOLO testo informativo in più nei
messaggi Telegram: non entrano nel prompt dell'AI che genera bias/motivo
e non toccano le chiusure protettive. Vedi Fase 2 in memoria: solo se la
validazione statistica lo confermerà, potranno un giorno influenzare
anche il bias vero e proprio.
"""

from __future__ import annotations

import pandas as pd

# Stessa soglia di CONFIRM_THRESHOLD_USD in gold_bot.py (sotto, un
# movimento è rumore non un vero trend) — duplicata qui invece che
# importata per non creare un import circolare (gold_bot.py importa
# questo modulo, non il contrario).
_TREND_THRESHOLD_USD = 2.0


def _find_swing_highs(df: pd.DataFrame, window: int = 3) -> list[tuple]:
    """Un massimo locale: il high più alto tra `window` candele prima e
    dopo, STRETTAMENTE più alto degli altri (non solo pari) — altrimenti
    un tratto piatto di N candele identiche si marcherebbe da solo come
    swing high in ogni suo punto, un pool di liquidità che non esiste
    davvero. Esclude le ultime `window` candele (non ancora confermate
    come swing, potrebbero ancora essere superate dalla candela
    successiva)."""
    highs = df["High"].values
    n = len(highs)
    swings = []
    for i in range(window, n - window):
        segment = highs[i - window : i + window + 1]
        others_max = max(segment[:window].max(), segment[window + 1 :].max())
        if highs[i] > others_max:
            swings.append((df.index[i], float(highs[i])))
    return swings


def _find_swing_lows(df: pd.DataFrame, window: int = 3) -> list[tuple]:
    lows = df["Low"].values
    n = len(lows)
    swings = []
    for i in range(window, n - window):
        segment = lows[i - window : i + window + 1]
        others_min = min(segment[:window].min(), segment[window + 1 :].min())
        if lows[i] < others_min:
            swings.append((df.index[i], float(lows[i])))
    return swings


def _nearest_liquidity(df: pd.DataFrame, current_price: float, window: int = 3) -> dict:
    """Liquidità = pool di stop oltre l'ultimo swing high/low non ancora
    "preso". Ritorna lo swing più vicino sopra e sotto il prezzo attuale
    (quello che verrebbe cacciato per primo da un movimento in quella
    direzione)."""
    highs_above = [h for _, h in _find_swing_highs(df, window) if h > current_price]
    lows_below = [l for _, l in _find_swing_lows(df, window) if l < current_price]
    return {
        "above": min(highs_above) if highs_above else None,
        "below": max(lows_below) if lows_below else None,
    }


def _find_fvgs(df: pd.DataFrame) -> list[dict]:
    """Fair Value Gap (inefficienza): 3 candele consecutive dove la
    candela centrale si muove così forte da lasciare un vuoto tra l'high
    della prima e il low della terza (rialzista) o viceversa (ribassista)
    — un prezzo mai scambiato, che il mercato tende a "richiudere" in
    seguito."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    fvgs = []
    for i in range(1, n - 1):
        if highs[i - 1] < lows[i + 1]:
            fvgs.append({
                "type": "bullish", "zone_low": float(highs[i - 1]),
                "zone_high": float(lows[i + 1]), "idx": i,
            })
        elif lows[i - 1] > highs[i + 1]:
            fvgs.append({
                "type": "bearish", "zone_low": float(highs[i + 1]),
                "zone_high": float(lows[i - 1]), "idx": i,
            })
    return fvgs


def _fvg_is_open(df: pd.DataFrame, fvg: dict) -> bool:
    """"Aperta" = nessuna candela successiva alla formazione è mai
    rientrata nella zona — un minimo controllo conservativo: anche un solo
    tocco parziale la considera già "toccata", non più uno spazio vuoto."""
    after = df.iloc[fvg["idx"] + 2 :]
    if after.empty:
        return True
    if fvg["type"] == "bullish":
        return not (after["Low"] <= fvg["zone_high"]).any()
    return not (after["High"] >= fvg["zone_low"]).any()


def _last_open_fvg(df: pd.DataFrame, fvg_type: str) -> dict | None:
    candidates = [f for f in _find_fvgs(df) if f["type"] == fvg_type and _fvg_is_open(df, f)]
    return candidates[-1] if candidates else None


def _find_last_order_block(df: pd.DataFrame, direction: str, atr_mult: float = 1.5,
                            atr_window: int = 14) -> dict | None:
    """Order block semplificato: l'ultima candela di colore opposto prima
    di un impulso "forte" (corpo > atr_mult * ATR) nella direzione
    richiesta. Non validato oltre questo: nessun controllo se il livello
    è già stato "mitigato" da allora, per ora è solo un riferimento
    informativo nel messaggio, non un livello operativo."""
    if len(df) < atr_window + 2:
        return None
    highs, lows, closes, opens = df["High"], df["Low"], df["Close"], df["Open"]
    prev_close = closes.shift()
    true_range = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(atr_window).mean()

    last_ob = None
    for i in range(atr_window, len(df)):
        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        move = closes.iloc[i] - opens.iloc[i]
        if abs(move) <= atr_mult * atr_i:
            continue
        prev_open, prev_close_val = opens.iloc[i - 1], closes.iloc[i - 1]
        if direction == "bullish" and move > 0 and prev_close_val < prev_open:
            last_ob = {"low": float(prev_close_val), "high": float(prev_open), "idx": i - 1}
        elif direction == "bearish" and move < 0 and prev_close_val > prev_open:
            last_ob = {"low": float(prev_open), "high": float(prev_close_val), "idx": i - 1}
    return last_ob


def compute_market_structure(df: pd.DataFrame, current_price: float) -> dict:
    """Punto d'ingresso principale: tutti i concetti SMC calcolati in un
    colpo solo su un DataFrame OHLC (colonne Open/High/Low/Close, indice
    temporale crescente — stesso formato di analyzer.get_data)."""
    return {
        "liquidity": _nearest_liquidity(df, current_price),
        "fvg_bullish": _last_open_fvg(df, "bullish"),
        "fvg_bearish": _last_open_fvg(df, "bearish"),
        "order_block_bullish": _find_last_order_block(df, "bullish"),
        "order_block_bearish": _find_last_order_block(df, "bearish"),
    }


def format_market_structure(structure: dict, current_price: float, bias: str = "") -> str:
    """Blocco di testo pronto per Telegram — solo fatti (livelli e
    distanze), nessun giudizio o previsione: quello resta al bias
    dell'AI, calcolato separatamente e senza vedere questo blocco.

    `bias` (2026-09-17, richiesta esplicita dell'utente — "mi interessano
    solo le zone che vanno nella direzione del bias evento"): con BUY
    mostra solo liquidità sopra/FVG rialzista/order block rialzista (le
    zone coerenti con un movimento al rialzo), con SELL solo le
    corrispondenti ribassiste — le zone nella direzione opposta non sono
    tolte perché "sbagliate", semplicemente non interessano quando c'è
    già una direzione attesa. Con bias NEUTRO o non riconosciuto (nessuna
    direzione su cui filtrare) mostra tutto, come prima."""
    bias_upper = (bias or "").strip().upper()
    show_bullish = bias_upper != "SELL"
    show_bearish = bias_upper != "BUY"

    lines = []
    liq = structure.get("liquidity", {})
    if show_bullish and liq.get("above") is not None:
        lines.append(f"🧲 Liquidità sopra: ${liq['above']:.2f} (+{liq['above'] - current_price:.2f}$)")
    if show_bearish and liq.get("below") is not None:
        lines.append(f"🧲 Liquidità sotto: ${liq['below']:.2f} ({liq['below'] - current_price:.2f}$)")

    fvg_b = structure.get("fvg_bullish")
    if show_bullish and fvg_b:
        lines.append(f"⬜ FVG rialzista aperta: ${fvg_b['zone_low']:.2f}–${fvg_b['zone_high']:.2f}")
    fvg_s = structure.get("fvg_bearish")
    if show_bearish and fvg_s:
        lines.append(f"⬜ FVG ribassista aperta: ${fvg_s['zone_low']:.2f}–${fvg_s['zone_high']:.2f}")

    ob_b = structure.get("order_block_bullish")
    if show_bullish and ob_b:
        lines.append(f"🟩 Order block rialzista: ${ob_b['low']:.2f}–${ob_b['high']:.2f}")
    ob_s = structure.get("order_block_bearish")
    if show_bearish and ob_s:
        lines.append(f"🟥 Order block ribassista: ${ob_s['low']:.2f}–${ob_s['high']:.2f}")

    if not lines:
        return ""
    return "📐 *Struttura di mercato (15min)*\n" + "\n".join(lines)


def get_market_structure_snapshot(current_price: float, bias: str = "", interval: str = "15min",
                                   outputsize: int = 120) -> str:
    """Wrapper usato da gold_bot.py: scarica i dati reali e ritorna il
    blocco di testo pronto, oppure stringa vuota se qualcosa va storto —
    non deve MAI far fallire o ritardare l'invio dell'alert principale."""
    try:
        from analyzer import get_data
        df = get_data(interval=interval, outputsize=outputsize)
        if df is None or len(df) < 20:
            return ""
        structure = compute_market_structure(df, current_price)
        return format_market_structure(structure, current_price, bias)
    except Exception:
        return ""


def compute_pre_event_trend(df: pd.DataFrame, current_price: float, bars_back: int) -> dict | None:
    """Prezzo `bars_back` candele fa vs adesso — puro contesto oggettivo,
    NESSUN bias da confermare qui (non c'è una previsione da verificare
    su un movimento avvenuto prima ancora che uscisse la notizia).

    Aggiunto il 2026-09-17 su segnalazione dell'utente (Philly Fed
    Manufacturing Index + Unemployment Claims): l'oro era già in un
    forte rally per tutt'altro motivo quando è uscita la notizia, e il
    confronto puramente aritmetico pre/post-evento ha marcato
    "CONFERMATO" un bias BUY la cui reazione REALE era invece un sell —
    il trend preesistente ha semplicemente coperto la vera reazione.
    Questo blocco non risolve il problema (non isola l'effetto della
    singola notizia da un trend più ampio, non è possibile farlo con un
    solo prezzo prima/dopo), ma lo rende visibile a chi legge, invece di
    lasciarlo silenzioso come finora."""
    if df is None or len(df) <= bars_back:
        return None
    reference_price = float(df["Close"].iloc[-1 - bars_back])
    return {"reference_price": reference_price, "change": current_price - reference_price}


def format_pre_event_trend(trend: dict | None, lookback_minutes: int) -> str:
    if not trend:
        return ""
    change = trend["change"]
    sign = "+" if change >= 0 else ""
    if change > _TREND_THRESHOLD_USD:
        direction = "RIALZISTA"
    elif change < -_TREND_THRESHOLD_USD:
        direction = "RIBASSISTA"
    else:
        direction = "LATERALE"
    return (f"📈 Trend pre-evento (ultimi {lookback_minutes} min): "
            f"{sign}{change:.2f}$ ({direction})")


def get_pre_event_trend_snapshot(current_price: float, interval: str = "15min",
                                  outputsize: int = 120, lookback_minutes: int = 30) -> str:
    """Wrapper usato da gold_bot.py, stesso principio di
    get_market_structure_snapshot sopra: mai far fallire l'alert
    principale. bars_back assume candele da 15 minuti (stesso interval
    di default) — se interval cambiasse andrebbe ricalcolato di
    conseguenza, per ora non serve renderlo generico."""
    try:
        from analyzer import get_data
        df = get_data(interval=interval, outputsize=outputsize)
        if df is None or len(df) < 5:
            return ""
        bars_back = max(1, round(lookback_minutes / 15))
        trend = compute_pre_event_trend(df, current_price, bars_back)
        return format_pre_event_trend(trend, lookback_minutes)
    except Exception:
        return ""
