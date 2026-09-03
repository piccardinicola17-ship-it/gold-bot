"""
agent_orchestrator.py — Multi-Agent Orchestration (Fase 6)
Pipeline sequenziale di 5 agenti specializzati che collaborano
per prendere ogni decisione di trading su XAU/USD.

Agenti:
  1. DataCollector     — raccoglie dati multi-TF e stato mercato
  2. StructureAnalyst  — analizza SMC, regime, setup validi
  3. RiskAgent         — verifica regole risk management
  4. NewsAgent         — controlla news e calendario economico
  5. DecisionMaker     — sintesi finale → EXECUTE / WAIT / SKIP

Ogni agente produce un "AgentResult" con stato, dati, e log.
Il tutto gira senza LangGraph — pipeline Python pura, zero dipendenze extra.
"""

import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pytz

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

# Cache candele condivisa tra tutti i TF nello stesso ciclo di 5 minuti.
# Evita di scaricare le stesse candele più volte (es. MTF che richiede M5 da H1 pipeline).
# Si svuota ogni 4 minuti automaticamente.
import time as _time
_candle_cache: dict = {}
_candle_cache_ts: dict = {}
CANDLE_CACHE_TTL = 240  # 4 minuti

def _get_cached_data(interval: str, outputsize: int):
    """Ritorna candele dalla cache se fresche, altrimenti None."""
    key = f"{interval}_{outputsize}"
    now = _time.time()
    if key in _candle_cache and now - _candle_cache_ts.get(key, 0) < CANDLE_CACHE_TTL:
        return _candle_cache[key]
    return None

def _set_cached_data(interval: str, outputsize: int, df):
    key = f"{interval}_{outputsize}"
    _candle_cache[key] = df
    _candle_cache_ts[key] = _time.time()

# Soglia confidenza AI per eseguire il trade
AI_CONFIDENCE_THRESHOLD = 65


# ── STATO CONDIVISO ───────────────────────────────────────────────────────────

@dataclass
class TradingState:
    """
    Stato condiviso che viene arricchito da ogni agente in sequenza.
    Inizia vuoto e si riempie man mano che la pipeline avanza.
    """
    # Input
    symbol:        str   = "XAU/USD"
    timeframe:     str   = "5min"
    timestamp:     str   = ""

    # Agente 1 — dati mercato
    current_price: float        = 0.0
    market_data:   dict         = field(default_factory=dict)
    data_timestamp: str         = ""

    # Agente 2 — analisi struttura
    signal:        str          = "NEUTRAL"   # BUY / SELL / NEUTRAL
    order_type:    str          = ""
    entry:         float        = 0.0
    sl:            float        = 0.0
    tp1:           float        = 0.0
    tp2:           float        = 0.0
    tp3:           float        = 0.0
    prob:          int          = 0
    regime:        str          = ""
    rr:            float        = 0.0
    structure_ok:  bool         = False
    # Livello del precedente swing high (BUY) / swing low (SELL) sul
    # timeframe del segnale, misurato al momento dell'apertura. Se il prezzo
    # lo rompe a favore prima di TP1, il monitor arma il break-even in
    # anticipo (vedi trade_manager._monitor_single) invece di aspettare il
    # target pieno — pensato per contenere il drawdown di H4/D1.
    early_be_level: float       = 0.0
    strategies:    dict         = field(default_factory=dict)

    # Agente 3 — risk
    risk_ok:           bool     = False
    risk_pct:          float    = 1.0
    risk_reason:       str      = ""
    trades_today:      int      = 0
    consecutive_loss:  int      = 0

    # Agente 4 — news
    news_safe:     bool         = True
    news_reason:   str          = ""
    news_summary:  str          = ""
    high_impact:   bool         = False
    news_error:    bool         = False

    # Agente 5 — decisione
    final_decision:    str      = "SKIP"   # EXECUTE / WAIT / SKIP
    decision_reason:   str      = ""
    decision_conf:     float    = 0.0

    # Log dell'intera pipeline
    log: list = field(default_factory=list)

    def add_log(self, agent: str, msg: str):
        ts = datetime.now(TIMEZONE).strftime("%H:%M:%S")
        entry = f"[{ts}] {agent}: {msg}"
        self.log.append(entry)
        logger.info(entry)


# ── RISULTATO AGENTE ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    success: bool
    data:    dict = field(default_factory=dict)
    error:   str  = ""


# ── AGENTE 1 — DATA COLLECTOR ─────────────────────────────────────────────────

async def agent_data_collector(state: TradingState) -> AgentResult:
    """
    Raccoglie dati di mercato multi-timeframe e prezzo corrente.
    Popola: current_price, market_data
    FIX: requests.get() gira in executor per non bloccare l'event loop.
    """
    state.add_log("📊 DataCollector", "Raccolta dati multi-TF...")
    try:
        from trade_manager import get_current_price_async
        from analyzer import get_data, compute_indicators

        # Prezzo live — non bloccante
        price = await get_current_price_async()
        if price <= 0:
            raise ValueError("Prezzo live non disponibile")
        state.current_price = price
        state.timestamp = datetime.now(TIMEZONE).isoformat()

        # Stessi outputsize di analyzer.get_multi_timeframe_data() (usata da
        # full_analyze, chiamata subito dopo da agent_structure_analyst):
        # allineare i due significa che il fetch qui sotto e quello dentro
        # full_analyze condividono la stessa chiave di cache in
        # analyzer._data_cache, quindi il secondo trova un cache hit invece
        # di riscaricare le stesse candele una seconda volta. Prima i due
        # numeri erano diversi per ogni timeframe: ogni singolo controllo
        # pipeline faceva sempre DUE fetch reali dello stesso TF, anche a
        # sorgenti dati perfettamente sane — scoperto analizzando la causa
        # dell'esaurimento quota Twelve Data del 1 settembre 2026 (quel bug
        # era il moltiplicatore ×6 su tutti i TF; questo è il raddoppio sul
        # solo TF principale, più piccolo ma non ancora sistemato allora).
        OUTPUTSIZE = {
            "1min": 120, "5min": 300, "15min": 200,
            "1h": 150, "4h": 100, "1day": 100,
        }
        outputsize = OUTPUTSIZE.get(state.timeframe, 150)

        # Usa cache se disponibile — evita 429 quando più TF girano in sequenza
        cached = _get_cached_data(state.timeframe, outputsize)
        if cached is not None:
            df = cached
            state.add_log("📊 DataCollector", f"✅ Cache hit {state.timeframe} | {len(df)} candele")
        else:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: compute_indicators(get_data(interval=state.timeframe, outputsize=outputsize))
            )
            _set_cached_data(state.timeframe, outputsize, df)

        state.market_data = {
            "rows":       len(df),
            "last_close": float(df["Close"].iloc[-1]),
            "last_high":  float(df["High"].iloc[-1]),
            "last_low":   float(df["Low"].iloc[-1]),
        }

        state.add_log("📊 DataCollector", f"✅ Prezzo: ${price} | Candele: {len(df)}")
        return AgentResult(success=True, data={"price": price})

    except Exception as e:
        state.add_log("📊 DataCollector", f"❌ Errore: {e}")
        return AgentResult(success=False, error=str(e))


# ── AGENTE 2 — STRUCTURE ANALYST ─────────────────────────────────────────────

async def agent_structure_analyst(state: TradingState) -> AgentResult:
    """
    Esegue full_analyze e valuta se il setup è strutturalmente valido.
    Popola: signal, order_type, entry, sl, tp1/2/3, prob, regime, rr, structure_ok
    FIX: full_analyze() è sincrona → gira in executor per non bloccare l'event loop.
    """
    state.add_log("🔍 StructureAnalyst", f"Analisi SMC su {state.timeframe}...")
    try:
        from analyzer import full_analyze

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: full_analyze(timeframe_focus=state.timeframe)
        )

        state.signal     = data.get("signal", "NEUTRAL")
        raw_order = data.get("order_type", state.signal)
        if raw_order.startswith(state.signal + " "):
            state.order_type = raw_order[len(state.signal):].strip()
        elif raw_order == state.signal:
            state.order_type = state.signal
        else:
            state.order_type = raw_order
        state.entry      = float(data.get("entry", 0))
        state.sl         = float(data.get("sl", 0))
        state.tp1        = float(data.get("tp1", 0))
        state.tp2        = float(data.get("tp2", 0))
        state.tp3        = float(data.get("tp3", 0))
        state.prob       = int(data.get("prob", 0))
        state.regime     = data.get("regime", "UNKNOWN")
        state.strategies = data.get("strategies", {})
        state.data_timestamp = str(data.get("data_timestamp") or "")

        # Livello strutturale per il break-even anticipato: il precedente
        # swing high/low sul TF del segnale. Solo se è ancora "davanti" al
        # prezzo (non già superato dall'entry stessa) — altrimenti armerebbe
        # il BE al primo tick, senza che il prezzo si sia mosso a favore.
        last_high = data.get("last_high")
        last_low  = data.get("last_low")
        if state.signal == "BUY" and last_high and float(last_high) > state.entry:
            state.early_be_level = float(last_high)
        elif state.signal == "SELL" and last_low and float(last_low) < state.entry:
            state.early_be_level = float(last_low)

        # Calcola R:R su TP2
        if state.entry and state.sl and state.tp2:
            diff_entry_sl = abs(state.entry - state.sl)
            diff_entry_tp = abs(state.tp2 - state.entry)
            state.rr = round(diff_entry_tp / diff_entry_sl, 2) if diff_entry_sl > 0 else 0

        # Soglia prob differenziata per TF
        _tf = state.timeframe.lower() if state.timeframe else ""
        if _tf in ("5min", "1min", "15min"):
            MIN_PROB = 65
        else:
            MIN_PROB = int(__import__("os").environ.get("MIN_PROB", "55"))

        # Regimi bloccati per timeframe — non è lo stesso regime ovunque.
        # H1: RANGING resta il peggiore (WR 23.7%, avg -0.297R, n=38 su 1y).
        # TRENDING_UP aggiunto il 1 set 2026: negativo in modo coerente su
        # tre finestre annidate (3m WR 27.6% avg -0.052R n=29; 6m WR 28.0%
        # avg -0.074R n=50; 1y WR 27.4% avg -0.113R n=84) — non un fluke di
        # un solo campione.
        # H4: RANGING bloccato, ma il campione (n=7, avg positivo) è troppo
        # piccolo per confermarlo o toglierlo — lasciato invariato finché
        # non c'è più storico.
        # D1: RANGING è quasi inesistente come regime lì (sotto la soglia
        # minima di 3 trade sia su 5y sia su 20y); il regime davvero debole
        # è TRENDING_DOWN, confermato su entrambi i campioni (WR 25-27%,
        # avg negativo, n=15 e n=67).
        _BLOCKED_REGIMES_BY_TF = {
            "1h":   ("RANGING", "TRENDING_UP"),
            "4h":   ("RANGING",),
            "1day": ("TRENDING_DOWN",),
        }
        _regime_up = str(state.regime).upper().replace(" ","_")
        _blocked = _BLOCKED_REGIMES_BY_TF.get(_tf, ())
        if _regime_up in _blocked:
            state.final_decision  = "SKIP"
            # _regime_up serve solo al confronto con _BLOCKED_REGIMES_BY_TF
            # (chiavi con underscore, es. "TRENDING_DOWN"): NON va mai in un
            # messaggio Telegram Markdown così com'è, vedi il commento su
            # format_pipeline_report più sotto sullo stesso bug.
            state.decision_reason = f"Regime {_regime_up.replace('_', ' ')} bloccato per {state.timeframe}"
            state.decision_conf   = 90.0
            state.add_log("🎯 DecisionMaker", f"SKIP — {_regime_up} bloccato su {state.timeframe}")
            return AgentResult(success=True, data={"decision": "SKIP"})

        state.structure_ok = (
            state.signal != "NEUTRAL" and
            state.prob >= MIN_PROB and
            state.entry > 0 and
            state.sl > 0 and
            state.tp2 > 0 and
            state.rr > 0
        )

        state.add_log(
            "🔍 StructureAnalyst",
            f"{'✅' if state.structure_ok else '⏭️'} "
            f"{state.signal} {state.order_type} @ {state.entry} | "
            f"Prob: {state.prob}% | Regime: {state.regime} | R:R {state.rr}"
        )
        return AgentResult(success=True, data={"signal": state.signal, "prob": state.prob})

    except Exception as e:
        state.add_log("🔍 StructureAnalyst", f"❌ Errore: {e}")
        return AgentResult(success=False, error=str(e))


# ── AGENTE 3 — RISK AGENT ─────────────────────────────────────────────────────

async def agent_risk(state: TradingState) -> AgentResult:
    """
    Verifica tutte le regole di risk management.
    Popola: risk_ok, risk_pct, risk_reason, trades_today, consecutive_loss
    """
    state.add_log("⚖️ RiskAgent", "Verifica risk management...")
    try:
        from risk_manager import check_can_trade, is_weekend_now

        result = check_can_trade(
            prob=state.prob,
            rr=state.rr,
            is_weekend=is_weekend_now(),
            near_news=state.high_impact,  # aggiornato dall'agente news se già girato
            signal=state.signal,
            news_error=state.news_error,
        )

        state.risk_ok     = result.allowed
        state.risk_pct    = result.risk_pct
        state.risk_reason = result.reason

        from risk_manager import get_session_stats, get_consecutive_losses
        session = get_session_stats()
        state.trades_today     = session.get("trades_count", 0)
        state.consecutive_loss = get_consecutive_losses()

        state.add_log(
            "⚖️ RiskAgent",
            f"{'✅ OK' if state.risk_ok else '🛑 BLOCCATO'} | "
            f"Trade oggi: {state.trades_today} | "
            f"Loss consecutive: {state.consecutive_loss} | "
            f"Risk: {state.risk_pct}%"
        )
        return AgentResult(success=True, data={"risk_ok": state.risk_ok})

    except Exception as e:
        state.add_log("⚖️ RiskAgent", f"❌ Errore: {e}")
        # In caso di errore del risk agent, blocca per sicurezza
        state.risk_ok     = False
        state.risk_reason = f"Errore risk agent: {e}"
        return AgentResult(success=False, error=str(e))


# ── AGENTE 4 — NEWS AGENT ─────────────────────────────────────────────────────

async def agent_news(state: TradingState) -> AgentResult:
    """
    Analizza news e calendario economico.
    Popola: news_safe, news_reason, news_summary, high_impact
    FIX: tutte le chiamate HTTP sono sincrone → executor per non bloccare.
    FIX: errore calendario = blocco prudenziale (non assume sicuro).
    """
    state.add_log("📰 NewsAgent", "Analisi news e calendario...")
    try:
        from analyzer import get_economic_events, get_news_sentiment
        from risk_manager import is_near_news

        loop = asyncio.get_event_loop()

        # Calendario in executor
        cal = await loop.run_in_executor(None, get_economic_events)
        calendar_error = cal.get("error", False)
        events         = cal.get("events", []) if not calendar_error else []
        state.news_error = bool(calendar_error)

        # Blackout notizie — passa news_error per fail-safe
        near = is_near_news(events, news_error=calendar_error)
        # high_impact significa "evento dentro il blackout", non "esiste oggi".
        # In precedenza bastava un evento in qualunque ora per bloccare tutto il giorno.
        state.high_impact = near
        state.news_safe   = not near
        state.news_reason = (
            "Evento macro ad alto impatto nelle prossime/ultime 30 min"
            if near and not calendar_error
            else "Errore calendario — blackout prudenziale"
            if calendar_error
            else "Nessun evento macro imminente"
        )

        if calendar_error:
            state.add_log("📰 NewsAgent", "⛔ Errore calendario → blocco prudenziale")

        # Sentiment in executor
        try:
            sentiment = await loop.run_in_executor(None, get_news_sentiment)
            label = sentiment.get("label", "NEUTRAL")
            score = sentiment.get("score", 0)
            state.news_summary = f"Sentiment: {label} ({score:+d})"
        except Exception:
            state.news_summary = "Sentiment non disponibile"

        state.add_log(
            "📰 NewsAgent",
            f"{'✅ SICURO' if state.news_safe else '⛔ PERICOLOSO'} | "
            f"{state.news_reason} | {state.news_summary}"
        )
        return AgentResult(success=True, data={"news_safe": state.news_safe})

    except Exception as e:
        # FAIL-SAFE: errore totale → blocca il trade
        state.add_log("📰 NewsAgent", f"⛔ Errore critico — blocco prudenziale: {e}")
        state.news_safe   = False
        state.high_impact = True
        state.news_error  = True
        state.news_reason = f"Errore news agent — blocco prudenziale: {e}"
        return AgentResult(success=True, data={"news_safe": False})


# ── AGENTE 5 — DECISION MAKER ─────────────────────────────────────────────────

async def agent_decision_maker(state: TradingState) -> AgentResult:
    """
    Sintetizza tutto e produce la decisione finale.
    Popola: final_decision, decision_reason, decision_conf

    Logica:
      SKIP  → nessun setup valido, o risk bloccato con news safe
      WAIT  → news pericolose (riprova dopo)
      EXECUTE → tutto OK, setup valido, risk OK, news OK
    """
    state.add_log("🎯 DecisionMaker", "Sintesi finale...")

    # Regola 1 — nessun setup strutturale
    if not state.structure_ok:
        state.final_decision  = "SKIP"
        state.decision_reason = f"Nessun setup valido ({state.signal} | prob={state.prob}%)"
        state.decision_conf   = 95.0
        state.add_log("🎯 DecisionMaker", f"⏭️ SKIP — {state.decision_reason}")
        return AgentResult(success=True, data={"decision": "SKIP"})

    # Regola 2 — news pericolose (riprova dopo, non skippa definitivamente)
    if not state.news_safe:
        state.final_decision  = "WAIT"
        state.decision_reason = state.news_reason
        state.decision_conf   = 100.0
        state.add_log("🎯 DecisionMaker", f"⏳ WAIT — {state.decision_reason}")
        return AgentResult(success=True, data={"decision": "WAIT"})

    # Regola 3 — risk bloccato
    if not state.risk_ok:
        state.final_decision  = "SKIP"
        state.decision_reason = state.risk_reason
        state.decision_conf   = 100.0
        state.add_log("🎯 DecisionMaker", f"🛑 SKIP (risk) — {state.decision_reason}")
        return AgentResult(success=True, data={"decision": "SKIP"})

    # Regola 4 — R:R insufficiente
    from risk_manager import MIN_RR
    if state.rr < MIN_RR:
        state.final_decision  = "SKIP"
        state.decision_reason = f"R:R {state.rr:.1f} < minimo {MIN_RR}"
        state.decision_conf   = 90.0
        state.add_log("🎯 DecisionMaker", f"⏭️ SKIP — {state.decision_reason}")
        return AgentResult(success=True, data={"decision": "SKIP"})

    # Regola 5 — Entry troppo lontana dal prezzo attuale (solo MARKET orders)
    # Per XAU/USD: 1 "pip" = $0.10 → 50 pip = $5.0
    # L'oro si può muovere di $3-8 in 5 minuti tra analisi e esecuzione.
    # Con MAX=15 si bloccava il 90% dei MARKET order — portato a 50.
    MAX_SLIPPAGE_PIPS = 50
    if "LIMIT" not in state.order_type.upper() and "STOP" not in state.order_type.upper():
        if state.current_price > 0 and state.entry > 0:
            distance_usd  = abs(state.current_price - state.entry)
            distance_pips = distance_usd * 10
            if distance_pips > MAX_SLIPPAGE_PIPS:
                state.final_decision  = "SKIP"
                state.decision_reason = (
                    f"Entry ${state.entry} troppo lontana dal prezzo attuale "
                    f"${state.current_price} ({distance_usd:.1f}$ = {distance_pips:.0f} pip)"
                )
                state.decision_conf = 95.0
                state.add_log("🎯 DecisionMaker", f"⏭️ SKIP — {state.decision_reason}")
                return AgentResult(success=True, data={"decision": "SKIP"})

    # Regola 5 — tutto OK: esegui
    state.final_decision  = "EXECUTE"
    state.decision_reason = (
        f"{state.signal} {state.order_type} @ {state.entry} | "
        f"Prob: {state.prob}% | R:R: {state.rr} | "
        f"Risk: {state.risk_pct}% | Regime: {state.regime}"
    )
    state.decision_conf = float(state.prob)

    state.add_log("🎯 DecisionMaker", f"✅ EXECUTE — {state.decision_reason}")
    return AgentResult(success=True, data={"decision": "EXECUTE"})


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

async def run_pipeline(timeframe: str = "5min") -> TradingState:
    """
    Esegue la pipeline completa dei 5 agenti in sequenza.
    Ogni agente può bloccare la pipeline se il suo risultato
    lo rende inutile proseguire (early exit per efficienza).

    Ritorna lo stato finale con la decisione.
    """
    state = TradingState(timeframe=timeframe)
    state.add_log("🚀 Pipeline", f"Avvio su {timeframe}...")

    agents = [
        ("DataCollector",    agent_data_collector),
        ("StructureAnalyst", agent_structure_analyst),
        ("NewsAgent",        agent_news),          # News prima del risk (high_impact → risk)
        ("RiskAgent",        agent_risk),
        ("DecisionMaker",    agent_decision_maker),
    ]

    for name, agent_fn in agents:
        try:
            result = await agent_fn(state)
            if not result.success:
                state.add_log("🚀 Pipeline", f"⚠️ {name} fallito: {result.error}")
                # Non bloccare la pipeline per errori non critici
        except Exception as e:
            state.add_log("🚀 Pipeline", f"❌ Eccezione in {name}: {e}")
            logger.exception(f"Eccezione in agente {name}")

        # Early exit: se non c'è setup, non serve andare oltre StructureAnalyst
        if name == "StructureAnalyst" and not state.structure_ok:
            state.add_log("🚀 Pipeline", "⏭️ Early exit — nessun setup")
            state.final_decision  = "SKIP"
            state.decision_reason = f"Nessun setup ({state.signal}, prob={state.prob}%)"
            break

    state.add_log("🚀 Pipeline", f"Fine — Decisione: {state.final_decision}")
    return state


def format_pipeline_report(state: TradingState) -> str:
    """
    Formatta il report segnale per Telegram — compatto e leggibile.
    Niente pipeline log, niente ridondanze.
    """
    from trade_manager import _fmt  # entry/sl/tp passano per una sottrazione
    # float (basis GC=F-spot, vedi gold_bot._check_single_timeframe) che può
    # lasciare artefatti di arrotondamento tipo "$4382.009999999999" nel
    # messaggio del segnale — stesso bug visto e corretto oggi (2026-09-02)
    # nei messaggi di trade_manager.py.

    tf_label = {
        "5min": "5MIN", "15min": "15MIN", "1h": "1H",
        "4h": "4H", "1day": "D1"
    }.get(state.timeframe, state.timeframe.upper())

    # Tipo ordine pulito (senza ripetere il signal)
    raw_order = state.order_type
    if raw_order == state.signal:
        order_label = "MARKET"
    elif raw_order.upper() in ("BUY LIMIT", "SELL LIMIT", "BUY STOP", "SELL STOP"):
        order_label = raw_order.split()[-1]  # solo "LIMIT" o "STOP"
    else:
        order_label = raw_order

    return (
        f"🤖 *MULTI-AGENT REPORT — {tf_label}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *{state.signal} {order_label}* @ *${_fmt(state.entry)}*\n"
        f"🛑 SL: ${_fmt(state.sl)} | 🎯 TP1: ${_fmt(state.tp1)}\n"
        f"🎯 TP2: ${_fmt(state.tp2)} | 🏆 TP3: ${_fmt(state.tp3)}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Prob: *{state.prob}%* | R:R: *{state.rr}* | Risk: *{state.risk_pct:.2f}%*\n"
        # state.regime (es. "TRENDING_DOWN"/"TRENDING_UP") contiene un
        # underscore che Telegram in Markdown legge come apertura di corsivo:
        # sommato all'unico "_...IT_" della riga sotto fa un numero dispari
        # di underscore nel messaggio, e l'intero invio fallisce con "Can't
        # parse entities" — bug reale in produzione, trovato il 2 settembre
        # 2026: bloccava OGNI segnale quando il regime era TRENDING_UP/DOWN
        # (0 segnali per ore nonostante setup validi). Sostituito con uno
        # spazio solo per la visualizzazione.
        f"📈 Regime: {str(state.regime).replace('_', ' ')}\n"
        f"_5 agenti | {state.timestamp[11:16]} IT_"
    )
