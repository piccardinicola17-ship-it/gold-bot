"""
self_learning.py — Auto-ottimizzazione pesi strategie GoldMind v2.
Legge dal DB unico goldbot.db (schema v2: colonna 'result', status='CLOSED').
"""

import os
import json
import logging
import sqlite3
import time
import tempfile
from datetime import datetime, timedelta
from typing import Optional
import pytz
import requests

logger   = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

DB_PATH         = os.environ.get("DB_PATH", os.path.join(os.environ.get("BOT_DIR", "/tmp"), "goldbot.db"))
LEARNED_WEIGHTS = os.path.join(os.environ.get("BOT_DIR", "/tmp"), "learned_weights.json")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "groq/compound-mini"

MIN_TRADES_FOR_LEARNING = 10
MIN_TRADES_PER_STRATEGY = 3

# Soglia per bloccare del tutto un regime (vedi optimize_strategy_weights):
# deliberatamente molto più alta di MIN_TRADES_PER_STRATEGY. Bloccare un intero
# regime è un'azione consequenziale quanto quella presa nel backtest storico
# del 1 settembre 2026 (filtro _BLOCKED_REGIMES_BY_TF in agent_orchestrator.py),
# dove n=7 è stato esplicitamente giudicato "troppo poco per decidere" anche
# con un segnale positivo — lì si è deciso solo con n=15+ su dati storici
# verificati. Qui i trade sono live, non backtestati: stesso standard minimo.
MIN_TRADES_FOR_REGIME_BLOCK = 20

_weights_cache      = {}
_weights_cache_mtime = 0.0


def load_learned_weights() -> dict:
    global _weights_cache, _weights_cache_mtime
    if not os.path.exists(LEARNED_WEIGHTS):
        return {}
    try:
        mtime = os.path.getmtime(LEARNED_WEIGHTS)
        if mtime == _weights_cache_mtime and _weights_cache:
            return _weights_cache
        with open(LEARNED_WEIGHTS) as f:
            data = json.load(f)
        _weights_cache       = data
        _weights_cache_mtime = mtime
        return data
    except Exception as e:
        logger.error(f"Errore caricamento pesi: {e}")
        return {}


def save_learned_weights(weights: dict):
    global _weights_cache, _weights_cache_mtime
    try:
        dir_ = os.path.dirname(LEARNED_WEIGHTS) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(weights, f, indent=2)
            tmp = f.name
        os.replace(tmp, LEARNED_WEIGHTS)
        _weights_cache       = {}
        _weights_cache_mtime = 0.0
        logger.info(f"[SELF-LEARNING] Pesi salvati: {LEARNED_WEIGHTS}")
    except Exception as e:
        logger.error(f"Errore salvataggio pesi: {e}")


def _get_closed_trades(days: Optional[int] = None) -> list:
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        if days:
            since = (datetime.now(TIMEZONE) - timedelta(days=days)).isoformat()
            rows  = conn.execute(
                "SELECT * FROM trades WHERE status='CLOSED' AND result IS NOT NULL AND timestamp >= ? ORDER BY id ASC",
                (since,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='CLOSED' AND result IS NOT NULL ORDER BY id ASC"
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Errore lettura trade: {e}")
        return []


def _is_win(t: dict) -> bool:
    """Stesso criterio di dashboard.py compute_stats: un WIN_BE con TP1 già
    raggiunto conta come vittoria (profitto parziale incassato), non viene
    escluso dal win rate come i BE "puri" (mai arrivati a TP1)."""
    return t.get("result") != "LOSS" and (
        bool(t.get("tp1_hit")) or t.get("result") in ("WIN_TP1", "WIN_TP2", "WIN_TP3")
    )


def _call_groq(system: str, user: str, max_tokens: int = 500) -> str:
    if not GROQ_API_KEY:
        return "{}"
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "temperature":0.2, "max_tokens":max_tokens},
            timeout=25,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Errore Groq: {e}")
        return "{}"


def analyze_last_trade(trade_id: str = "") -> str:
    """
    Analizza l'ultimo trade chiuso.
    Se trade_id è specificato, analizza quel trade specifico
    invece dell'ultimo del DB — evita di analizzare il trade sbagliato
    quando più trade si chiudono quasi contemporaneamente.
    """
    trades = _get_closed_trades()
    if not trades:
        return "Nessun trade nel database ancora."

    # Cerca il trade specifico per ID se fornito
    trade = None
    if trade_id:
        for t in reversed(trades):
            if t.get("trade_id", "").startswith(trade_id[:8]):
                trade = t
                break
    if trade is None:
        trade = trades[-1]  # fallback all'ultimo
    result  = trade.get("result", "")
    is_win  = result in ("WIN_TP1", "WIN_TP2", "WIN_TP3")
    is_be   = result == "WIN_BE"
    pnl_r   = trade.get("pnl_r", 0) or 0
    signal  = trade.get("signal", "")
    regime  = trade.get("regime", "")
    timeframe = trade.get("timeframe", "")
    entry   = trade.get("entry", 0)
    sl      = trade.get("sl", 0)
    tp1     = trade.get("tp1", 0)
    tp2     = trade.get("tp2", 0)
    tp3     = trade.get("tp3", 0)
    tp1_hit = bool(trade.get("tp1_hit"))
    be_hit  = bool(trade.get("be_hit"))
    exit_price = trade.get("exit_price", 0)
    prob    = trade.get("prob", 0)
    timestamp = (trade.get("timestamp") or "")[:16]

    all_trades = _get_closed_trades()
    wins_all_l = [t for t in all_trades if _is_win(t)]
    losses_all = [t for t in all_trades if t.get("result") == "LOSS"]
    n          = len(wins_all_l) + len(losses_all)
    wins_all   = len(wins_all_l)
    wr_all     = round(wins_all / n * 100, 1) if n > 0 else 0

    # Mappa result → descrizione chiara per l'AI
    result_map = {
        "WIN_TP1": "VITTORIA — ha raggiunto TP1 (+1R)",
        "WIN_TP2": "VITTORIA — ha raggiunto TP2 (+2R)",
        "WIN_TP3": "VITTORIA — ha raggiunto TP3 (+3R)",
        "WIN_BE":  "PAREGGIO — chiuso a break even (0R)",
        "LOSS":    "PERDITA — stop loss colpito (-1R)",
    }
    result_desc = result_map.get(result, f"RISULTATO: {result}")
    esito_label = "VINTO" if is_win else "CHIUSO A PAREGGIO" if is_be else "PERSO"
    tp1_txt = "TP1 toccato" if tp1_hit else "TP1 non toccato"
    be_txt  = "BE toccato"  if be_hit  else "BE non toccato"

    user = f"""ANALISI POST-TRADE — LEGGI ATTENTAMENTE:

RISULTATO REALE: {result_desc}
Questo trade ha {esito_label}. NON inventare un esito diverso.

Dati del trade:
Trade: {signal} {timeframe} in regime {regime}
Entry: ${entry} | SL: ${sl} | TP1: ${tp1} | TP2: ${tp2} | TP3: ${tp3}
Exit: ${exit_price} | P&L: {pnl_r:+.1f}R
Tappe raggiunte: {tp1_txt} | {be_txt}
Confidenza bot: {prob}% (score qualitativo, non probabilità statistica)
Orario: {timestamp}
Storico sistema: {n} trade decisivi, WR {wr_all}%

Rispondi SOLO con questa struttura (adatta il contenuto al fatto che il trade ha {esito_label}):

PERCHE HA {esito_label}:
[2-3 righe — causa principale basata sui DATI REALI sopra]

IL SETUP ERA VALIDO:
[Si/No e perché — considera il risultato reale]

COSA SI PUO MIGLIORARE:
[Anche sui trade vincenti c'è sempre qualcosa da ottimizzare]

LEZIONE CONCRETA:
[1 azione specifica da applicare al prossimo trade]

IMPATTO SUL SISTEMA:
[Come questo trade influenza la strategia complessiva]"""

    return _call_groq(
        system=(
            "Sei un trading coach esperto su XAU/USD. "
            "REGOLA FONDAMENTALE: analizza SEMPRE il trade basandoti sul risultato reale fornito. "
            "Se il trade ha vinto (WIN_TP1/TP2/TP3), NON dire che ha perso. "
            "Se ha perso (LOSS), NON dire che ha vinto. "
            "Rispondi in italiano, analisi brevi e azionabili."
        ),
        user=user,
        max_tokens=500,
    )


def weekly_review() -> str:
    trades = _get_closed_trades(days=7)
    if not trades:
        return "Nessun trade nell'ultima settimana."

    wins     = [t for t in trades if _is_win(t)]
    losses   = [t for t in trades if t.get("result") == "LOSS"]
    be_list  = [t for t in trades if t.get("result") == "WIN_BE"]
    decisivi = wins + losses
    wr       = round(len(wins) / len(decisivi) * 100, 1) if decisivi else 0
    pnl_r    = sum(t.get("pnl_r") or 0 for t in trades)

    regime_stats = {}
    for t in decisivi:
        r = t.get("regime", "?")
        regime_stats.setdefault(r, {"wins":0,"total":0})
        regime_stats[r]["total"] += 1
        if _is_win(t): regime_stats[r]["wins"] += 1
    regime_txt = "\n".join(f"  {r}: {d['wins']}/{d['total']} ({round(d['wins']/d['total']*100)}%)" for r,d in regime_stats.items()) or "  Nessun dato"

    tf_stats = {}
    for t in decisivi:
        tf = t.get("timeframe","?")
        tf_stats.setdefault(tf, {"wins":0,"total":0})
        tf_stats[tf]["total"] += 1
        if _is_win(t): tf_stats[tf]["wins"] += 1
    tf_txt = "\n".join(f"  {tf}: {d['wins']}/{d['total']} ({round(d['wins']/d['total']*100)}%)" for tf,d in tf_stats.items()) or "  Nessun dato"

    dir_stats = {}
    for t in decisivi:
        sig = t.get("signal","?")
        dir_stats.setdefault(sig, {"wins":0,"total":0})
        dir_stats[sig]["total"] += 1
        if _is_win(t): dir_stats[sig]["wins"] += 1
    dir_txt = "\n".join(f"  {sig}: {d['wins']}/{d['total']} ({round(d['wins']/d['total']*100)}%)" for sig,d in dir_stats.items()) or "  Nessun dato"

    user = f"""REVIEW SETTIMANALE XAU/USD:
Trade totali: {len(trades)} (decisivi: {len(decisivi)}, BE: {len(be_list)})
Win Rate: {wr}% ({len(wins)}W / {len(losses)}L) — calcolato solo su trade decisivi
P&L totale: {pnl_r:+.1f}R

Per regime:\n{regime_txt}
Per timeframe:\n{tf_txt}
Per direzione:\n{dir_txt}

Fornisci:
1. PERFORMANCE SETTIMANALE: valutazione generale
2. PUNTI DI FORZA: cosa ha funzionato
3. AREE DI MIGLIORAMENTO: cosa non ha funzionato
4. PATTERN IDENTIFICATI: tendenze nelle performance
5. OBIETTIVI SETTIMANA PROSSIMA: 2-3 azioni concrete"""

    ai_review = _call_groq(
        system="Sei un trading coach esperto su XAU/USD. Weekly review dettagliata e azionabile. Rispondi in italiano.",
        user=user,
        max_tokens=600,
    )

    header = (
        f"WEEKLY REVIEW — {datetime.now(TIMEZONE).strftime('%d/%m/%Y')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Trade: {len(trades)} totali | Win {len(wins)} | Loss {len(losses)} | BE {len(be_list)}\n"
        f"WR: {wr}% (su {len(decisivi)} decisivi) | P&L: {pnl_r:+.1f}R\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    return header + ai_review


def optimize_strategy_weights() -> dict:
    trades   = _get_closed_trades()
    wins_l   = [t for t in trades if _is_win(t)]
    losses_l = [t for t in trades if t.get("result") == "LOSS"]
    decisivi = wins_l + losses_l

    if len(decisivi) < MIN_TRADES_FOR_LEARNING:
        return {"status":"insufficient_data", "message":f"Servono almeno {MIN_TRADES_FOR_LEARNING} trade decisivi ({len(decisivi)} disponibili).", "weights":{}}

    regime_wr = {}
    for t in decisivi:
        r = t.get("regime","UNKNOWN")
        regime_wr.setdefault(r, {"wins":0,"total":0})
        regime_wr[r]["total"] += 1
        if _is_win(t): regime_wr[r]["wins"] += 1
    regime_performance = {r: round(d["wins"]/d["total"]*100,1) for r,d in regime_wr.items() if d["total"] >= MIN_TRADES_PER_STRATEGY}

    tf_wr = {}
    for t in decisivi:
        tf = t.get("timeframe","?")
        tf_wr.setdefault(tf, {"wins":0,"total":0})
        tf_wr[tf]["total"] += 1
        if _is_win(t): tf_wr[tf]["wins"] += 1
    tf_performance = {tf: round(d["wins"]/d["total"]*100,1) for tf,d in tf_wr.items() if d["total"] >= MIN_TRADES_PER_STRATEGY}

    dir_wr = {}
    for t in decisivi:
        sig = t.get("signal","?")
        dir_wr.setdefault(sig, {"wins":0,"total":0})
        dir_wr[sig]["total"] += 1
        if _is_win(t): dir_wr[sig]["wins"] += 1
    dir_performance = {sig: round(d["wins"]/d["total"]*100,1) for sig,d in dir_wr.items() if d["total"] >= MIN_TRADES_PER_STRATEGY}

    regime_r_values = {}
    for t in decisivi:
        r = t.get("regime","UNKNOWN")
        regime_r_values.setdefault(r, [])
        regime_r_values[r].append(t.get("pnl_r") or 0)
    regime_avg_r = {r: round(sum(v)/len(v),2) for r,v in regime_r_values.items() if len(v) >= MIN_TRADES_PER_STRATEGY}

    # Bloccare un intero regime viene deciso qui dal codice, MAI dall'LLM: a
    # differenza dei moltiplicatori sotto (limitati a +-20%, rischio contenuto
    # anche se l'LLM sbaglia), azzerare un regime è un'azione netta e binaria.
    # Serve lo stesso standard di campione usato per le decisioni sui regimi
    # nel backtest storico (vedi MIN_TRADES_FOR_REGIME_BLOCK) più un criterio
    # oggettivo — R medio negativo, non solo winrate basso: un regime può
    # avere WR modesto ma R medio comunque positivo (visto nel backtest di
    # ieri, H4 RANGING: WR 42.9% ma +0.311R) e non avrebbe senso bloccarlo.
    auto_blocked_regimes = [
        r for r, values in regime_r_values.items()
        if len(values) >= MIN_TRADES_FOR_REGIME_BLOCK and (sum(values) / len(values)) < 0
    ]

    perf_summary = json.dumps({"total_decisivi":len(decisivi),"regime_wr_pct":regime_performance,"regime_avg_r":regime_avg_r,"timeframe_wr_pct":tf_performance,"direction_wr_pct":dir_performance}, indent=2)

    groq_raw = _call_groq(
        system="Sei un quantitative analyst su XAU/USD. Rispondi SOLO con JSON valido, zero testo extra.",
        user=f"Performance reali:\n{perf_summary}\n\nRestituisci SOLO questo JSON:\n{{\n  \"regime_multipliers\": {{\"TRENDING_DOWN\": 1.2}},\n  \"direction_bias\": {{\"BUY\": 1.0, \"SELL\": 1.1}},\n  \"preferred_timeframe\": \"1h\",\n  \"rationale\": \"max 2 righe\"\n}}\nValori tra 0.5 e 1.5. WR>65% e avg_r>0.5 → 1.2-1.4. WR<50% → 0.6-0.8. Non decidere quali regimi bloccare: ci pensa il codice con una soglia più severa.",
        max_tokens=450,
    )

    try:
        clean = groq_raw.replace("```json","").replace("```","").strip()
        learned = json.loads(clean)
    except Exception:
        learned = {}

    # Sovrascrive sempre qualunque "blocked_regimes" l'LLM avesse comunque
    # restituito nonostante il prompt — la decisione è solo quella calcolata
    # sopra, deterministica.
    learned["blocked_regimes"] = auto_blocked_regimes

    learned["updated_at"]     = datetime.now(TIMEZONE).isoformat()
    learned["based_on"]       = len(decisivi)
    learned["based_on_total"] = len(trades)
    learned["regime_data"]    = regime_performance
    learned["regime_avg_r"]   = regime_avg_r
    learned["tf_data"]        = tf_performance
    learned["direction_data"] = dir_performance

    save_learned_weights(learned)
    return {"status":"optimized", "weights":learned, "message":f"Pesi ottimizzati su {len(decisivi)} trade decisivi."}


def format_learning_report(result: dict) -> str:
    if result.get("status") == "insufficient_data":
        return f"Dati insufficienti per l'ottimizzazione.\n{result.get('message','')}"

    weights  = result.get("weights", {})
    regime_m = weights.get("regime_multipliers", {})
    dir_bias = weights.get("direction_bias", {})
    pref_tf  = weights.get("preferred_timeframe", "N/A")
    blocked  = weights.get("blocked_regimes", [])
    rationale = weights.get("rationale", "")
    based_on = weights.get("based_on", 0)

    regime_txt = "\n".join(f"  {r}: {v}x" for r,v in regime_m.items()) or "  Nessuno"
    dir_txt    = "\n".join(f"  {d}: {v}x" for d,v in dir_bias.items()) or "  Neutro"
    blocked_txt = ", ".join(blocked) if blocked else "Nessuno"

    return (
        f"SELF-LEARNING — Pesi Aggiornati\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Basato su: {based_on} trade decisivi\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Moltiplicatori regime:\n{regime_txt}\n"
        f"Bias direzionale:\n{dir_txt}\n"
        f"TF preferito: {pref_tf}\n"
        f"Regimi bloccati: {blocked_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Rationale: {rationale}"
    )
