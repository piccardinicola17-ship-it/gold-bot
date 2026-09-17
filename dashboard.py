"""
Dashboard live GoldMind.

Legge lo stesso database SQLite usato dal bot. Ogni trade è un unico record e
ogni livello raggiunto (TP1, TP2, TP3, BE o SL) è mostrato nel proprio blocco.
"""

from __future__ import annotations

import hmac
import logging
import os
import sqlite3
from datetime import datetime

import pytz
from flask import Flask, abort, g, jsonify, redirect, render_template_string, request

# DB_PATH e _connect() importati da trade_manager (unica fonte di verità):
# prima dashboard.py ricalcolava DB_PATH per conto proprio (stesso schema,
# ma duplicato) e apriva le connessioni con la propria _connect(), con
# timeout più basso (10s vs 15s) e senza i pragma synchronous=NORMAL e
# foreign_keys=ON che trade_manager applica — es. /api/reset cancellava
# righe passando per una connessione che non forzava i vincoli FK che
# trade_manager applica scrivendo le stesse tabelle.
from trade_manager import (
    is_decisive_win, amend_closed_trade, RESULT_PNL, DB_PATH, _connect,
    load_broker_orders_pending, ack_broker_order, get_broker_fills,
    XAUUSD_PIP_SIZE, get_macro_event_outcomes, save_macro_event_pre,
    save_macro_event_post, get_trade_by_id, enqueue_broker_order,
)
# XAUUSD_OZ_PER_LOT vive in risk_manager (unica fonte di verità già usata
# per il sizing reale, vedi calculate_lot_size) — importata qui invece di
# ridefinirla per non aggiungere una terza copia della stessa costante
# (XAUUSD_PIP_SIZE sopra era già duplicata in due file, stesso pattern:
# vedi feedback_dual_mechanism_drift_pattern in memoria).
from risk_manager import XAUUSD_OZ_PER_LOT

app = Flask(__name__)
logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "").strip()
ALLOW_RESET = os.environ.get("ALLOW_DASHBOARD_RESET", "false").lower() == "true"

# Budget simulato in $ (2026-09-16, richiesta esplicita dell'utente): la
# dashboard mostra quanto avrebbe fruttato/perso ogni segnale REALE del
# bot con un lotto FISSO di riferimento su un conto da SIM_STARTING_BUDGET_USD
# — non serve un conto demo MT5 per "vedere i soldi", i prezzi entry/exit
# sono già quelli reali salvati per ogni trade. Lotto fisso (non calcolato
# dinamicamente sull'1% di rischio come fa calculate_lot_size) per scelta
# esplicita dell'utente: con un budget piccolo il sizing dinamico
# risulterebbe spesso "non tradabile" (lotto arrotondato a 0) su stop loss
# larghi, mentre l'utente vuole vedere un numero per OGNI segnale.
# Mostrato in $ (non convertito in €) perché l'oro è comunque quotato in
# dollari — evita un secondo tasso di cambio da tenere aggiornato.
SIM_STARTING_BUDGET_USD = float(os.environ.get("SIM_STARTING_BUDGET_USD", "500"))
SIM_LOT_SIZE = float(os.environ.get("SIM_LOT_SIZE", "0.02"))

# Spread fisso simulato (2026-09-16, richiesta esplicita dell'utente): una
# media onesta tra broker MT5 reali per XAUUSD, non il minimo teorico di un
# conto ECN con commissione a parte. Ricerca fatta su piu' fonti: i conti
# ECN "raw" (Exness Zero, IC Markets Raw, Pepperstone Razor) mostrano
# spread quasi a zero ma caricano una commissione fissa per lotto (~$7)
# che nei fatti riporta il costo reale nella stessa fascia; i conti
# standard senza commissione (il tipo più comune per chi copia segnali,
# come in questo bot) mostrano tipicamente 0.20-0.50$ (2-5 pip in questa
# convenzione, XAUUSD_PIP_SIZE=0.10) durante le sessioni Londra/New York,
# più larghi in sessione asiatica o su news. 0.30$ (3 pip) è il valore
# citato più spesso come "spread standard tipico" ed è il centro onesto
# di quella fascia — non il best-case di un solo broker specifico.
# Sottratto UNA volta per trade (non due): lo spread è già la differenza
# bid/ask pagata una sola volta nel round-trip apertura+chiusura.
SIM_SPREAD_USD = float(os.environ.get("SIM_SPREAD_USD", "0.30"))


def _is_loopback() -> bool:
    return request.remote_addr in ("127.0.0.1", "::1")


@app.before_request
def protect_dashboard():
    if request.path == "/health":
        return None
    # Le richieste che scrivono/distruggono dati (oggi /api/reset e
    # /api/correct-trade, entrambe POST) richiedono sempre il token, anche
    # da loopback. L'esenzione loopback esiste per comodità di lettura
    # locale, non deve coprire un endpoint che muta lo stato — altrimenti
    # qualunque processo nello stesso container (non solo l'utente)
    # potrebbe azzerare il DB o falsificare un trade senza presentare
    # alcuna credenziale. Bug reale trovato 2026-09-03 (per /api/reset).
    #
    # FIX: prima l'elenco era un allowlist di path scritta a mano
    # (_WRITE_ENDPOINTS) — un futuro endpoint di scrittura sarebbe rimasto
    # coperto dall'esenzione loopback finché qualcuno non si fosse ricordato
    # di aggiungerlo alla lista. Classificare per metodo HTTP (GET/HEAD non
    # mutano, tutto il resto sì) rende sicuro ogni nuovo endpoint per
    # costruzione, senza bisogno di ricordarselo.
    is_write = request.method not in ("GET", "HEAD", "OPTIONS")
    if not is_write and _is_loopback():
        return None
    if not DASHBOARD_TOKEN:
        abort(503, "Configura DASHBOARD_TOKEN nelle variabili Railway")

    bearer = request.headers.get("Authorization", "")
    header_token = bearer[7:] if bearer.startswith("Bearer ") else ""
    candidate = (
        request.args.get("token", "")
        or request.cookies.get("goldmind_dashboard", "")
        or header_token
    )
    if not hmac.compare_digest(candidate, DASHBOARD_TOKEN):
        abort(401)
    g.set_dashboard_cookie = bool(request.args.get("token"))
    return None


@app.after_request
def persist_dashboard_login(response):
    if getattr(g, "set_dashboard_cookie", False):
        response.set_cookie(
            "goldmind_dashboard",
            DASHBOARD_TOKEN,
            httponly=True,
            secure=request.is_secure,
            samesite="Strict",
            max_age=8 * 60 * 60,
        )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok", "database": os.path.exists(DB_PATH)})


def _trade_pips(trade: dict) -> float:
    saved = trade.get("pips")
    if saved is not None:
        try:
            return round(float(saved), 1)
        except (TypeError, ValueError):
            pass

    try:
        entry = float(trade.get("entry"))
        exit_price = float(trade.get("exit_price"))
    except (TypeError, ValueError):
        return 0.0
    if XAUUSD_PIP_SIZE <= 0:
        return 0.0
    direction = 1.0 if trade.get("signal") == "BUY" else -1.0
    return round(((exit_price - entry) * direction) / XAUUSD_PIP_SIZE, 1)


def _trade_pnl_usd(trade: dict, lot_size: float = SIM_LOT_SIZE,
                    spread_usd: float = SIM_SPREAD_USD) -> float:
    """Profitto/perdita in $ che il segnale REALE (entry/exit_price già
    salvati) avrebbe fatto con un lotto fisso di riferimento, al netto
    dello spread simulato — vedi i commenti sopra SIM_STARTING_BUDGET_USD
    e SIM_SPREAD_USD. Calcolato dal prezzo grezzo, non dai pip già
    arrotondati a 1 decimale (_trade_pips), per non accumulare un doppio
    arrotondamento sulla conversione in dollari.
    Formula: $ = (movimento_prezzo_a_favore - spread) × once_per_lotto ×
    lotto — lo spread si paga una sola volta a round-trip (non due),
    sottratto qui direttamente in unità di prezzo prima di convertire in $
    così il segno funziona per BUY e SELL senza doverlo duplicare."""
    try:
        entry = float(trade.get("entry"))
        exit_price = float(trade.get("exit_price"))
    except (TypeError, ValueError):
        return 0.0
    direction = 1.0 if trade.get("signal") == "BUY" else -1.0
    net_move = (exit_price - entry) * direction - spread_usd
    return round(net_move * XAUUSD_OZ_PER_LOT * lot_size, 2)


def _get_trades() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                  FROM trades
                 ORDER BY id DESC
                """
            ).fetchall()
        trades = [dict(row) for row in rows]
        for trade in trades:
            trade["sim_lot_size"] = SIM_LOT_SIZE
            if trade.get("status") == "CLOSED":
                # Un CANCELLED non ha un P&L reale (sempre 0R) — i pip salvati
                # sono solo la distanza ipotetica fino al prezzo di
                # cancellazione e confondono in dashboard. Solo visivo: il
                # dato salvato in DB resta intatto. Stesso principio per
                # sim_pnl_usd, aggiunto il 2026-09-16.
                is_cancelled = trade.get("result") == "CANCELLED"
                trade["pips"] = 0.0 if is_cancelled else _trade_pips(trade)
                trade["sim_pnl_usd"] = 0.0 if is_cancelled else _trade_pnl_usd(trade)
        return trades
    except sqlite3.Error:
        logger.exception("Lettura trade dashboard fallita")
        return []


def _get_session() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE date=?", (today,)
            ).fetchone()
        return dict(row) if row else {}
    except sqlite3.Error:
        logger.exception("Lettura sessione dashboard fallita")
        return {}


def compute_stats(trades: list[dict]) -> dict:
    valid = [trade for trade in trades if trade.get("result") != "CANCELLED"]
    closed = [
        trade for trade in valid
        if trade.get("status") == "CLOSED" and trade.get("result")
    ]
    losses = [trade for trade in closed if trade.get("result") == "LOSS"]
    wins = [trade for trade in closed if is_decisive_win(trade)]
    be_trades = [
        trade for trade in closed
        if bool(trade.get("be_hit")) or trade.get("result") == "WIN_BE"
    ]
    decisive = len(wins) + len(losses)

    return {
        "closed_total": len(closed),
        "open_count": sum(
            1 for trade in valid if trade.get("status") == "OPEN"
        ),
        "wins": len(wins),
        "losses": len(losses),
        "be_total": len(be_trades),
        "win_rate": round(len(wins) / decisive * 100, 1) if decisive else 0,
        "total_pips": round(sum(_trade_pips(trade) for trade in closed), 1),
        "total_r": round(
            sum(float(trade.get("pnl_r") or 0) for trade in closed), 2
        ),
        "total_usd": round(sum(_trade_pnl_usd(trade) for trade in closed), 2),
        "sim_budget_usd": round(
            SIM_STARTING_BUDGET_USD + sum(_trade_pnl_usd(trade) for trade in closed), 2
        ),
        "sim_starting_budget_usd": SIM_STARTING_BUDGET_USD,
        "sim_lot_size": SIM_LOT_SIZE,
        "sim_spread_usd": SIM_SPREAD_USD,
        "tp1_total": sum(
            1 for trade in valid
            if bool(trade.get("tp1_hit")) or trade.get("result") in ("WIN_TP1","WIN_TP2","WIN_TP3")
        ),
        "tp2_total": sum(
            1 for trade in valid
            if bool(trade.get("tp2_hit")) or trade.get("result") in ("WIN_TP2","WIN_TP3")
        ),
        "tp3_total": sum(
            1 for trade in valid
            if bool(trade.get("tp3_hit")) or trade.get("result") == "WIN_TP3"
        ),
        "tp_total": sum(
            (1 if (bool(trade.get("tp1_hit")) or trade.get("result") in ("WIN_TP1","WIN_TP2","WIN_TP3")) else 0)
            + (1 if (bool(trade.get("tp2_hit")) or trade.get("result") in ("WIN_TP2","WIN_TP3")) else 0)
            + (1 if (bool(trade.get("tp3_hit")) or trade.get("result") == "WIN_TP3") else 0)
            for trade in valid
        ),
    }


@app.route("/api/data")
def api_data():
    trades = _get_trades()
    return jsonify(
        {
            "stats": compute_stats(trades),
            "trades": trades,
            "session": _get_session(),
            "macro_events": get_macro_event_outcomes(20),
            "updated": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
        }
    )


@app.route("/api/ea/pending")
def api_ea_pending():
    """Ordini in attesa di essere copiati sul conto demo MT5 reale —
    interrogato dall'Expert Advisor (mql5/GoldMindCopier.mq5) ogni pochi
    secondi. Stessa coda che trade_manager.open_trade() riempie, nessun
    secondo calcolo qui: la dashboard si limita a esporla via HTTP.

    Array JSON (non un oggetto chiave/valore): MQL5 non ha un parser JSON
    nativo, un array di oggetti piatti è molto più semplice da spezzare a
    mano lato Expert Advisor di un dizionario con chiavi dinamiche (i
    trade_id, degli UUID)."""
    return jsonify(list(load_broker_orders_pending().values()))


@app.route("/api/ea/ack", methods=["POST"])
def api_ea_ack():
    """L'EA chiama questo endpoint subito dopo aver aperto (o tentato di
    aprire) l'ordine sul broker, cosi da non riceverlo di nuovo al prossimo
    polling. fill_price (opzionale, solo ordini a mercato): prezzo di
    esecuzione reale riportato da CTrade.ResultPrice() — usato per
    registrare lo slippage vs l'entry teorica (vedi trade_manager.
    ack_broker_order)."""
    payload = request.get_json(silent=True) or {}
    trade_id = str(payload.get("trade_id", "")).strip()
    if not trade_id:
        return jsonify({"error": "trade_id mancante"}), 400
    fill_price = payload.get("fill_price")
    ack_broker_order(trade_id, float(fill_price) if fill_price else None)
    return jsonify({"ok": True})


@app.route("/api/ea/fills")
def api_ea_fills():
    """Storico slippage reale (entry teorica vs fill vero sul conto demo
    MT5) — vedi trade_manager.get_broker_fills."""
    return jsonify(get_broker_fills())


@app.route("/api/ea/requeue", methods=["POST"])
def api_ea_requeue():
    """Rimette in coda per l'EA un trade già aperto lato bot (paper) ma
    mai arrivato sul conto MT5 reale — caso reale 2026-09-17: un bug
    nell'EA (GoldMindCopier.mq5, PickSupportedExpiration) confermava
    /api/ea/ack anche quando l'apertura falliva, quindi un BUY LIMIT
    spariva per sempre dalla coda senza mai essere piazzato sul broker,
    pur restando regolarmente aperto/pending nella simulazione paper.
    Il bug nell'EA è corretto (non confermerà più un fallimento), ma un
    ordine già tolto dalla coda PRIMA della correzione non si ripresenta
    da solo — questo endpoint lo re-inserisce a mano, prendendo entry/
    sl/tp/risk_pct dal trade già registrato. Stesso token di tutta la
    dashboard, stesso principio di /api/correct-trade: non cancella
    nulla, resta sempre tracciabile."""
    payload = request.get_json(silent=True) or {}
    trade_id = str(payload.get("trade_id", "")).strip()
    if not trade_id:
        return jsonify({"status": "error", "message": "trade_id mancante"}), 400
    trade = get_trade_by_id(trade_id)
    if not trade:
        return jsonify({"status": "error", "message": "trade non trovato"}), 404
    if trade.get("status") != "OPEN":
        return jsonify({"status": "error", "message": "il trade non è più OPEN, non ha senso rimetterlo in coda"}), 400
    enqueue_broker_order(trade_id, trade)
    return jsonify({"status": "ok"})


@app.route("/api/correct-trade", methods=["POST"])
def api_correct_trade():
    """Correzione amministrativa di un trade già chiuso — es. un LOSS preso
    da uno SL durante un evento macro, da correggere a CLOSED_EARLY come se
    la chiusura protettiva pre-evento (2026-09-04) fosse già esistita a
    quel momento. Protetto dallo stesso token di tutta la dashboard
    (before_request) — non serve un flag separato come ALLOW_RESET perché,
    a differenza del reset, qui non si cancella nulla: resta sempre
    tracciabile e reversibile con un'altra correzione."""
    payload = request.get_json(silent=True) or {}
    trade_id = str(payload.get("trade_id", "")).strip()
    result = str(payload.get("result", "")).strip().upper()
    notes = str(payload.get("notes", "Corretto manualmente"))
    try:
        exit_price = float(payload.get("exit_price"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "exit_price mancante o non numerico"}), 400
    if not trade_id or result not in RESULT_PNL:
        return jsonify({"status": "error", "message": f"trade_id o result non validi (result deve essere uno tra {sorted(RESULT_PNL)})"}), 400
    try:
        ok = amend_closed_trade(trade_id, result, exit_price, notes)
    except Exception as exc:
        logger.exception("Correzione trade fallita")
        return jsonify({"status": "error", "message": str(exc)}), 500
    if not ok:
        return jsonify({"status": "error", "message": "Trade non trovato o non ancora chiuso"}), 404
    return jsonify({"status": "ok"})


@app.route("/api/macro-event/manual", methods=["POST"])
def api_macro_event_manual():
    """Inserimento/correzione amministrativa di un evento macro nel
    tracciamento dashboard (2026-09-16, richiesta esplicita: "aggiungi
    nella dashboard gli eventi che si verificano a mercato e se il bot li
    prende... per ora il FOMC il bot l'ha preso in pieno... da segnare
    manualmente"). Serve per il caso in cui il resoconto automatico non è
    partito (bot riavviato durante la finestra, vedi save_macro_event_pre/
    post in gold_bot.py) ma l'utente ha comunque i dati reali a
    disposizione (es. da screenshot Telegram) — stesso principio di
    /api/correct-trade: niente si cancella, resta sempre tracciabile.
    Protetto dallo stesso token di tutta la dashboard (before_request)."""
    payload = request.get_json(silent=True) or {}
    group_key = str(payload.get("group_key", "")).strip()
    title = str(payload.get("title", "")).strip()
    currency = str(payload.get("currency", "USD")).strip().upper()
    impact = str(payload.get("impact", "HIGH")).strip().upper()
    event_time = str(payload.get("event_time", "")).strip()
    bias = str(payload.get("bias", "")).strip().upper()
    try:
        price_pre_event = float(payload.get("price_pre_event"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "price_pre_event mancante o non numerico"}), 400
    if not group_key or not title or not event_time or not bias:
        return jsonify({"status": "error", "message": "group_key, title, event_time e bias sono obbligatori"}), 400

    try:
        save_macro_event_pre(group_key, title, currency, impact, event_time, price_pre_event, bias)

        # I campi post-evento sono opzionali: un evento può essere seminato
        # solo con il bias pre-evento (in attesa dell'esito reale) oppure
        # con l'intero esito già noto, come nel caso del FOMC del
        # 2026-09-16 recuperato manualmente da screenshot Telegram.
        if payload.get("price_post") is not None:
            price_immediate = payload.get("price_immediate")
            change_immediate = payload.get("change_immediate")
            price_post = float(payload.get("price_post"))
            change_post = float(payload.get("change_post"))
            minutes_post = int(payload.get("minutes_post", 0))
            save_macro_event_post(
                group_key,
                float(price_immediate) if price_immediate is not None else None,
                float(change_immediate) if change_immediate is not None else None,
                (str(payload.get("esito_immediate")).strip().upper() or None)
                    if payload.get("esito_immediate") is not None else None,
                price_post, change_post,
                str(payload.get("esito_post", "")).strip().upper(),
                minutes_post,
            )
    except Exception as exc:
        logger.exception("Inserimento manuale evento macro fallito")
        return jsonify({"status": "error", "message": str(exc)}), 500
    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if not ALLOW_RESET:
        return jsonify({"status": "disabled"}), 403
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM sessions")
            conn.commit()
        return jsonify({"status": "ok"})
    except sqlite3.Error as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GoldMind Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

:root {
  --bg: #070807;
  --surface: #101210;
  --surface-2: #151815;
  --border: #252925;
  --gold: #d3ae4b;
  --text: #f2f4f2;
  --muted: #8d958d;
  --green: #39d98a;
  --red: #ff5b62;
  --amber: #f2bf4f;
  --blue: #61a8ff;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, system-ui, sans-serif;
}

header {
  height: 72px;
  padding: 0 34px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand {
  color: var(--gold);
  font-family: "IBM Plex Mono", monospace;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: .18em;
}
.live {
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}
.live::before {
  content: "";
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 12px rgba(57, 217, 138, .55);
}

main {
  width: min(1500px, calc(100% - 40px));
  margin: 26px auto 60px;
}

.session {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 26px;
  padding: 15px 18px;
  margin-bottom: 18px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  background: var(--surface);
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}
.session strong { color: var(--text); }

.metrics {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}
.metric {
  min-height: 122px;
  padding: 18px;
  border: 1px solid var(--border);
  border-top: 2px solid var(--gold);
  background: var(--surface);
}
.metric-label {
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  letter-spacing: .12em;
}
.metric-value {
  display: block;
  margin: 13px 0 7px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 28px;
  font-weight: 600;
}
.metric-note {
  color: var(--muted);
  font-size: 12px;
}
.positive { color: var(--green) !important; }
.negative { color: var(--red) !important; }
.protected { color: var(--amber) !important; }

.equity-wrap {
  position: relative;
  height: 220px;
  margin-bottom: 28px;
  padding: 4px;
  border: 1px solid var(--border);
  background: var(--surface);
}
.equity-wrap svg { width: 100%; height: 100%; display: block; }
.equity-wrap .empty {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
}

.section-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin: 0 0 12px;
}
.section-head h2 {
  margin: 0 0 4px;
  font-size: 18px;
}
.section-head p,
.legend {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.trade-list {
  display: grid;
  gap: 12px;
}
.trade-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 170px;
  background: var(--surface);
  border: 1px solid var(--border);
}
.trade-body { padding: 20px; min-width: 0; }
.trade-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 16px;
}
.trade-title {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.direction {
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  font-weight: 600;
}
.direction.buy { color: var(--green); }
.direction.sell { color: var(--red); }
.result {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
}

.facts,
.levels {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 8px 14px;
  margin-bottom: 14px;
}
.levels { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.fact span,
.level span {
  display: block;
  margin-bottom: 4px;
  color: var(--muted);
  font-size: 11px;
}
.fact strong,
.level strong {
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}
.trade-foot {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
}

.milestones {
  padding: 18px;
  background: var(--surface-2);
  border-left: 1px solid var(--border);
  display: grid;
  align-content: center;
  gap: 3px;
}
.milestone {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
  border-bottom: 1px solid var(--border);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}
.milestone:last-child { border-bottom: 0; }
.mark {
  width: 22px;
  text-align: center;
  font-size: 18px;
  font-weight: 600;
}
.mark.hit { color: var(--green); }
.mark.loss { color: var(--red); }
.mark.waiting { color: var(--muted); }
.mark.armed { color: var(--amber); }

.macro-list {
  display: grid;
  gap: 10px;
}
.macro-card {
  padding: 16px 20px;
  background: var(--surface);
  border: 1px solid var(--border);
}
.macro-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-bottom: 10px;
}
.macro-head strong { font-size: 14px; }
.macro-time {
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
}
.macro-rows {
  display: grid;
  gap: 6px;
}
.macro-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: var(--muted);
}
.badge {
  display: inline-block;
  padding: 2px 9px;
  border: 1px solid var(--border);
  border-radius: 2px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  letter-spacing: .02em;
  white-space: nowrap;
}
.badge.confermato { color: var(--green); border-color: var(--green); }
.badge.non-confermato { color: var(--red); border-color: var(--red); }
.badge.neutro,
.badge.non-significativo { color: var(--muted); }
.badge.pending { color: var(--amber); border-color: var(--amber); }

.empty {
  padding: 70px 20px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  text-align: center;
}
.error {
  display: none;
  margin-bottom: 15px;
  padding: 12px 16px;
  border: 1px solid var(--red);
  color: var(--red);
  font-size: 13px;
}

@media (max-width: 1050px) {
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .facts { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 700px) {
  header { padding: 0 18px; }
  main { width: min(100% - 24px, 1500px); }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trade-card { grid-template-columns: 1fr; }
  .milestones {
    grid-template-columns: repeat(5, minmax(0, 1fr));
    border-left: 0;
    border-top: 1px solid var(--border);
  }
  .milestone {
    flex-direction: column;
    justify-content: center;
    border-bottom: 0;
  }
}
@media (max-width: 440px) {
  .metrics,
  .facts,
  .levels { grid-template-columns: 1fr 1fr; }
  .metric { min-height: 110px; }
}
</style>
</head>
<body>
<header>
  <div class="brand">GOLDMIND</div>
  <div class="live"><span id="updated">--:--:--</span></div>
</header>
<main>
  <div id="error" class="error"></div>

  <section class="session" aria-label="Sessione odierna">
    <span>Sessione <strong id="session-status">ATTIVA</strong></span>
    <span>Trade oggi <strong id="session-trades">0</strong></span>
    <span>Win <strong id="session-wins">0</strong></span>
    <span>Loss <strong id="session-losses">0</strong></span>
    <span>Loss consecutive <strong id="session-consecutive">0</strong></span>
    <span>P&amp;L oggi <strong id="session-pnl">+0R</strong></span>
  </section>

  <section class="metrics" aria-label="Statistiche generali">
    <div class="metric">
      <span class="metric-label">BUDGET SIMULATO</span>
      <strong id="sim-budget" class="metric-value">$0.00</strong>
      <span id="sim-budget-note" class="metric-note">$0 iniziali · lotto 0.00</span>
    </div>
    <div class="metric">
      <span class="metric-label">PIPS TOTALI</span>
      <strong id="total-pips" class="metric-value">0</strong>
      <span class="metric-note">Trade virtuali chiusi</span>
    </div>
    <div class="metric">
      <span class="metric-label">WIN RATE</span>
      <strong id="win-rate" class="metric-value">0%</strong>
      <span id="win-note" class="metric-note">0 win · 0 loss</span>
    </div>
    <div class="metric">
      <span class="metric-label">TP PRESI</span>
      <strong id="tp-total" class="metric-value positive">0</strong>
      <span id="tp-note" class="metric-note">TP1 0 · TP2 0 · TP3 0</span>
    </div>
    <div class="metric">
      <span class="metric-label">SL PRESI</span>
      <strong id="sl-total" class="metric-value negative">0</strong>
      <span class="metric-note">Trade chiusi in perdita</span>
    </div>
    <div class="metric">
      <span class="metric-label">BREAK EVEN</span>
      <strong id="be-total" class="metric-value">0</strong>
      <span class="metric-note">Chiusure reali a pareggio</span>
    </div>
  </section>

  <div class="section-head">
    <div>
      <h2>Equity curve</h2>
      <p>R cumulato sui trade chiusi, in ordine cronologico</p>
    </div>
    <div class="legend">
      <span>Picco <strong id="equity-peak">0R</strong></span>
      <span>Drawdown attuale <strong id="equity-dd">0R</strong></span>
    </div>
  </div>
  <section class="equity-wrap" aria-label="Equity curve">
    <svg id="equity-svg" viewBox="0 0 1000 220" preserveAspectRatio="none"></svg>
    <div id="equity-empty" class="empty" style="display:none;">Ancora nessun trade chiuso.</div>
  </section>

  <div class="section-head">
    <div>
      <h2>Trade registrati</h2>
      <p>Un blocco per ogni segnale · aggiornamento automatico</p>
    </div>
    <div class="legend">
      <span class="positive">✓ raggiunto</span>
      <span class="negative">× stop loss</span>
      <span class="protected">◆ BE protetto</span>
      <span>○ non raggiunto</span>
    </div>
  </div>

  <section id="trade-list" class="trade-list" aria-live="polite"></section>

  <div class="section-head">
    <div>
      <h2>Eventi macro</h2>
      <p>Bias del bot pre-evento confrontato col movimento di prezzo reale</p>
    </div>
    <div class="legend">
      <span class="positive">✓ confermato</span>
      <span class="negative">× non confermato</span>
      <span>○ in attesa dell'esito</span>
    </div>
  </div>
  <section id="macro-event-list" class="macro-list" aria-live="polite"></section>
</main>

<script>
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function num(value, digits = 2) {
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString("it-IT", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "—";
}

function signed(value, suffix = "") {
  const parsed = Number(value || 0);
  const sign = parsed > 0 ? "+" : "";
  return `${sign}${num(parsed, parsed % 1 === 0 ? 0 : 1)}${suffix}`;
}

function signedUsd(value) {
  const parsed = Number(value || 0);
  const sign = parsed > 0 ? "+" : parsed < 0 ? "-" : "";
  return `${sign}$${num(Math.abs(parsed), 2)}`;
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? esc(value)
    : date.toLocaleString("it-IT");
}

function milestone(label, state) {
  const marks = {
    hit: ["✓", "hit"],
    loss: ["×", "loss"],
    armed: ["◆", "armed"],
    waiting: ["○", "waiting"]
  };
  const [icon, css] = marks[state] || marks.waiting;
  return `
    <div class="milestone">
      <span>${label}</span>
      <strong class="mark ${css}">${icon}</strong>
    </div>`;
}

function getTp1State(trade) {
  if (trade.tp1_hit) return "hit";
  if (["WIN_TP2", "WIN_TP3"].includes(trade.result)) return "hit";
  return "waiting";
}

function getTp2State(trade) {
  if (trade.tp2_hit) return "hit";
  if (trade.result === "WIN_TP3") return "hit";
  return "waiting";
}

function getTp3State(trade) {
  return trade.tp3_hit ? "hit" : "waiting";
}

function getBeState(trade) {
  if (trade.be_hit || trade.result === "WIN_BE") return "hit";
  if (trade.be_armed && trade.status === "OPEN") return "armed";
  return "waiting";
}

function getSlState(trade) {
  if (trade.result === "LOSS") return "loss";
  return "waiting";
}

function highestTarget(trade) {
  if (trade.tp3_hit) return "TP3";
  if (trade.tp2_hit) return "TP2";
  if (trade.tp1_hit) return "TP1";
  return "";
}

function resultInfo(trade) {
  if (trade.status === "OPEN") {
    if (!trade.activated) return ["IN ATTESA", "protected"];
    if (trade.be_armed) return ["ATTIVO · PROTETTO BE", "protected"];
    return ["ATTIVO", "positive"];
  }
  if (trade.result === "LOSS") return ["CHIUSO · SL", "negative"];
  if (trade.result === "WIN_BE") {
    const target = highestTarget(trade);
    return [`CHIUSO · ${target ? target + " + " : ""}BE`, ""];
  }
  if (trade.result === "CANCELLED") return ["ANNULLATO", ""];
  if (trade.result === "CLOSED_EARLY") {
    const cls = Number(trade.pips || 0) > 0 ? "positive" : Number(trade.pips || 0) < 0 ? "negative" : "";
    return ["CHIUSO · PRE-EVENTO", cls];
  }
  const target = highestTarget(trade) || String(trade.result || "").replace("WIN_", "");
  return [`CHIUSO · ${target}`, "positive"];
}

function tradeCard(trade) {
  const [resultText, resultClass] = resultInfo(trade);
  const direction = trade.signal === "BUY" ? "buy" : "sell";
  const pips = trade.status === "CLOSED"
    ? signed(trade.pips || 0)
    : "—";
  const pipsClass = Number(trade.pips || 0) > 0
    ? "positive"
    : Number(trade.pips || 0) < 0 ? "negative" : "";
  const simUsd = trade.status === "CLOSED"
    ? signedUsd(trade.sim_pnl_usd || 0)
    : "—";
  const simUsdClass = Number(trade.sim_pnl_usd || 0) > 0
    ? "positive"
    : Number(trade.sim_pnl_usd || 0) < 0 ? "negative" : "";
  const identifier = String(trade.trade_id || "—");
  const shortId = identifier.length > 12
    ? `${identifier.slice(0, 6)}…${identifier.slice(-4)}`
    : identifier;

  return `
    <article class="trade-card">
      <div class="trade-body">
        <div class="trade-head">
          <div class="trade-title">
            <span class="direction ${direction}">${esc(trade.signal)}</span>
            <strong>XAU/USD · ${esc(String(trade.timeframe || "?").toUpperCase())}</strong>
          </div>
          <span class="result ${resultClass}">${esc(resultText)}</span>
        </div>

        <div class="facts">
          <div class="fact"><span>Entry</span><strong>$${num(trade.entry)}</strong></div>
          <div class="fact"><span>SL</span><strong>$${num(trade.sl)}</strong></div>
          <div class="fact"><span>Probabilità</span><strong>${num(trade.prob, 0)}%</strong></div>
          <div class="fact"><span>Risk</span><strong>${num(trade.risk_pct)}%</strong></div>
          <div class="fact"><span>Risultato</span><strong>${esc(trade.result || "APERTO")}</strong></div>
          <div class="fact"><span>Pips</span><strong class="${pipsClass}">${pips}</strong></div>
          <div class="fact"><span>Lotto sim.</span><strong>${num(trade.sim_lot_size || 0, 2)}</strong></div>
          <div class="fact"><span>P&amp;L sim. $</span><strong class="${simUsdClass}">${simUsd}</strong></div>
        </div>

        <div class="levels">
          <div class="level"><span>TP1</span><strong>$${num(trade.tp1)}</strong></div>
          <div class="level"><span>TP2</span><strong>$${num(trade.tp2)}</strong></div>
          <div class="level"><span>TP3</span><strong>$${num(trade.tp3)}</strong></div>
        </div>

        <div class="trade-foot">
          <span>${esc(trade.regime || "REGIME N/D")} · ${formatTime(trade.timestamp)}</span>
          <span title="${esc(identifier)}">ID ${esc(shortId)}</span>
        </div>
      </div>

      <div class="milestones">
        ${milestone("TP1", getTp1State(trade))}
        ${milestone("TP2", getTp2State(trade))}
        ${milestone("TP3", getTp3State(trade))}
        ${milestone("BE", getBeState(trade))}
        ${milestone("SL", getSlState(trade))}
      </div>
    </article>`;
}

const MACRO_STATUS_LABELS = {
  CONFERMATO: ["badge confermato", "✓ CONFERMATO"],
  NON_CONFERMATO: ["badge non-confermato", "× NON CONFERMATO"],
  NEUTRO: ["badge neutro", "— NEUTRO"],
  NON_SIGNIFICATIVO: ["badge non-significativo", "— NON SIGNIFICATIVO"],
};

function macroStatusBadge(status) {
  if (!status) return '<span class="badge pending">○ IN ATTESA</span>';
  const [cls, label] = MACRO_STATUS_LABELS[status] || ["badge", esc(status)];
  return `<span class="${cls}">${label}</span>`;
}

function macroEventCard(ev) {
  const currencyTag = ev.currency && ev.currency !== "USD" ? ` [${esc(ev.currency)}]` : "";
  return `
    <article class="macro-card">
      <div class="macro-head">
        <strong>${esc(ev.title)}${currencyTag}</strong>
        <span class="macro-time">${formatTime(ev.event_time)}</span>
      </div>
      <div class="macro-rows">
        <div class="macro-row">
          <span>Bias evento: ${esc(ev.bias || "?")} (reazione ~1-7 min)</span>
          ${macroStatusBadge(ev.esito_immediate)}
        </div>
        <div class="macro-row">
          <span>Bias post-evento: ${esc(ev.bias || "?")} (assestamento)</span>
          ${macroStatusBadge(ev.esito_post)}
        </div>
      </div>
    </article>`;
}

const EQUITY_SVG_W = 1000;
const EQUITY_SVG_H = 220;
const EQUITY_PAD = 14;

function renderEquityCurve(trades) {
  // Solo chiusi con un pnl_r reale, in ordine cronologico (l'API li da'
  // piu' recenti-prima) — CANCELLED escluso, stesso criterio di
  // compute_stats() lato server (mai un doppio calcolo divergente).
  const closed = trades
    .filter((t) => t.status === "CLOSED" && t.result && t.result !== "CANCELLED")
    .slice()
    .reverse();

  const svg = $("equity-svg");
  const empty = $("equity-empty");
  if (!closed.length) {
    svg.innerHTML = "";
    empty.style.display = "flex";
    $("equity-peak").textContent = "0R";
    $("equity-dd").textContent = "0R";
    return;
  }
  empty.style.display = "none";

  let equity = 0, peak = 0, maxDd = 0;
  const points = [{ x: 0, y: 0 }];
  closed.forEach((t, i) => {
    equity += Number(t.pnl_r || 0);
    peak = Math.max(peak, equity);
    maxDd = Math.max(maxDd, peak - equity);
    points.push({ x: i + 1, y: equity });
  });

  const minY = Math.min(0, ...points.map((p) => p.y));
  const maxY = Math.max(0, ...points.map((p) => p.y));
  const spanY = maxY - minY || 1;
  const toX = (i) => EQUITY_PAD + (i / (points.length - 1)) * (EQUITY_SVG_W - 2 * EQUITY_PAD);
  const toY = (y) => EQUITY_SVG_H - EQUITY_PAD - ((y - minY) / spanY) * (EQUITY_SVG_H - 2 * EQUITY_PAD);

  const zeroY = toY(0).toFixed(1);
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(p.y).toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${toX(points.length - 1).toFixed(1)},${zeroY} L${toX(0).toFixed(1)},${zeroY} Z`;
  const lineColor = equity >= 0 ? "var(--green)" : "var(--red)";

  svg.innerHTML = `
    <line x1="${EQUITY_PAD}" y1="${zeroY}" x2="${EQUITY_SVG_W - EQUITY_PAD}" y2="${zeroY}"
          stroke="var(--border)" stroke-width="1" />
    <path d="${areaPath}" fill="${lineColor}" opacity="0.12" stroke="none" />
    <path d="${linePath}" fill="none" stroke="${lineColor}" stroke-width="2" />
  `;

  $("equity-peak").textContent = signed(peak, "R");
  $("equity-dd").textContent = `-${num(maxDd, 2)}R`;
}

function render(data) {
  const stats = data.stats || {};
  const session = data.session || {};

  $("updated").textContent = data.updated || "--:--:--";

  const simBudget = Number(stats.sim_budget_usd || 0);
  const simStart = Number(stats.sim_starting_budget_usd || 0);
  const simLot = Number(stats.sim_lot_size || 0);
  const simSpread = Number(stats.sim_spread_usd || 0);
  $("sim-budget").textContent = `$${num(simBudget, 2)}`;
  $("sim-budget").className = `metric-value ${
    simBudget > simStart ? "positive" : simBudget < simStart ? "negative" : ""
  }`;
  $("sim-budget-note").textContent =
    `$${num(simStart, 0)} iniziali · lotto ${num(simLot, 2)} · spread $${num(simSpread, 2)} · ${signedUsd(stats.total_usd)}`;

  $("total-pips").textContent = signed(stats.total_pips || 0);
  $("total-pips").className = `metric-value ${
    Number(stats.total_pips || 0) > 0
      ? "positive"
      : Number(stats.total_pips || 0) < 0 ? "negative" : ""
  }`;
  $("win-rate").textContent = `${num(stats.win_rate || 0, 1)}%`;
  $("win-note").textContent = `${stats.wins || 0} win · ${stats.losses || 0} loss`;
  $("tp-total").textContent = stats.tp_total || 0;
  $("tp-note").textContent =
    `TP1 ${stats.tp1_total || 0} · TP2 ${stats.tp2_total || 0} · TP3 ${stats.tp3_total || 0}`;
  $("sl-total").textContent = stats.losses || 0;
  $("be-total").textContent = stats.be_total || 0;

  $("session-status").textContent = session.session_stopped ? "FERMATA" : "ATTIVA";
  $("session-status").className = session.session_stopped ? "negative" : "positive";
  $("session-trades").textContent = session.trades_count || 0;
  $("session-wins").textContent = session.wins || 0;
  $("session-losses").textContent = session.losses || 0;
  $("session-consecutive").textContent = session.consecutive_losses || 0;
  $("session-pnl").textContent = signed(session.pnl_r || 0, "R");

  const trades = data.trades || [];
  renderEquityCurve(trades);
  $("trade-list").innerHTML = trades.length
    ? trades.map(tradeCard).join("")
    : '<div class="empty">Nessun trade registrato nel database.</div>';

  const macroEvents = data.macro_events || [];
  $("macro-event-list").innerHTML = macroEvents.length
    ? macroEvents.map(macroEventCard).join("")
    : '<div class="empty">Nessun evento macro registrato.</div>';
}

async function refresh() {
  try {
    const response = await fetch("/api/data", {
      credentials: "same-origin",
      cache: "no-store"
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    $("error").style.display = "none";
  } catch (error) {
    $("error").textContent = `Aggiornamento fallito: ${error.message}`;
    $("error").style.display = "block";
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    if request.args.get("token"):
        return redirect("/")
    return render_template_string(DASHBOARD_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("DASHBOARD_PORT", "5050")))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    logger.info("Dashboard avviata su %s:%s - DB %s", host, port, DB_PATH)
    app.run(host=host, port=port, debug=False, threaded=True)
