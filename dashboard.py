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
from pathlib import Path

import pytz
from flask import Flask, abort, g, jsonify, redirect, render_template_string, request

app = Flask(__name__)
logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone("Europe/Rome")

PROJECT_DIR = Path(__file__).resolve().parent
BOT_DIR = Path(os.environ.get("BOT_DIR", PROJECT_DIR / "data")).expanduser().resolve()
DB_PATH = os.environ.get("DB_PATH", str(BOT_DIR / "goldbot.db"))
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "").strip()
ALLOW_RESET = os.environ.get("ALLOW_DASHBOARD_RESET", "false").lower() == "true"
XAUUSD_PIP_SIZE = float(os.environ.get("XAUUSD_PIP_SIZE", "0.10"))


def _is_loopback() -> bool:
    return request.remote_addr in ("127.0.0.1", "::1")


@app.before_request
def protect_dashboard():
    if request.path == "/health":
        return None
    # /api/reset cancella tutti i trade: richiede sempre il token, anche da
    # loopback. L'esenzione loopback esiste per comodità di lettura locale,
    # non deve coprire un endpoint distruttivo — altrimenti qualunque
    # processo nello stesso container (non solo l'utente) potrebbe azzerare
    # il DB senza presentare alcuna credenziale. Bug reale trovato 2026-09-03.
    if request.path != "/api/reset" and _is_loopback():
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


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


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
            if trade.get("status") == "CLOSED":
                # Un CANCELLED non ha un P&L reale (sempre 0R) — i pip salvati
                # sono solo la distanza ipotetica fino al prezzo di
                # cancellazione e confondono in dashboard. Solo visivo: il
                # dato salvato in DB resta intatto.
                trade["pips"] = 0.0 if trade.get("result") == "CANCELLED" else _trade_pips(trade)
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
    wins = [
        trade for trade in closed
        if trade.get("result") != "LOSS"
        and (
            bool(trade.get("tp1_hit"))
            or trade.get("result") in ("WIN_TP1", "WIN_TP2", "WIN_TP3")
        )
    ]
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
            "updated": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
        }
    )


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
  const beState = trade.be_hit
    ? "hit"
    : trade.be_armed && trade.status === "OPEN" ? "armed" : "waiting";
  const slState = trade.result === "LOSS" ? "loss" : "waiting";
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

function render(data) {
  const stats = data.stats || {};
  const session = data.session || {};

  $("updated").textContent = data.updated || "--:--:--";
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
  $("trade-list").innerHTML = trades.length
    ? trades.map(tradeCard).join("")
    : '<div class="empty">Nessun trade registrato nel database.</div>';
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
