"""
weekly_chart.py — genera l'immagine con le zone chiave (supporto,
resistenza, prezzo attuale, trend) per l'analisi weekend della domenica.

compute_weekly_zones() è l'UNICA fonte dei livelli: sia il testo del
messaggio (gold_bot._build_weekend_outlook) sia il grafico qui sotto
leggono dallo stesso dict, quindi non possono mai raccontare due cose
diverse (vedi memoria "pattern bug: doppio meccanismo che diverge").
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyzer


def compute_weekly_zones(df: pd.DataFrame) -> dict:
    """Zone chiave calcolate sulle candele 4h passate (ultime ~15gg)."""
    sr = analyzer.get_support_resistance(df)
    week = df.tail(42)  # ~7 giorni di candele 4h
    return {
        "current_price": round(float(df["Close"].iloc[-1]), 2),
        "resistance":     sr["r_near"],
        "support":        sr["s_near"],
        "week_high":      round(float(week["High"].max()), 2),
        "week_low":       round(float(week["Low"].min()), 2),
        "pivot":          sr["pivot"],
    }


def compute_smc_context(df: pd.DataFrame) -> dict:
    """
    Fatti tecnici Smart Money Concepts (struttura BOS/CHoCH, order block,
    fair value gap, liquidità EQH/EQL, zona premium/discount, regime) sulle
    stesse candele 4h usate per il grafico e le zone chiave — nessun nuovo
    calcolo indipendente, solo le funzioni SMC già validate in analyzer.py
    (le stesse che decidono gli order type dei segnali live).

    Passato a news_analyst.get_weekly_smc_narrative() perché lo trasformi
    in una spiegazione discorsiva senza inventare numeri: qui ci sono SOLO
    i fatti calcolati, mai un'interpretazione.
    """
    d = analyzer.compute_indicators(df.copy())
    d = analyzer.detect_swing_points(d)
    structure = analyzer.detect_bos_choch(d)
    ob        = analyzer.detect_order_blocks(d)
    fvg       = analyzer.detect_fvg(d)
    liq       = analyzer.detect_liquidity(d)
    regime    = analyzer.detect_market_regime(d)
    return {
        "structure":        structure,
        "order_blocks":     ob,
        "fvg":              fvg,
        "liquidity":        liq,
        "premium_discount": analyzer.detect_premium_discount(d, structure),
        "mitigation":       analyzer.detect_mitigation(d, ob),
        "regime":           regime["regime"],
        "adx":              regime["adx"],
    }


def render_weekly_outlook_chart(df: pd.DataFrame, zones: dict, bias: str) -> str:
    """
    Disegna le candele 4h delle ultime ~2 settimane con le zone di
    supporto/resistenza, il prezzo attuale, una linea di trend e una
    freccia per lo scenario più probabile (bias prevalente in grassetto,
    scenario opposto più sfumato) — stesso spirito dei recap settimanali
    dei canali Telegram di trading, ma generato dal bot con i suoi numeri.

    Ritorna il path del PNG temporaneo: il chiamante deve rimuoverlo dopo
    l'invio su Telegram.
    """
    plot_df = df.tail(60).copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    span = max(zones["week_high"] - zones["week_low"], 1.0)
    up = (plot_df["Close"] >= plot_df["Open"]).values
    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        color = "#26a69a" if up[i] else "#ef5350"
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1, zorder=2)
        y0, y1 = sorted([row["Open"], row["Close"]])
        h = max(y1 - y0, span * 0.002)
        ax.add_patch(plt.Rectangle((i - 0.3, y0), 0.6, h, color=color, zorder=3))

    # Zone supporto/resistenza (bande) — stessi valori del testo
    ax.axhspan(zones["resistance"], zones["week_high"], color="#ef5350", alpha=0.12, zorder=1)
    ax.axhspan(zones["week_low"], zones["support"], color="#26a69a", alpha=0.12, zorder=1)
    ax.axhline(zones["current_price"], color="#37474f", linestyle=":", linewidth=1.2, zorder=2)

    # Linea di trend (regressione lineare sulle chiusure mostrate)
    z = np.polyfit(x, plot_df["Close"].values, 1)
    trend_color = "#2e7d32" if z[0] > 0 else ("#c62828" if z[0] < 0 else "#757575")
    ax.plot(x, np.polyval(z, x), color=trend_color, linestyle="--", linewidth=1.6, alpha=0.8, zorder=3)

    last_x = len(plot_df) - 1
    arrow_tip_x = last_x + 6
    label_x = last_x + 8
    ax.set_xlim(-1, last_x + 17)

    bullish_target = max(zones["resistance"] + (zones["resistance"] - zones["current_price"]) * 0.3, zones["resistance"])
    bearish_target = min(zones["support"] - (zones["current_price"] - zones["support"]) * 0.3, zones["support"])
    main_is_bull = bias == "BUY"
    main_is_bear = bias == "SELL"

    # Ylim esplicito: gli endpoint delle frecce non entrano nell'autoscale
    # di matplotlib (annotate() non aggiorna i datalim), quindi senza
    # forzarlo lo scenario ribassista può finire tagliato fuori dal grafico.
    y_min = min(zones["week_low"], bearish_target, float(plot_df["Low"].min()))
    y_max = max(zones["week_high"], bullish_target, float(plot_df["High"].max()))
    pad = (y_max - y_min) * 0.06
    ax.set_ylim(y_min - pad, y_max + pad)

    def arrow(y_target, main, color):
        ax.annotate(
            "", xy=(arrow_tip_x, y_target), xytext=(last_x, zones["current_price"]),
            arrowprops=dict(
                arrowstyle="-|>", color=color,
                lw=2.4 if main else 1.4,
                alpha=1.0 if main else 0.35,
                linestyle="-" if main else "--",
            ),
            zorder=4,
        )

    arrow(bullish_target, main_is_bull, "#2e7d32")
    arrow(bearish_target, main_is_bear, "#c62828")

    def label(y, text, color):
        ax.text(label_x, y, f"{text}  {y:,.2f}", va="center", fontsize=9,
                 color=color, fontweight="bold", zorder=5,
                 bbox=dict(facecolor="#ffffff", edgecolor="none", pad=1.5))

    label(zones["resistance"], "Resistenza", "#c62828")
    label(zones["support"], "Supporto", "#2e7d32")
    label(zones["current_price"], "Prezzo attuale", "#37474f")

    ax.set_title("GOLD WEEKLY OUTLOOK — XAU/USD (4H)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks([])
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.text(0.5, 0.01, "GoldMind v2 · analisi automatica, non è consiglio finanziario",
              ha="center", fontsize=7.5, color="#9e9e9e")

    fd, path = tempfile.mkstemp(suffix=".png", prefix="weekly_outlook_")
    os.close(fd)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
