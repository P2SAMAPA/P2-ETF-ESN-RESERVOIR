"""app.py — Echo State Network · Reservoir Computing Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="ESN Reservoir · P2Quant",
    layout="wide",
    page_icon="🌊",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PALETTE = [
    "#1B4F8A", "#27AE60", "#E74C3C", "#F39C12",
    "#8E44AD", "#148F77", "#CA6F1E", "#2471A3",
    "#CB4335", "#1A5276", "#117A65", "#B7950B",
]

SCORE_COLOURS = {
    "positive_high": "#1D9E75",
    "positive_low":  "#82C3A9",
    "negative_low":  "#F0A07A",
    "negative_high": "#E74C3C",
}


def score_colour(v: float) -> str:
    if v >= 0.5:  return SCORE_COLOURS["positive_high"]
    if v >= 0.0:  return SCORE_COLOURS["positive_low"]
    if v >= -0.5: return SCORE_COLOURS["negative_low"]
    return SCORE_COLOURS["negative_high"]


def fmt_pct(v: float) -> str:
    return f"{v:+.4f}"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading ESN results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🌊 Echo State Network · Reservoir Computing Engine")
st.caption(
    "1 000-node fixed random reservoir · Leaky-integrator neurons · "
    "Ridge readout only · 5-ensemble average · "
    "Score = 70% ESN forecast + 30% EWM momentum"
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**ETFs in universe:** {len(config.UNIVERSES[universe])}")
    st.markdown(f"**Reservoir size:** {config.RESERVOIR_SIZE:,} nodes")
    st.markdown(f"**Spectral radius:** {config.SPECTRAL_RADIUS}")
    st.markdown(f"**Leak rate (α):** {config.LEAK_RATE}")
    st.markdown(f"**Ensembles:** {config.N_ENSEMBLES}")
    st.markdown(f"**Input window:** {config.INPUT_WINDOW} days")
    st.markdown(f"**Refit every:** {config.REFIT_FREQ} days")
    st.markdown(f"**Top N:** {config.TOP_N}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Data split:**")
    st.markdown(f"- Warm-up: 2008 → {config.WARMUP_END}")
    st.markdown(f"- Train:   → {config.TRAIN_END}")
    st.markdown(f"- Val:     → {config.VALIDATE_END}")
    st.markdown(f"- OOS:     {config.VALIDATE_END[:4]} → present")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

slug       = universe.lower().replace("_", "-")
data       = load_json(universe)
daily_df   = load_csv(f"daily_{slug}.csv")
score_df   = load_csv(f"scores_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

latest_scores  = data.get("latest_scores", {})
latest_ranked  = data.get("latest_ranked", [])
latest_date    = data.get("latest_date", "?")
run_date       = data.get("run_date", "?")
cfg            = data.get("config", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run Date",    run_date)
k2.metric("Latest Date", latest_date)
k3.metric("Universe",    universe)
k4.metric("ETFs scored", len(latest_scores))

if latest_ranked:
    top       = latest_ranked[0]
    cash_flag = data.get("latest_ranked", [{}])[0].get("composite_score", 0) < config.CASH_THRESHOLD
    bottom    = latest_ranked[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Top Pick",
              "CASH" if cash_flag else top["ticker"],
              help="Highest composite ESN score")
    m2.metric("Top Score",   fmt_pct(top.get("composite_score", 0)))
    m3.metric("Bottom Pick", bottom["ticker"],
              delta=fmt_pct(bottom.get("composite_score", 0)),
              delta_color="inverse")
    m4.metric("CASH Signal", "Yes ⚠️" if cash_flag else "No ✅")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "📈 Score History",
    "🔁 Refit Calendar",
    "📊 Cross-Sectional Spread",
    "📋 Full Scores Table",
])

# ── Tab 1: Rankings & Scores ──────────────────────────────────────────────────
with tab1:
    st.subheader(f"ESN Rankings as of {latest_date}")

    if latest_ranked:
        tickers_r = [r["ticker"] for r in latest_ranked]
        scores_r  = [r.get("composite_score", 0) for r in latest_ranked]
        colours_r = [score_colour(s) for s in scores_r]

        left, right = st.columns(2)

        with left:
            st.markdown("**Composite Score (ESN forecast + Momentum)**")
            fig_rank = go.Figure(go.Bar(
                y=tickers_r,
                x=scores_r,
                orientation="h",
                marker_color=colours_r,
                text=[fmt_pct(s) for s in scores_r],
                textposition="outside",
            ))
            fig_rank.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_rank.update_layout(
                title="Composite Score = 70% ESN z-score + 30% Momentum z-score",
                xaxis_title="Score",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(tickers_r) * 30),
                margin=dict(t=50, b=20, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rank, use_container_width=True, key="rank_bar")

        with right:
            st.markdown("**ESN Raw Forecast vs EWM Momentum Contribution**")
            esn_vals = [r.get("esn_forecast", 0) for r in latest_ranked]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="ESN Forecast",
                y=tickers_r,
                x=esn_vals,
                orientation="h",
                marker_color="#1B4F8A",
            ))
            fig_comp.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_comp.update_layout(
                title="Raw ESN predicted return (before z-scoring)",
                xaxis_title="Predicted return",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(tickers_r) * 30),
                margin=dict(t=50, b=20, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_comp, use_container_width=True, key="esn_bar")

        # Top N recommendation cards
        st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
        cols = st.columns(config.TOP_N)
        for i, row in enumerate(latest_ranked[: config.TOP_N]):
            with cols[i]:
                sc = row.get("composite_score", 0)
                bg = score_colour(sc)
                rank_label = f"Rank #{row.get('rank', i+1)}"
                st.markdown(
                    f"**#{i+1} {row['ticker']}**\n\n"
                    f"Score: `{fmt_pct(sc)}`\n\n"
                    f"ESN pred: `{fmt_pct(row.get('esn_forecast', 0))}`\n\n"
                    f'<span style="background:{bg};color:white;padding:2px 8px;'
                    f'border-radius:8px;font-size:11px">{rank_label}</span>',
                    unsafe_allow_html=True,
                )

# ── Tab 2: Score History ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Composite Score History Over Time")

    if score_df is not None:
        avail_tickers = [
            c for c in score_df.columns if c in config.UNIVERSES[universe]
        ]
        selected = st.multiselect(
            "Select ETFs to display",
            avail_tickers,
            default=avail_tickers[:6],
            key="score_select",
        )

        if selected:
            # OOS period only by default
            show_period = st.radio(
                "Period", ["OOS only (2022+)", "Validate + OOS (2020+)", "Full history"],
                horizontal=True,
            )
            df_plot = score_df.copy()
            if show_period == "OOS only (2022+)":
                df_plot = df_plot[df_plot.index >= "2022-01-01"]
            elif show_period == "Validate + OOS (2020+)":
                df_plot = df_plot[df_plot.index >= "2020-01-01"]

            fig_hist = go.Figure()
            for i, tkr in enumerate(selected):
                if tkr in df_plot.columns:
                    fig_hist.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot[tkr],
                        mode="lines",
                        name=tkr,
                        line=dict(width=1.5, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_hist.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_hist.update_layout(
                title="Composite ESN Score Over Time (cross-sectionally z-scored)",
                yaxis_title="Composite Score",
                height=430,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="score_hist")

            # Score heatmap — last 252 days
            recent = score_df[avail_tickers].tail(252)
            fig_heat = go.Figure(go.Heatmap(
                z=recent.values.T,
                x=recent.index.strftime("%Y-%m-%d"),
                y=avail_tickers,
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Score"),
                hoverongaps=False,
            ))
            fig_heat.update_layout(
                title="Score Heatmap — last 252 days (green=buy, red=avoid)",
                height=max(300, len(avail_tickers) * 22 + 80),
                margin=dict(t=40, b=60, l=60, r=20),
                xaxis=dict(tickangle=-45, nticks=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="score_heat")
    else:
        st.info("No score history found.")

# ── Tab 3: Refit Calendar ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Reservoir Refit Calendar")
    st.caption(
        f"Readout weights are re-estimated every {config.REFIT_FREQ} trading days "
        f"on a rolling {config.REFIT_WINDOW}-day window. "
        "The reservoir weights (W, W_in) are fixed for life — only W_out changes."
    )

    if daily_df is not None and "refit_day" in daily_df.columns:
        refit_dates = daily_df[daily_df["refit_day"] == True].index

        # Timeline of top picks
        fig_cal = go.Figure()

        if "top_ticker" in daily_df.columns:
            cash_mask = daily_df["top_ticker"] == "CASH"
            fig_cal.add_trace(go.Scatter(
                x=daily_df[~cash_mask].index,
                y=daily_df[~cash_mask]["top_score"],
                mode="lines",
                name="Top Score",
                line=dict(color="#1B4F8A", width=1.2),
            ))
            # CASH signals
            if cash_mask.any():
                fig_cal.add_trace(go.Scatter(
                    x=daily_df[cash_mask].index,
                    y=daily_df[cash_mask]["top_score"],
                    mode="markers",
                    name="CASH signal",
                    marker=dict(color="#E74C3C", size=5, symbol="x"),
                ))

        # Refit vertical lines — x must be a string; Timestamps cause Plotly to crash
        for rd in refit_dates[-20:]:
            rd_str = rd.strftime("%Y-%m-%d") if hasattr(rd, "strftime") else str(rd)
            fig_cal.add_vline(
                x=rd_str,
                line_dash="dash",
                line_color="rgba(100,100,100,0.3)",
            )

        fig_cal.update_layout(
            title="Top-1 ESN Score Over Time with Refit Events",
            yaxis_title="Composite Score",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cal, use_container_width=True, key="refit_cal")

        # Top-pick frequency bar chart
        if "top_ticker" in daily_df.columns:
            oos_daily = daily_df[daily_df.index >= config.VALIDATE_END]
            pick_counts = oos_daily["top_ticker"].value_counts()
            fig_freq = go.Figure(go.Bar(
                x=pick_counts.index,
                y=pick_counts.values,
                marker_color="#1D9E75",
                text=pick_counts.values,
                textposition="outside",
            ))
            fig_freq.update_layout(
                title=f"Top-Pick Frequency (OOS: {config.VALIDATE_END}→present)",
                yaxis_title="Days as #1 pick",
                height=300,
                margin=dict(t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_freq, use_container_width=True, key="pick_freq")
    else:
        st.info("No daily history found.")

# ── Tab 4: Cross-Sectional Spread ─────────────────────────────────────────────
with tab4:
    st.subheader("Cross-Sectional Score Spread")
    st.caption(
        "Spread = max(score) − min(score) per day. "
        "Wide spread = high signal conviction. "
        "Narrow spread → ESN sees little differentiation (CASH-leaning regime)."
    )

    if score_df is not None:
        etf_cols = [c for c in score_df.columns if c in config.UNIVERSES[universe]]
        spread   = score_df[etf_cols].max(axis=1) - score_df[etf_cols].min(axis=1)
        spread   = spread[spread.index >= "2020-01-01"]

        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            line=dict(color="#8E44AD", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(142,68,173,0.08)",
            name="Cross-sectional spread",
        ))
        # Rolling 63-day mean spread
        roll_mean = spread.rolling(63).mean()
        fig_spread.add_trace(go.Scatter(
            x=roll_mean.index,
            y=roll_mean.values,
            mode="lines",
            line=dict(color="#E74C3C", width=1.5, dash="dot"),
            name="63d rolling mean",
        ))
        fig_spread.update_layout(
            title="Daily Cross-Sectional Score Spread (max − min)",
            yaxis_title="Spread",
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_spread, use_container_width=True, key="spread_chart")

        # Rank stability: how often does #1 stay #1 day-over-day?
        if ranking_df is not None:
            rank_cols = [c for c in ranking_df.columns if c in config.UNIVERSES[universe]]
            top1_series = ranking_df[rank_cols].idxmin(axis=1)
            stability   = (top1_series == top1_series.shift(1)).mean()
            st.metric(
                "Rank-1 Day-over-Day Stability (OOS)",
                f"{stability:.1%}",
                help="Fraction of days where the top-ranked ETF stays the same as previous day",
            )

            # Rolling 21-day rank of each ETF
            st.markdown("**Rolling 63-day average rank per ETF (lower = better)**")
            avg_ranks = ranking_df[rank_cols].rolling(63).mean().tail(252)
            fig_ranks = go.Figure()
            for i, tkr in enumerate(rank_cols[:8]):
                fig_ranks.add_trace(go.Scatter(
                    x=avg_ranks.index,
                    y=avg_ranks[tkr],
                    mode="lines",
                    name=tkr,
                    line=dict(width=1.2, color=PALETTE[i % len(PALETTE)]),
                ))
            fig_ranks.update_layout(
                title="Rolling 63-day Average Rank (1=best) — last 252 days",
                yaxis_title="Avg Rank",
                yaxis_autorange="reversed",
                height=360,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_ranks, use_container_width=True, key="rank_roll")
    else:
        st.info("No score history found.")

# ── Tab 5: Full Scores Table ──────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full Scores Table — {latest_date}")

    if latest_ranked:
        rows = []
        for i, row in enumerate(latest_ranked):
            rows.append({
                "Rank":           i + 1,
                "Ticker":         row["ticker"],
                "Composite Score":fmt_pct(row.get("composite_score", 0)),
                "ESN Forecast":   fmt_pct(row.get("esn_forecast", 0)),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=600,
        )

    st.divider()
    # Reservoir config summary
    st.markdown("**Reservoir Configuration**")
    cfg_rows = [
        {"Parameter": k, "Value": str(v)}
        for k, v in cfg.items()
    ]
    st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True,
                 hide_index=True, height=350)

    st.divider()
    st.caption(
        f"P2Quant ESN Engine · Run: {run_date} · "
        f"Reservoir Computing / Echo State Network · "
        f"No backprop — fixed W, W_in; ridge-trained W_out · "
        f"Data: {config.HF_DATA_REPO}"
    )
