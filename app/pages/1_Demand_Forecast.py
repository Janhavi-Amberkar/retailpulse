import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from ui import P

ui.setup("Demand Forecast", "📈")
ui.page_view("forecast")
ui.hero(
    "Demand Forecasting",
    "Prophet + PyTorch-Lightning LSTM ensemble · 30-day horizon · rolling-origin validation "
    "on 7 folds plus an untouched peak-season hold-out.",
    "F-03 · Time-series",
)
if not ui.need("forecast_future", "forecast_backtest", "daily_sales"):
    st.stop()

daily = ui.parquet("daily_sales")
fut = ui.parquet("forecast_future")
bt = ui.parquet("forecast_backtest")
m = ui.jreport("forecast_metrics.json")
open_d = daily[daily["is_open"].astype(bool)]

t1, t2, t3, t4 = st.tabs(["🔮 30-day forecast", "🎯 Accuracy & validation", "🧬 Decomposition", "🧪 What-if scenarios"])

# ------------------------------------------------------------ forecast ------
with t1:
    c = st.columns([1, 1, 1, 2])
    model = c[0].selectbox("Model", ["ensemble", "prophet", "lstm"], help="Ensemble = weighted Prophet + LSTM")
    hist_days = c[1].slider("History shown (days)", 30, 365, 120, 15)
    band = c[2].toggle("80% interval", True)
    h = open_d.tail(int(hist_days * 6 / 7))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h["date"], y=h["units_capped"], name="Actual units", mode="lines", line=dict(color=P["cyan"], width=2)
        )
    )
    if band:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fut["date"], fut["date"][::-1]]),
                y=pd.concat([fut["upper"], fut["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(124,92,255,.22)",
                line=dict(width=0),
                name="80% interval",
                hoverinfo="skip",
            )
        )
    fut_p = ui.break_gaps(fut)
    fig.add_trace(
        go.Scatter(
            x=fut_p["date"],
            y=fut_p[model],
            name=f"{model.title()} forecast",
            mode="lines+markers",
            line=dict(color=P["violet"], width=2.5),
            marker=dict(size=5),
        )
    )
    fig.add_vline(x=daily["date"].max(), line=dict(color=P["amber"], dash="dot"))
    fig.add_annotation(
        x=daily["date"].max(),
        y=0.03,
        yref="paper",
        text=" forecast origin",
        showarrow=False,
        font=dict(color=P["amber"], size=11),
        xanchor="left",
    )
    ui.show(ui.style(fig, 400, title="Daily units — history and next 30 days (trading days only)"), key="f1")

    tot = fut[model].sum()
    last_tot = open_d.tail(26)["units_capped"].sum()
    avg_price = daily.tail(60)["revenue"].sum() / max(daily.tail(60)["units"].sum(), 1)
    ui.kpis(
        [
            ("Forecast units (30d)", ui.num(tot), f"{len(fut)} trading days", P["violet"]),
            (
                "Per trading day vs last 26",
                f"{(tot / len(fut) / (last_tot / 26) - 1) * 100:+.1f}%",
                f"{ui.num(tot / len(fut))}/day vs {ui.num(last_tot / 26)}/day (pre-Christmas wind-down)",
                P["cyan"],
            ),
            ("Implied revenue", ui.money(tot * avg_price), f"@ £{avg_price:.2f}/unit recent avg", P["emerald"]),
            ("Weekly MAPE (CV)", f'{m["cv"]["ensemble"]["mape_weekly"]:.1f}%', "target ≤ 12%", P["amber"]),
        ]
    )
    st.markdown(
        '<span class="rp-note">The shop is closed on Saturdays, UK bank holidays and 24 Dec – 3 Jan, so the '
        "calendar rule removes those days from the horizon (model learns this from 2 years of data).</span>",
        unsafe_allow_html=True,
    )
    out = fut[["date", "prophet", "lstm", "ensemble", "lower", "upper"]].copy()
    out[["prophet", "lstm", "ensemble", "lower", "upper"]] = out[
        ["prophet", "lstm", "ensemble", "lower", "upper"]
    ].round(0)
    ui.df(out, hide_index=True, height=260)
    ui.csv_button(out, "⬇ Download forecast CSV", "retailpulse_forecast_30d.csv", "forecast")

# ------------------------------------------------------------ accuracy ------
with t2:
    cv, te = m["cv"], m["test"]
    names = {
        "naive": "Seasonal naive",
        "prophet": "Prophet",
        "lstm_raw": "LSTM (raw)",
        "lstm": "LSTM (calibrated)",
        "ensemble": "Ensemble ★",
    }
    rows = []
    for k, label in names.items():
        for part, d in (("CV (7 folds)", cv), ("Hold-out", te)):
            if k in d:
                rows.append(
                    {
                        "Model": label,
                        "Split": part,
                        "Daily MAPE %": d[k]["mape_daily"],
                        "Weekly MAPE %": d[k]["mape_weekly"],
                        "30-day total error %": d[k]["ape_30d_total"],
                        "Daily WAPE %": d[k]["wape_daily"],
                    }
                )
    tbl = pd.DataFrame(rows)
    ui.df(
        tbl.style.format({c: "{:.1f}" for c in tbl.columns if "%" in c}).highlight_min(
            subset=["Weekly MAPE %", "30-day total error %"], color="rgba(52,211,153,.35)"
        ),
        hide_index=True,
    )
    st.markdown(
        f'<div class="rp-card"><b>How to read this.</b> The spec asks for MAPE ≤ 12 %. Daily unit totals are dominated by '
        f'a handful of wholesale orders, so <b>daily</b> MAPE stays near {cv["ensemble"]["mape_daily"]:.0f} % (CV) for every model, '
        f"including the naive baseline. At the granularity that drives replenishment the ensemble reaches "
        f'<b>{cv["ensemble"]["mape_weekly"]:.1f} % weekly MAPE</b> (hold-out {te["ensemble"]["mape_weekly"]:.1f} %) and '
        f'<b>{cv["ensemble"]["ape_30d_total"]:.1f} % error on the 30-day total</b> (hold-out {te["ensemble"]["ape_30d_total"]:.1f} %). '
        f'Ensemble weight on LSTM = {m["w_lstm"]:.2f}; LSTM bias calibration k = {m["k_lstm"]:.2f}, both fitted on CV folds only.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.5, 1])
    with c1:
        folds = sorted(bt["fold"].unique())
        fsel = st.select_slider(
            "Back-test fold",
            folds,
            value=folds[-1],
            format_func=lambda f: f"#{f}" + (" · hold-out" if f == folds[-1] else ""),
        )
        g = bt[bt["fold"] == fsel].sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["date"], y=g["actual"], name="Actual", line=dict(color=P["cyan"], width=2.5)))
        for k, col in (("prophet", P["violet"]), ("lstm", P["amber"]), ("ensemble", P["emerald"])):
            fig.add_trace(
                go.Scatter(
                    x=g["date"],
                    y=g[k],
                    name=k.title(),
                    line=dict(color=col, width=1.8, dash="dot" if k != "ensemble" else "solid"),
                )
            )
        ui.show(
            ui.style(fig, 360, title=f"Fold #{fsel}: forecast vs actual (origin {g['origin'].iloc[0]:%d %b %Y})"),
            key="f2",
        )
    with c2:
        pf = ui.parquet("forecast_per_fold")
        pe = pf[pf["model"].isin(["prophet", "ensemble", "naive"])]
        fig = go.Figure()
        for k, col in (("naive", "#64748b"), ("prophet", P["violet"]), ("ensemble", P["emerald"])):
            d = pe[pe["model"] == k]
            fig.add_trace(go.Bar(x=d["fold"].astype(str), y=d["mape_weekly"], name=k.title(), marker_color=col))
        fig.add_hline(y=12, line=dict(color=P["rose"], dash="dash"), annotation_text="12% target")
        fig.update_layout(barmode="group")
        ui.show(ui.style(fig, 360, title="Weekly MAPE per fold"), key="f3")

    wg = ui.parquet("ensemble_weight_grid")
    fig = go.Figure()
    for k, col in (
        ("mape_daily", P["amber"]),
        ("mape_weekly", P["cyan"]),
        ("ape_30d_total", P["emerald"]),
        ("composite", P["violet"]),
    ):
        fig.add_trace(
            go.Scatter(x=wg["w_lstm"], y=wg[k], name=k, line=dict(color=col, width=3 if k == "composite" else 1.6))
        )
    fig.add_vline(x=m["w_lstm"], line=dict(color="#fff", dash="dot"))
    ui.show(ui.style(fig, 300, title="Ensemble weight search (CV folds) — white line = chosen LSTM weight"), key="f4")

# --------------------------------------------------------- decomposition ----
with t3:
    dec = ui.parquet("ts_decomposition")
    diag = m["diagnostics"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Observed (units)", "Trend (STL)", "Weekly seasonality + residual"),
    )
    fig.add_trace(go.Scatter(x=dec["date"], y=dec["observed"], line=dict(color=P["cyan"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=dec["date"], y=dec["trend"], line=dict(color=P["violet"], width=2.5)), 2, 1)
    fig.add_trace(
        go.Scatter(x=dec["date"], y=dec["seasonal"], line=dict(color=P["emerald"], width=1), name="seasonal"), 3, 1
    )
    fig.add_trace(go.Scatter(x=dec["date"], y=dec["resid"], line=dict(color=P["rose"], width=1), name="residual"), 3, 1)
    ui.show(ui.style(fig, 560, False), key="dec")
    ui.kpis(
        [
            (
                "ADF p-value (level)",
                f'{diag["adf_p"]:.3f}',
                "non-stationary" if not diag["stationary_level"] else "stationary",
                P["rose"] if not diag["stationary_level"] else P["emerald"],
            ),
            ("ADF p-value (1st diff)", f'{diag["adf_diff_p"]:.1e}', "stationary after differencing", P["emerald"]),
            (
                "KPSS p-value",
                f'{diag["kpss_p"]:.3f}',
                "rejects stationarity" if diag["kpss_p"] < 0.05 else "stationary",
                P["amber"],
            ),
            ("Trend strength", f'{diag["strength_trend"]:.2f}', "STL, 0-1", P["violet"]),
            ("Seasonal strength", f'{diag["strength_seasonal"]:.2f}', "weekly cycle", P["cyan"]),
        ]
    )
    st.caption(
        "Series is non-stationary (trend + Q4 peak) → Prophet models trend/seasonality explicitly and the LSTM sees "
        "log-scaled windows with calendar features instead of raw levels."
    )

# --------------------------------------------------------------- what-if ----
with t4:
    st.markdown(
        "Stress-test the forecast. Scenario maths is applied to the model output (no retraining), so it is instant."
    )
    c = st.columns(4)
    promo = c[0].slider("Promotion uplift %", 0, 60, 0, 5, help="Extra demand on promoted days")
    promo_share = c[1].slider("Share of days promoted %", 0, 100, 40, 10)
    price = c[2].slider("Price change %", -30, 30, 0, 1)
    elast = c[3].slider("Price elasticity", -3.0, -0.2, -1.2, 0.1, help="Assumed - not estimated from data")
    c = st.columns(3)
    shock = c[0].slider("Market shock %", -40, 40, 0, 5, help="Macro demand shift")
    lost_days = c[1].slider("Extra closed trading days", 0, 10, 0, 1, help="Strike / supply disruption")
    start_late = c[2].slider("Shift forecast later (days)", 0, 14, 0, 1)
    base = fut.copy()
    mult = (1 + promo / 100 * promo_share / 100) * (1 + price / 100) ** elast * (1 + shock / 100)
    scen = base[["date", "ensemble", "lower", "upper"]].copy()
    for col in ("ensemble", "lower", "upper"):
        scen[col] = scen[col] * mult
    if lost_days:
        scen.loc[scen.index[-lost_days:], ["ensemble", "lower", "upper"]] = 0
    avg_price = daily.tail(60)["revenue"].sum() / max(daily.tail(60)["units"].sum(), 1)
    base_rev = base["ensemble"].sum() * avg_price
    scen_rev = scen["ensemble"].sum() * avg_price * (1 + price / 100)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=base["date"], y=base["ensemble"], name="Baseline", line=dict(color=P["violet"], width=2))
    )
    fig.add_trace(
        go.Scatter(x=scen["date"], y=scen["ensemble"], name="Scenario", line=dict(color=P["amber"], width=2.5))
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([scen["date"], scen["date"][::-1]]),
            y=pd.concat([scen["upper"], scen["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(251,191,36,.12)",
            line=dict(width=0),
            name="scenario 80%",
        )
    )
    ui.show(ui.style(fig, 340, title="Baseline vs scenario demand"), key="wi")
    ui.kpis(
        [
            (
                "Scenario units",
                ui.num(scen["ensemble"].sum()),
                f'{(scen["ensemble"].sum() / base["ensemble"].sum() - 1) * 100:+.1f}% vs baseline',
                P["amber"],
            ),
            (
                "Scenario revenue",
                ui.money(scen_rev),
                f"{(scen_rev / base_rev - 1) * 100:+.1f}% vs {ui.money(base_rev)}",
                P["emerald"],
            ),
            ("Demand multiplier", f"{mult:.2f}×", "promo × price effect × shock", P["cyan"]),
        ]
    )
    st.caption("Elasticity and promo uplift are user assumptions: the dataset has no price experiments or promo flags.")
