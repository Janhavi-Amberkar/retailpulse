"""RetailPulse - Overview page (routed by app/streamlit_app.py)."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import P

ui.setup("Overview", "📊")
ui.page_view("overview")

daily = ui.parquet("daily_sales")
etl = ui.jreport("etl_report.json")
fcm = ui.jreport("forecast_metrics.json")
chm = ui.jreport("churn_metrics.json")

ui.hero(
    "RetailPulse",
    "AI-powered customer analytics &amp; demand forecasting — demand forecasts, customer segments, churn risk "
    "and inventory recommendations from <b>1.0M+ real retail transactions</b>.",
    "Zidio Data Science & Analytics · v2.0",
)

# ------------------------------------------------------------------ KPIs ----
open_days = daily[daily["is_open"].astype(bool)]
rev = daily["revenue"].sum()
inv = daily["invoices"].sum()
last30, prev30 = open_days.tail(26), open_days.iloc[-52:-26]
d_rev = (last30["revenue"].sum() / prev30["revenue"].sum() - 1) * 100
ret_rate = daily["return_value"].sum() / rev * 100
ui.kpis(
    [
        (
            "Total revenue",
            ui.money(rev),
            f'<span class="{"up" if d_rev >= 0 else "down"}">{d_rev:+.1f}%</span> last 26 trading days',
            P["violet"],
        ),
        ("Orders", ui.num(inv), f"avg basket {ui.money(rev / inv)}", P["cyan"]),
        ("Customers", f'{etl.get("customers", 0):,}', "identified (pseudonymised)", P["emerald"]),
        ("Active SKUs", f'{etl.get("skus", 0):,}', "after cleaning", P["amber"]),
        ("Return rate", f"{ret_rate:.1f}%", "of revenue (cancelled invoices)", P["rose"]),
        (
            "Data retained",
            f'{etl.get("retention_pct", 0):.1f}%',
            f'{etl.get("rows_clean", 0):,} clean rows',
            P["violet"],
        ),
    ]
)

# --------------------------------------------------------------- alerts -----
ui.section("Live alerts")
a1, a2 = st.columns([1.25, 1])
with a1:
    msgs = []
    if ui.have("forecast_future"):
        fut = ui.parquet("forecast_future")
        f30 = fut["ensemble"].sum()
        per_day, last_day = f30 / len(fut), open_days.tail(26)["units_capped"].mean()
        msgs.append(
            (
                "ok",
                f"Next-30-day demand forecast: <b>{ui.num(f30)} units</b> over {len(fut)} trading days = "
                f"<b>{ui.num(per_day)}/day</b> vs {ui.num(last_day)}/day in the last 26 trading days "
                f"(pre-Christmas wind-down, shop closed 24 Dec – 3 Jan).",
            )
        )
    if ui.have("inventory_recommendations"):
        rec = ui.parquet("inventory_recommendations")
        n_out = int((rec["status"] == "Stockout risk").sum())
        n_re = int((rec["status"] == "Reorder now").sum())
        n_ov = int((rec["status"] == "Overstock").sum())
        msgs.append(
            (
                "crit" if n_out else "ok",
                f"<b>{n_out}</b> SKUs at stock-out risk · <b>{n_re}</b> need reordering · "
                f"<b>{n_ov}</b> overstocked (simulated stock levels).",
            )
        )
    if ui.have("churn_scores"):
        cs = ui.parquet("churn_scores")
        crit = cs[cs["risk_tier"] == "Critical"]
        msgs.append(
            (
                "warn",
                f"<b>{len(crit):,}</b> customers in the <i>Critical</i> churn tier (top 15 % of active customers by predicted risk) — "
                f"<b>{ui.money(crit['expected_loss'].sum())}</b> expected 180-day revenue at risk.",
            )
        )
    dr = ui.jreport("drift_summary.json")
    if dr:
        d = dr["decision"]
        msgs.append(
            (
                "warn" if d["retrain"] else "ok",
                (
                    ("Drift monitor recommends <b>retraining</b>: " + "; ".join(d["reasons"]))
                    if d["retrain"]
                    else "Drift monitor: feature distributions stable — no retraining needed."
                ),
            )
        )
    for lvl, m in msgs:
        ui.alert(m, lvl)

with a2:

    @st.fragment(run_every="4s")
    def live_stream():
        """Replays the last 40 trading days as a simulated real-time feed (demo of the alerting loop)."""
        d = open_days.tail(40).reset_index(drop=True)
        i = st.session_state.get("tick", 8)
        st.session_state["tick"] = i + 1 if i + 1 < len(d) else 8
        cur = d.iloc[i]
        base = d.iloc[max(0, i - 7) : i]["revenue"].mean()
        delta = cur["revenue"] / base - 1
        fig = go.Figure(
            go.Scatter(
                x=d["date"][: i + 1],
                y=d["revenue"][: i + 1],
                mode="lines",
                fill="tozeroy",
                line=dict(color=P["cyan"], width=2),
                fillcolor="rgba(34,211,238,.15)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cur["date"]],
                y=[cur["revenue"]],
                mode="markers",
                marker=dict(
                    size=11, color=P["rose"] if delta < -0.3 else P["emerald"], line=dict(width=2, color="#fff")
                ),
            )
        )
        ui.show(ui.style(fig, 190, False), key="live")
        lvl = "crit" if delta < -0.3 else ("warn" if delta < -0.15 else "ok")
        ui.alert(
            f"<b>{cur['date']:%a %d %b %Y}</b> · revenue {ui.money(cur['revenue'])} "
            f"({delta:+.0%} vs 7-day avg) · {int(cur['invoices'])} orders",
            lvl,
        )
        st.caption("● simulated live stream — replay of historical trading days (refreshes every 4 s)")

    live_stream()

# --------------------------------------------------------------- trends -----
ui.section("Revenue & demand trends")
c1, c2 = st.columns([1.55, 1])
with c1:
    mr = ui.parquet("monthly_region")
    fig = px.area(
        mr,
        x="month_start",
        y="revenue",
        color="region_group",
        color_discrete_sequence=[P["violet"], P["cyan"], P["emerald"], P["amber"]],
    )
    fig.update_traces(line=dict(width=1.2))
    ui.show(ui.style(fig, 360, title="Monthly revenue by region (custom column: region_group)"), key="mr")
with c2:
    ws = open_days.set_index("date")["units_capped"].resample("W").sum()
    ws = ws.iloc[1:-1]
    fig = go.Figure(
        go.Bar(x=ws.index, y=ws.values, marker=dict(color=ws.values, colorscale=[[0, P["violet"]], [1, P["cyan"]]]))
    )
    ui.show(ui.style(fig, 360, False, "Weekly units sold"), key="wk")

c3, c4 = st.columns([1.2, 1])
with c3:
    cs = ui.parquet("country_summary").copy()
    cs["iso_name"] = cs["country"].map(lambda c: ui.COUNTRY_FIX.get(c, c))
    cs = cs.dropna(subset=["iso_name"])
    fig = px.choropleth(
        cs,
        locations="iso_name",
        locationmode="country names",
        color="revenue",
        color_continuous_scale=[[0, "#1b2250"], [0.25, P["violet"]], [1, P["cyan"]]],
        hover_name="country",
        range_color=(0, cs[cs["country"] != "United Kingdom"]["revenue"].max() * 1.2),
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        landcolor="#10162f",
        showcountries=True,
        countrycolor="rgba(255,255,255,.08)",
        projection_type="natural earth",
        lakecolor="rgba(0,0,0,0)",
    )
    fig.update_coloraxes(showscale=False)
    ui.show(ui.style(fig, 380, False, "Revenue by country (UK excluded from colour scale)"), key="geo")
with c4:
    prod = ui.parquet("products").head(10).sort_values("total_revenue")
    fig = go.Figure(
        go.Bar(
            x=prod["total_revenue"],
            y=prod["description"].str.title().str.slice(0, 28),
            orientation="h",
            marker=dict(color=prod["total_revenue"], colorscale=[[0, P["violet"]], [1, P["emerald"]]]),
        )
    )
    ui.show(ui.style(fig, 380, False, "Top 10 products by revenue"), key="top")

hp = ui.parquet("hourly_profile")
piv = hp.pivot(index="dow", columns="hour", values="revenue").reindex(range(7)).fillna(0)
piv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig = px.imshow(piv, aspect="auto", color_continuous_scale=[[0, "#0d1233"], [0.5, P["violet"]], [1, P["cyan"]]])
fig.update_coloraxes(showscale=False)
ui.show(
    ui.style(fig, 270, False, "When customers buy — day × hour revenue heat-map (Saturday: shop closed)"), key="heat"
)

# ---------------------------------------------------- scoreboard ------------
ui.section("Spec scoreboard — measured, not claimed")
if fcm and chm:
    t = fcm["test"]["ensemble"]
    cv = fcm["cv"]["ensemble"]
    fin = chm["final"]
    ok = lambda b: "up" if b else "warn"  # noqa: E731
    ui.kpis(
        [
            (
                "Forecast MAPE · weekly (CV)",
                f'{cv["mape_weekly"]:.1f}%',
                f'target ≤ 12% · hold-out {t["mape_weekly"]:.1f}%',
                P["emerald"],
                ok(cv["mape_weekly"] <= 12),
            ),
            (
                "Forecast MAPE · daily (CV)",
                f'{cv["mape_daily"]:.1f}%',
                f'hold-out {t["mape_daily"]:.1f}% · lumpy bulk orders',
                P["amber"],
                "warn",
            ),
            (
                "30-day total error",
                f'{t["ape_30d_total"]:.1f}%',
                f'CV {cv["ape_30d_total"]:.1f}% · drives replenishment',
                P["cyan"],
                "up",
            ),
            (
                "Churn AUC-ROC",
                f'{fin["auc_roc"]:.3f}',
                "target ≥ 0.88 (out-of-time test)",
                P["rose"],
                ok(fin["auc_roc"] >= 0.88),
            ),
            (
                "Precision @ top 20%",
                f'{fin["precision_top20"]:.2f}',
                f'target ≥ 0.75 · lift {fin["lift_top20"]:.2f}×',
                P["emerald"],
                ok(fin["precision_top20"] >= 0.75),
            ),
        ]
    )
st.caption(
    "Navigate with the sidebar → Demand Forecast · Customer Segments · Churn Risk · Inventory · MLOps & Drift · Data & ETL · Reports"
)
