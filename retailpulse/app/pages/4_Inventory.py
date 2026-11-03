import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from retailpulse import config, inventory
from ui import P

ui.setup("Inventory Optimisation", "📦")
ui.page_view("inventory")
ui.hero(
    "Inventory Optimisation",
    "Reorder points, safety stock, EOQ and order quantities driven by hierarchical Prophet demand forecasts — "
    "with a back-test against moving-average policies on real demand.",
    "F-05 · Prescriptive analytics",
)
if not ui.need(
    "inventory_recommendations",
    "inventory_stock_sim",
    "inventory_policy_summary",
    "sku_weekly",
    "inventory_final_forecast",
):
    st.stop()

im = ui.jreport("inventory_metrics.json")
stock = ui.parquet("inventory_stock_sim")
weekly = ui.parquet("sku_weekly").set_index("week_end")
pol = ui.parquet("inventory_policy_summary")
_ff = ui.parquet("inventory_final_forecast")
final = {c: g.drop(columns="stock_code").reset_index(drop=True) for c, g in _ff.groupby("stock_code")}

st.markdown(
    '<div class="rp-alert warn">🧪 <b>Transparency:</b> the public dataset contains sales only. Demand, forecasts, safety stock, ROP, EOQ and the '
    "policy back-test use <b>real</b> sales. Current on-hand stock and supplier lead times are <b>simulated</b> (seeded) so the logic can be demonstrated; "
    "plug in your ERP stock table to use real values.</div>",
    unsafe_allow_html=True,
)

t1, t2, t3 = st.tabs(["🧮 Recommendations & what-if", "🔬 SKU deep-dive", "🏁 Policy back-test"])

with t1:
    c = st.columns(4)
    sl = c[0].slider("Service level", 0.80, 0.995, float(config.SERVICE_LEVEL), 0.005, format="%.3f")
    lead = c[1].slider("Lead-time stress ×", 0.5, 2.5, 1.0, 0.1, help="1.5 = suppliers 50% slower")
    dm = c[2].slider("Demand multiplier ×", 0.5, 1.8, 1.0, 0.05, help="What if demand is higher/lower than forecast")
    stat = c[3].multiselect("Status filter", ["Stockout risk", "Reorder now", "Healthy", "Overstock"], default=[])
    rec = inventory.recommend(stock, final, service_level=sl, lead_scale=lead, demand_multiplier=dm)
    counts = rec["status"].value_counts()
    ui.kpis(
        [
            (
                "Stock-out risk",
                f'{counts.get("Stockout risk", 0)}',
                "on-hand < demand during lead time",
                P["rose"],
                "down",
            ),
            ("Reorder now", f'{counts.get("Reorder now", 0)}', "below reorder point", P["amber"], "warn"),
            ("Healthy", f'{counts.get("Healthy", 0)}', "inside the target band", P["emerald"], "up"),
            (
                "Overstock",
                f'{counts.get("Overstock", 0)}',
                f'{ui.money(rec["excess_value"].sum())} tied up in excess',
                P["violet"],
            ),
            (
                "Purchase plan",
                ui.money(rec["order_value"].sum()),
                f'{int(rec["recommended_order"].sum()):,} units',
                P["cyan"],
            ),
        ]
    )
    c1, c2 = st.columns([1, 1.6])
    with c1:
        order = ["Stockout risk", "Reorder now", "Healthy", "Overstock"]
        colors = {
            "Stockout risk": P["rose"],
            "Reorder now": P["amber"],
            "Healthy": P["emerald"],
            "Overstock": P["violet"],
        }
        fig = go.Figure(go.Bar(x=order, y=[counts.get(o, 0) for o in order], marker_color=[colors[o] for o in order]))
        ui.show(ui.style(fig, 330, False, "SKUs by inventory status"), key="st")
    with c2:
        top = rec.sort_values("order_value", ascending=False).head(12).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top["order_value"],
                y=top["description"].str.title().str.slice(0, 30),
                orientation="h",
                marker=dict(color=top["order_value"], colorscale=[[0, P["violet"]], [1, P["cyan"]]]),
            )
        )
        ui.show(ui.style(fig, 330, False, "Largest recommended purchase orders (£ at cost)"), key="po")
    t = rec if not stat else rec[rec["status"].isin(stat)]
    cols = [
        "stock_code",
        "description",
        "status",
        "on_hand",
        "forecast_30d",
        "safety_stock",
        "reorder_point",
        "order_up_to",
        "eoq",
        "recommended_order",
        "cover_days",
        "lead_days",
        "order_value",
    ]
    ui.df(
        t[cols].round(0),
        hide_index=True,
        height=380,
        column_config={"description": st.column_config.TextColumn(width="medium")},
    )
    ui.csv_button(t[cols].round(1), "⬇ Download purchase plan CSV", "retailpulse_purchase_plan.csv", "inventory")

with t2:
    sku = st.selectbox(
        "SKU",
        rec["stock_code"],
        format_func=lambda c: f'{c} · {rec.set_index("stock_code").loc[c, "description"].title()[:45]}',
    )
    r = rec.set_index("stock_code").loc[sku]
    f = final[sku]
    h = weekly[sku].tail(40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Weekly units", line=dict(color=P["cyan"], width=2.5)))
    fig.add_trace(
        go.Scatter(
            x=f["week_end"],
            y=f["yhat"] * dm,
            name="Forecast",
            line=dict(color=P["violet"], width=2.5),
            mode="lines+markers",
        )
    )
    z = 1.645
    fig.add_trace(
        go.Scatter(
            x=pd.concat([f["week_end"], f["week_end"][::-1]]),
            y=pd.concat([(f["yhat"] + z * f["sigma"]) * dm, ((f["yhat"] - z * f["sigma"]).clip(lower=0) * dm)[::-1]]),
            fill="toself",
            fillcolor="rgba(124,92,255,.18)",
            line=dict(width=0),
            name="90% interval",
        )
    )
    ui.show(ui.style(fig, 360, title="Weekly demand and forecast"), key="sku")
    ui.kpis(
        [
            ("On hand (sim)", f'{int(r["on_hand"]):,}', f'{r["cover_days"]:.0f} days cover', P["cyan"]),
            ("Reorder point", f'{r["reorder_point"]:,.0f}', f'lead {r["lead_days"]:.0f} d', P["amber"]),
            ("Safety stock", f'{r["safety_stock"]:,.0f}', f"z at {sl * 100:.1f}% service", P["violet"]),
            ("EOQ", f'{r["eoq"]:,.0f}', "min cost order size", P["emerald"]),
            (
                "Order now",
                f'{r["recommended_order"]:,.0f}',
                f'{r["status"]}',
                P["rose"] if r["status"] == "Stockout risk" else P["emerald"],
            ),
        ]
    )

with t3:
    ref = pol.set_index("policy")
    rp = ref.loc["RetailPulse (hierarchical + safety stock)"]
    ma = ref.loc["Moving average (no buffer)"]
    mas = ref.loc["Moving average + safety stock"]
    st.markdown(
        f"Rolling-origin back-test: **{im['n_skus']} SKUs × {im['origins']} weekly origins = {im['n_cycles']:,} replenishment cycles** on real weekly demand. "
        "For each cycle the stock level each policy would have held is compared with the demand that actually occurred over lead time + review period."
    )
    order = [
        "Moving average (no buffer)",
        "Moving average + safety stock",
        "Per-SKU Prophet (no buffer)",
        "Hierarchical forecast (no buffer)",
        "RetailPulse (hierarchical + safety stock)",
    ]
    pp = pol.set_index("policy").loc[order].reset_index()
    cols = [P["amber"], P["cyan"], "#64748b", P["violet"], P["emerald"]]
    c1, c2, c3 = st.columns(3)
    for col, (met, ttl) in zip(
        (c1, c2, c3),
        (
            ("fill_rate_pct", "Fill rate % (higher better)"),
            ("overstock_pct_of_demand", "Overstock % of demand (lower better)"),
            ("total_cost", "Total cost £ = lost margin + holding"),
        ),
    ):
        with col:
            fig = go.Figure(
                go.Bar(x=["MA", "MA+SS", "SKU-Prophet", "Hier.", "RetailPulse"], y=pp[met], marker_color=cols)
            )
            ui.show(ui.style(fig, 330, False, ttl), key=f"b_{met}")
    disp = pp[
        [
            "policy",
            "fill_rate_pct",
            "understock_pct_of_demand",
            "overstock_pct_of_demand",
            "imbalance_units",
            "total_cost",
            "stockout_cycles",
        ]
    ].copy()
    disp["stockout_cycles"] *= 100
    disp.columns = [
        "Policy",
        "Fill rate %",
        "Understock %",
        "Overstock %",
        "Under+over units",
        "Total cost £",
        "Cycles with stock-out %",
    ]
    ui.df(disp.style.format({c: "{:,.1f}" for c in disp.columns[1:]}), hide_index=True)

    def red(a, b):
        return (1 - a / b) * 100

    st.markdown(
        f'<div class="rp-card"><b>Result vs the spec target (reduce over/under-stock by 25–40 %).</b><br>'
        f'RetailPulse vs <i>moving average + safety stock</i>: under-stock {red(rp["understock_units"], mas["understock_units"]):+.0f}% · '
        f'over-stock {red(rp["overstock_units"], mas["overstock_units"]):+.0f}% · total units out of balance {red(rp["imbalance_units"], mas["imbalance_units"]):+.0f}% · '
        f'cost {red(rp["total_cost"], mas["total_cost"]):+.0f}%.<br>'
        f'RetailPulse vs <i>moving average (no buffer)</i>: under-stock {red(rp["understock_units"], ma["understock_units"]):+.0f}% · '
        f'total units out of balance {red(rp["imbalance_units"], ma["imbalance_units"]):+.0f}% · cost {red(rp["total_cost"], ma["total_cost"]):+.0f}%.'
        f'<br><span class="rp-note">Positive = improvement (reduction). Measured on real demand; cost assumptions: unit cost {config.UNIT_COST_RATIO:.0%} of price, '
        f'holding {config.HOLDING_RATE:.0%}/yr. Median per-SKU 4-week WAPE — moving average {im["sku_wape_median"]["ma"]:.0f}% · per-SKU Prophet {im["sku_wape_median"]["direct"]:.0f}% · hierarchical {im["sku_wape_median"]["hier"]:.0f}%.</span></div>',
        unsafe_allow_html=True,
    )
