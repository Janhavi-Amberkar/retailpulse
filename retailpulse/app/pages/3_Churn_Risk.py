import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import P

ui.setup("Churn Risk", "⚠️")
ui.page_view("churn")
ui.hero(
    "Churn Prediction",
    "XGBoost tuned with Optuna, benchmarked against Logistic Regression and Random Forest, "
    "explained with SHAP. Evaluated out-of-time: trained on the past, tested on the most recent 90 days.",
    "F-04 · Classification",
)
if not ui.need("churn_scores", "churn_benchmark", "churn_shap_importance"):
    st.stop()

m = ui.jreport("churn_metrics.json")
fin = m["final"]
scores = ui.parquet("churn_scores")
bench = ui.parquet("churn_benchmark")
imp = ui.parquet("churn_shap_importance")

auc_ok = fin["auc_roc"] >= m["targets"]["auc_min"]
p20_ok = fin["precision_top20"] >= m["targets"]["precision_top20_min"]
ui.kpis(
    [
        (
            "AUC-ROC (test)",
            f'{fin["auc_roc"]:.3f}',
            f'target ≥ {m["targets"]["auc_min"]} · {"met" if auc_ok else "not met"}',
            P["emerald"] if auc_ok else P["rose"],
            "up" if auc_ok else "down",
        ),
        (
            "Precision @ top 20%",
            f'{fin["precision_top20"]:.3f}',
            f'target ≥ {m["targets"]["precision_top20_min"]} · {"met" if p20_ok else "not met"}',
            P["emerald"] if p20_ok else P["rose"],
            "up" if p20_ok else "down",
        ),
        ("Lift @ top 20%", f'{fin["lift_top20"]:.2f}×', f'base churn rate {fin["base_rate"] * 100:.0f}%', P["cyan"]),
        (
            "F1 score",
            f'{fin["f1"]:.3f}',
            f'recall {fin["recall"]:.2f} · precision {fin["precision"]:.2f} · thr {fin["threshold"]:.2f}',
            P["violet"],
        ),
        ("PR-AUC", f'{fin["auc_pr"]:.3f}', f'Brier {fin["brier"]:.3f} · {m["n_features"]} features', P["amber"]),
    ]
)
st.markdown(
    f'<span class="rp-note">Churn = no purchase in the next {m["label_window_days"]} days. Train: {m["n_train"]:,} weekly customer-snapshots (Mar-2010→Feb-2011), '
    f'validation: {m["n_valid"]:,} (Jun-2011), test: {m["n_test"]:,} (snapshot {m["test_snapshot"]}). Label windows never overlap later snapshots → no leakage.</span>',
    unsafe_allow_html=True,
)
if not auc_ok:
    ui.alert(
        f'AUC {fin["auc_roc"]:.3f} is below the 0.88 spec target. With only two years of data, one Christmas season to learn from and strictly out-of-time '
        f"testing, this is the measured ceiling of the current feature set — shown honestly rather than inflated with a random split.",
        "warn",
    )

t1, t2, t3, t4 = st.tabs(
    ["📊 Model quality", "🔍 SHAP explainability", "🎯 At-risk customers", "💸 Retention simulator"]
)

with t1:
    c1, c2 = st.columns(2)
    cv = m["curves"]
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cv["fpr"],
                y=cv["tpr"],
                fill="tozeroy",
                fillcolor="rgba(124,92,255,.18)",
                line=dict(color=P["violet"], width=3),
                name=f'XGBoost AUC {fin["auc_roc"]:.3f}',
            )
        )
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#64748b", dash="dash"), name="Chance"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        ui.show(ui.style(fig, 340, title="ROC curve (test snapshot)"), key="roc")
    with c2:
        dec = pd.DataFrame(cv["deciles"])
        fig = go.Figure(
            go.Bar(
                x=dec["decile"],
                y=dec["churn_rate"] * 100,
                marker=dict(color=dec["churn_rate"], colorscale=[[0, P["emerald"]], [1, P["rose"]]]),
            )
        )
        fig.add_hline(y=fin["base_rate"] * 100, line=dict(color="#fff", dash="dot"), annotation_text="base rate")
        fig.update_xaxes(title="Risk decile (1 = highest predicted risk)")
        fig.update_yaxes(title="Actual churn %")
        ui.show(ui.style(fig, 340, False, "Decile lift — does predicted risk rank real churn?"), key="dec")
    ui.section("Model selection (all fit on train, thresholded on validation, scored on test)")
    cols = ["model", "auc_roc", "auc_pr", "precision_top20", "lift_top20", "f1", "recall", "brier"]
    ui.df(
        bench[cols]
        .style.format({c: "{:.3f}" for c in cols[1:]})
        .highlight_max(subset=["auc_roc"], color="rgba(52,211,153,.35)"),
        hide_index=True,
    )
    bp = m["best_params"]
    st.markdown(
        f'<div class="rp-card"><b>Optuna ({m["optuna_trials"]} TPE trials, best validation AUC {m["optuna_best_valid_auc"]:.3f}):</b> '
        + " · ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in bp.items())
        + f" · n_estimators={m['n_estimators']}</div>",
        unsafe_allow_html=True,
    )

with t2:
    cust_imp = imp[~imp["seasonal_baseline"]] if "seasonal_baseline" in imp else imp
    c1, c2 = st.columns([1, 1.15])
    with c1:
        top = cust_imp.head(15).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top["mean_abs_shap"],
                y=top["feature"],
                orientation="h",
                marker=dict(color=top["mean_abs_shap"], colorscale=[[0, P["violet"]], [1, P["cyan"]]]),
            )
        )
        ui.show(ui.style(fig, 470, False, "Customer-level importance — mean |SHAP|"), key="imp")
    with c2:
        img = ui.config.FIGURES / "shap_summary.png"
        if img.exists():
            ui.image(
                str(img),
                caption="SHAP beeswarm — each dot is a customer; colour = feature value",
            )
    names = ", ".join(f"`{f}`" for f in cust_imp["feature"].head(4))
    st.caption(
        f"Positive SHAP pushes towards churn. Strongest customer-level drivers: {names}. "
        f"Three snapshot-level seasonal features (`snap_sin`, `snap_cos`, `days_to_xmas`) are kept in the model because churn is strongly seasonal, "
        f"but they shift every customer's baseline equally, so they are excluded from rankings and per-customer explanations."
    )
    sbr = pd.DataFrame(m.get("snapshot_base_rates", []))
    if len(sbr):
        sbr["snapshot"] = pd.to_datetime(sbr["snapshot"])
        fig = go.Figure()
        for sp, col in (("train", P["violet"]), ("valid", P["amber"]), ("test", P["cyan"])):
            d_ = sbr[sbr["split"] == sp]
            fig.add_trace(
                go.Scatter(
                    x=d_["snapshot"],
                    y=d_["churn_rate"] * 100,
                    mode="lines+markers",
                    name=sp,
                    line=dict(color=col, width=2),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(m["current_snapshot"])],
                y=[m["current_mean_pred"] * 100],
                mode="markers",
                name="live prediction",
                marker=dict(size=14, color=P["rose"], symbol="diamond", line=dict(width=2, color="#fff")),
            )
        )
        fig.update_yaxes(title="% churned")
        ui.show(
            ui.style(fig, 300, title="Churn is seasonal — base rate by snapshot (Dec–Jan lull ≈ 70 %)"), key="season"
        )

with t3:
    shp = ui.parquet("churn_shap_current")
    c = st.columns([1.3, 1, 1, 1])
    thr = c[0].slider("Risk threshold", 0.3, 0.99, 0.80, 0.01)
    tiers = c[1].multiselect("Risk tier", ["Critical", "High", "Medium", "Low", "Lapsed"], default=["Critical", "High"])
    segs = c[2].multiselect("Segment", sorted(scores["segment"].dropna().unique()))
    min_val = c[3].number_input("Min 180-day spend £", 0, 100000, 0, 100)
    d = scores[(scores["churn_prob"] >= thr) & scores["risk_tier"].isin(tiers) & (scores["monetary_180"] >= min_val)]
    if segs:
        d = d[d["segment"].isin(segs)]
    d = d.sort_values("expected_loss", ascending=False)
    ui.kpis(
        [
            ("Customers flagged", f"{len(d):,}", f"{len(d) / len(scores) * 100:.0f}% of base", P["rose"]),
            ("Expected revenue at risk", ui.money(d["expected_loss"].sum()), "prob × 180-day spend", P["amber"]),
            (
                "Avg churn probability",
                f'{d["churn_prob"].mean() * 100:.0f}%' if len(d) else "–",
                "flagged set",
                P["violet"],
            ),
        ]
    )
    t = (
        d[
            [
                "customer_key",
                "segment",
                "churn_prob",
                "risk_tier",
                "recency",
                "frequency",
                "monetary_180",
                "expected_loss",
                "top_drivers",
            ]
        ]
        .head(300)
        .copy()
    )
    t["customer_key"] = ui.mask_key(t["customer_key"])
    ui.df(
        t.round(2),
        hide_index=True,
        height=360,
        column_config={
            "churn_prob": st.column_config.ProgressColumn("churn prob", min_value=0, max_value=1, format="%.2f")
        },
    )
    ui.csv_button(
        d.assign(top_drivers=d["top_drivers"])[
            ["customer_key", "segment", "churn_prob", "risk_tier", "expected_loss", "top_drivers"]
        ].round(3),
        "⬇ Download at-risk list",
        "retailpulse_at_risk_customers.csv",
        "churn",
    )

    ui.section("Why is this customer at risk?")
    if len(d):
        keys = d["customer_key"].head(200).tolist()
        sel = st.selectbox(
            "Customer",
            keys,
            format_func=lambda k: (k if ui.has("Analyst") else k[:4] + "••••••")
            + f"  ·  p={d.set_index('customer_key').loc[k, 'churn_prob']:.2f}",
        )
        row = shp[shp["customer_key"] == sel].iloc[0].drop("customer_key")
        row.index = [i.replace("shap_", "") for i in row.index]
        season = float(row[[c for c in row.index if c in ("snap_sin", "snap_cos", "days_to_xmas")]].sum())
        row = row.drop([c for c in row.index if c in ("snap_sin", "snap_cos", "days_to_xmas")])
        row = row.reindex(row.abs().sort_values(ascending=False).index)[:9]
        row["season (same for everyone)"] = season
        row = row.iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=row.values,
                y=row.index,
                orientation="h",
                marker_color=[P["rose"] if v > 0 else P["emerald"] for v in row.values],
            )
        )
        ui.show(
            ui.style(fig, 340, False, "Top SHAP contributions (red ↑ raises churn risk, green ↓ lowers it)"), key="wf"
        )

with t4:
    st.markdown("Estimate the payoff of a win-back campaign on the flagged customers.")
    c = st.columns(4)
    budget = c[0].number_input("Campaign budget £", 500, 200000, 10000, 500)
    cost_per = c[1].number_input("Cost per contact £", 0.5, 100.0, 5.0, 0.5)
    save = c[2].slider("Save rate of contacted %", 1, 50, 15, 1, help="Assumption — measure with an A/B test")
    margin = c[3].slider("Gross margin %", 10, 70, 40, 1)
    pool = d if len(d) else scores
    n_contact = int(min(len(pool), budget // cost_per))
    ranked = pool.sort_values("expected_loss", ascending=False).head(n_contact)
    saved = ranked["expected_loss"].sum() * save / 100
    profit = saved * margin / 100 - n_contact * cost_per
    ui.kpis(
        [
            ("Customers contacted", f"{n_contact:,}", f"{ui.money(n_contact * cost_per)} spend", P["violet"]),
            (
                "Revenue retained",
                ui.money(saved),
                f"{save}% of {ui.money(ranked['expected_loss'].sum())} at risk",
                P["emerald"],
            ),
            (
                "Net profit",
                ui.money(profit),
                f"ROI {profit / max(n_contact * cost_per, 1) * 100:+.0f}%",
                P["cyan"],
                "up" if profit > 0 else "down",
            ),
        ]
    )
    x = np.arange(0, min(len(pool), 1500), 25)
    cum = pool.sort_values("expected_loss", ascending=False)["expected_loss"].cumsum().values
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=[cum[min(i, len(cum) - 1)] * save / 100 * margin / 100 - i * cost_per for i in x],
            fill="tozeroy",
            line=dict(color=P["emerald"], width=3),
            fillcolor="rgba(52,211,153,.15)",
        )
    )
    fig.add_vline(x=n_contact, line=dict(color=P["amber"], dash="dot"))
    fig.update_xaxes(title="Customers contacted (highest expected loss first)")
    fig.update_yaxes(title="Net profit £")
    ui.show(ui.style(fig, 320, False, "Profit curve — where to stop contacting"), key="roi")
