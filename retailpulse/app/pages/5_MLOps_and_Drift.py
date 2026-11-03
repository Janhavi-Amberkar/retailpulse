import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import ui
from retailpulse import config, drift, monitoring
from ui import P

ui.setup("MLOps & Drift", "🛰️")
ui.page_view("mlops")
ui.hero(
    "MLOps · Drift · Monitoring",
    "MLflow experiment tracking, PSI/KS + Evidently AI drift detection, a retraining policy, "
    "Prometheus metrics and an audit trail.",
    "Production readiness",
)

dr = ui.jreport("drift_summary.json")
fm, cm = ui.jreport("forecast_metrics.json"), ui.jreport("churn_metrics.json")
bb = ui.jreport("batch_benchmark.json")
if monitoring.HAVE_PROM and fm and cm:
    monitoring.MODEL_MAPE.labels("weekly").set(fm["cv"]["ensemble"]["mape_weekly"])
    monitoring.MODEL_MAPE.labels("daily").set(fm["cv"]["ensemble"]["mape_daily"])
    monitoring.CHURN_AUC.set(cm["final"]["auc_roc"])
    if dr:
        monitoring.DRIFT_SHARE.set(dr["current"]["share_drifted"])

t1, t2, t3, t4, t5 = st.tabs(
    ["🧪 Experiments (MLflow)", "🌊 Drift monitor", "🔁 Retraining loop", "❤️ System health", "🛡️ Security & audit"]
)

# ------------------------------------------------------------ mlflow --------
with t1:
    p = config.REPORTS / "mlflow_runs.csv"
    if p.exists():
        runs = pd.read_csv(p)
        runs["start_time"] = pd.to_datetime(runs["start_time"], errors="coerce")
        name = runs.get("tags.mlflow.runName", runs["run_id"])
        ui.kpis(
            [
                ("Tracked runs", f"{len(runs)}", "MLflow experiment 'RetailPulse'", P["violet"]),
                ("Models", f"{name.nunique()}", "segmentation · forecast · churn · inventory · drift", P["cyan"]),
                ("Backend", "SQLite", "swap MLFLOW_TRACKING_URI for a server", P["emerald"]),
            ]
        )
        show = runs.dropna(axis=1, how="all").copy()
        show.insert(0, "run", name)
        ui.df(show.drop(columns=["run_id"], errors="ignore"), hide_index=True, height=320)
    else:
        st.info(
            "Run `python -m retailpulse.train --stage all` to create MLflow runs, then `mlflow ui --backend-store-uri sqlite:///mlflow.db`."
        )
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000", language="bash")

# ------------------------------------------------------------- drift --------
with t2:
    if not dr or not ui.have("drift_current"):
        st.warning("Drift artefacts missing — run the drift stage.")
    else:
        cur = dr["current"]
        tbl = ui.parquet("drift_current")
        lvl = "warn" if dr["decision"]["retrain"] else "ok"
        ui.kpis(
            [
                ("Features tested", f'{cur["n_features"]}', "churn model inputs", P["violet"]),
                (
                    "ALERT (PSI ≥ 0.2)",
                    f'{cur["n_alert"]}',
                    f'{cur["share_drifted"] * 100:.0f}% of features',
                    P["rose"] if cur["n_alert"] else P["emerald"],
                ),
                ("WARN (0.1–0.2)", f'{cur["n_warn"]}', "watch-list", P["amber"]),
                ("Max PSI", f'{cur["max_psi"]:.3f}', "most-shifted feature", P["cyan"]),
                (
                    "Retrain?",
                    "YES" if dr["decision"]["retrain"] else "NO",
                    "policy: ≥30% features in ALERT",
                    P["rose"] if dr["decision"]["retrain"] else P["emerald"],
                ),
            ]
        )
        ui.alert(
            (
                "Retraining recommended — " + "; ".join(dr["decision"]["reasons"])
                if dr["decision"]["retrain"]
                else "No significant drift: reference (latest training period) vs current customers."
            ),
            lvl,
        )
        colors = {"OK": P["emerald"], "WARN": P["amber"], "ALERT": P["rose"]}
        t = tbl.sort_values("psi")
        fig = go.Figure(
            go.Bar(x=t["psi"], y=t["feature"], orientation="h", marker_color=[colors[s] for s in t["status"]])
        )
        fig.add_vline(x=drift.PSI_WARN, line=dict(color=P["amber"], dash="dot"))
        fig.add_vline(x=drift.PSI_ALERT, line=dict(color=P["rose"], dash="dot"))
        ui.show(ui.style(fig, 520, False, "Population Stability Index per feature (reference vs current)"), key="psi")

        ui.section("🎛️ Drift simulation — inject a market shock and watch the monitor react")
        sev = st.slider(
            "Shock severity",
            0.0,
            1.0,
            0.4,
            0.1,
            help="Customers buy less often, spend less, wait longer between orders",
        )
        ref = ui.parquet("drift_reference")
        cust_now = ui.parquet("churn_scores")
        feats = list(ref.columns)
        shifted = drift.simulate_drift(cust_now, sev, feats)
        t2_ = drift.drift_table(ref, shifted, feats)
        s2 = drift.drift_summary(t2_)
        dec = drift.should_retrain(s2)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            sim = ui.parquet("drift_simulation")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sim["severity"],
                    y=sim["share_drifted"] * 100,
                    mode="lines+markers",
                    line=dict(color=P["violet"], width=3),
                    name="share of features in ALERT %",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[sev],
                    y=[s2["share_drifted"] * 100],
                    mode="markers",
                    marker=dict(size=16, color=P["amber"], line=dict(width=2, color="#fff")),
                    name="your setting",
                )
            )
            fig.add_hline(y=30, line=dict(color=P["rose"], dash="dash"), annotation_text="retrain trigger 30%")
            fig.update_xaxes(title="shock severity")
            fig.update_yaxes(title="% features drifted")
            ui.show(ui.style(fig, 330, title="Drift response curve"), key="simc")
        with c2:
            ui.kpis(
                [
                    (
                        "Drifted features",
                        f'{s2["n_alert"]}/{s2["n_features"]}',
                        f'max PSI {s2["max_psi"]:.2f}',
                        P["rose"] if dec["retrain"] else P["emerald"],
                    ),
                    (
                        "Decision",
                        "RETRAIN" if dec["retrain"] else "KEEP MODEL",
                        "automatic policy",
                        P["rose"] if dec["retrain"] else P["emerald"],
                    ),
                ]
            )
            ui.df(t2_.head(6)[["feature", "psi", "status"]].round(3), hide_index=True)

        ui.section("Evidently AI reports")
        which = st.radio(
            "Report", ["Current data vs training reference", "Simulated shock (severity 0.6)"], horizontal=True
        )
        f = config.DRIFT_DIR / (
            "evidently_current.html" if which.startswith("Current") else "evidently_simulated_shift.html"
        )
        if not f.exists():
            st.info("Evidently HTML report not generated in this environment.")
        elif st.toggle("Load interactive Evidently report (≈5 MB)", value=False, key="load_evidently"):
            components.html(f.read_text(encoding="utf-8"), height=700, scrolling=True)
        else:
            st.caption(
                "Toggle to embed the full Evidently AI data-drift report (kept off by default for fast page loads)."
            )

# ------------------------------------------------------------ retrain -------
with t3:
    st.markdown("**Automated retraining loop** (Airflow DAG `retailpulse_retrain`, scheduled daily): ")
    steps = [
        "Ingest & ETL",
        "Data-quality gate",
        "Drift check (Evidently/PSI)",
        "Policy: retrain?",
        "Retrain forecast + churn",
        "Validate vs champion",
        "Promote / alert",
    ]
    fig = go.Figure()
    for i, s in enumerate(steps):
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                text=[s.replace(" ", "<br>", 1)],
                textposition="bottom center",
                marker=dict(
                    size=44,
                    color=[P["violet"], P["cyan"], P["amber"], P["rose"], P["violet"], P["emerald"], P["cyan"]][i],
                    line=dict(width=2, color="#fff"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(steps))),
            y=[0] * len(steps),
            mode="lines",
            line=dict(color="rgba(255,255,255,.3)", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False, range=[-0.6, len(steps) - 0.4])
    fig.update_yaxes(visible=False, range=[-1, 0.7])
    ui.show(ui.style(fig, 230, False), key="dag")
    st.markdown(
        '<div class="rp-card"><b>Policy</b> — retrain when ≥ 30 % of features have PSI ≥ 0.2, <i>or</i> weekly MAPE degrades &gt; 25 % vs baseline, <i>or</i> churn AUC falls &gt; 0.03. '
        "A challenger only replaces the champion if it is better on the latest hold-out.</div>",
        unsafe_allow_html=True,
    )
    lp = config.REPORTS / "retrain_log.json"
    hist = json.loads(lp.read_text()) if lp.exists() else []
    if hist:
        ui.df(pd.DataFrame(hist[::-1]), hide_index=True)
    if ui.gate("Admin", "manual retrain trigger"):
        if st.button("▶ Log a manual drift check"):
            d = drift.should_retrain(dr["current"]) if dr else {"retrain": False, "reasons": [], "checked_at": ""}
            drift.append_retrain_log(
                {
                    "checked_at": d["checked_at"],
                    "retrain": d["retrain"],
                    "reasons": d["reasons"],
                    "share_drifted": dr["current"]["share_drifted"] if dr else 0,
                    "trigger": "manual (dashboard)",
                }
            )
            monitoring.audit(st.session_state.user, ui.role(), "manual_drift_check", "")
            st.success("Logged. Run `python scripts/retrain.py` to execute the full loop.")
    st.code(
        (
            open(config.ROOT / "airflow" / "dags" / "retailpulse_retrain.py").read()
            if (config.ROOT / "airflow" / "dags" / "retailpulse_retrain.py").exists()
            else "# DAG file not found"
        ),
        language="python",
    )

# ------------------------------------------------------------ health --------
with t4:
    if bb:
        r = bb["runs"]
        ui.kpis(
            [
                ("Throughput", f'{bb["rows_per_second"]:,.0f} rows/s', "daily aggregation step", P["cyan"]),
                (f'{r[-1]["rows"] / 1e6:.1f}M rows', f'{r[-1]["seconds"]:.1f}s', "measured", P["emerald"], "up"),
                (
                    "10M rows (extrapolated)",
                    f'{bb["extrapolated_10M_seconds"]:.0f}s',
                    "target < 300 s · linear extrapolation",
                    P["violet"],
                ),
            ]
        )
        fig = go.Figure(
            go.Scatter(
                x=[x["rows"] / 1e6 for x in r],
                y=[x["seconds"] for x in r],
                mode="lines+markers",
                line=dict(color=P["cyan"], width=3),
            )
        )
        fig.update_xaxes(title="rows (millions)")
        fig.update_yaxes(title="seconds")
        ui.show(ui.style(fig, 280, False, "Load test — daily batch aggregation vs data volume"), key="lt")
    targets = []
    if fm:
        targets.append(
            (
                "Forecast MAPE (weekly, CV)",
                "≤ 12%",
                f'{fm["cv"]["ensemble"]["mape_weekly"]:.1f}%',
                fm["cv"]["ensemble"]["mape_weekly"] <= 12,
            )
        )
        targets.append(
            (
                "Forecast MAPE (daily, CV)",
                "≤ 12%",
                f'{fm["cv"]["ensemble"]["mape_daily"]:.1f}%',
                fm["cv"]["ensemble"]["mape_daily"] <= 12,
            )
        )
    if cm:
        targets.append(("Churn AUC-ROC", "≥ 0.88", f'{cm["final"]["auc_roc"]:.3f}', cm["final"]["auc_roc"] >= 0.88))
        targets.append(
            (
                "Churn precision@top20%",
                "≥ 0.75",
                f'{cm["final"]["precision_top20"]:.3f}',
                cm["final"]["precision_top20"] >= 0.75,
            )
        )
    sm = ui.jreport("segmentation_metrics.json")
    if sm:
        targets.append(("Segments", "6–8", str(sm["k"]), 6 <= sm["k"] <= 8))
    if bb:
        targets.append(
            (
                "Batch time (10M rows, extrapolated)",
                "< 300 s",
                f'{bb["extrapolated_10M_seconds"]:.0f} s',
                bb["extrapolated_10M_seconds"] < 300,
            )
        )
    if targets:
        df = pd.DataFrame(targets, columns=["Acceptance criterion", "Target", "Measured", "Met"])
        df["Met"] = df["Met"].map({True: "✅ met", False: "❌ not met"})
        ui.df(df, hide_index=True)
    st.caption(
        f"Prometheus metrics endpoint: http://localhost:{__import__('os').getenv('METRICS_PORT', '8000')}/metrics · scraped by monitoring/prometheus.yml · Grafana dashboard in monitoring/grafana/."
    )

# ------------------------------------------------------------ security ------
with t5:
    st.markdown("""
- **Data anonymisation** – customer IDs are salted SHA-256 pseudonyms created inside the ETL; raw IDs never reach any downstream file.
- **Role-based access** – Viewer (dashboards, masked IDs) · Analyst (+ exports, full pseudonyms) · Admin (+ MLOps controls, audit log).
- **Audit logging** – logins, failed logins, exports and manual operations are written to an append-only JSON-lines log.
- **JWT-secured API** – `api/main.py` exposes forecast/churn endpoints behind OAuth2-password + signed JWT (see README).
""")
    if ui.gate("Admin", "the audit log"):
        log = monitoring.read_audit(300)
        if log:
            ui.df(pd.DataFrame(log), hide_index=True, height=360)
        else:
            st.info("No audit events yet.")
