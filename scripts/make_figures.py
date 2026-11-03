"""Static, print-quality figures for the PDF report and README (reads the pipeline artefacts).

python scripts/make_figures.py       ->  reports/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from retailpulse import config  # noqa: E402

V, C, G, A, R, INK, MUTE = "#7C5CFF", "#22D3EE", "#34D399", "#FBBF24", "#FB7185", "#1b2142", "#6b7396"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9cde3",
        "axes.labelcolor": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": "#e8eaf4",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)
OUT = config.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
P = config.DATA_PROC


def J(name):
    return json.loads((config.REPORTS / name).read_text())


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ 1 architecture
def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, sub="", color=V, fc="#f6f4ff"):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=color, lw=1.8)
        )
        ax.text(
            x + w / 2,
            y + h * (0.62 if sub else 0.5),
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=INK,
        )
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7.3, color=MUTE)

    def arrow(x1, y1, x2, y2, color=MUTE):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4))

    ax.text(0.1, 5.95, "RetailPulse – end-to-end architecture", fontsize=13, weight="bold", color=INK)
    # layer labels
    for y, t in ((4.55, "DATA"), (3.05, "ML"), (1.55, "SERVE"), (0.15, "MLOPS")):
        ax.text(0.05, y + 0.55, t, rotation=90, va="center", fontsize=8, color=MUTE, weight="bold")
    box(0.5, 4.5, 2.0, 1.0, "Online Retail II", "1.07M raw transactions", C, "#effcff")
    box(3.0, 4.5, 2.0, 1.0, "ETL", "extract · clean · custom col", V)
    box(5.5, 4.5, 2.0, 1.0, "Quality gate", "Great Expectations", G, "#effdf7")
    box(8.0, 4.5, 2.5, 1.0, "Parquet + SQL", "PostgreSQL / SQLite · Redis", A, "#fffaec")
    arrow(2.5, 5.0, 3.0, 5.0)
    arrow(5.0, 5.0, 5.5, 5.0)
    arrow(7.5, 5.0, 8.0, 5.0)

    box(0.5, 3.0, 2.0, 1.0, "Segmentation", "RFM · K-Means · DBSCAN", V)
    box(3.0, 3.0, 2.0, 1.0, "Demand forecast", "Prophet + LSTM ensemble", V)
    box(5.5, 3.0, 2.0, 1.0, "Churn", "XGBoost · Optuna · SHAP", V)
    box(8.0, 3.0, 2.5, 1.0, "Inventory", "hier. forecast · ROP · EOQ", V)
    for x in (1.5, 4.0, 6.5, 9.25):
        arrow(x if x < 9 else 9.25, 4.5, x if x < 9 else 9.25, 4.0)

    box(0.5, 1.5, 3.0, 1.0, "Streamlit dashboard", "8 pages · what-if · exports · RBAC", C, "#effcff")
    box(4.0, 1.5, 3.0, 1.0, "JWT REST API", "FastAPI · OAuth2 · roles", C, "#effcff")
    box(7.5, 1.5, 3.0, 1.0, "Audit log", "logins · exports · changes", C, "#effcff")
    for x in (2.0, 5.5, 9.0):
        arrow(x, 3.0, x, 2.5)

    box(0.5, 0.1, 2.0, 1.0, "MLflow", "runs · params · artefacts", R, "#fff3f5")
    box(3.0, 0.1, 2.0, 1.0, "Evidently + PSI", "drift reports", R, "#fff3f5")
    box(5.5, 0.1, 2.0, 1.0, "Airflow / CronJob", "retrain + champion gate", R, "#fff3f5")
    box(8.0, 0.1, 2.5, 1.0, "Docker · K8s · CI/CD", "Prometheus + Grafana", R, "#fff3f5")
    for x in (1.5, 4.0, 6.5, 9.25):
        arrow(x, 1.5, x, 1.1, R)
    save(fig, "architecture.png")


# ------------------------------------------------------------------ 2 ETL lineage
def lineage():
    rep = J("etl_report.json")
    lin = [x for x in rep["lineage"] if x["rows_removed"] > 0]
    labels = ["Raw rows"] + [x["step"].replace("drop ", "").replace("split ", "") for x in lin] + ["Clean rows"]
    vals = [rep["rows_raw"]] + [-x["rows_removed"] for x in lin]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    run = rep["rows_raw"]
    ax.bar(0, run, color=V)
    for i, v in enumerate(vals[1:], 1):
        ax.bar(i, v, bottom=run, color=R)
        ax.text(i, run + 8000, f"{v:,}", ha="center", fontsize=8, color=INK)
        run += v
    ax.bar(len(vals), run, color=G)
    ax.text(0, rep["rows_raw"] + 8000, f"{rep['rows_raw']:,}", ha="center", fontsize=8)
    ax.text(len(vals), run + 8000, f"{run:,}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([t[:26] for t in labels], rotation=28, ha="right", fontsize=8)
    ax.set_ylim(rep["rows_raw"] * 0.88, rep["rows_raw"] * 1.03)
    ax.set_title(f"ETL row lineage – {rep['retention_pct']}% of raw rows retained (y-axis truncated)")
    save(fig, "etl_lineage.png")


# ------------------------------------------------------------------ 3 EDA
def eda():
    d = pd.read_parquet(P / "daily_sales.parquet")
    mr = pd.read_parquet(P / "monthly_region.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    o = d[d["is_open"].astype(bool)]
    ax[0].plot(o["date"], o["revenue"], color=C, lw=0.8, alpha=0.7)
    ax[0].plot(o["date"], o["revenue"].rolling(28, min_periods=7).mean(), color=V, lw=2.4, label="28-day mean")
    ax[0].set_title("Daily revenue (£) – closed days omitted")
    ax[0].legend()
    piv = mr.pivot(index="month_start", columns="region_group", values="revenue").fillna(0)
    piv.plot.area(ax=ax[1], color=[V, C, G, A][: piv.shape[1]], lw=0)
    ax[1].set_title("Monthly revenue by region_group (custom column)")
    ax[1].set_xlabel("")
    save(fig, "eda_revenue.png")

    c = pd.read_parquet(P / "customers.parquet")
    cols = ["recency", "frequency", "monetary", "avg_order_value", "n_skus", "mean_gap", "return_rate", "tenure_days"]
    corr = np.log1p(c[cols].clip(lower=0)).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#fff" if abs(corr.iloc[i, j]) > 0.5 else INK,
            )
    ax.grid(False)
    ax.set_title("Spearman correlation of customer features")
    fig.colorbar(im, shrink=0.8)
    save(fig, "eda_correlation.png")

    # dow x hour + distribution of order value
    h = (
        pd.read_parquet(P / "hourly_profile.parquet")
        .pivot(index="dow", columns="hour", values="revenue")
        .reindex(range(7))
        .fillna(0)
    )
    prod = pd.read_parquet(P / "products.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    im = ax[0].imshow(h.values, aspect="auto", cmap="Purples")
    ax[0].set_yticks(range(7))
    ax[0].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[0].set_xticks(range(0, h.shape[1], 2))
    ax[0].set_xticklabels(h.columns[::2])
    ax[0].grid(False)
    ax[0].set_title("Revenue by weekday × hour (Saturday closed)")
    ax[1].hist(np.log10(prod["total_revenue"]), bins=25, color=V)
    ax[1].set_title("Top-180 SKUs: distribution of lifetime revenue")
    ax[1].set_xlabel("log10(£)")
    save(fig, "eda_patterns.png")


# ------------------------------------------------------------------ 4 segmentation
def segmentation():
    c = pd.read_parquet(P / "customers.parquet")
    prof = pd.read_parquet(P / "segment_profile.parquet").sort_values("revenue_share", ascending=False)
    cols = dict(zip(prof["segment"], [G, V, C, A, R, "#A78BFA", "#38BDF8", "#F472B6"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    s = c.sample(min(3000, len(c)), random_state=1)
    for seg, g in s.groupby("segment"):
        ax[0].scatter(g["pc1"], g["pc2"], s=7, alpha=0.55, color=cols[seg], label=seg)
    ax[0].legend(markerscale=3, fontsize=8, ncol=2)
    ax[0].set_title("Customer segments in PCA space (K-Means)")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[1].barh(prof["segment"], prof["revenue_share"] * 100, color=[cols[x] for x in prof["segment"]])
    ax[1].invert_yaxis()
    for i, (v, n) in enumerate(zip(prof["revenue_share"] * 100, prof["size"])):
        ax[1].text(v + 1, i, f"{v:.0f}%  ({int(n):,})", va="center", fontsize=8)
    ax[1].set_xlim(0, 100)
    ax[1].set_title("Revenue share (customers)")
    save(fig, "segments.png")


# ------------------------------------------------------------------ 5 forecast
def forecast():
    bt = pd.read_parquet(P / "forecast_backtest.parquet")
    fut = pd.read_parquet(P / "forecast_future.parquet")
    d = pd.read_parquet(P / "daily_sales.parquet")
    m = J("forecast_metrics.json")
    t = bt[bt["is_test"]].sort_values("date")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot(t["date"], t["actual"], color=C, lw=2.4, label="Actual")
    ax[0].plot(t["date"], t["prophet"], color=V, lw=1.3, ls="--", label="Prophet")
    ax[0].plot(t["date"], t["lstm"], color=A, lw=1.3, ls="--", label="LSTM (calibrated)")
    ax[0].plot(t["date"], t["ensemble"], color=G, lw=2.4, label="Ensemble")
    e = m["test"]["ensemble"]
    ax[0].set_title(
        f"Hold-out (peak season): weekly MAPE {e['mape_weekly']:.1f}% · 30-day total error {e['ape_30d_total']:.1f}%"
    )
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=8)
    o = d[d["is_open"].astype(bool)].tail(90)
    ax[1].plot(o["date"], o["units_capped"], color=C, lw=1.6, label="Actual")
    gap = fut["date"].diff().dt.days > 3
    segs = np.split(np.arange(len(fut)), np.flatnonzero(gap.values))  # break the line at the Christmas closure
    for i, ix in enumerate(segs):
        f = fut.iloc[ix]
        ax[1].fill_between(
            f["date"], f["lower"], f["upper"], color=V, alpha=0.2, label="80% interval" if i == 0 else None
        )
        ax[1].plot(f["date"], f["ensemble"], color=V, lw=2.2, marker="o", ms=3, label="Forecast" if i == 0 else None)
    ax[1].set_title("Next 30 days (trading days only)")
    ax[1].tick_params(axis="x", rotation=30)
    ax[1].legend(fontsize=8)
    save(fig, "forecast_holdout.png")

    pf = pd.read_parquet(P / "forecast_per_fold.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
    w = 0.26
    folds = sorted(pf["fold"].unique())
    for k, (mod, col) in enumerate((("naive", "#9aa3c7"), ("prophet", V), ("ensemble", G))):
        dd = pf[pf["model"] == mod].sort_values("fold")
        ax[0].bar(np.arange(len(folds)) + (k - 1) * w, dd["mape_weekly"], w, color=col, label=mod)
    ax[0].axhline(12, color=R, ls="--", lw=1.2)
    ax[0].text(len(folds) - 0.5, 12.4, "12% target", color=R, fontsize=8, ha="right")
    ax[0].set_xticks(range(len(folds)))
    ax[0].set_xticklabels([f"#{f}" + ("\nhold-out" if f == folds[-1] else "") for f in folds], fontsize=8)
    ax[0].set_title("Weekly MAPE per back-test fold")
    ax[0].legend(fontsize=8)
    names = ["naive", "prophet", "lstm", "ensemble"]
    cv = m["cv"]
    x = np.arange(3)
    for k, mod in enumerate(names):
        vals = [cv[mod]["mape_daily"], cv[mod]["mape_weekly"], cv[mod]["ape_30d_total"]]
        ax[1].bar(x + (k - 1.5) * 0.2, vals, 0.2, color=["#9aa3c7", V, A, G][k], label=mod)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["daily MAPE", "weekly MAPE", "30-day total error"])
    ax[1].set_title("Cross-validation (7 folds): error by aggregation level")
    ax[1].legend(fontsize=8)
    save(fig, "forecast_cv.png")

    dec = pd.read_parquet(P / "ts_decomposition.parquet")
    fig, ax = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True)
    ax[0].plot(dec["date"], dec["observed"], color=C, lw=0.8)
    ax[0].set_ylabel("observed")
    ax[1].plot(dec["date"], dec["trend"], color=V, lw=2)
    ax[1].set_ylabel("trend")
    ax[2].plot(dec["date"], dec["seasonal"], color=G, lw=0.8, label="seasonal")
    ax[2].plot(dec["date"], dec["resid"], color=R, lw=0.8, label="residual")
    ax[2].legend(fontsize=8, ncol=2)
    dg = m["diagnostics"]
    ax[0].set_title(
        f"STL decomposition · ADF p={dg['adf_p']:.2f} (non-stationary) → p={dg['adf_diff_p']:.0e} after differencing"
    )
    save(fig, "ts_decomposition.png")


# ------------------------------------------------------------------ 6 churn
def churn():
    m = J("churn_metrics.json")
    cv = m["curves"]
    fin = m["final"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].plot(cv["fpr"], cv["tpr"], color=V, lw=2.6, label=f"XGBoost AUC {fin['auc_roc']:.3f}")
    ax[0].fill_between(cv["fpr"], cv["tpr"], color=V, alpha=0.12)
    ax[0].plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1)
    ax[0].set_title("ROC (out-of-time test)")
    ax[0].legend(loc="lower right")
    dec = pd.DataFrame(cv["deciles"])
    ax[1].bar(
        dec["decile"], dec["churn_rate"] * 100, color=[R if v > fin["base_rate"] else G for v in dec["churn_rate"]]
    )
    ax[1].axhline(fin["base_rate"] * 100, color=INK, ls=":", lw=1.2)
    ax[1].set_title("Decile lift – actual churn %")
    ax[1].set_xlabel("risk decile (1 = highest)")
    b = pd.DataFrame(m["benchmark"])
    colr = ["#9aa3c7", "#b9a8ff", V, G]
    ax[2].barh(b["model"], b["auc_roc"], color=colr[: len(b)])
    ax[2].set_xlim(0.7, 0.85)
    for i, v in enumerate(b["auc_roc"]):
        ax[2].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax[2].axvline(0.88, color=R, ls="--")
    ax[2].set_title("Model benchmark (test AUC) · 0.88 target")
    save(fig, "churn_quality.png")


# ------------------------------------------------------------------ 7 inventory
def inventory():
    pol = pd.DataFrame(J("inventory_metrics.json")["policy_summary"])
    order = [
        "Moving average (no buffer)",
        "Moving average + safety stock",
        "Per-SKU Prophet (no buffer)",
        "Hierarchical forecast (no buffer)",
        "RetailPulse (hierarchical + safety stock)",
    ]
    pol = pol.set_index("policy").loc[order].reset_index()
    short = ["MA", "MA + SS", "SKU-Prophet", "Hier.", "RetailPulse"]
    colr = [A, C, "#9aa3c7", V, G]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for a, (col, ttl) in zip(
        ax,
        (
            ("fill_rate_pct", "Fill rate % (↑)"),
            ("overstock_pct_of_demand", "Overstock % of demand (↓)"),
            ("total_cost", "Total cost £ = lost margin + holding (↓)"),
        ),
    ):
        a.bar(short, pol[col], color=colr)
        for i, v in enumerate(pol[col]):
            a.text(i, v, f"{v:,.0f}" if col == "total_cost" else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        a.set_title(ttl)
        a.tick_params(axis="x", labelsize=8, rotation=15)
    save(fig, "inventory_policies.png")


# ------------------------------------------------------------------ 8 drift
def drift():
    t = pd.read_parquet(P / "drift_current.parquet").sort_values("psi")
    sim = pd.read_parquet(P / "drift_simulation.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    colr = {"OK": G, "WARN": A, "ALERT": R}
    ax[0].barh(t["feature"], t["psi"], color=[colr[s] for s in t["status"]])
    ax[0].axvline(0.1, color=A, ls=":")
    ax[0].axvline(0.2, color=R, ls=":")
    ax[0].set_title("PSI per feature (current vs training reference)")
    ax[0].tick_params(axis="y", labelsize=7)
    ax[1].plot(sim["severity"], sim["share_drifted"] * 100, color=V, lw=2.6, marker="o")
    ax[1].axhline(30, color=R, ls="--")
    ax[1].text(0.02, 31.5, "retrain trigger (30% of features)", color=R, fontsize=8)
    ax[1].set_title("Drift simulation – injected market shock")
    ax[1].set_xlabel("shock severity")
    ax[1].set_ylabel("% features in ALERT")
    save(fig, "drift.png")


if __name__ == "__main__":
    for fn in (architecture, lineage, eda, segmentation, forecast, churn, inventory, drift):
        fn()
