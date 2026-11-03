"""Build and EXECUTE the analysis notebooks so they ship with outputs.

python scripts/build_notebooks.py        ->  notebooks/01_EDA.ipynb, notebooks/02_Model_Results.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from retailpulse import config
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
PAL = ["#7C5CFF", "#22D3EE", "#34D399", "#FBBF24", "#FB7185"]
P = config.DATA_PROC
def J(n): return json.loads((config.REPORTS / n).read_text())
"""


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def eda() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 01 · Exploratory Data Analysis – RetailPulse\n**Dataset:** UCI *Online Retail II* (1,067,371 raw transactions, 1 Dec 2009 → 9 Dec 2011) · "
            "**Goals (Day 1–2):** dataset selection, distributions, missing values, correlation heat-map, cleaning decisions and the engineered columns.\n\n"
            "> Run `make pipeline` first. The raw CSV is only needed for the *missing-value audit* cell (it is skipped if the file is absent)."
        ),
        code(SETUP),
        md(
            "## 1. Why this dataset?\nIt is a *real* transactional log (not synthetic) that contains all three analytical grains the brief asks for: "
            "**sales** (invoice lines), **customers** (IDs, wholesalers vs. retail) and **products** (≈4.7k SKUs). It spans two Christmas seasons, "
            "which makes seasonality/forecasting non-trivial. It has **no inventory table**, so stock levels are simulated and labelled as such."
        ),
        code("""raw_path = config.RAW_FILE
if raw_path.exists():
    raw = pd.read_csv(raw_path, dtype={"Invoice": str, "StockCode": str})
    print(raw.shape)
    miss = raw.isna().mean().mul(100).sort_values(ascending=False)
    display(raw.head())
    ax = miss.plot.barh(color=PAL[0], figsize=(7, 3)); ax.set_title("Missing values in the RAW file (%)"); ax.invert_yaxis()
    for i, v in enumerate(miss): ax.text(v + .3, i, f"{v:.1f}%", va="center")
    print("exact duplicate rows:", raw.duplicated().sum())
    print("cancellations (Invoice starts with 'C'):", raw["Invoice"].str.startswith("C").sum())
    print("quantity <= 0:", (raw["Quantity"] <= 0).sum(), "| price <= 0:", (raw["Price"] <= 0).sum())
    print(raw["StockCode"][~raw["StockCode"].str.match(r"^\\d{5}")].value_counts().head(8).to_string())
else:
    print("raw file not present - skipping raw audit; see reports/etl_report.json lineage below")"""),
        md(
            "**Findings →** 22.8 % of lines have no `Customer ID` (guest checkouts) – kept for *demand* modelling, excluded from CRM models; "
            "≈34k exact duplicates; ≈19k cancellations (split into a returns table); non-product service codes (`POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE` …) removed."
        ),
        md("## 2. ETL lineage (what each cleaning step removed)"),
        code(
            """rep = J("etl_report.json")
lin = pd.DataFrame(rep["lineage"])
display(lin[["step", "rows_before", "rows_after", "rows_removed", "note"]])
print(f"retained {rep['retention_pct']}%  |  quality checks passed: {rep['quality_passed']}  |  custom column: {rep['custom_column']}")"""
        ),
        md("## 3. Distributions (clean data)"),
        code(
            """sales = pd.read_parquet(P / "clean_transactions.parquet", columns=["quantity", "quantity_capped", "unit_price", "revenue", "date", "region_group", "country", "is_guest", "invoice_no", "customer_key", "dow", "hour"])
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(np.log10(sales["quantity"]), bins=40, color=PAL[0]); ax[0].set_title("log10 line quantity (heavy right tail → winsorise p99.9)")
ax[1].hist(np.log10(sales["unit_price"]), bins=40, color=PAL[1]); ax[1].set_title("log10 unit price (£)")
ax[2].hist(np.log10(sales["revenue"].clip(lower=.01)), bins=40, color=PAL[2]); ax[2].set_title("log10 line revenue (£)")
plt.tight_layout(); plt.show()
print(sales[["quantity", "quantity_capped", "unit_price", "revenue"]].describe(percentiles=[.5, .99, .999]).round(2))"""
        ),
        md("## 4. Time structure – the key to forecasting"),
        code(
            """d = pd.read_parquet(P / "daily_sales.parquet")
o = d[d["is_open"].astype(bool)]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2, 1]})
ax[0].plot(o["date"], o["revenue"], lw=.8, color=PAL[1]); ax[0].plot(o["date"], o["revenue"].rolling(28, min_periods=7).mean(), lw=2.4, color=PAL[0])
ax[0].set_title("Daily revenue (£) with 28-day mean – note the Q4 peak and Christmas closure")
dow = o.groupby(o["date"].dt.day_name())["revenue"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
ax[1].bar(dow.index.str[:3], dow.values.astype(float), color=PAL[0]); ax[1].set_title("Mean revenue by weekday (Saturday = 0, shop closed)")
plt.tight_layout(); plt.show()
print("calendar days:", len(d), "| trading days:", int(d["is_open"].sum()), "| closed (Sat, bank holidays, 24 Dec-3 Jan):", int((~d["is_open"].astype(bool)).sum()))"""
        ),
        code("""mr = pd.read_parquet(P / "monthly_region.parquet")
piv = mr.pivot(index="month_start", columns="region_group", values="revenue").fillna(0)
ax = piv.plot.area(figsize=(11, 3.6), color=PAL[:piv.shape[1]], lw=0); ax.set_title("Monthly revenue by region_group  (← custom column added in ETL)")
display((piv.sum() / piv.sum().sum() * 100).round(1).rename("% of revenue"))"""),
        md("## 5. Customer-level features & correlation heat-map"),
        code(
            """c = pd.read_parquet(P / "customers.parquet")
cols = ["recency", "frequency", "monetary", "avg_order_value", "n_skus", "mean_gap", "return_rate", "tenure_days"]
corr = np.log1p(c[cols].clip(lower=0)).corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.5, 5.2)); im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right"); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)): ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im); ax.set_title("Spearman correlation (log-scaled)"); plt.show()
share = c.nlargest(len(c) // 100, "monetary")["monetary"].sum() / c["monetary"].sum() * 100
print(f"Customers: {len(c)} | median orders: {c['frequency'].median()} | top 1 pct of customers hold {share:.0f}% of revenue")"""
        ),
        md(
            "**Findings →** (1) frequency, monetary and SKU breadth are strongly correlated (wholesalers buy often, big and broad) → one `Champions` segment drives most revenue; "
            "(2) recency is negatively related to frequency; (3) revenue is extremely concentrated, so we log-scale before clustering; "
            "(4) strong weekly + annual seasonality and a Christmas closure → trading-day calendar + holiday effects in the forecasters."
        ),
    ]
    return nb


def results() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 02 · Model results – segmentation, forecasting, churn, inventory, drift\nEvery number below is read from the artefacts written by `python -m retailpulse.train --stage all`."
        ),
        code(SETUP),
        md("## Segmentation (F-02)"),
        code(
            """sm = J("segmentation_metrics.json"); prof = pd.read_parquet(P / "segment_profile.parquet").sort_values("revenue_share", ascending=False)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in sm.items() if k != "pca_explained_variance"})
display(prof[["segment", "size", "revenue_share", "recency", "frequency", "monetary", "avg_order_value", "description"]].round(2))"""
        ),
        md("## Demand forecasting (F-03)"),
        code("""fm = J("forecast_metrics.json")
rows = []
for split in ("cv", "test"):
    for m, v in fm[split].items(): rows.append({"split": split, "model": m, **{k: round(x, 2) for k, x in v.items()}})
display(pd.DataFrame(rows))
print("ensemble weight on LSTM:", fm["w_lstm"], "| LSTM calibration k:", round(fm["k_lstm"], 3), "| hold-out origin:", fm["test_origin"])
bt = pd.read_parquet(P / "forecast_backtest.parquet"); t = bt[bt["is_test"]].sort_values("date")
plt.figure(figsize=(11, 3.6)); plt.plot(t["date"], t["actual"], lw=2.5, label="actual", color=PAL[1])
for k, c in (("prophet", PAL[0]), ("lstm", PAL[3]), ("ensemble", PAL[2])): plt.plot(t["date"], t[k], lw=1.6, ls="--" if k != "ensemble" else "-", label=k, color=c)
plt.legend(); plt.title("Peak-season hold-out (never used for tuning)"); plt.xticks(rotation=30); plt.show()"""),
        md("## Churn (F-04)"),
        code("""cm = J("churn_metrics.json"); b = pd.DataFrame(cm["benchmark"])
display(b[["model", "auc_roc", "auc_pr", "precision_top20", "lift_top20", "f1", "recall", "brier"]].round(3))
print("final:", {k: round(v, 3) for k, v in cm["final"].items()}); print("targets:", cm["targets"])
display(pd.read_parquet(P / "churn_shap_importance.parquet").head(10).round(4))"""),
        md("## Inventory (F-05)"),
        code(
            """im = J("inventory_metrics.json"); pol = pd.DataFrame(im["policy_summary"])
display(pol[["policy", "fill_rate_pct", "understock_pct_of_demand", "overstock_pct_of_demand", "lost_margin", "holding_cost", "total_cost"]].round(1))
print("median per-SKU 4-week WAPE %:", {k: round(v, 1) for k, v in im["sku_wape_median"].items()}); print(im["assumptions"])"""
        ),
        md("## Drift & MLOps"),
        code(
            """dr = J("drift_summary.json"); print(json.dumps({k: dr[k] for k in ("current", "decision")}, indent=1, default=str)[:1500])
display(pd.read_parquet(P / "drift_simulation.parquet").round(3))
print("batch benchmark:", J("batch_benchmark.json"))"""
        ),
    ]
    return nb


def run(nb: nbf.NotebookNode, name: str) -> None:
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)
    print("executed", name)


if __name__ == "__main__":
    for builder, fname in ((eda, "01_EDA.ipynb"), (results, "02_Model_Results.ipynb")):
        run(builder(), fname)
    sys.exit(0)
