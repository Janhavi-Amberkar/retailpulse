import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from retailpulse import config
from ui import P

ui.setup("Data & ETL", "🗄️")
ui.page_view("etl")
ui.hero(
    "Data & ETL",
    "Automated Extract → Validate → Transform → Load pipeline with a lineage audit trail, a data-quality gate "
    "(Great Expectations) and one custom engineered column.",
    "F-01 · Data engineering",
)

etl = ui.jreport("etl_report.json")
qc = ui.jreport("data_quality.json")
if not etl:
    st.error("Run `python -m retailpulse.etl.pipeline` first.")
    st.stop()

ui.kpis(
    [
        ("Raw rows", f'{etl["rows_raw"]:,}', "UCI Online Retail II", P["violet"]),
        ("Clean rows", f'{etl["rows_clean"]:,}', f'{etl["retention_pct"]}% retained', P["emerald"]),
        ("Quality checks", etl["quality_passed"], "Great Expectations + pandas", P["cyan"], "up"),
        ("ETL time", f'{etl["elapsed_seconds"]}s', "1M rows end-to-end", P["amber"]),
        (
            "Guest lines",
            f'{etl["guest_line_share"] * 100:.0f}%',
            "no customer ID → excluded from CRM models",
            P["rose"],
        ),
    ]
)

t1, t2, t3, t4 = st.tabs(["🔗 Lineage", "✅ Data quality", "🧬 Schema & custom column", "📚 Source"])
with t1:
    lin = pd.DataFrame(etl["lineage"])
    rem = lin[lin["rows_removed"] > 0]
    fig = go.Figure(
        go.Waterfall(
            x=["Raw rows"] + rem["step"].tolist() + ["Clean rows"],
            measure=["absolute"] + ["relative"] * len(rem) + ["total"],
            y=[etl["rows_raw"]] + (-rem["rows_removed"]).tolist() + [0],
            connector=dict(line=dict(color="rgba(255,255,255,.2)")),
            increasing=dict(marker=dict(color=P["emerald"])),
            decreasing=dict(marker=dict(color=P["rose"])),
            totals=dict(marker=dict(color=P["violet"])),
        )
    )
    fig.update_xaxes(tickangle=-25)
    ui.show(ui.style(fig, 420, False, "Row lineage — what each cleaning step removed"), key="wf")
    ui.df(lin[["step", "rows_before", "rows_after", "rows_removed", "note"]], hide_index=True)
with t2:
    q = pd.DataFrame(qc["checks"])
    q["result"] = q["passed"].map({True: "✅ pass", False: "❌ fail"})
    ui.df(q[["result", "name", "column", "detail", "engine"]], hide_index=True, height=500)
with t3:
    st.markdown(
        f'<div class="rp-card">⭐ <b>Custom column added: <code>{etl["custom_column"]}</code></b> — maps each country to '
        f"<i>UK / Europe / Rest of World / Unspecified</i>. It is used for regional revenue trends, filtering and reporting. "
        f"Other engineered columns (<code>revenue, date, year, month, quarter, dow, hour, quantity_capped, is_guest, customer_key</code>) are standard derivations.</div>",
        unsafe_allow_html=True,
    )
    sample = config.DATA_SAMPLE / "clean_sample.csv"
    if sample.exists():
        df = pd.read_csv(sample)
        ui.df(df.head(200), hide_index=True, height=360)
        ui.csv_button(df, "⬇ Download clean sample (5,000 rows)", "retailpulse_clean_sample.csv", "sample", "Viewer")
with t4:
    st.markdown(f"""
**Dataset:** [UCI Online Retail II]({config.UCI_PAGE}) — real transactions of a UK-based, non-store online gift retailer, 1 Dec 2009 → 9 Dec 2011
(1,067,371 rows). Licence: CC BY 4.0 (Chen, D., 2012). Many customers are wholesalers.

**Cleaning rules applied** (see lineage): exact duplicates · cancellations split to a returns table · bad-debt adjustments · non-positive quantity/price ·
non-product service codes (POST, DOT, M, BANK CHARGES…) · description normalisation + back-fill · duplicates exposed by normalisation ·
quantity winsorised at the 99.9th percentile for demand modelling · customer IDs salted-hashed.

**Inventory note:** the dataset has no stock table, so on-hand stock and lead times are simulated (clearly flagged) in the Inventory module.
""")
