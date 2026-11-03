import io
import zipfile

import streamlit as st

import ui
from retailpulse import config, monitoring, reporting
from ui import P

ui.setup("Reports & Export", "🧾")
ui.page_view("reports")
ui.hero("Reports & Export", "One-click executive PDF and a CSV bundle of every analytical output.", "Day 20 · Export")

if ui.gate("Analyst", "report exports"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="rp-card"><b>📄 Executive PDF</b><br><span class="rp-note">KPIs, forecast accuracy, churn model, segments, '
            "purchase plan, policy back-test and monitoring status.</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("Generate executive PDF"):
            with st.spinner("Building report…"):
                st.session_state["pdf"] = reporting.build_executive_pdf()
            monitoring.audit(st.session_state.user, ui.role(), "generate_pdf", "executive report")
        if "pdf" in st.session_state:
            st.download_button(
                "⬇ Download RetailPulse_Executive_Report.pdf",
                st.session_state["pdf"],
                "RetailPulse_Executive_Report.pdf",
                "application/pdf",
            )
    with c2:
        st.markdown(
            '<div class="rp-card"><b>🗂️ CSV bundle (.zip)</b><br><span class="rp-note">Forecast, segments, churn scores (pseudonymised), '
            "purchase plan, drift table.</span></div>",
            unsafe_allow_html=True,
        )
        files = {
            "forecast_30d": "forecast_future",
            "customer_segments": "customers",
            "churn_scores": "churn_scores",
            "purchase_plan": "inventory_recommendations",
            "drift_table": "drift_current",
            "daily_sales": "daily_sales",
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, art in files.items():
                if ui.have(art):
                    df = ui.parquet(art)
                    z.writestr(f"{fname}.csv", df.to_csv(index=False))
        if st.download_button("⬇ Download CSV bundle", buf.getvalue(), "RetailPulse_exports.zip", "application/zip"):
            monitoring.audit(st.session_state.user, ui.role(), "export_zip", ",".join(files))
st.caption("All exports are audit-logged (user, role, action, timestamp). Customer identifiers are pseudonyms.")
