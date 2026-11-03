"""RetailPulse - entry point / router.   Run:  streamlit run app/streamlit_app.py"""

import streamlit as st

PAGES = [
    st.Page("pages/0_Overview.py", title="Overview", icon="📊", default=True),
    st.Page("pages/1_Demand_Forecast.py", title="Demand Forecast", icon="📈"),
    st.Page("pages/2_Customer_Segments.py", title="Customer Segments", icon="👥"),
    st.Page("pages/3_Churn_Risk.py", title="Churn Risk", icon="⚠️"),
    st.Page("pages/4_Inventory.py", title="Inventory", icon="📦"),
    st.Page("pages/5_MLOps_and_Drift.py", title="MLOps & Drift", icon="🛰️"),
    st.Page("pages/6_Data_and_ETL.py", title="Data & ETL", icon="🗄️"),
    st.Page("pages/7_Reports.py", title="Reports", icon="🧾"),
]
st.navigation(PAGES).run()
