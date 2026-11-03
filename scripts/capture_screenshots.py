"""Capture dashboard screenshots with Playwright for the report / README.

streamlit run app/streamlit_app.py &            # in another terminal
python scripts/capture_screenshots.py [base_url]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"

# name, url path, tab to click (or None), scroll_y, viewport height
SHOTS = [
    ("01_overview", "", None, 0, 1000),
    ("02_overview_trends", "", None, 1050, 1000),
    ("03_forecast_30d", "Demand_Forecast", None, 0, 1100),
    ("04_forecast_accuracy", "Demand_Forecast", "Accuracy & validation", 380, 1100),
    ("05_forecast_whatif", "Demand_Forecast", "What-if scenarios", 380, 1000),
    ("06_segments_profiles", "Customer_Segments", None, 560, 1100),
    ("07_segments_3d", "Customer_Segments", "3-D cluster map", 560, 1000),
    ("08_churn_quality", "Churn_Risk", None, 640, 1100),
    ("09_churn_shap", "Churn_Risk", "SHAP explainability", 640, 1000),
    ("10_inventory_plan", "Inventory", None, 380, 1100),
    ("11_inventory_backtest", "Inventory", "Policy back-test", 380, 1100),
    ("12_mlops_drift", "MLOps_and_Drift", "Drift monitor", 380, 1100),
    ("13_etl_lineage", "Data_and_ETL", None, 380, 1000),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        pg = ctx.new_page()
        for name, path, tab, scroll, h in SHOTS:
            pg.set_viewport_size({"width": 1440, "height": h})
            pg.goto(f"{BASE}/{path}", wait_until="networkidle")
            pg.wait_for_selector('[data-testid="stMain"]', timeout=30000)
            pg.wait_for_timeout(3500)
            if tab:
                pg.get_by_role("tab", name=tab).first.click()
                pg.wait_for_timeout(2500)
            if scroll:
                pg.evaluate(f"document.querySelector('[data-testid=\"stMain\"]').scrollTo(0, {scroll})")
                pg.wait_for_timeout(1500)
            exc = pg.locator('[data-testid="stException"]').count()
            pg.screenshot(path=str(OUT / f"{name}.png"))
            print(f"{name}: exceptions={exc}")
        b.close()


if __name__ == "__main__":
    main()
