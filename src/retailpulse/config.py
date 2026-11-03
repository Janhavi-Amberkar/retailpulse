"""Central configuration for RetailPulse (paths, constants, business assumptions)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "data").exists():  # pragma: no cover - installed-package fallback
    ROOT = Path.cwd()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DRIFT_DIR = REPORTS / "drift"
LOGS = ROOT / "logs"

RAW_FILE = DATA_RAW / "online_retail_II.csv"
# Public mirror of the UCI "Online Retail II" dataset (CC BY 4.0, Chen 2012).
RAW_URL = "https://raw.githubusercontent.com/omidreza-amrollahi/online_retail_data/" "main/online_retail_II.csv"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"

SEED = 42

# --- Privacy -----------------------------------------------------------------
ANON_SALT = os.getenv("RETAILPULSE_SALT", "retailpulse-demo-salt")

# --- Modelling constants -----------------------------------------------------
FORECAST_HORIZON = 30  # days ahead (spec F03)
CHURN_WINDOW_DAYS = 90  # customer is "churned" if no purchase in next 90 days
QTY_CAP_QUANTILE = 0.999  # winsorisation of line quantity for demand modelling
N_SKU_FORECAST = 60  # top-N SKUs by revenue for SKU-level forecasts / inventory

# --- Inventory assumptions (no stock table exists in the public dataset) ----
SERVICE_LEVEL = 0.95
ORDER_COST = 25.0  # GBP per purchase order (assumption)
HOLDING_RATE = 0.20  # annual holding cost as share of unit cost (assumption)
UNIT_COST_RATIO = 0.60  # unit cost = 60 % of selling price (assumption)

# --- Acceptance targets from the Zidio spec ---------------------------------
TARGETS = {
    "mape_max": 12.0,
    "auc_min": 0.88,
    "precision_top20_min": 0.75,
    "segments_min": 6,
    "segments_max": 8,
    "batch_seconds_max": 300,
}

# --- Brand palette used across charts / dashboard ---------------------------
PALETTE = {
    "bg": "#0B0F1F",
    "panel": "#131A33",
    "violet": "#7C5CFF",
    "cyan": "#22D3EE",
    "emerald": "#34D399",
    "amber": "#FBBF24",
    "rose": "#FB7185",
    "text": "#E6E9F5",
    "muted": "#8E97B8",
}

EUROPE = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Channel Islands",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "EIRE",
    "European Community",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Iceland",
    "Italy",
    "Lithuania",
    "Malta",
    "Netherlands",
    "Norway",
    "Poland",
    "Portugal",
    "Spain",
    "Sweden",
    "Switzerland",
}


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROC, DATA_SAMPLE, MODELS, REPORTS, FIGURES, DRIFT_DIR, LOGS):
        p.mkdir(parents=True, exist_ok=True)
