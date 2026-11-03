"""EXTRACT: read raw Online Retail II transactions (CSV or XLSX)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from retailpulse import config

log = logging.getLogger(__name__)

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def download_raw(dest: Path | None = None, url: str = config.RAW_URL) -> Path:
    """Download the public dataset mirror when the raw file is not present locally."""
    dest = dest or config.RAW_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL
    return dest


def extract(path: Path | None = None) -> pd.DataFrame:
    """Load the raw transaction log with explicit dtypes (no silent coercion)."""
    path = Path(path or config.RAW_FILE)
    if not path.exists():
        download_raw(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frames = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        df = pd.concat(frames.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str})
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")
    log.info("Extracted %s rows x %s cols", *df.shape)
    return df[RAW_COLUMNS].copy()
