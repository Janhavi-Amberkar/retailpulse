"""TRANSFORM: clean raw transactions, engineer features, add the custom column.

Every cleaning step records how many rows it removed (lineage), so the ETL report
is a faithful audit trail rather than a hand-written summary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from retailpulse import config

# Product codes look like 85048, 79323P, 15056BL ... plus the DCGS* gift-set family.
PRODUCT_CODE = r"^(?:\d{5}[A-Za-z]{0,2}|DCGS\w+)$"

CUSTOM_COLUMN = "region_group"  # <- the extra column added on top of the source schema


@dataclass
class TransformResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    lineage: list[dict] = field(default_factory=list)


def region_group(country: str) -> str:
    """CUSTOM COLUMN: map each country to a commercial region."""
    if country == "United Kingdom":
        return "UK"
    if country in config.EUROPE:
        return "Europe"
    if country == "Unspecified":
        return "Unspecified"
    return "Rest of World"


def anonymise_ids(ids: pd.Series, salt: str = config.ANON_SALT) -> pd.Series:
    """Salted SHA-256 pseudonym, e.g. 13085.0 -> 'C-9f2a41c07b'. Raw IDs never leave ETL."""

    def _h(v: float) -> str | None:
        if pd.isna(v):
            return None
        return "C-" + hashlib.sha256(f"{salt}:{int(v)}".encode()).hexdigest()[:10]

    uniq = ids.dropna().unique()
    mapping = {v: _h(v) for v in uniq}
    return ids.map(mapping)


def transform(raw: pd.DataFrame) -> TransformResult:
    lineage: list[dict] = []

    def step(name: str, before: int, after: int, note: str = "") -> None:
        lineage.append(
            {
                "step": name,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "note": note,
            }
        )

    df = raw.rename(
        columns={
            "Invoice": "invoice_no",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "unit_price",
            "Customer ID": "customer_raw",
            "Country": "country",
        }
    )
    lineage.append(
        {"step": "extract", "rows_before": len(df), "rows_after": len(df), "rows_removed": 0, "note": "raw rows loaded"}
    )

    # 1. types ----------------------------------------------------------------
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    n = len(df)
    df = df.dropna(subset=["invoice_date"])
    step("drop unparsable dates", n, len(df))
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["country"] = df["country"].astype(str).str.strip()

    # 2. exact duplicates -----------------------------------------------------
    n = len(df)
    df = df.drop_duplicates()
    step("drop exact duplicate rows", n, len(df), "same invoice/item/qty/time/price")

    # 3. cancellations (invoice starts with 'C') are kept apart as returns ------
    is_cancel = df["invoice_no"].str.upper().str.startswith("C")
    returns = df[is_cancel].copy()
    n = len(df)
    df = df[~is_cancel]
    step("split cancellations -> returns table", n, len(df), "invoice prefix 'C'")

    # 4. bad-debt adjustments (invoice prefix 'A') ----------------------------
    n = len(df)
    df = df[~df["invoice_no"].str.upper().str.startswith("A")]
    step("drop bad-debt adjustments", n, len(df), "invoice prefix 'A'")

    # 5. impossible values ----------------------------------------------------
    n = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    step("drop qty<=0 or price<=0", n, len(df), "stock write-offs / free samples")

    # 6. non-product service lines (POST, DOT, M, BANK CHARGES, AMAZONFEE ...) --
    n = len(df)
    df = df[df["stock_code"].str.match(PRODUCT_CODE)]
    step("drop non-product service codes", n, len(df), "postage, fees, manual, samples")

    # 7. descriptions: normalise, back-fill from StockCode mode, drop leftovers ---
    df["description"] = df["description"].astype("string").str.strip().str.upper()
    mode_desc = df.dropna(subset=["description"]).groupby("stock_code")["description"].agg(lambda s: s.mode().iat[0])
    df["description"] = df["description"].fillna(df["stock_code"].map(mode_desc))
    n = len(df)
    df = df.dropna(subset=["description"])
    step("fix / drop missing descriptions", n, len(df), "back-filled from StockCode mode")
    # canonical description per code (latest spelling wins -> avoids duplicate SKUs)
    df["description"] = df["stock_code"].map(mode_desc).fillna(df["description"])

    # 7b. duplicates that only differed by description spelling/whitespace ----------
    n = len(df)
    df = df.drop_duplicates(["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_raw"])
    step("drop duplicates after text normalisation", n, len(df), "same line, different description text")

    # 8. winsorise quantity for demand modelling (keeps raw quantity too) --------
    cap = float(df["quantity"].quantile(config.QTY_CAP_QUANTILE))
    df["quantity_capped"] = df["quantity"].clip(upper=cap)
    n_cap = int((df["quantity"] > cap).sum())
    lineage.append(
        {
            "step": "winsorise quantity",
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_removed": 0,
            "note": f"{n_cap} lines capped at p{config.QTY_CAP_QUANTILE * 100:.1f} = {cap:.0f} units",
        }
    )

    # 9. customers: guest checkouts have no ID -> flagged, not dropped ----------
    df["is_guest"] = df["customer_raw"].isna()
    df["customer_key"] = anonymise_ids(df["customer_raw"])
    df = df.drop(columns=["customer_raw"])  # privacy: no raw IDs downstream
    lineage.append(
        {
            "step": "anonymise customer IDs",
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_removed": 0,
            "note": f"{int(df['is_guest'].sum())} guest lines keep NULL key (excluded from CRM models)",
        }
    )

    # 10. feature engineering + CUSTOM COLUMN ----------------------------------
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["date"] = df["invoice_date"].dt.normalize()
    df["year"] = df["invoice_date"].dt.year.astype("int16")
    df["month"] = df["invoice_date"].dt.month.astype("int8")
    df["quarter"] = df["invoice_date"].dt.quarter.astype("int8")
    df["dow"] = df["invoice_date"].dt.dayofweek.astype("int8")
    df["hour"] = df["invoice_date"].dt.hour.astype("int8")
    df[CUSTOM_COLUMN] = df["country"].map(region_group)
    lineage.append(
        {
            "step": f"add custom column '{CUSTOM_COLUMN}'",
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_removed": 0,
            "note": "UK / Europe / Rest of World / Unspecified, derived from country",
        }
    )

    for c in ("country", "region_group", "stock_code"):
        df[c] = df[c].astype("category")

    # returns, cleaned in the same way (only product lines, with customer key) ----
    returns = returns[returns["stock_code"].str.match(PRODUCT_CODE)].copy()
    returns["customer_key"] = anonymise_ids(returns["customer_raw"])
    returns["date"] = returns["invoice_date"].dt.normalize()
    returns["units"] = returns["quantity"].abs()
    returns["value"] = (returns["units"] * returns["unit_price"]).round(2)
    returns = returns[["invoice_no", "stock_code", "date", "units", "value", "customer_key", "country"]].reset_index(
        drop=True
    )

    df = df.sort_values("invoice_date").reset_index(drop=True)
    return TransformResult(sales=df, returns=returns, lineage=lineage)


def build_daily_sales(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day (closed days are kept with zero sales + `is_open` flag)."""
    g = sales.groupby("date").agg(
        units=("quantity", "sum"),
        units_capped=("quantity_capped", "sum"),
        revenue=("revenue", "sum"),
        invoices=("invoice_no", "nunique"),
        lines=("invoice_no", "size"),
        customers=("customer_key", "nunique"),
        guest_lines=("is_guest", "sum"),
    )
    r = returns.groupby("date").agg(return_units=("units", "sum"), return_value=("value", "sum"))
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    daily = g.join(r, how="outer").reindex(full)
    daily.index.name = "date"
    daily["is_open"] = daily["invoices"].notna() & (daily["invoices"] > 0)
    daily = daily.fillna(0.0)
    daily["guest_share"] = np.where(daily["lines"] > 0, daily["guest_lines"] / daily["lines"].replace(0, 1), 0)
    return daily.reset_index()
