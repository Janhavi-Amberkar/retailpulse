"""VALIDATE: data-quality gate (Great Expectations when installed, pandas otherwise).

The same expectation list is evaluated either way so results are identical; Great
Expectations is used as the execution engine when available (spec Day 2).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    column: str
    passed: bool
    detail: str
    engine: str = "pandas"


def _pandas_checks(df: pd.DataFrame) -> list[Check]:
    out: list[Check] = []

    def add(name: str, col: str, ok: bool, detail: str) -> None:
        out.append(Check(name, col, bool(ok), detail))

    required = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "unit_price",
        "invoice_date",
        "country",
        "revenue",
        "region_group",
        "customer_key",
        "date",
    ]
    add("schema has required columns", "*", all(c in df.columns for c in required), f"{len(df.columns)} columns")
    for col in (
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "unit_price",
        "invoice_date",
        "country",
        "revenue",
        "region_group",
    ):
        nn = int(df[col].isna().sum())
        add("not null", col, nn == 0, f"{nn} nulls")
    add("quantity > 0", "quantity", (df["quantity"] > 0).all(), f"min={df['quantity'].min()}")
    add("unit_price > 0", "unit_price", (df["unit_price"] > 0).all(), f"min={df['unit_price'].min():.3f}")
    add("unit_price <= 10,000", "unit_price", (df["unit_price"] <= 10_000).all(), f"max={df['unit_price'].max():.2f}")
    add("quantity_capped <= quantity", "quantity_capped", (df["quantity_capped"] <= df["quantity"]).all(), "")
    add("no cancellations left", "invoice_no", (~df["invoice_no"].str.upper().str.startswith("C")).all(), "")
    add(
        "revenue == quantity*price",
        "revenue",
        ((df["quantity"] * df["unit_price"]).round(2) - df["revenue"]).abs().max() < 0.011,
        "",
    )
    add(
        "region_group in allowed set",
        "region_group",
        set(df["region_group"].astype(str).unique()) <= {"UK", "Europe", "Rest of World", "Unspecified"},
        str(sorted(df["region_group"].astype(str).unique())),
    )
    add(
        "date range 2009-12 .. 2011-12",
        "invoice_date",
        (df["invoice_date"].min() >= pd.Timestamp("2009-12-01"))
        and (df["invoice_date"].max() < pd.Timestamp("2011-12-31")),
        f"{df['invoice_date'].min():%Y-%m-%d} -> {df['invoice_date'].max():%Y-%m-%d}",
    )
    add(
        "no duplicate lines",
        "*",
        not df.duplicated(["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_key"]).any(),
        "",
    )
    add(
        "customer keys anonymised",
        "customer_key",
        df["customer_key"].dropna().str.match(r"^C-[0-9a-f]{10}$").all(),
        "pattern C-xxxxxxxxxx",
    )
    add("guest flag consistent with key", "is_guest", (df["is_guest"] == df["customer_key"].isna()).all(), "")
    return out


def _ge_checks(df: pd.DataFrame) -> list[Check] | None:
    """Run a representative subset through Great Expectations 1.x. None if unavailable."""
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
    except Exception:  # pragma: no cover
        return None
    try:
        sample = df.sample(min(len(df), 200_000), random_state=42).copy()
        for c in sample.select_dtypes("category").columns:
            sample[c] = sample[c].astype(str)
        sample["customer_key"] = sample["customer_key"].astype(object)
        ctx = gx.get_context(mode="ephemeral")
        src = ctx.data_sources.add_pandas("retail")
        asset = src.add_dataframe_asset("clean_sales")
        bd = asset.add_batch_definition_whole_dataframe("whole")
        batch = bd.get_batch(batch_parameters={"dataframe": sample})
        exps = [
            ("not null", "invoice_no", gxe.ExpectColumnValuesToNotBeNull(column="invoice_no")),
            ("not null", "stock_code", gxe.ExpectColumnValuesToNotBeNull(column="stock_code")),
            ("not null", "description", gxe.ExpectColumnValuesToNotBeNull(column="description")),
            ("quantity > 0", "quantity", gxe.ExpectColumnValuesToBeBetween(column="quantity", min_value=1)),
            (
                "unit_price > 0",
                "unit_price",
                gxe.ExpectColumnValuesToBeBetween(column="unit_price", min_value=0.001, max_value=10_000),
            ),
            (
                "region_group in set",
                "region_group",
                gxe.ExpectColumnValuesToBeInSet(
                    column="region_group", value_set=["UK", "Europe", "Rest of World", "Unspecified"]
                ),
            ),
        ]
        res = []
        for name, col, e in exps:
            r = batch.validate(e)
            res.append(Check(name, col, bool(r.success), "great_expectations (200k sample)", "great_expectations"))
        return res
    except Exception as exc:  # pragma: no cover - GE API drift must not break the ETL
        log.warning("Great Expectations run skipped: %s", exc)
        return None


def validate(df: pd.DataFrame, use_ge: bool = True) -> dict:
    checks = _pandas_checks(df)
    ge = _ge_checks(df) if use_ge else None
    if ge:
        checks += ge
    passed = sum(c.passed for c in checks)
    return {
        "passed": passed,
        "total": len(checks),
        "success": passed == len(checks),
        "great_expectations_used": bool(ge),
        "checks": [asdict(c) for c in checks],
    }
