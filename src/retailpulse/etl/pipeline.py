"""End-to-end ETL runner:  python -m retailpulse.etl.pipeline

Outputs (data/processed):
  clean_transactions.parquet   full cleaned line items (git-ignored, ~800k rows)
  daily_sales.parquet          one row per calendar day
  sku_daily.parquet            daily units for top-N SKUs
  products.parquet             product master (top SKUs)
  country_summary.parquet      revenue by country / region
  hourly_profile.parquet       dow x hour heat-map data
  returns.parquet              cancelled lines
Reports: reports/etl_report.json, reports/data_quality.json, data/sample/clean_sample.csv
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from retailpulse import config
from retailpulse.etl import extract, load, transform, validate

log = logging.getLogger("retailpulse.etl")


def build_aggregates(sales: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily = transform.build_daily_sales(sales, returns)

    top = sales.groupby("stock_code", observed=True)["revenue"].sum().nlargest(config.N_SKU_FORECAST * 3).index
    sub = sales[sales["stock_code"].isin(top)]
    sku_daily = (
        sub.groupby(["date", "stock_code"], observed=True)
        .agg(units=("quantity_capped", "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )
    sku_daily["stock_code"] = sku_daily["stock_code"].astype(str)

    products = (
        sub.groupby("stock_code", observed=True)
        .agg(
            description=("description", "first"),
            avg_price=("unit_price", "median"),
            total_units=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
            active_days=("date", "nunique"),
            buyers=("customer_key", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    products["stock_code"] = products["stock_code"].astype(str)
    products["description"] = products["description"].astype(str)
    q = products["avg_price"].quantile([0.25, 0.5, 0.75]).values
    products["price_band"] = pd.cut(
        products["avg_price"], [-1, q[0], q[1], q[2], 1e9], labels=["Budget", "Standard", "Premium", "Luxury"]
    ).astype(str)

    country = (
        sales.groupby(["country", "region_group"], observed=True)
        .agg(
            revenue=("revenue", "sum"),
            units=("quantity", "sum"),
            invoices=("invoice_no", "nunique"),
            customers=("customer_key", "nunique"),
        )
        .reset_index()
    )
    country["country"] = country["country"].astype(str)
    country["region_group"] = country["region_group"].astype(str)
    country = country[country["invoices"] > 0].sort_values("revenue", ascending=False)

    hourly = (
        sales.groupby(["dow", "hour"]).agg(revenue=("revenue", "sum"), invoices=("invoice_no", "nunique")).reset_index()
    )
    monthly_region = (
        sales.assign(month_start=sales["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month_start", "region_group"], observed=True)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
        .reset_index()
    )
    monthly_region["region_group"] = monthly_region["region_group"].astype(str)
    return {
        "daily_sales": daily,
        "sku_daily": sku_daily,
        "products": products,
        "country_summary": country,
        "hourly_profile": hourly,
        "monthly_region": monthly_region,
        "returns": returns,
    }


def run(path=None, write_sql: bool = True) -> dict:
    config.ensure_dirs()
    t0 = time.time()
    raw = extract.extract(path)
    n_raw = len(raw)
    res = transform.transform(raw)
    del raw
    sales = res.sales

    qc = validate.validate(sales)
    (config.REPORTS / "data_quality.json").write_text(json.dumps(qc, indent=2))
    if not qc["success"]:
        failed = [c for c in qc["checks"] if not c["passed"]]
        raise RuntimeError(f"Data-quality gate failed: {failed}")

    aggs = build_aggregates(sales, res.returns)
    for name, df in aggs.items():
        load.save_parquet(df, name)
    load.save_parquet(sales, "clean_transactions")
    sales.head(5000).to_csv(config.DATA_SAMPLE / "clean_sample.csv", index=False)

    if write_sql:
        try:
            load.save_sql(aggs["daily_sales"], "daily_sales")
            load.save_sql(aggs["products"], "products")
            load.save_sql(aggs["country_summary"], "country_summary")
        except Exception as exc:  # DB is optional for local runs
            log.warning("SQL load skipped: %s", exc)

    elapsed = time.time() - t0
    report = {
        "rows_raw": int(n_raw),
        "rows_clean": int(len(sales)),
        "rows_returns": int(len(res.returns)),
        "retention_pct": round(100 * len(sales) / n_raw, 2),
        "customers": int(sales["customer_key"].nunique()),
        "skus": int(sales["stock_code"].nunique()),
        "date_min": str(sales["date"].min().date()),
        "date_max": str(sales["date"].max().date()),
        "revenue_total": float(sales["revenue"].sum()),
        "guest_line_share": float(sales["is_guest"].mean()),
        "custom_column": transform.CUSTOM_COLUMN,
        "columns": list(sales.columns),
        "lineage": res.lineage,
        "quality_passed": f"{qc['passed']}/{qc['total']}",
        "elapsed_seconds": round(elapsed, 1),
    }
    (config.REPORTS / "etl_report.json").write_text(json.dumps(report, indent=2))
    log.info("ETL done in %.1fs -> %s rows kept of %s", elapsed, len(sales), n_raw)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rep = run()
    print(json.dumps({k: v for k, v in rep.items() if k not in ("lineage", "columns")}, indent=2))
