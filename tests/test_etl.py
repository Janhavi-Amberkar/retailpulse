import numpy as np
import pandas as pd
import pytest

from retailpulse.etl import transform, validate


def raw_frame():
    rows = [
        # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        ("500001", "85048", "glass ball", 12, "2010-01-04 10:00", 6.95, 13085.0, "United Kingdom"),
        ("500001", "85048", "glass ball", 12, "2010-01-04 10:00", 6.95, 13085.0, "United Kingdom"),  # exact dup
        ("500002", "79323P", "Pink Lights", 5, "2010-01-05 11:00", 2.00, 12583.0, "France"),
        ("C500003", "85048", "glass ball", -2, "2010-01-06 09:00", 6.95, 13085.0, "United Kingdom"),  # cancellation
        ("500004", "POST", "POSTAGE", 1, "2010-01-06 09:30", 18.0, 13085.0, "United Kingdom"),  # service code
        ("500005", "22041", "frame", -3, "2010-01-07 09:30", 2.1, 13085.0, "United Kingdom"),  # negative qty
        ("500006", "22064", "doughnut pot", 24, "2010-01-08 12:00", 0.0, 13085.0, "United Kingdom"),  # price 0
        ("500007", "21232", None, 6, "2010-01-09 12:00", 1.25, np.nan, "Brazil"),  # guest, missing description
        ("500008", "21232", "STRAWBERRY BOX", 6, "2010-01-10 12:00", 1.25, 14000.0, "EIRE"),
        ("A500009", "21232", "adjust bad debt", 1, "2010-01-10 12:30", 1.25, np.nan, "Unspecified"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
    )


@pytest.fixture(scope="module")
def res():
    return transform.transform(raw_frame())


def test_removes_duplicates_returns_service_and_invalid(res):
    s = res.sales
    assert len(s) == 4  # 85048, 79323P, 21232 (guest, desc back-filled), 21232 (EIRE)
    assert (s["quantity"] > 0).all() and (s["unit_price"] > 0).all()
    assert not s["invoice_no"].str.startswith(("C", "A")).any()
    assert "POST" not in set(s["stock_code"].astype(str))
    assert len(res.returns) == 1 and res.returns["units"].iloc[0] == 2


def test_description_backfilled_from_stockcode(res):
    s = res.sales
    assert s["description"].notna().all()
    assert (s[s["stock_code"] == "21232"]["description"] == "STRAWBERRY BOX").all()


def test_custom_column_region_group(res):
    m = dict(zip(res.sales["country"].astype(str), res.sales["region_group"].astype(str)))
    assert m == {"United Kingdom": "UK", "France": "Europe", "Brazil": "Rest of World", "EIRE": "Europe"}
    assert transform.region_group("Unspecified") == "Unspecified"


def test_customer_ids_are_pseudonymised_and_stable(res):
    k = res.sales["customer_key"].dropna()
    assert k.str.match(r"^C-[0-9a-f]{10}$").all()
    assert "13085" not in " ".join(k)
    assert res.sales["is_guest"].sum() == 1
    a = transform.anonymise_ids(pd.Series([1.0, 2.0, 1.0]))
    assert a.iloc[0] == a.iloc[2] != a.iloc[1]


def test_revenue_and_lineage(res):
    s = res.sales
    assert np.allclose(s["revenue"], (s["quantity"] * s["unit_price"]).round(2))
    removed = sum(l["rows_removed"] for l in res.lineage)
    # 10 raw rows: 1 duplicate + 1 cancellation + 1 bad-debt + 1 qty<=0 + 1 price<=0 + 1 service code = 6 removed
    assert removed == 6 and len(s) == 10 - removed
    steps = {l["step"]: l["rows_removed"] for l in res.lineage}
    assert steps["drop exact duplicate rows"] == 1 and steps["drop non-product service codes"] == 1
    assert any(l["step"].startswith("add custom column") for l in res.lineage)


def test_quality_gate_passes_on_clean_data(res):
    rep = validate.validate(res.sales, use_ge=False)
    assert rep["success"], [c for c in rep["checks"] if not c["passed"]]


def test_quality_gate_catches_bad_data(res):
    bad = res.sales.copy()
    bad.loc[bad.index[0], "quantity"] = -5
    assert not validate.validate(bad, use_ge=False)["success"]
