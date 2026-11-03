import pandas as pd

from retailpulse import features


def inv_frame():
    d = pd.to_datetime
    rows = []
    # customer A: buys every 30 days until 2011-03-01 then stops; customer B: keeps buying; C: first buy late
    for i, day in enumerate(pd.date_range("2010-01-04", "2011-03-01", freq="30D")):
        rows.append(("A", f"A{i}", day, 100.0, 10, 3, day.dayofweek, "United Kingdom"))
    for i, day in enumerate(pd.date_range("2010-01-04", "2011-12-01", freq="20D")):
        rows.append(("B", f"B{i}", day, 50.0, 5, 2, day.dayofweek, "France"))
    rows.append(("C", "C0", d("2011-05-01"), 10.0, 1, 1, 6, "United Kingdom"))
    return pd.DataFrame(
        rows, columns=["customer_key", "invoice_no", "date", "value", "units", "skus", "dow", "country"]
    )


RET = pd.DataFrame({"customer_key": pd.Series([], dtype=str), "date": pd.to_datetime([]), "value": [], "units": []})


def test_features_use_only_past_data():
    inv = inv_frame()
    snap = pd.Timestamp("2011-06-01")
    f = features.rfm_features(inv, RET, snap).set_index("customer_key")
    assert set(f.index) == {"A", "B", "C"}
    last_a = inv[inv["customer_key"] == "A"]["date"].max()
    assert f.loc["A", "recency"] == (snap - last_a).days  # exact: days since the last pre-snapshot order
    assert f.loc["A", "frequency"] == (inv["customer_key"] == "A").sum()
    assert f.loc["C", "tenure_days"] == (snap - pd.Timestamp("2011-05-01")).days
    # adding FUTURE data must not change features at the snapshot
    future = pd.DataFrame(
        [("A", "AX", pd.Timestamp("2011-07-01"), 999.0, 1, 1, 4, "United Kingdom")], columns=inv.columns
    )
    f2 = features.rfm_features(pd.concat([inv, future]), RET, snap).set_index("customer_key")
    pd.testing.assert_frame_equal(f.sort_index(), f2.sort_index())


def test_churn_label_definition():
    inv = inv_frame()
    snap = pd.Timestamp("2011-06-01")
    df = features.churn_snapshot(inv, RET, snap).set_index("customer_key")
    assert df.loc["A", "churn"] == 1  # stopped buying
    assert df.loc["B", "churn"] == 0  # buys every 20 days
    assert "churn" not in features.churn_snapshot(inv, RET, snap, labelled=False).columns


def test_temporal_split_has_no_overlap():
    inv = inv_frame()
    last = pd.Timestamp("2011-12-09")
    tr, va, te, test_snap = features.build_churn_frames(inv, RET, last)
    assert tr["snapshot"].max() + pd.Timedelta(days=90) <= va["snapshot"].iloc[0]
    assert va["snapshot"].iloc[0] + pd.Timedelta(days=90) <= test_snap
    assert test_snap + pd.Timedelta(days=89) == last
    assert set(features.CHURN_FEATURES) <= set(tr.columns)
