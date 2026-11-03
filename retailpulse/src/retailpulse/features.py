"""Feature engineering: RFM / behavioural customer features and churn snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailpulse import config

CHURN_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_basket_units",
    "n_skus",
    "tenure_days",
    "mean_gap",
    "std_gap",
    "recency_gap_ratio",
    "freq_90",
    "freq_180",
    "monetary_90",
    "monetary_180",
    "freq_trend",
    "monetary_trend",
    "return_rate",
    "sunday_share",
    "is_uk",
    "max_order_value",
    "days_since_first_to_second",
    # short-window + seasonality-aware features
    "freq_30",
    "freq_60",
    "last_order_value",
    "ly_orders",
    "ly_value",
    "q4_share",
    "snap_sin",
    "snap_cos",
    "days_to_xmas",
    "gap_cv",
    "orders_per_month",
]


def invoice_table(sales: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items into one row per invoice for identified customers."""
    s = sales[~sales["is_guest"]]
    inv = (
        s.groupby(["customer_key", "invoice_no"], observed=True)
        .agg(
            date=("date", "first"),
            value=("revenue", "sum"),
            units=("quantity", "sum"),
            skus=("stock_code", "nunique"),
            dow=("dow", "first"),
            country=("country", "first"),
        )
        .reset_index()
    )
    inv["country"] = inv["country"].astype(str)
    return inv


def rfm_features(inv: pd.DataFrame, returns: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """RFM + behavioural features as of `snapshot` (uses only data strictly before it)."""
    h = inv[inv["date"] < snapshot]
    h = h.assign(_sun=(h["dow"] == 6).astype(float), _uk=(h["country"] == "United Kingdom").astype(float))
    g = h.groupby("customer_key")
    f = pd.DataFrame(
        {
            "recency": (snapshot - g["date"].max()).dt.days,
            "frequency": g["invoice_no"].nunique(),
            "monetary": g["value"].sum(),
            "avg_order_value": g["value"].mean(),
            "avg_basket_units": g["units"].mean(),
            "n_skus": g["skus"].sum(),
            "tenure_days": (snapshot - g["date"].min()).dt.days,
            "max_order_value": g["value"].max(),
            "sunday_share": g["_sun"].mean(),
            "is_uk": g["_uk"].mean(),
        }
    )

    # inter-purchase gap statistics ------------------------------------------
    hh = h.sort_values(["customer_key", "date"])
    hh["gap"] = hh.groupby("customer_key")["date"].diff().dt.days
    gap = hh.groupby("customer_key")["gap"].agg(["mean", "std"])
    f["mean_gap"] = gap["mean"]
    f["std_gap"] = gap["std"]
    f["days_since_first_to_second"] = hh.dropna(subset=["gap"]).groupby("customer_key")["gap"].first()

    # windowed activity ------------------------------------------------------
    def win(days: int) -> pd.DataFrame:
        w = h[h["date"] >= snapshot - pd.Timedelta(days=days)].groupby("customer_key")
        return pd.DataFrame({"f": w["invoice_no"].nunique(), "m": w["value"].sum()})

    w90, w180 = win(90), win(180)
    f["freq_90"], f["monetary_90"] = w90["f"], w90["m"]
    f["freq_180"], f["monetary_180"] = w180["f"], w180["m"]
    f[["freq_90", "monetary_90", "freq_180", "monetary_180"]] = f[
        ["freq_90", "monetary_90", "freq_180", "monetary_180"]
    ].fillna(0)
    prev_f = (f["freq_180"] - f["freq_90"]).clip(lower=0)
    prev_m = (f["monetary_180"] - f["monetary_90"]).clip(lower=0)
    f["freq_trend"] = (f["freq_90"] - prev_f) / (prev_f + 1)
    f["monetary_trend"] = (f["monetary_90"] - prev_m) / (prev_m + 1)

    # short windows, last order value -------------------------------------------
    f["freq_30"] = win(30)["f"].reindex(f.index).fillna(0)
    f["freq_60"] = win(60)["f"].reindex(f.index).fillna(0)
    f["last_order_value"] = hh.groupby("customer_key")["value"].last()

    # same 90-day window one year earlier (captures annual / seasonal buyers) ------
    ly0, ly1 = snapshot - pd.Timedelta(days=365), snapshot - pd.Timedelta(days=365 - config.CHURN_WINDOW_DAYS)
    ly = h[(h["date"] >= ly0) & (h["date"] < ly1)].groupby("customer_key")
    f["ly_orders"] = ly["invoice_no"].nunique().reindex(f.index).fillna(0)
    f["ly_value"] = ly["value"].sum().reindex(f.index).fillna(0)
    q4 = h.assign(q4=h["date"].dt.month.isin([10, 11, 12])).groupby("customer_key")["q4"].mean()
    f["q4_share"] = q4.reindex(f.index).fillna(0)

    # time-of-year of the snapshot (retail demand is strongly seasonal) ------------
    doy = snapshot.dayofyear
    f["snap_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["snap_cos"] = np.cos(2 * np.pi * doy / 365.25)
    xmas = pd.Timestamp(year=snapshot.year, month=12, day=25)
    f["days_to_xmas"] = (xmas - snapshot).days if snapshot <= xmas else 365 - (snapshot - xmas).days

    # returns -----------------------------------------------------------------
    r = returns[(returns["date"] < snapshot) & returns["customer_key"].notna()]
    rv = r.groupby("customer_key")["value"].sum()
    f["return_rate"] = (rv.reindex(f.index).fillna(0) / f["monetary"].replace(0, np.nan)).clip(0, 1).fillna(0)

    f["mean_gap"] = f["mean_gap"].fillna(f["tenure_days"].clip(lower=1))
    f["std_gap"] = f["std_gap"].fillna(0)
    f["days_since_first_to_second"] = f["days_since_first_to_second"].fillna(-1)
    f["recency_gap_ratio"] = f["recency"] / f["mean_gap"].clip(lower=1)
    f["gap_cv"] = f["std_gap"] / f["mean_gap"].clip(lower=1)
    f["orders_per_month"] = f["frequency"] / (f["tenure_days"].clip(lower=30) / 30.0)
    return f.reset_index()


def churn_snapshot(
    inv: pd.DataFrame,
    returns: pd.DataFrame,
    snapshot: pd.Timestamp,
    window: int = config.CHURN_WINDOW_DAYS,
    labelled: bool = True,
) -> pd.DataFrame:
    """Features at `snapshot`; label = 1 if the customer buys nothing in the next `window` days."""
    f = rfm_features(inv, returns, snapshot)
    f["snapshot"] = snapshot
    if labelled:
        fut = inv[(inv["date"] >= snapshot) & (inv["date"] < snapshot + pd.Timedelta(days=window))]
        active = set(fut["customer_key"])
        f["churn"] = (~f["customer_key"].isin(active)).astype(int)
    return f


def build_churn_frames(inv: pd.DataFrame, returns: pd.DataFrame, last_date: pd.Timestamp):
    """Temporal split without leakage: label windows never overlap later snapshots.

    train: weekly snapshots 2010-03-01 .. 2011-02-28  (labels end <= 2011-06-01)
    valid: 2011-06-09                                  (labels end 2011-09-07)
    test : last_date - 90d  (= 2011-09-10)             (labels end at last_date)
    """
    train_snaps = pd.date_range("2010-03-01", "2011-03-01", freq="7D")
    valid_snap = pd.Timestamp("2011-06-09")
    test_snap = (last_date - pd.Timedelta(days=config.CHURN_WINDOW_DAYS - 1)).normalize()
    tr = pd.concat([churn_snapshot(inv, returns, s) for s in train_snaps], ignore_index=True)
    va = churn_snapshot(inv, returns, valid_snap)
    te = churn_snapshot(inv, returns, test_snap)
    return tr, va, te, test_snap
