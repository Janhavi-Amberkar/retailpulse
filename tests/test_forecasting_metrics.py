import numpy as np
import pandas as pd

from retailpulse import forecasting as F


def test_trading_calendar():
    d = pd.DatetimeIndex(
        ["2011-12-10", "2011-12-11", "2011-12-25", "2011-12-31", "2012-01-02", "2012-01-04", "2011-04-22"]
    )
    # Sat closed | Sun open | Xmas closed | NYE closed | Jan 2 closed | Jan 4 open | Good Friday (UK holiday) closed
    assert F.is_trading_day(d).tolist() == [False, True, False, False, False, True, False]


def test_mape_wape_basic():
    y, p = np.array([100, 200, 0]), np.array([110, 180, 5])
    assert abs(F.mape(y, p) - 10.0) < 1e-9  # zero actuals excluded
    assert abs(F.wape([100, 100], [90, 130]) - 20.0) < 1e-9


def test_horizon_metrics_weekly_blocks_and_total():
    dates = pd.date_range("2011-01-01", periods=28)
    df = pd.DataFrame({"date": dates, "actual": 100.0, "p": 110.0})
    m = F.horizon_metrics(df, "p")
    assert (
        abs(m["mape_daily"] - 10) < 1e-9 and abs(m["mape_weekly"] - 10) < 1e-9 and abs(m["ape_30d_total"] - 10) < 1e-9
    )


def test_calendar_features_shape_and_ranges():
    c = F.calendar_features(pd.date_range("2011-11-01", periods=60))
    assert c.shape[1] == 6 and c["is_open"].isin([0, 1]).all()
    assert c["pre_xmas"].between(0, 1).all() and c["dow_sin"].abs().max() <= 1


def test_best_weight_prefers_better_model():
    n = 28
    folds = pd.DataFrame(
        {"fold": 1, "date": pd.date_range("2011-01-01", periods=n), "actual": 100.0, "prophet": 140.0, "lstm": 100.0}
    )
    w, _ = F.best_weight(folds)
    assert w == 1.0
