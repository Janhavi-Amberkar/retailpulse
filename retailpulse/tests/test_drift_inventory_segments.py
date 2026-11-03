import numpy as np
import pandas as pd

from retailpulse import drift, inventory, segmentation


def test_psi_identical_and_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert drift.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert drift.psi(a, rng.normal(1.5, 1, 5000)) > 0.2


def test_drift_policy_and_simulation_monotone():
    rng = np.random.default_rng(1)
    feats = ["recency", "frequency", "monetary", "mean_gap"]
    ref = pd.DataFrame({f: rng.lognormal(3, 0.5, 3000) for f in feats})
    shares = []
    for sev in (0.0, 0.5, 1.0):
        t = drift.drift_table(ref, drift.simulate_drift(ref.sample(2000, random_state=2), sev, feats), feats)
        shares.append(drift.drift_summary(t)["share_drifted"])
    assert shares[0] == 0 and shares[-1] >= shares[1] >= shares[0] and shares[-1] > 0.3
    assert drift.should_retrain({"dataset_drift": True, "n_alert": 3, "n_features": 4})["retrain"]
    assert not drift.should_retrain({"dataset_drift": False, "n_alert": 0, "n_features": 4})["retrain"]
    assert drift.should_retrain({"dataset_drift": False, "n_alert": 0, "n_features": 4}, 20, 10)["retrain"]


def _final(code="X", yhat=100.0, sigma=20.0, n=5):
    return {
        code: pd.DataFrame(
            {"week_end": pd.date_range("2011-12-10", periods=n, freq="W-SAT"), "yhat": [yhat] * n, "sigma": [sigma] * n}
        )
    }


def _stock(on_hand, lead=14):
    return pd.DataFrame(
        {
            "stock_code": ["X"],
            "description": ["TEST"],
            "on_hand": [on_hand],
            "lead_days": [lead],
            "avg_price": [2.0],
            "synthetic": [True],
        }
    )


def test_inventory_formulas_and_monotonicity():
    lo = inventory.recommend(_stock(50), _final(), service_level=0.90).iloc[0]
    hi = inventory.recommend(_stock(50), _final(), service_level=0.99).iloc[0]
    assert hi["safety_stock"] > lo["safety_stock"] > 0
    # lead 14d = 2 weeks, review 2 weeks -> P = 4 weeks of 100 units
    assert abs(lo["order_up_to"] - (400 + lo["safety_stock"])) < 1e-6
    assert abs(lo["reorder_point"] - (200 + 1.2816 * 20 * np.sqrt(2))) < 0.5
    assert lo["recommended_order"] == np.ceil(lo["order_up_to"] - 50)
    more = inventory.recommend(_stock(50), _final(), demand_multiplier=1.5).iloc[0]
    assert more["order_up_to"] > lo["order_up_to"]
    slow = inventory.recommend(_stock(50), _final(), lead_scale=2.0).iloc[0]
    assert slow["reorder_point"] > lo["reorder_point"]


def test_inventory_status_logic():
    assert inventory.recommend(_stock(10), _final()).iloc[0]["status"] == "Stockout risk"
    assert inventory.recommend(_stock(5000), _final()).iloc[0]["status"] == "Overstock"
    assert inventory.recommend(_stock(0), _final()).iloc[0]["recommended_order"] > 0


def test_segmentation_on_synthetic_rfm():
    rng = np.random.default_rng(3)
    n = 900
    grp = rng.integers(0, 6, n)
    rfm = pd.DataFrame(
        {
            "customer_key": [f"C-{i:010x}" for i in range(n)],
            "recency": rng.gamma(2 + grp, 20),
            "frequency": rng.poisson(1 + 3 * (5 - grp)) + 1,
            "monetary": rng.lognormal(5 + (5 - grp) * 0.4, 0.4),
            "avg_order_value": rng.lognormal(5, 0.3),
            "n_skus": rng.poisson(20, n) + 1,
            "mean_gap": rng.gamma(2, 20, n),
            "return_rate": rng.beta(1, 20, n),
            "tenure_days": rng.integers(30, 700, n),
        }
    )
    res = segmentation.run_segmentation(rfm)
    assert 6 <= res["metrics"]["k"] <= 8
    names = res["profile"]["segment"].tolist()
    assert len(set(names)) == len(names)  # every cluster gets a unique business name
    assert abs(res["profile"]["revenue_share"].sum() - 1) < 1e-9


def test_churn_tiers_and_spend_at_risk():
    from retailpulse import churn

    p = np.linspace(0.99, 0.01, 100)
    rec = np.where(np.arange(100) < 20, 400, 30)  # 20 highest-risk customers are already lapsed
    t = churn.rank_tiers(p, rec)
    assert (t[:20] == "Lapsed").all() and (t[20:] != "Lapsed").all()
    assert (t == "Critical").sum() == 12 and t[20] == "Critical"  # 15 % of the 80 active customers
    df = pd.DataFrame({"monetary": [3600.0, 100.0], "tenure_days": [360, 360], "monetary_180": [0.0, 500.0]})
    s = churn.spend_at_risk(df)
    assert abs(s[0] - 1800) < 1e-6 and s[1] == 500  # run-rate beats zero recent spend; recent beats low run-rate
