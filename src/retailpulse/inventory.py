"""F05 – Inventory optimisation driven by SKU-level demand forecasts.

The public dataset has *sales* but no stock table.  Therefore:
  * demand, forecasts, safety stock, reorder point, EOQ and the policy back-test use REAL sales;
  * the current on-hand stock / lead times used for the demo recommendations are SIMULATED
    (seeded, documented, flagged `synthetic=True`) so that the optimisation logic can be shown.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from retailpulse import config

warnings.filterwarnings("ignore")
for n in ("cmdstanpy", "prophet"):
    logging.getLogger(n).setLevel(logging.ERROR)

REVIEW_WEEKS = 2
Z_INTERVAL = 1.645  # Prophet interval_width=0.90 -> 95th percentile one-sided


# ------------------------------------------------------------------ data -----
def weekly_matrix(sku_daily: pd.DataFrame, skus: list[str]) -> pd.DataFrame:
    """Weeks run Sunday..Saturday (shop is closed Saturdays, so Sun-Fri weeks are complete).
    The first partial week is dropped; the dataset ends on a Friday so the last week is complete."""
    d = sku_daily[sku_daily["stock_code"].isin(skus)]
    p = d.pivot_table(index="date", columns="stock_code", values="units", aggfunc="sum")
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="D")).fillna(0)
    w = p.resample("W-SAT").sum()
    return w.iloc[1:]


# -------------------------------------------------------------- forecasting --
def _fit_forecast(y: pd.Series, horizon: int) -> pd.DataFrame:
    from prophet import Prophet

    df = pd.DataFrame({"ds": y.index, "y": np.log1p(y.values)})
    m = Prophet(
        yearly_seasonality=4,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.90,
        uncertainty_samples=150,
    )
    m.fit(df)
    fut = pd.DataFrame({"ds": pd.date_range(y.index.max() + pd.Timedelta(days=7), periods=horizon, freq="W-SAT")})
    f = m.predict(fut)
    yhat = np.expm1(f["yhat"].values).clip(min=0)
    hi = np.expm1(f["yhat_upper"].values).clip(min=0)
    sigma = np.maximum((hi - yhat) / Z_INTERVAL, 0.05 * yhat + 1.0)
    return pd.DataFrame({"week_end": fut["ds"], "yhat": yhat, "sigma": sigma})


def _origin_job(code: str, y: pd.Series, origin_idx: int, horizon: int) -> tuple[str, int, pd.DataFrame]:
    return code, origin_idx, _fit_forecast(y.iloc[:origin_idx], horizon)


def forecast_all(weekly: pd.DataFrame, origins: list[int], horizon: int, n_jobs: int = 2) -> dict:
    """origins: integer positions (exclusive end of training slice) in the weekly index."""
    jobs = [(c, weekly[c], o, horizon) for c in weekly.columns for o in origins]
    res = Parallel(n_jobs=n_jobs, backend="loky", batch_size=4)(delayed(_origin_job)(*j) for j in jobs)
    return {(c, o): f for c, o, f in res}


def total_weekly(daily: pd.DataFrame) -> pd.Series:
    """Business-wide weekly units on the same Sunday-Saturday grid as `weekly_matrix`."""
    d = daily.set_index("date")["units_capped"]
    return d.resample("W-SAT").sum().iloc[1:]


def _fit_total(y: pd.Series, horizon: int) -> np.ndarray:
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=6,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        uncertainty_samples=0,
    )
    m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
    fut = pd.DataFrame({"ds": pd.date_range(y.index.max() + pd.Timedelta(days=7), periods=horizon, freq="W-SAT")})
    return m.predict(fut)["yhat"].clip(lower=0).values, fut["ds"]


def hier_forecast(
    weekly: pd.DataFrame, total: pd.Series, o: int, horizon: int, rho: float = 0.12, share_weeks: int = 8
) -> dict:
    """Hierarchical (top-down) SKU forecast at origin `o` (uses weeks < o only).

    forecast_sku(h) = recent demand share of the SKU  x  Prophet forecast of TOTAL weekly demand(h)
    The total carries seasonality estimated from ~1M transactions; SKU series alone are too sparse.
    sigma(h)^2 = share-noise^2 (back-cast residual std over last 13 weeks) + (rho * forecast)^2,
    where rho = relative error of the total forecast (weekly MAPE ~12 % from the demand module).
    """
    tot_hat, wk = _fit_total(total.iloc[:o], horizon)
    sku_roll = weekly.rolling(share_weeks).sum().shift(1)
    tot_roll = total.rolling(share_weeks).sum().shift(1)
    share_hist = sku_roll.div(tot_roll, axis=0)
    resid = weekly - share_hist.mul(total, axis=0)  # back-cast with actual totals -> pure share noise
    share_now = weekly.iloc[o - share_weeks : o].sum() / total.iloc[o - share_weeks : o].sum()
    sig_share = resid.iloc[o - 13 : o].std(ddof=0)
    out = {}
    for c in weekly.columns:
        yhat = share_now[c] * tot_hat
        sigma = np.sqrt(sig_share[c] ** 2 + (rho * yhat) ** 2)
        out[(c, o)] = pd.DataFrame({"week_end": wk, "yhat": yhat, "sigma": np.maximum(sigma, 1.0)})
    return out


def hier_forecast_all(weekly: pd.DataFrame, total: pd.Series, origins: list[int], horizon: int) -> dict:
    res: dict = {}
    for o in origins:
        res.update(hier_forecast(weekly, total, o, horizon))
    return res


# ------------------------------------------------------------- policy tests --
def backtest_policies(
    weekly: pd.DataFrame,
    lead_weeks: dict[str, int],
    fc_hier: dict,
    fc_direct: dict,
    origins: list[int],
    service_level: float = config.SERVICE_LEVEL,
) -> pd.DataFrame:
    """Order-up-to simulation over protection interval P = lead + review weeks.

    For each (SKU, origin): realised demand over P vs the stock position each policy would hold.
      understock = max(0, realised - S)   (lost sales)      overstock = max(0, S - realised) (excess)
    """
    z = norm.ppf(service_level)
    rows = []
    for (code, o), f in fc_hier.items():
        lw = lead_weeks[code]
        P = lw + REVIEW_WEEKS
        if o + P > len(weekly):
            continue
        y = weekly[code]
        real = float(y.iloc[o : o + P].sum())
        hist = y.iloc[:o]
        ma = float(hist.tail(4).mean())
        sd = float(hist.tail(12).std(ddof=0))
        d = fc_direct[(code, o)]
        S = {
            "Moving average (no buffer)": ma * P,
            "Moving average + safety stock": ma * P + z * sd * np.sqrt(P),
            "Per-SKU Prophet (no buffer)": float(d["yhat"].iloc[:P].sum()),
            "Hierarchical forecast (no buffer)": float(f["yhat"].iloc[:P].sum()),
            "RetailPulse (hierarchical + safety stock)": float(f["yhat"].iloc[:P].sum())
            + z * float(np.sqrt((f["sigma"].iloc[:P] ** 2).sum())),
        }
        for pol, s in S.items():
            rows.append(
                {
                    "sku": code,
                    "origin": int(o),
                    "week_end": weekly.index[o - 1],
                    "policy": pol,
                    "protect_weeks": P,
                    "realised": real,
                    "stock_level": s,
                    "understock": max(0.0, real - s),
                    "overstock": max(0.0, s - real),
                }
            )
    return pd.DataFrame(rows)


def summarise_policies(bt: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    b = bt.copy()
    price = b["sku"].map(prices).fillna(prices.median())
    margin = price * (1 - config.UNIT_COST_RATIO)
    cost = price * config.UNIT_COST_RATIO
    hold = cost * config.HOLDING_RATE * b["protect_weeks"] / 52.0
    b["lost_margin"] = b["understock"] * margin
    b["holding_cost"] = b["overstock"] * hold
    b["wasted_capital"] = b["overstock"] * cost
    g = b.groupby("policy").agg(
        cycles=("sku", "size"),
        demand=("realised", "sum"),
        understock_units=("understock", "sum"),
        overstock_units=("overstock", "sum"),
        lost_margin=("lost_margin", "sum"),
        holding_cost=("holding_cost", "sum"),
        stockout_cycles=("understock", lambda s: float((s > 0).mean())),
    )
    g["fill_rate_pct"] = 100 * (1 - g["understock_units"] / g["demand"])
    g["understock_pct_of_demand"] = 100 * g["understock_units"] / g["demand"]
    g["overstock_pct_of_demand"] = 100 * g["overstock_units"] / g["demand"]
    g["imbalance_units"] = g["understock_units"] + g["overstock_units"]
    g["total_cost"] = g["lost_margin"] + g["holding_cost"]
    return g.reset_index()


# ------------------------------------------------------- recommendations -----
def synthetic_stock(weekly: pd.DataFrame, products: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """SIMULATED current stock + supplier lead times (no stock table in the public data)."""
    rng = np.random.default_rng(seed)
    avg = weekly.tail(8).mean()
    skus = list(weekly.columns)
    cover_weeks = rng.uniform(0.4, 9.0, len(skus))
    lead_days = rng.choice([7, 14, 14, 21], len(skus))
    st = pd.DataFrame(
        {
            "stock_code": skus,
            "on_hand": np.round(avg.values * cover_weeks).astype(int),
            "lead_days": lead_days,
            "synthetic": True,
        }
    )
    return st.merge(products[["stock_code", "description", "avg_price"]], on="stock_code", how="left")


def recommend(
    stock: pd.DataFrame,
    final_fc: dict,
    service_level: float = config.SERVICE_LEVEL,
    lead_scale: float = 1.0,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Re-computable (cheap, no model fitting) so the dashboard can run what-if scenarios live."""
    z = norm.ppf(service_level)
    rows = []
    for _, r in stock.iterrows():
        f = final_fc[r["stock_code"]]
        lead_days = r["lead_days"] * lead_scale
        lw = lead_days / 7.0
        P = lw + REVIEW_WEEKS
        wk = f["yhat"].values * demand_multiplier
        sg = f["sigma"].values * demand_multiplier

        def cum(w: float, arr):  # fractional-week cumulative sum
            full = int(np.floor(w))
            frac = w - full
            tot = arr[:full].sum() + (arr[full] * frac if full < len(arr) else 0)
            return tot

        mu_L, mu_P = cum(lw, wk), cum(P, wk)
        sig_week = float(np.sqrt((sg**2).mean()))
        ss = z * sig_week * np.sqrt(P)
        rop = mu_L + z * sig_week * np.sqrt(max(lw, 0.3))
        target = mu_P + ss
        annual = 52.0 * wk.mean()
        unit_cost = r["avg_price"] * config.UNIT_COST_RATIO
        eoq = float(np.sqrt(2 * annual * config.ORDER_COST / max(unit_cost * config.HOLDING_RATE, 1e-6)))
        order_qty = max(0.0, target - r["on_hand"])
        wk_avg = max(wk.mean(), 1e-6)
        cover_days = r["on_hand"] / (wk_avg / 6.0)
        if r["on_hand"] < mu_L:
            status = "Stockout risk"
        elif r["on_hand"] < rop:
            status = "Reorder now"
        elif r["on_hand"] > target + eoq:
            status = "Overstock"
        else:
            status = "Healthy"
        rows.append(
            {
                "stock_code": r["stock_code"],
                "description": r["description"],
                "on_hand": int(r["on_hand"]),
                "lead_days": float(lead_days),
                "forecast_30d": float(cum(30 / 7.0, wk)),
                "safety_stock": float(ss),
                "reorder_point": float(rop),
                "order_up_to": float(target),
                "eoq": eoq,
                "recommended_order": float(np.ceil(order_qty)),
                "cover_days": float(cover_days),
                "status": status,
                "unit_cost": float(unit_cost),
                "order_value": float(np.ceil(order_qty) * unit_cost),
                "excess_units": float(max(0.0, r["on_hand"] - (target + eoq))),
                "excess_value": float(max(0.0, r["on_hand"] - (target + eoq)) * unit_cost),
            }
        )
    return pd.DataFrame(rows)
