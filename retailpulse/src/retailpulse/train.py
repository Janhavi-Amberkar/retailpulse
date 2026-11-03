"""Training / scoring orchestrator with MLflow tracking.

    python -m retailpulse.train --stage all
    python -m retailpulse.train --stage segment|forecast|churn|inventory|drift|bench

Every stage writes its artefacts to data/processed, models/ and reports/ and logs params,
metrics and artefacts to MLflow (sqlite backend, ./mlflow.db).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from retailpulse import churn, config, drift, features, inventory, segmentation
from retailpulse import forecasting as fc

warnings.filterwarnings("ignore")
log = logging.getLogger("retailpulse.train")

MLFLOW_URI = f"sqlite:///{config.ROOT / 'mlflow.db'}"


# ------------------------------------------------------------------ mlflow ---
@contextmanager
def mlrun(name: str, tags: dict | None = None):
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("RetailPulse")
        with mlflow.start_run(run_name=name, tags=tags or {}):
            yield mlflow
    except Exception as exc:  # MLflow must never break training
        log.warning("MLflow unavailable (%s) - continuing without tracking", exc)

        class _Noop:
            def __getattr__(self, _):
                return lambda *a, **k: None

        yield _Noop()


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
            out[f"{prefix}{k}"] = float(v)
    return out


def _save_json(obj, name: str) -> None:
    (config.REPORTS / name).write_text(json.dumps(obj, indent=2, default=float))


# -------------------------------------------------------------------- data ---
def load_inputs():
    sales = pd.read_parquet(config.DATA_PROC / "clean_transactions.parquet")
    returns = pd.read_parquet(config.DATA_PROC / "returns.parquet")
    daily = pd.read_parquet(config.DATA_PROC / "daily_sales.parquet")
    inv = features.invoice_table(sales)
    return sales, returns, daily, inv


# ----------------------------------------------------------------- segment ---
def stage_segment(inv, returns, last_date):
    t0 = time.time()
    rfm = features.rfm_features(inv, returns, last_date + pd.Timedelta(days=1))
    res = segmentation.run_segmentation(rfm)
    cust = res["customers"]
    keep = (
        ["customer_key", "segment", "segment_desc", "cluster", "dbscan", "pc1", "pc2", "pc3"]
        + segmentation.SEG_FEATURES
        + ["avg_basket_units", "max_order_value", "freq_90", "monetary_90"]
    )
    cust[keep].to_parquet(config.DATA_PROC / "customers.parquet", index=False)
    res["profile"].to_parquet(config.DATA_PROC / "segment_profile.parquet", index=False)
    res["k_eval"].to_parquet(config.DATA_PROC / "segment_k_eval.parquet", index=False)
    m = res["metrics"] | {"seconds": time.time() - t0}
    _save_json(m, "segmentation_metrics.json")
    with mlrun("segmentation-kmeans-dbscan", {"feature": "F02"}) as ml:
        ml.log_params({"algo": "KMeans", "k": m["k"], "dbscan_eps": round(m["dbscan_eps"], 3)})
        ml.log_metrics(_flat(m))
        ml.log_artifact(str(config.REPORTS / "segmentation_metrics.json"))
    log.info("segmentation: k=%s silhouette=%.3f", m["k"], m["silhouette"])
    return m


# ---------------------------------------------------------------- forecast ---
def stage_forecast(daily: pd.DataFrame, lstm_epochs: int = 120):
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss

    t0 = time.time()
    s = fc.series_from_daily(daily)
    last = s.index.max()

    # --- Day 4: stationarity tests + decomposition ------------------------------
    adf = adfuller(s.values, autolag="AIC")
    adf_d = adfuller(np.diff(s.values), autolag="AIC")
    kp = kpss(s.values, regression="c", nlags="auto")
    stl = STL(np.log1p(s.values), period=6, robust=True).fit()
    pd.DataFrame(
        {
            "date": s.index,
            "observed": s.values,
            "trend": np.expm1(stl.trend),
            "seasonal": stl.seasonal,
            "resid": stl.resid,
        }
    ).to_parquet(config.DATA_PROC / "ts_decomposition.parquet", index=False)
    diag = {
        "adf_stat": float(adf[0]),
        "adf_p": float(adf[1]),
        "adf_diff_p": float(adf_d[1]),
        "kpss_stat": float(kp[0]),
        "kpss_p": float(kp[1]),
        "stationary_level": bool(adf[1] < 0.05),
        "stationary_diff": bool(adf_d[1] < 0.05),
        "strength_trend": float(max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))),
        "strength_seasonal": float(max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))),
    }
    _save_json(diag, "ts_diagnostics.json")

    # --- rolling-origin back-test ------------------------------------------------
    tune_origins = [
        pd.Timestamp(x)
        for x in ["2011-04-10", "2011-05-10", "2011-06-10", "2011-07-10", "2011-08-10", "2011-09-09", "2011-10-09"]
    ]
    test_origin = (last - pd.Timedelta(days=fc.H - 1)).normalize()
    folds = []
    for i, o in enumerate(tune_origins + [test_origin]):
        t1 = time.time()
        f = fc.run_fold(s, o, lstm_epochs)
        f["fold"] = i + 1
        f["is_test"] = o == test_origin
        folds.append(f)
        log.info("fold %s origin %s done in %.0fs", i + 1, o.date(), time.time() - t1)
    bt = pd.concat(folds, ignore_index=True).dropna(subset=["actual"])
    # LSTM is trained on log1p -> its output is a conditional median and under-shoots totals.
    # Calibrate with one scalar estimated on the TUNING folds only (hold-out fold stays untouched).
    tune = bt[~bt["is_test"]]
    k_lstm = float(tune["actual"].sum() / tune["lstm"].sum())
    bt["lstm_raw"] = bt["lstm"]
    bt["lstm"] = bt["lstm_raw"] * k_lstm
    tune = bt[~bt["is_test"]]
    w, wgrid = fc.best_weight(tune)
    bt["ensemble"] = (1 - w) * bt["prophet"] + w * bt["lstm"]
    tune = bt[~bt["is_test"]]
    test = bt[bt["is_test"]]

    def fold_table(df):
        rows = []
        for model in ["naive", "prophet", "lstm_raw", "lstm", "ensemble"]:
            per = [fc.horizon_metrics(g, model) for _, g in df.groupby("fold")]
            agg = pd.DataFrame(per).mean().to_dict()
            rows.append({"model": model, **{k: float(v) for k, v in agg.items()}})
        return pd.DataFrame(rows)

    cv_tbl, test_tbl = fold_table(tune), fold_table(test)
    per_fold = []
    for fold, g in bt.groupby("fold"):
        for model in ["naive", "prophet", "lstm_raw", "lstm", "ensemble"]:
            per_fold.append(
                {
                    "fold": int(fold),
                    "origin": str(g["origin"].iloc[0].date()),
                    "model": model,
                    "is_test": bool(g["is_test"].iloc[0]),
                    **fc.horizon_metrics(g, model),
                }
            )
    per_fold = pd.DataFrame(per_fold)

    # --- final model: refit on ALL data, forecast the next 30 days ------------------
    origin = last + pd.Timedelta(days=1)
    fdates = pd.date_range(origin, periods=fc.H, freq="D")
    fdates = fdates[fc.is_trading_day(fdates)]
    m = fc.fit_prophet(pd.DataFrame({"date": s.index, "y": s.values}))
    fp = fc.predict_prophet(m, fdates)
    lf = fc.LSTMForecaster(epochs=lstm_epochs).fit(s)
    fl = lf.predict(s, origin)
    fl["lstm"] = fl["lstm"] * k_lstm
    fut = fp.merge(fl, on="date", how="left")
    fut["ensemble"] = (1 - w) * fut["prophet"] + w * fut["lstm"]
    ratio = (fut["ensemble"] / fut["prophet"].replace(0, np.nan)).fillna(1.0)
    fut["lower"], fut["upper"] = fut["prophet_lo"] * ratio, fut["prophet_hi"] * ratio
    fut.to_parquet(config.DATA_PROC / "forecast_future.parquet", index=False)
    bt.to_parquet(config.DATA_PROC / "forecast_backtest.parquet", index=False)
    per_fold.to_parquet(config.DATA_PROC / "forecast_per_fold.parquet", index=False)
    wgrid.to_parquet(config.DATA_PROC / "ensemble_weight_grid.parquet", index=False)

    try:
        import torch
        from prophet.serialize import model_to_json

        torch.save(lf.model.state_dict(), config.MODELS / "lstm_demand.pt")
        (config.MODELS / "prophet_demand.json").write_text(model_to_json(m))
        (config.MODELS / "lstm_scaler.json").write_text(
            json.dumps({"mu": lf.mu, "sd": lf.sd, "w_lstm": w, "k_lstm": k_lstm})
        )
    except Exception as exc:
        log.warning("model save skipped: %s", exc)

    def to_dict(tbl):
        return {r["model"]: {k: float(v) for k, v in r.items() if k != "model"} for _, r in tbl.iterrows()}

    out = {
        "w_lstm": w,
        "k_lstm": k_lstm,
        "cv_origins": [str(o.date()) for o in tune_origins],
        "test_origin": str(test_origin.date()),
        "cv": to_dict(cv_tbl),
        "test": to_dict(test_tbl),
        "diagnostics": diag,
        "seconds": time.time() - t0,
        "target_mape": config.TARGETS["mape_max"],
    }
    _save_json(out, "forecast_metrics.json")

    with mlrun("forecast-prophet-lstm-ensemble", {"feature": "F03"}) as ml:
        ml.log_params(
            {
                "horizon_days": fc.H,
                "w_lstm": round(w, 2),
                "lstm_epochs": lstm_epochs,
                "prophet_mode": "multiplicative",
                "n_folds": len(tune_origins),
            }
        )
        for model in ["naive", "prophet", "lstm", "ensemble"]:
            ml.log_metric("k_lstm", k_lstm)
            for k, v in out["cv"][model].items():
                ml.log_metric(f"cv_{model}_{k}", v)
            for k, v in out["test"][model].items():
                ml.log_metric(f"test_{model}_{k}", v)
        ml.log_artifact(str(config.REPORTS / "forecast_metrics.json"))
    log.info("forecast done: ensemble test %s", out["test"]["ensemble"])
    return out


# ------------------------------------------------------------------- churn ---
def stage_churn(inv, returns, last_date, n_trials: int = 40):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import xgboost as xgb

    t0 = time.time()
    tr, va, te, test_snap = features.build_churn_frames(inv, returns, last_date)
    Xtr, ytr = churn.feature_matrix(tr), tr["churn"].values
    Xva, yva = churn.feature_matrix(va), va["churn"].values
    Xte, yte = churn.feature_matrix(te), te["churn"].values

    best, study = churn.tune_xgb(Xtr, ytr, Xva, yva, n_trials=n_trials)
    bench, fitted = churn.benchmark(Xtr, ytr, Xva, yva, Xte, yte, best)

    # best iteration on validation -> refit on train+val for the reported test score
    m0 = churn._xgb(best)
    m0.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    n_est = int(m0.best_iteration + 1)
    thr = churn.best_f1_threshold(yva, m0.predict_proba(Xva)[:, 1])

    def refit(X, y):
        m = xgb.XGBClassifier(
            n_estimators=n_est,
            objective="binary:logistic",
            tree_method="hist",
            random_state=config.SEED,
            n_jobs=2,
            **best,
        )
        return m.fit(X, y)

    Xtv, ytv = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
    m_eval = refit(Xtv, ytv)
    p_te = m_eval.predict_proba(Xte)[:, 1]
    final_metrics = churn.evaluate(yte, p_te, thr)
    curves = churn.curves(yte, p_te)
    bench = pd.concat(
        [bench, pd.DataFrame([{"model": "XGBoost (final: train+valid refit)", **final_metrics}])], ignore_index=True
    )
    bench.to_parquet(config.DATA_PROC / "churn_benchmark.parquet", index=False)
    pd.DataFrame({"customer_key": te["customer_key"], "y": yte, "p": p_te}).to_parquet(
        config.DATA_PROC / "churn_test_pred.parquet", index=False
    )

    # SHAP (evaluation model, test snapshot) -----------------------------------------
    sv_te, imp = churn.shap_analysis(m_eval, Xte)
    imp["seasonal_baseline"] = imp["feature"].isin(churn.SEASONAL_FEATURES)
    imp.to_parquet(config.DATA_PROC / "churn_shap_importance.parquet", index=False)
    cust_cols = [c for c in Xte.columns if c not in churn.SEASONAL_FEATURES]
    ci = [list(Xte.columns).index(c) for c in cust_cols]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv_te[:, ci], Xte[cust_cols], show=False, max_display=15, plot_size=None)
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    # production model: all labelled snapshots, scores the live customer base ------------
    Xall = pd.concat([Xtv, Xte])
    yall = np.concatenate([ytv, yte])
    m_prod = refit(Xall, yall)
    m_prod.save_model(config.MODELS / "churn_xgb.json")
    cur = features.rfm_features(inv, returns, last_date + pd.Timedelta(days=1))
    Xcur = churn.feature_matrix(cur)
    p_cur = m_prod.predict_proba(Xcur)[:, 1]
    sv_cur, _ = churn.shap_analysis(m_prod, Xcur)
    cur["churn_prob"] = p_cur
    cur["risk_tier"] = churn.rank_tiers(p_cur, cur["recency"].values)
    cur["top_drivers"] = churn.top_drivers(sv_cur, Xcur)
    seg = pd.read_parquet(config.DATA_PROC / "customers.parquet")[["customer_key", "segment"]]
    cur = cur.merge(seg, on="customer_key", how="left")
    cur["spend_at_risk"] = churn.spend_at_risk(cur)
    cur["expected_loss"] = cur["churn_prob"] * cur["spend_at_risk"]
    cur.to_parquet(config.DATA_PROC / "churn_scores.parquet", index=False)
    shp = pd.DataFrame(sv_cur.astype("float32"), columns=[f"shap_{c}" for c in Xcur.columns])
    shp.insert(0, "customer_key", cur["customer_key"].values)
    shp.to_parquet(config.DATA_PROC / "churn_shap_current.parquet", index=False)

    metrics = {
        "best_params": best,
        "n_estimators": n_est,
        "optuna_trials": n_trials,
        "optuna_best_valid_auc": float(study.best_value),
        "test_snapshot": str(test_snap.date()),
        "label_window_days": config.CHURN_WINDOW_DAYS,
        "n_train": int(len(Xtr)),
        "n_valid": int(len(Xva)),
        "n_test": int(len(Xte)),
        "n_features": int(Xtr.shape[1]),
        "final": final_metrics,
        "curves": curves,
        "benchmark": bench.to_dict("records"),
        "targets": {"auc_min": config.TARGETS["auc_min"], "precision_top20_min": config.TARGETS["precision_top20_min"]},
        "top_features": imp[~imp["seasonal_baseline"]].head(10).to_dict("records"),
        "seasonal_baseline_shap": float(imp.loc[imp["seasonal_baseline"], "mean_abs_shap"].sum()),
        "snapshot_base_rates": [
            {"snapshot": str(k.date()), "churn_rate": float(v), "split": sp}
            for sp, fr in (("train", tr), ("valid", va), ("test", te))
            for k, v in fr.groupby("snapshot")["churn"].mean().items()
        ],
        "current_mean_pred": float(p_cur.mean()),
        "current_snapshot": str((last_date + pd.Timedelta(days=1)).date()),
        "seconds": time.time() - t0,
    }
    _save_json(metrics, "churn_metrics.json")
    with mlrun("churn-xgboost-optuna-shap", {"feature": "F04"}) as ml:
        ml.log_params({k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()})
        ml.log_params({"n_estimators": n_est, "optuna_trials": n_trials, "test_snapshot": str(test_snap.date())})
        ml.log_metrics(_flat(final_metrics, "test_"))
        ml.log_metric("valid_auc_optuna_best", float(study.best_value))
        ml.log_artifact(str(config.REPORTS / "churn_metrics.json"))
        ml.log_artifact(str(config.FIGURES / "shap_summary.png"))
        try:
            ml.xgboost.log_model(m_prod, name="churn_model")
        except Exception:
            pass
    log.info("churn done: %s", {k: round(v, 3) for k, v in final_metrics.items()})
    return metrics


# --------------------------------------------------------------- inventory ---
def stage_inventory(daily, n_jobs: int = 2):
    t0 = time.time()
    sku_daily = pd.read_parquet(config.DATA_PROC / "sku_daily.parquet")
    products = pd.read_parquet(config.DATA_PROC / "products.parquet")
    top = products.sort_values("total_revenue", ascending=False).head(config.N_SKU_FORECAST)["stock_code"].tolist()
    weekly = inventory.weekly_matrix(sku_daily, top)
    top = [c for c in top if c in weekly.columns and weekly[c].tail(26).sum() > 0]
    weekly = weekly[top]
    total = inventory.total_weekly(daily).reindex(weekly.index)
    n = len(weekly)
    stock = inventory.synthetic_stock(weekly, products)
    lead_weeks = {r["stock_code"]: int(np.ceil(r["lead_days"] / 7)) for _, r in stock.iterrows()}

    horizon = 5
    origins = list(range(n - 12 - 4, n - 4 + 1))  # 13 weekly origins, each leaves >= 4 wks of actuals
    fc_hier = inventory.hier_forecast_all(weekly, total, origins, horizon)
    fc_direct = inventory.forecast_all(weekly, origins, horizon, n_jobs)
    bt = inventory.backtest_policies(weekly, lead_weeks, fc_hier, fc_direct, origins)
    prices = products.set_index("stock_code")["avg_price"]
    pol = inventory.summarise_policies(bt, prices)

    # per-SKU 4-week forecast accuracy pooled over origins: moving average vs per-SKU Prophet vs hierarchical
    rows = []
    for code in top:
        err = {"ma": 0.0, "direct": 0.0, "hier": 0.0}
        bias = {"ma": 0.0, "direct": 0.0, "hier": 0.0}
        act = 0.0
        for o in origins:
            real = weekly[code].iloc[o : o + 4].values
            k = len(real)
            if k < 4:
                continue
            preds = {
                "ma": np.repeat(weekly[code].iloc[o - 4 : o].mean(), k),
                "direct": fc_direct[(code, o)]["yhat"].values[:k],
                "hier": fc_hier[(code, o)]["yhat"].values[:k],
            }
            act += real.sum()
            for m_, p_ in preds.items():
                err[m_] += np.abs(real - p_).sum()
                bias[m_] += (p_ - real).sum()
        rows.append(
            {
                "stock_code": code,
                **{f"wape_{k}": 100 * v / max(act, 1) for k, v in err.items()},
                **{f"bias_{k}": 100 * v / max(act, 1) for k, v in bias.items()},
            }
        )
    acc = pd.DataFrame(rows)

    # final forecasts on all data ------------------------------------------------------
    final_fc = inventory.hier_forecast(weekly, total, n, horizon)
    final = {c: f for (c, _), f in final_fc.items()}
    rec = inventory.recommend(stock, final)
    rec.to_parquet(config.DATA_PROC / "inventory_recommendations.parquet", index=False)
    stock.to_parquet(config.DATA_PROC / "inventory_stock_sim.parquet", index=False)
    pd.concat([f.assign(stock_code=c) for c, f in final.items()], ignore_index=True).to_parquet(
        config.DATA_PROC / "inventory_final_forecast.parquet", index=False
    )
    bt.to_parquet(config.DATA_PROC / "inventory_backtest_cycles.parquet", index=False)
    pol.to_parquet(config.DATA_PROC / "inventory_policy_summary.parquet", index=False)
    acc.to_parquet(config.DATA_PROC / "sku_forecast_accuracy.parquet", index=False)
    weekly.assign(total_units=total).rename_axis("week_end").reset_index().to_parquet(
        config.DATA_PROC / "sku_weekly.parquet", index=False
    )

    out = {
        "n_skus": len(top),
        "n_cycles": int(pol["cycles"].iloc[0]),
        "origins": len(origins),
        "policy_summary": pol.to_dict("records"),
        "sku_wape_median": {k: float(acc[f"wape_{k}"].median()) for k in ("ma", "direct", "hier")},
        "service_level": config.SERVICE_LEVEL,
        "seconds": time.time() - t0,
        "status_counts": rec["status"].value_counts().to_dict(),
        "assumptions": {
            "stock": "SIMULATED (no stock table in public dataset)",
            "unit_cost_ratio": config.UNIT_COST_RATIO,
            "holding_rate": config.HOLDING_RATE,
            "order_cost": config.ORDER_COST,
        },
    }
    _save_json(out, "inventory_metrics.json")
    with mlrun("inventory-optimisation", {"feature": "F05"}) as ml:
        ml.log_params({"service_level": config.SERVICE_LEVEL, "n_skus": len(top), "method": "hierarchical top-down"})
        for k, v in out["sku_wape_median"].items():
            ml.log_metric(f"sku_wape_median_{k}", v)
        for r in out["policy_summary"]:
            tag = r["policy"].split("(")[0].strip().replace(" ", "_").replace("+", "plus")
            ml.log_metric(f"{tag}_fill_rate", r["fill_rate_pct"])
            ml.log_metric(f"{tag}_total_cost", r["total_cost"])
    log.info("inventory done: %s", out["sku_wape_median"])
    return out


# ------------------------------------------------------------------- drift ---
def stage_drift(inv, returns, last_date):
    t0 = time.time()
    tr, va, te, test_snap = features.build_churn_frames(inv, returns, last_date)
    feats = features.CHURN_FEATURES
    seasonal = ["snap_sin", "snap_cos", "days_to_xmas"]  # constants within a snapshot -> not drift-tested
    tested = [f for f in feats if f not in seasonal]
    ref = tr[tr["snapshot"] >= tr["snapshot"].max() - pd.Timedelta(days=60)]  # latest training period
    cur = features.rfm_features(inv, returns, last_date + pd.Timedelta(days=1))
    tbl = drift.drift_table(ref, cur, tested)
    summ = drift.drift_summary(tbl)
    tbl.to_parquet(config.DATA_PROC / "drift_current.parquet", index=False)
    ref[tested].sample(min(len(ref), 3000), random_state=config.SEED).to_parquet(
        config.DATA_PROC / "drift_reference.parquet", index=False
    )

    sims = []
    for sev in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        shifted = drift.simulate_drift(cur, sev, tested)
        t = drift.drift_table(ref, shifted, tested)
        s = drift.drift_summary(t)
        sims.append({"severity": sev, **s})
    pd.DataFrame(sims).to_parquet(config.DATA_PROC / "drift_simulation.parquet", index=False)
    sim_tbl = drift.drift_table(ref, drift.simulate_drift(cur, 0.6, tested), tested)
    sim_tbl.to_parquet(config.DATA_PROC / "drift_sim_table_0.6.parquet", index=False)

    ok1 = drift.evidently_html(ref, cur, tested, config.DRIFT_DIR / "evidently_current.html")
    ok2 = drift.evidently_html(
        ref, drift.simulate_drift(cur, 0.6, tested), tested, config.DRIFT_DIR / "evidently_simulated_shift.html"
    )

    fm = json.loads((config.REPORTS / "forecast_metrics.json").read_text())
    cm = json.loads((config.REPORTS / "churn_metrics.json").read_text())
    decision = drift.should_retrain(
        summ,
        fm["test"]["ensemble"]["mape_weekly"],
        fm["cv"]["ensemble"]["mape_weekly"],
        cm["final"]["auc_roc"],
        cm["benchmark"][2]["auc_roc"],
    )
    out = {
        "current": summ | {"top": tbl.head(5).to_dict("records")},
        "decision": decision,
        "evidently_reports": bool(ok1 and ok2),
        "psi_warn": drift.PSI_WARN,
        "psi_alert": drift.PSI_ALERT,
        "seconds": time.time() - t0,
    }
    _save_json(out, "drift_summary.json")
    drift.append_retrain_log(
        {
            "checked_at": decision["checked_at"],
            "retrain": decision["retrain"],
            "reasons": decision["reasons"],
            "share_drifted": summ["share_drifted"],
            "trigger": "scheduled-check",
        }
    )
    with mlrun("drift-check", {"tool": "evidently+psi"}) as ml:
        ml.log_metrics(
            {
                "share_drifted": summ["share_drifted"],
                "max_psi": summ["max_psi"],
                "retrain_flag": float(decision["retrain"]),
            }
        )
    log.info("drift: %s | decision %s", summ, decision)
    return out


# ------------------------------------------------------------- scalability ---
def stage_bench(sales: pd.DataFrame | None = None):
    """Day 27 load test: time the FULL ETL transform + aggregation at 1x / 2x / 3x the raw volume.

    Replicas get unique invoice numbers so de-duplication does not collapse them."""
    from retailpulse.etl import extract, pipeline, transform

    raw = extract.extract()
    rows = []
    for mult in (1, 2, 3):
        big = (
            raw
            if mult == 1
            else pd.concat([raw.assign(Invoice=raw["Invoice"] + str(i)) for i in range(mult)], ignore_index=True)
        )
        t = time.time()
        res = transform.transform(big)
        pipeline.build_aggregates(res.sales, res.returns)
        rows.append({"rows": int(len(big)), "seconds": round(time.time() - t, 2)})
        del big, res
    rps = rows[-1]["rows"] / max(rows[-1]["seconds"], 1e-9)
    out = {
        "runs": rows,
        "rows_per_second": rps,
        "extrapolated_10M_seconds": 1e7 / rps,
        "note": "full ETL transform + aggregation measured at 1-3x raw volume; 10M value is a linear extrapolation",
    }
    _save_json(out, "batch_benchmark.json")
    log.info("bench: %s", out)
    return out


# ------------------------------------------------------------ mlflow export ---
def export_mlflow_runs() -> None:
    """Snapshot MLflow runs to reports/mlflow_runs.csv so the deployed dashboard can show them."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        runs = mlflow.search_runs(experiment_names=["RetailPulse"], order_by=["start_time DESC"])
        keep = [
            c
            for c in runs.columns
            if c in ("run_id", "start_time", "status", "tags.mlflow.runName")
            or c.startswith("metrics.")
            or c.startswith("params.")
        ]
        runs[keep].to_csv(config.REPORTS / "mlflow_runs.csv", index=False)
        log.info("exported %s MLflow runs", len(runs))
    except Exception as exc:
        log.warning("MLflow export skipped: %s", exc)


# --------------------------------------------------------------------- main --
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", default="all", choices=["all", "segment", "forecast", "churn", "inventory", "drift", "bench"]
    )
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()
    config.ensure_dirs()
    sales, returns, daily, inv = load_inputs()
    last = sales["date"].max()
    st = a.stage
    if st in ("all", "segment"):
        stage_segment(inv, returns, last)
    if st in ("all", "forecast"):
        stage_forecast(daily, a.epochs)
    if st in ("all", "churn"):
        stage_churn(inv, returns, last, a.trials)
    if st in ("all", "inventory"):
        stage_inventory(daily)
    if st in ("all", "drift"):
        stage_drift(inv, returns, last)
    if st in ("all", "bench"):
        stage_bench()
    export_mlflow_runs()


if __name__ == "__main__":
    main()
