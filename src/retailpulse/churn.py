"""F04 – Churn prediction: XGBoost (Optuna-tuned) benchmarked against LR / RandomForest, with SHAP."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retailpulse import config
from retailpulse.features import CHURN_FEATURES

log = logging.getLogger(__name__)

# Snapshot-level features: identical for every customer in a snapshot, so they only shift the BASELINE churn
# rate with the season (they cannot rank customers) and must not be reported as per-customer drivers.
SEASONAL_FEATURES = ["snap_sin", "snap_cos", "days_to_xmas"]


def precision_at_top(y, p, frac: float = 0.20) -> tuple[float, float, float]:
    """Precision, recall and lift in the top `frac` highest-risk customers."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(int(len(y) * frac), 1)
    idx = np.argsort(-p)[:k]
    prec = float(y[idx].mean())
    rec = float(y[idx].sum() / max(y.sum(), 1))
    lift = prec / max(float(y.mean()), 1e-9)
    return prec, rec, lift


def evaluate(y, p, threshold: float) -> dict:
    prec20, rec20, lift20 = precision_at_top(y, p, 0.20)
    yhat = (p >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "precision_top20": prec20,
        "recall_top20": rec20,
        "lift_top20": lift20,
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "base_rate": float(np.mean(y)),
    }


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def _xgb(params: dict, n_estimators: int = 2000) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=config.SEED,
        n_jobs=2,
        early_stopping_rounds=60,
        **params,
    )


def tune_xgb(Xtr, ytr, Xva, yva, n_trials: int = 40) -> tuple[dict, object]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        m = _xgb(params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return float(m.best_score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def benchmark(Xtr, ytr, Xva, yva, Xte, yte, xgb_params) -> tuple[pd.DataFrame, dict]:
    """Model selection table: every model is fit on train, thresholded on valid, scored on test."""
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=2, random_state=config.SEED
        ),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        thr = best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
        met = evaluate(yte, m.predict_proba(Xte)[:, 1], thr)
        rows.append({"model": name, **met})
        fitted[name] = m
    m = _xgb(xgb_params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    thr = best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
    rows.append({"model": "XGBoost (Optuna-tuned)", **evaluate(yte, m.predict_proba(Xte)[:, 1], thr)})
    fitted["XGBoost (Optuna-tuned)"] = m
    return pd.DataFrame(rows), fitted


def shap_analysis(model: xgb.XGBClassifier, X: pd.DataFrame):
    """TreeSHAP values -> (shap_values array, mean |SHAP| table)."""
    import shap

    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
        "mean_abs_shap", ascending=False
    )
    return sv, imp


def top_drivers(sv: np.ndarray, X: pd.DataFrame, k: int = 3) -> list[str]:
    """Readable per-customer explanation: the k customer-level features pushing churn risk up the most."""
    names = np.array(X.columns)
    keep = np.array([n not in SEASONAL_FEATURES for n in names])
    out = []
    for row in sv:
        r = np.where(keep, row, -np.inf)
        idx = np.argsort(-r)[:k]
        out.append(", ".join(f"{names[i]}" for i in idx if r[i] > 0) or "no strong driver")
    return out


def spend_at_risk(df: pd.DataFrame) -> pd.Series:
    """Estimated 180-day spend: last-180-day spend, or the customer's historical run-rate if that is higher."""
    run_rate = df["monetary"] / df["tenure_days"].clip(lower=30) * 180
    return np.maximum(df["monetary_180"].clip(lower=0), run_rate)


def rank_tiers(p: np.ndarray, recency: np.ndarray, lapsed_days: int = 180) -> np.ndarray:
    """Risk tiers for win-back prioritisation.

    Customers with no order for > `lapsed_days` are 'Lapsed' (already lost - different play-book).  Active customers
    are tiered by rank (top 15 % Critical, next 25 % High, next 30 % Medium, rest Low) so a seasonal shift of the
    whole population's baseline probability does not flood the Critical tier."""
    p = np.asarray(p, float)
    active = np.asarray(recency) <= lapsed_days
    pct = pd.Series(np.where(active, p, np.nan)).rank(pct=True, ascending=False).values
    tier = np.select([pct <= 0.15, pct <= 0.40, pct <= 0.70], ["Critical", "High", "Medium"], "Low")
    return np.where(active, tier, "Lapsed")


def curves(y, p) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    order = np.argsort(-np.asarray(p))
    yy = np.asarray(y)[order]
    deciles = []
    for d in range(10):
        seg = yy[int(len(yy) * d / 10) : int(len(yy) * (d + 1) / 10)]
        deciles.append({"decile": d + 1, "churn_rate": float(seg.mean()), "n": int(len(seg))})

    def thin(a):
        a = np.asarray(a)
        idx = np.unique(np.linspace(0, len(a) - 1, min(len(a), 200)).astype(int))
        return a[idx].tolist()

    return {"fpr": thin(fpr), "tpr": thin(tpr), "precision": thin(pr), "recall": thin(rc), "deciles": deciles}


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[CHURN_FEATURES].astype(float)
