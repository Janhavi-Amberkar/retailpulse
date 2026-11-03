"""MLOps: data / performance drift detection (own PSI+KS engine + Evidently AI HTML reports)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retailpulse import config

log = logging.getLogger(__name__)

PSI_WARN, PSI_ALERT = 0.10, 0.20


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins (robust to heavy tails)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        p = psi(ref[f].values, cur[f].values)
        ks = stats.ks_2samp(ref[f].dropna().values, cur[f].dropna().values)
        rows.append(
            {
                "feature": f,
                "psi": p,
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "ref_mean": float(ref[f].mean()),
                "cur_mean": float(cur[f].mean()),
                "status": "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "OK"),
            }
        )
    t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return t


def drift_summary(tbl: pd.DataFrame) -> dict:
    n = len(tbl)
    alert = int((tbl["status"] == "ALERT").sum())
    warn = int((tbl["status"] == "WARN").sum())
    return {
        "n_features": n,
        "n_alert": alert,
        "n_warn": warn,
        "share_drifted": alert / max(n, 1),
        "max_psi": float(tbl["psi"].max()),
        "dataset_drift": alert / max(n, 1) >= 0.30,
    }


def simulate_drift(df: pd.DataFrame, severity: float, features: list[str], seed: int = config.SEED) -> pd.DataFrame:
    """Inject a realistic market shock: customers buy less often, less money, longer gaps."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    s = float(severity)
    mult = {
        "recency": 1 + 1.2 * s,
        "frequency": 1 - 0.45 * s,
        "monetary": 1 - 0.5 * s,
        "avg_order_value": 1 - 0.35 * s,
        "mean_gap": 1 + 0.9 * s,
        "freq_90": 1 - 0.6 * s,
        "monetary_90": 1 - 0.6 * s,
        "freq_30": 1 - 0.6 * s,
        "freq_60": 1 - 0.6 * s,
    }
    for f, m in mult.items():
        if f in out and f in features:
            out[f] = out[f] * m * rng.normal(1.0, 0.03, len(out))
    return out


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], path: Path) -> bool:
    """Write an Evidently AI data-drift report. Supports the 0.7+ and the legacy API."""
    path.parent.mkdir(parents=True, exist_ok=True)
    r, c = ref[features].copy(), cur[features].copy()
    try:  # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snap = Report([DataDriftPreset()]).run(c, r)
        snap.save_html(str(path))
        return True
    except Exception as e1:
        try:  # legacy API
            from evidently.metric_preset import DataDriftPreset as LegacyPreset
            from evidently.report import Report as LegacyReport

            rep = LegacyReport(metrics=[LegacyPreset()])
            rep.run(reference_data=r, current_data=c)
            rep.save_html(str(path))
            return True
        except Exception as e2:
            log.warning("Evidently report skipped: %s | %s", e1, e2)
            return False


def should_retrain(
    summary: dict,
    recent_mape: float | None = None,
    baseline_mape: float | None = None,
    recent_auc: float | None = None,
    baseline_auc: float | None = None,
) -> dict:
    """Retraining policy used by the Airflow DAG and the dashboard."""
    reasons = []
    if summary["dataset_drift"]:
        reasons.append(f"{summary['n_alert']}/{summary['n_features']} features have PSI >= {PSI_ALERT}")
    if recent_mape is not None and baseline_mape and recent_mape > 1.25 * baseline_mape:
        reasons.append(f"forecast MAPE degraded {recent_mape:.1f}% vs baseline {baseline_mape:.1f}% (>25%)")
    if recent_auc is not None and baseline_auc and recent_auc < baseline_auc - 0.03:
        reasons.append(f"churn AUC fell {baseline_auc:.3f} -> {recent_auc:.3f}")
    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def append_retrain_log(entry: dict) -> None:
    p = config.REPORTS / "retrain_log.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append(entry)
    p.write_text(json.dumps(hist[-50:], indent=2))
