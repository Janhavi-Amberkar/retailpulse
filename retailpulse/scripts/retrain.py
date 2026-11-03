"""Automated retraining loop with a champion/challenger gate.

    python scripts/retrain.py            # ETL -> drift check -> retrain only if policy says so
    python scripts/retrain.py --force    # retrain regardless of drift
    python scripts/retrain.py --skip-etl

Steps (mirrors the Airflow DAG in airflow/dags/retailpulse_retrain.py):
  1. ETL + data-quality gate         2. drift check (PSI / Evidently)     3. policy decision
  4. snapshot champion artefacts     5. retrain forecast + churn          6. promote challenger only if it is
     not worse than the champion beyond a tolerance, otherwise roll back.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retailpulse import config, drift, monitoring, train  # noqa: E402
from retailpulse.etl import pipeline  # noqa: E402

log = logging.getLogger("retrain")
AUC_TOL, MAPE_TOL = 0.01, 1.0  # challenger may be at most this much worse (AUC points / MAPE pp)
TRACKED = ["churn_metrics.json", "forecast_metrics.json"]
TRACKED_DIRS = [config.MODELS]


def _read(name: str) -> dict:
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def challenger_ok(old_auc: float, new_auc: float, old_mape: float, new_mape: float) -> bool:
    """Champion/challenger gate: promote unless AUC dropped > AUC_TOL or weekly MAPE rose > MAPE_TOL."""
    return (new_auc >= old_auc - AUC_TOL) and (new_mape <= old_mape + MAPE_TOL)


def snapshot(tag: str) -> Path:
    dest = config.ROOT / "models" / "_champion" / tag
    dest.mkdir(parents=True, exist_ok=True)
    for f in TRACKED:
        if (config.REPORTS / f).exists():
            shutil.copy2(config.REPORTS / f, dest / f)
    for f in config.MODELS.glob("*"):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    for name in ("churn_scores", "forecast_future"):
        p = config.DATA_PROC / f"{name}.parquet"
        if p.exists():
            shutil.copy2(p, dest / p.name)
    return dest


def restore(src: Path) -> None:
    for f in src.glob("*"):
        if f.suffix == ".json" and f.name in TRACKED:
            shutil.copy2(f, config.REPORTS / f.name)
        elif f.suffix == ".parquet":
            shutil.copy2(f, config.DATA_PROC / f.name)
        else:
            shutil.copy2(f, config.MODELS / f.name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-etl", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()

    if not a.skip_etl:
        rep = pipeline.run()
        if monitoring.HAVE_PROM:
            monitoring.BATCH_SECONDS.observe(rep["elapsed_seconds"])

    sales, returns, daily, inv = train.load_inputs()
    last = sales["date"].max()
    out = train.stage_drift(inv, returns, last)
    decision = out["decision"]
    log.info("drift decision: %s", decision)
    if not (decision["retrain"] or a.force):
        log.info("No retraining needed.")
        return 0

    tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    champ = snapshot(tag)
    old_auc = _read("churn_metrics.json").get("final", {}).get("auc_roc", 0.0)
    old_mape = _read("forecast_metrics.json").get("cv", {}).get("ensemble", {}).get("mape_weekly", 999.0)

    train.stage_segment(inv, returns, last)
    train.stage_forecast(daily)
    train.stage_churn(inv, returns, last)
    new_auc = _read("churn_metrics.json")["final"]["auc_roc"]
    new_mape = _read("forecast_metrics.json")["cv"]["ensemble"]["mape_weekly"]

    ok = challenger_ok(old_auc, new_auc, old_mape, new_mape)
    entry = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrain": True,
        "reasons": decision["reasons"] or ["forced"],
        "trigger": "scripts/retrain.py",
        "champion_auc": old_auc,
        "challenger_auc": new_auc,
        "champion_mape_weekly": old_mape,
        "challenger_mape_weekly": new_mape,
        "promoted": ok,
    }
    if ok:
        log.info(
            "Challenger promoted (AUC %.3f -> %.3f, weekly MAPE %.1f -> %.1f)", old_auc, new_auc, old_mape, new_mape
        )
    else:
        log.warning("Challenger WORSE than champion -> rolling back to %s", champ)
        restore(champ)
    drift.append_retrain_log(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
