"""Observability: Prometheus metrics + structured audit log (security highlight: audit logging)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime

from retailpulse import config

log = logging.getLogger(__name__)
_started = False
_lock = threading.Lock()

try:  # optional dependency: dashboards degrade gracefully without it
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PAGE_VIEWS = Counter("retailpulse_page_views_total", "Dashboard page views", ["page", "role"])
    EXPORTS = Counter("retailpulse_exports_total", "Report exports", ["kind"])
    MODEL_MAPE = Gauge("retailpulse_forecast_mape", "Forecast MAPE (%) on latest hold-out", ["horizon"])
    CHURN_AUC = Gauge("retailpulse_churn_auc", "Churn model AUC-ROC on test snapshot")
    DRIFT_SHARE = Gauge("retailpulse_drift_share", "Share of features with PSI alert")
    BATCH_SECONDS = Histogram("retailpulse_batch_seconds", "ETL batch duration", buckets=(5, 30, 60, 120, 300, 600))
    HAVE_PROM = True
except Exception:  # pragma: no cover
    HAVE_PROM = False


def start_metrics_server(port: int = 8000) -> bool:
    """Idempotent; exposes /metrics for Prometheus (scraped by monitoring/prometheus.yml)."""
    global _started
    if not HAVE_PROM:
        return False
    with _lock:
        if not _started:
            try:
                start_http_server(port)
                _started = True
            except OSError:
                _started = True  # port taken by another Streamlit session in the same process
    return _started


def audit(user: str, role: str, action: str, detail: str = "") -> None:
    """Append-only JSON-lines audit trail for sensitive operations (logins, exports, role changes)."""
    config.LOGS.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "user": user,
        "role": role,
        "action": action,
        "detail": detail,
    }
    try:
        with open(config.LOGS / "audit.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as exc:  # read-only filesystems (some PaaS) must not break the app
        log.debug("audit log not writable: %s", exc)


def read_audit(n: int = 200) -> list[dict]:
    p = config.LOGS / "audit.log"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().splitlines()[-n:]
    return [json.loads(x) for x in lines][::-1]
