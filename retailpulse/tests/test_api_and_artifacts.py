import json

import pytest

from retailpulse import config

ART = config.DATA_PROC / "churn_scores.parquet"
pytestmark_art = pytest.mark.skipif(not ART.exists(), reason="run the pipeline first")


def _client():
    import api.main as m
    from fastapi.testclient import TestClient

    return TestClient(m.app), m


def _tok(c, user, pw):
    r = c.post("/token", data={"username": user, "password": pw})
    return r, {"Authorization": f"Bearer {r.json().get('access_token', '')}"}


def test_health_open_and_auth_required():
    c, _ = _client()
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/forecast").status_code == 401
    assert c.get("/forecast", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert c.post("/token", data={"username": "viewer", "password": "wrong"}).status_code == 401


@pytestmark_art
def test_role_based_access():
    c, _ = _client()
    _, viewer = _tok(c, "viewer", "viewer-demo")
    _, analyst = _tok(c, "analyst", "analyst-demo")
    _, admin = _tok(c, "admin", "admin-demo")
    assert c.get("/forecast?days=7", headers=viewer).status_code == 200
    assert len(c.get("/forecast?days=7", headers=viewer).json()["points"]) == 7
    assert c.get("/customers/C-0000000000/churn", headers=viewer).status_code == 403
    import pandas as pd

    key = pd.read_parquet(ART)["customer_key"].iloc[0]
    r = c.get(f"/customers/{key}/churn", headers=analyst)
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert c.get("/monitoring/drift", headers=analyst).status_code == 403
    assert c.get("/monitoring/drift", headers=admin).status_code == 200


def test_expired_token_rejected():
    import jwt

    c, m = _client()
    tok = jwt.encode({"sub": "admin", "role": "admin", "exp": 1}, m.JWT_SECRET, m.JWT_ALG)
    assert c.get("/forecast", headers={"Authorization": f"Bearer {tok}"}).status_code == 401


@pytestmark_art
def test_artifact_sanity():
    rep = json.loads((config.REPORTS / "etl_report.json").read_text())
    assert rep["rows_clean"] > 900_000 and rep["custom_column"] == "region_group"
    q = json.loads((config.REPORTS / "data_quality.json").read_text())
    assert q["success"]
    ch = json.loads((config.REPORTS / "churn_metrics.json").read_text())["final"]
    assert 0.7 < ch["auc_roc"] <= 1 and ch["precision_top20"] > ch["base_rate"]
    fm = json.loads((config.REPORTS / "forecast_metrics.json").read_text())
    assert fm["cv"]["ensemble"]["mape_weekly"] < fm["cv"]["naive"]["mape_weekly"] + 3
    sm = json.loads((config.REPORTS / "segmentation_metrics.json").read_text())
    assert 6 <= sm["k"] <= 8
