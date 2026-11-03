"""RetailPulse REST API – JWT-secured access to forecasts, churn scores and inventory plans.

Run:   uvicorn api.main:app --port 8080
Auth:  POST /token (OAuth2 password flow)  ->  Authorization: Bearer <jwt>
Roles: viewer < analyst < admin   (same model as the dashboard)

Secrets: set RETAILPULSE_JWT_SECRET (>=32 chars) and RETAILPULSE_API_USERS in production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jwt  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Query, status  # noqa: E402
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm  # noqa: E402

from retailpulse import config, monitoring  # noqa: E402

JWT_SECRET = os.getenv("RETAILPULSE_JWT_SECRET") or secrets.token_urlsafe(48)  # ephemeral if unset
JWT_ALG, JWT_TTL = "HS256", 3600
ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def _load_users() -> dict:
    """RETAILPULSE_API_USERS='{"alice": {"password": "...", "role": "admin"}}' else demo users."""
    raw = os.getenv("RETAILPULSE_API_USERS")
    cfg = (
        json.loads(raw)
        if raw
        else {
            "viewer": {"password": "viewer-demo", "role": "viewer"},
            "analyst": {"password": "analyst-demo", "role": "analyst"},
            "admin": {"password": "admin-demo", "role": "admin"},
        }
    )
    users = {}
    for name, u in cfg.items():
        salt = secrets.token_bytes(16)
        users[name] = {"salt": salt, "hash": _hash(u["password"], salt), "role": u["role"]}
    return users


USERS = _load_users()
app = FastAPI(
    title="RetailPulse API",
    version="2.0",
    description="JWT-secured analytics API for demand forecasts, churn risk and inventory plans.",
)
oauth2 = OAuth2PasswordBearer(tokenUrl="token")


def _artifact(name: str) -> pd.DataFrame:
    p = config.DATA_PROC / f"{name}.parquet"
    if not p.exists():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"artefact {name} not built – run the pipeline")
    return pd.read_parquet(p)


def current_user(token: str = Depends(oauth2)) -> dict:
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token", headers={"WWW-Authenticate": "Bearer"})
    return {"user": claims["sub"], "role": claims["role"]}


def require(min_role: str):
    def dep(user: dict = Depends(current_user)) -> dict:
        if ROLE_RANK[user["role"]] < ROLE_RANK[min_role]:
            monitoring.audit(user["user"], user["role"], "api_forbidden", min_role)
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"{min_role} role required")
        return user

    return dep


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "time": int(time.time())}


@app.post("/token", tags=["auth"])
def token(form: OAuth2PasswordRequestForm = Depends()) -> dict:
    u = USERS.get(form.username)
    ok = u is not None and hmac.compare_digest(u["hash"], _hash(form.password, u["salt"]))
    if not ok:
        monitoring.audit(form.username, "anonymous", "api_login_failed", "")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect username or password")
    now = int(time.time())
    tok = jwt.encode({"sub": form.username, "role": u["role"], "iat": now, "exp": now + JWT_TTL}, JWT_SECRET, JWT_ALG)
    monitoring.audit(form.username, u["role"], "api_login", "")
    return {"access_token": tok, "token_type": "bearer", "expires_in": JWT_TTL}


@app.get("/forecast", tags=["forecast"])
def forecast(days: int = Query(30, ge=1, le=30), user: dict = Depends(require("viewer"))) -> dict:
    f = _artifact("forecast_future").head(days)
    return {
        "unit": "units/day",
        "model": "prophet+lstm ensemble",
        "total": float(f["ensemble"].sum()),
        "points": [
            {
                "date": str(r.date.date()),
                "forecast": round(float(r.ensemble), 1),
                "lower": round(float(r.lower), 1),
                "upper": round(float(r.upper), 1),
            }
            for r in f.itertuples()
        ],
    }


@app.get("/inventory/{sku}", tags=["inventory"])
def inventory(sku: str, user: dict = Depends(require("viewer"))) -> dict:
    r = _artifact("inventory_recommendations")
    row = r[r["stock_code"] == sku.upper()]
    if row.empty:
        raise HTTPException(404, "SKU not in the optimised assortment")
    return {**row.iloc[0].to_dict(), "stock_is_simulated": True}


@app.get("/customers/{customer_key}/churn", tags=["churn"])
def churn(customer_key: str, user: dict = Depends(require("analyst"))) -> dict:
    s = _artifact("churn_scores")
    row = s[s["customer_key"] == customer_key]
    if row.empty:
        raise HTTPException(404, "unknown customer key")
    r = row.iloc[0]
    monitoring.audit(user["user"], user["role"], "api_churn_lookup", customer_key)
    return {
        "customer_key": customer_key,
        "churn_probability": round(float(r["churn_prob"]), 4),
        "risk_tier": r["risk_tier"],
        "segment": r["segment"],
        "top_drivers": r["top_drivers"],
        "expected_loss_gbp": round(float(r["expected_loss"]), 2),
    }


@app.get("/monitoring/drift", tags=["mlops"])
def drift(user: dict = Depends(require("admin"))) -> dict:
    p = config.REPORTS / "drift_summary.json"
    if not p.exists():
        raise HTTPException(503, "drift report not built")
    return json.loads(p.read_text())
