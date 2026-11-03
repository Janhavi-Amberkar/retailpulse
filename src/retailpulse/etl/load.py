"""LOAD: persist processed tables to Parquet and to a SQL database.

DATABASE_URL may point to PostgreSQL (production / docker-compose); it defaults to a
local SQLite file so the project runs with zero external services.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine

from retailpulse import config

log = logging.getLogger(__name__)


def get_engine():
    url = os.getenv("DATABASE_URL", f"sqlite:///{config.ROOT / 'data' / 'retailpulse.db'}")
    return create_engine(url)


def save_parquet(df: pd.DataFrame, name: str) -> str:
    config.DATA_PROC.mkdir(parents=True, exist_ok=True)
    path = config.DATA_PROC / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("wrote %s (%s rows, %.1f MB)", path.name, len(df), path.stat().st_size / 1e6)
    return str(path)


def save_sql(df: pd.DataFrame, table: str) -> None:
    eng = get_engine()
    out = df.copy()
    for c in out.select_dtypes("category").columns:
        out[c] = out[c].astype(str)
    out.to_sql(table, eng, if_exists="replace", index=False, chunksize=50_000)
    log.info("sql table %s loaded (%s rows)", table, len(out))
