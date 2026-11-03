"""Airflow DAG – daily ETL, drift check and (conditional) retraining for RetailPulse.

Deploy by mounting ./airflow/dags into the Airflow scheduler (see docker-compose.yml, profile "airflow").
Each task shells out to the same code the CLI uses, so the DAG and `make` targets never diverge.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

PROJECT = "/opt/retailpulse"
ENV = f"cd {PROJECT} && export PYTHONPATH={PROJECT}/src && "


def _decide(**_) -> str:
    """Branch on the drift report written by the drift_check task."""
    import json
    from pathlib import Path

    rep = json.loads((Path(PROJECT) / "reports" / "drift_summary.json").read_text())
    return "retrain_models" if rep["decision"]["retrain"] else "skip_retrain"


default_args = {
    "owner": "retailpulse",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="retailpulse_retrain",
    description="ETL -> quality gate -> drift check -> conditional retrain -> champion/challenger promotion",
    schedule="0 2 * * *",  # 02:00 daily batch (spec: daily batch < 5 min)
    start_date=datetime(2026, 3, 1),
    catchup=False,
    default_args=default_args,
    tags=["retailpulse", "mlops"],
) as dag:
    etl = BashOperator(task_id="etl_and_quality_gate", bash_command=ENV + "python -m retailpulse.etl.pipeline")
    drift_check = BashOperator(task_id="drift_check", bash_command=ENV + "python -m retailpulse.train --stage drift")
    branch = BranchPythonOperator(task_id="retrain_needed", python_callable=_decide)
    retrain = BashOperator(
        task_id="retrain_models", bash_command=ENV + "python scripts/retrain.py --skip-etl --force"
    )  # champion/challenger inside
    skip = EmptyOperator(task_id="skip_retrain")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    etl >> drift_check >> branch >> [retrain, skip] >> done
