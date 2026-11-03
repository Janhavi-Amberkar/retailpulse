.PHONY: install install-ml etl train pipeline test lint format app api docker report notebooks retrain clean
PY ?= python
export PYTHONPATH := src

install:            ## dashboard runtime only
	pip install -r requirements.txt
install-ml:         ## full training stack (prophet, torch, xgboost, mlflow, ...)
	pip install -r requirements-ml.txt
data:               ## download the public dataset (95 MB, not stored in git)
	$(PY) scripts/download_data.py
etl:                ## Extract -> Validate -> Transform -> Load
	$(PY) -m retailpulse.etl.pipeline
train:              ## segmentation, forecast, churn, inventory, drift, load test (+ MLflow)
	$(PY) -m retailpulse.train --stage all
pipeline: etl train ## everything from raw data to dashboard artefacts
test:
	pytest
lint:
	ruff check . && black --check --line-length 120 src app api tests scripts airflow
format:
	ruff check --fix . ; black --line-length 120 src app api tests scripts airflow
app:
	streamlit run app/streamlit_app.py
api:
	uvicorn api.main:app --port 8080 --reload
retrain:
	$(PY) scripts/retrain.py
docker:
	docker compose up --build
report:             ## regenerate report figures and the 10-18 page PDF
	$(PY) scripts/make_figures.py && $(PY) scripts/build_report.py
notebooks:
	$(PY) scripts/build_notebooks.py
clean:
	rm -rf mlruns mlflow.db .pytest_cache .ruff_cache logs/*.log
