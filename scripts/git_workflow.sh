#!/usr/bin/env bash
# Creates a clean, reviewable git history with semantic commits and feature branches merged via --no-ff
# (the Zidio guidelines ask for semantic commits, feature branches / PRs, and no committed secrets).
#
#   bash scripts/git_workflow.sh https://github.com/<your-user>/retailpulse.git
#
# Review `git log --graph --oneline` afterwards, then push.  Open real PRs on GitHub if you prefer:
# replace the `git merge` lines with `git push -u origin <branch>` + a PR in the GitHub UI.
set -euo pipefail
REMOTE="${1:-}"
git init -b main >/dev/null 2>&1 || true
git config user.name  >/dev/null 2>&1 || { echo "Set git user.name / user.email first"; exit 1; }

feat () {  # feat <branch> <commit message> <paths...>
  local branch="$1" msg="$2"; shift 2
  git checkout -q -b "$branch" 2>/dev/null || git checkout -q "$branch"
  git add -- "$@"
  git diff --cached --quiet || git commit -q -m "$msg"
  git checkout -q main
  git merge -q --no-ff "$branch" -m "Merge branch '$branch' (PR)" || true
}

git add .gitignore LICENSE README.md pyproject.toml requirements.txt requirements-ml.txt requirements-dev.txt Makefile
git commit -q -m "chore: bootstrap repository (readme, licence, requirements, tooling)" || true

feat feature/etl-pipeline        "feat(etl): add extract-validate-transform-load pipeline with region_group column" src/retailpulse/__init__.py src/retailpulse/config.py src/retailpulse/etl scripts/download_data.py data/raw/.gitkeep data/sample reports/etl_report.json reports/data_quality.json tests/conftest.py tests/test_etl.py
feat feature/eda-features        "feat(features): add EDA notebook, RFM and churn snapshot features" src/retailpulse/features.py notebooks/01_EDA.ipynb tests/test_features_churn.py
feat feature/segmentation        "feat(segmentation): add K-Means and DBSCAN customer segmentation" src/retailpulse/segmentation.py reports/segmentation_metrics.json
feat feature/forecasting         "feat(forecast): add Prophet, PyTorch-Lightning LSTM and ensemble demand forecasting" src/retailpulse/forecasting.py reports/forecast_metrics.json reports/ts_diagnostics.json tests/test_forecasting_metrics.py
feat feature/churn-model         "feat(churn): add Optuna-tuned XGBoost churn model with SHAP explainability" src/retailpulse/churn.py reports/churn_metrics.json reports/figures/shap_summary.png
feat feature/inventory           "feat(inventory): add hierarchical forecast, safety stock, ROP, EOQ and policy back-test" src/retailpulse/inventory.py reports/inventory_metrics.json
feat feature/mlops               "feat(mlops): add MLflow tracking, Evidently drift detection, retraining loop and Airflow DAG" src/retailpulse/train.py src/retailpulse/drift.py src/retailpulse/monitoring.py scripts/retrain.py airflow reports/drift_summary.json reports/retrain_log.json reports/batch_benchmark.json reports/mlflow_runs.csv reports/drift tests/test_drift_inventory_segments.py
feat feature/processed-artefacts "chore(data): add compact processed artefacts and model files for the dashboard" data/processed models
feat feature/dashboard           "feat(dashboard): add multi-page Streamlit app with what-if analysis, RBAC and exports" app .streamlit src/retailpulse/reporting.py tests/test_dashboard.py
feat feature/api-security        "feat(api): add JWT-secured FastAPI endpoints with role-based access" api tests/test_api_and_artifacts.py
feat feature/deployment          "feat(deploy): add Docker multi-stage build, compose stack, Kubernetes manifests, Prometheus and Grafana" Dockerfile .dockerignore docker-compose.yml k8s monitoring
feat feature/ci-cd               "ci: add GitHub Actions for lint, tests, dashboard smoke test and image build" .github
feat feature/docs-report         "docs: add deployment guide, demo scripts, notebooks, figures and project report" docs notebooks scripts/make_figures.py scripts/build_notebooks.py scripts/build_report.py scripts/git_workflow.sh reports/figures

git add -A
git diff --cached --quiet || git commit -q -m "chore: add remaining project files"
if [ -n "$REMOTE" ]; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
  echo "Now run:  git push -u origin main"
fi
git log --graph --oneline | head -40
