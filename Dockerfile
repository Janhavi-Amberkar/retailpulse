# syntax=docker/dockerfile:1.7
# Multi-stage build: wheels are compiled once, the runtime image stays small and runs as non-root.
ARG PYTHON_VERSION=3.11

# ---------- stage 1: build wheels ----------
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ml.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- target: app (Streamlit dashboard, ~lightweight) ----------
FROM python:${PYTHON_VERSION}-slim AS app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    STREAMLIT_SERVER_PORT=8501 STREAMLIT_SERVER_ADDRESS=0.0.0.0 METRICS_PORT=8000
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY src ./src
COPY app ./app
COPY api ./api
COPY .streamlit ./.streamlit
COPY data/processed ./data/processed
COPY data/sample ./data/sample
COPY reports ./reports
COPY models ./models
RUN useradd --create-home --uid 10001 retail && mkdir -p logs && chown -R retail /app
USER retail
EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true"]

# ---------- target: ml (training / retraining / API) ----------
FROM python:${PYTHON_VERSION}-slim AS ml
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ml.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
COPY . .
RUN useradd --create-home --uid 10001 retail && chown -R retail /app
USER retail
CMD ["python", "scripts/retrain.py"]
