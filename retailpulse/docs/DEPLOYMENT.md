# Deployment guide – getting a public HTTPS demo URL

The Zidio brief needs **one publicly reachable HTTPS link** (no sign-up, loads in < 8 s). The dashboard only needs the
light `requirements.txt` (no Prophet / PyTorch) because every model output is pre-computed into `data/processed/`.

## Option A – Streamlit Community Cloud (free, recommended, ~3 minutes)

1. Push the repo to GitHub as a **public** repository (see `scripts/git_workflow.sh`).
2. Go to <https://share.streamlit.io> → **Create app** → *Deploy a public app from GitHub*.
3. Fill in
   * Repository: `<your-user>/retailpulse`
   * Branch: `main`
   * **Main file path: `app/streamlit_app.py`**
   * *Advanced settings* → Python version **3.11** (the app was also verified on 3.13 / pandas 2.2 and 3.0)
4. (Optional) *Advanced settings → Secrets* – override the demo passcodes:
   ```toml
   ANALYST_PIN = "choose-something"
   ADMIN_PIN   = "choose-something-else"
   ```
   Without secrets the documented demo passcodes work (`analyst` / `admin`) so judges can try every role.
5. Click **Deploy**. Copy the `https://<name>.streamlit.app` URL into the Zidio **Live Deployment** slot.
6. Open the URL in an *incognito* window and click through all 8 pages once (Zidio checks that no login is needed).

> The Streamlit sidebar shows page names; the app has a built-in *Access* box (Viewer / Analyst / Admin) – judges do not need it for core features.

## Option B – Hugging Face Spaces

Create a Space with SDK **Streamlit**, upload the repo, and add this front-matter to the Space's `README.md`:

```yaml
---
title: RetailPulse
sdk: streamlit
app_file: app/streamlit_app.py
python_version: "3.11"
---
```

## Option C – Docker / cloud

```bash
docker build --target app -t retailpulse .
docker run -p 8501:8501 -p 8000:8000 retailpulse           # dashboard + Prometheus metrics
docker compose up --build                                  # + Postgres, Redis, MLflow, Prometheus, Grafana
kubectl apply -f k8s/secret.yaml && kubectl apply -k k8s/  # Kubernetes (edit image + host first)
```

`.github/workflows/deploy-cloudrun.yml` deploys the same image to **Google Cloud Run** (manual trigger, needs `GCP_*` secrets).

## Pre-submission checklist (Zidio dashboard – 5 links)

| # | Deliverable | Where it comes from |
|---|-------------|---------------------|
| 1 | Source code | GitHub repo URL (public) |
| 2 | Live deployment | Streamlit Cloud / HF Space URL |
| 3 | Demo video (4–8 min) | `docs/DEMO_SCRIPT.md` → YouTube (unlisted) / Loom / Drive |
| 4 | Feedback video | `docs/FEEDBACK_VIDEO_OUTLINE.md` |
| 5 | Project report | `docs/RetailPulse_Report_Zidio_Nov2026.pdf` uploaded to Drive ("anyone with the link") |

Open every link in an incognito window before pressing **Submit project** – you can only submit once.
