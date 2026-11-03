# Demo video script (target 6–7 min, limit 4–8)

Record the **live URL** (or `make app`) with Loom / OBS. Speak to the numbers on screen – they are measured, not claimed.

| Time | Screen | Say / do |
|------|--------|----------|
| 0:00–0:30 | Overview | "RetailPulse turns 1M real retail transactions into forecasts, customer segments, churn risk and inventory actions." Point at the KPI row, the live-stream tile and alerts. |
| 0:30–1:15 | Data & ETL | Lineage waterfall: 1,067,371 → 1,003,336 rows; show the 27 quality checks and the **custom column `region_group`**. Mention anonymised customer IDs. |
| 1:15–2:30 | Demand Forecast | 30-day forecast with interval; **Accuracy tab**: Prophet vs LSTM vs ensemble vs naive; explain *why daily MAPE is high but weekly/30-day is within target*; show the untouched hold-out fold. Try a **what-if** (promo +20 %, price −5 %). |
| 2:30–3:15 | Customer Segments | 3-D cluster map, radar, "Champions are ~73 % of revenue". K-Means vs DBSCAN. |
| 3:15–4:15 | Churn Risk | ROC / lift, model benchmark, **SHAP**; at-risk list; click a customer → "why at risk"; retention simulator ROI. Be honest: AUC vs target. |
| 4:15–5:15 | Inventory | Change *service level*, *lead-time stress*, *demand multiplier* → status counts and purchase plan update live. Policy back-test table. Say clearly: stock levels are **simulated**. |
| 5:15–6:30 | MLOps & Drift | **Drift simulation slider**: drag severity 0 → 0.6 → 1.0 and watch the monitor flip to RETRAIN; open the Evidently report; show MLflow runs and the retraining loop. Unlock **Admin** (passcode `admin`) → audit log. |
| 6:30–7:00 | Reports | Generate the executive PDF; mention Docker / K8s / CI / Airflow / Prometheus in the repo. |

Tips: close with the architecture diagram from the report; keep the browser at 1440×900; avoid reading tables aloud.
