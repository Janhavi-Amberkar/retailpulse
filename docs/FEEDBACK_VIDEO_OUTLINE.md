# Feedback / reflection video outline (2–4 min) – speak in your OWN words

Zidio asks for *what you learned, challenges you faced, key takeaways*. Use this as prompts, not a script – the programme's
rules say content must be your own, and mentors can tell. Suggested structure:

1. **What I built** (20 s) – one sentence per module (ETL, segments, forecast, churn, inventory, dashboard, MLOps).
2. **What I learned** (60 s) – pick 2–3 that genuinely surprised you, e.g.
   * why *time-based* validation matters (random splits would have inflated churn AUC);
   * why daily MAPE can look bad while weekly / monthly forecasts are good enough for replenishment;
   * what drift monitoring actually flags (try the slider) and when to retrain.
3. **Challenges** (60 s) – real ones from the project: lumpy wholesale orders; non-stationary series; no inventory table in the dataset
   (so stock is simulated); keeping the dashboard deployable with light dependencies; Python 3.11 vs 3.13 differences.
4. **What I'd do next** (20 s) – real stock data, promo/price features, per-SKU hierarchical models, online learning, A/B-tested win-back.
5. **Takeaway** (10 s).
