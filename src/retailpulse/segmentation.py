"""F02 – RFM + behavioural customer segmentation (K-Means primary, DBSCAN comparison)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from retailpulse import config

SEG_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "n_skus",
    "mean_gap",
    "return_rate",
    "tenure_days",
]
LOG_FEATURES = ["frequency", "monetary", "avg_order_value", "n_skus", "mean_gap"]


def prepare(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    x = rfm[SEG_FEATURES].copy()
    for c in LOG_FEATURES:
        x[c] = np.log1p(x[c].clip(lower=0))
    x["return_rate"] = x["return_rate"].clip(0, 1)
    sc = StandardScaler()
    return sc.fit_transform(x), sc


def select_k(X: np.ndarray, k_range=range(6, 9)) -> tuple[int, pd.DataFrame]:
    """Pick k in the spec's 6-8 band by silhouette (ties -> Davies-Bouldin)."""
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_, sample_size=4000, random_state=config.SEED),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            }
        )
    ev = pd.DataFrame(rows)
    cand = ev[ev["k"].isin(list(k_range))].sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    return int(cand.iloc[0]["k"]), ev


def name_segments(profile: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Business names derived from each cluster's centroid rank on R / F / M (not hard-coded ids)."""
    p = profile.copy()
    p["r_rank"] = p["recency"].rank(ascending=True)  # low recency = good
    p["f_rank"] = p["frequency"].rank(ascending=False)
    p["m_rank"] = p["monetary"].rank(ascending=False)
    p["score"] = (len(p) - p["r_rank"]) + (len(p) - p["f_rank"]) + (len(p) - p["m_rank"])
    names: dict[int, tuple[str, str]] = {}
    used: set[str] = set()
    order = p.sort_values("score", ascending=False).index.tolist()

    def pick(cid: int, name: str, desc: str) -> None:
        base, i = name, 2
        while name in used:
            name = f"{base} {i}"
            i += 1
        used.add(name)
        names[cid] = (name, desc)

    for rank, cid in enumerate(order):
        r = p.loc[cid]
        recent = r["recency"] <= p["recency"].median()
        if rank == 0:
            pick(cid, "Champions", "Buy often, spend the most, bought very recently - protect and reward.")
        elif rank == 1 and recent:
            pick(cid, "Loyal Regulars", "Frequent, recent buyers - upsell and cross-sell.")
        elif r["tenure_days"] < p["tenure_days"].median() and recent:
            pick(cid, "New & Promising", "Recent first purchases - nurture into loyalty.")
        elif recent and r["monetary"] >= p["monetary"].median():
            pick(cid, "Big Spenders", "Large baskets but irregular - offer volume incentives.")
        elif recent:
            pick(cid, "Potential Loyalists", "Active but low frequency - encourage repeat orders.")
        elif r["monetary"] >= p["monetary"].median():
            pick(cid, "At Risk High-Value", "Used to spend a lot, gone quiet - win-back priority.")
        elif rank == len(order) - 1:
            pick(cid, "Lost / Hibernating", "Long inactive and low value - low-cost reactivation only.")
        else:
            pick(cid, "Fading Occasionals", "Infrequent, drifting away - light-touch reminders.")
    return names


def run_segmentation(rfm: pd.DataFrame) -> dict:
    X, scaler = prepare(rfm)
    k, ev = select_k(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=config.SEED).fit(X)
    out = rfm.copy()
    out["cluster"] = km.labels_

    profile = out.groupby("cluster")[SEG_FEATURES].median()
    profile["size"] = out.groupby("cluster").size()
    profile["revenue_share"] = out.groupby("cluster")["monetary"].sum() / out["monetary"].sum()
    names = name_segments(profile)
    out["segment"] = out["cluster"].map(lambda c: names[c][0])
    out["segment_desc"] = out["cluster"].map(lambda c: names[c][1])
    profile["segment"] = [names[c][0] for c in profile.index]
    profile["description"] = [names[c][1] for c in profile.index]

    pca = PCA(n_components=3, random_state=config.SEED).fit(X)
    pcs = pca.transform(X)
    out[["pc1", "pc2", "pc3"]] = pcs

    # DBSCAN comparison (eps from the k-distance knee: 90th pct of 5-NN distance) -------
    nn = NearestNeighbors(n_neighbors=6).fit(X)
    dist = np.sort(nn.kneighbors(X)[0][:, -1])
    eps = float(np.quantile(dist, 0.90))
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    mask = db.labels_ != -1
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    db_sil = (
        float(silhouette_score(X[mask], db.labels_[mask], sample_size=4000, random_state=config.SEED))
        if n_db > 1 and mask.sum() > 50
        else float("nan")
    )
    out["dbscan"] = db.labels_

    chosen = ev[ev["k"] == k].iloc[0]
    metrics = {
        "k": k,
        "silhouette": float(chosen["silhouette"]),
        "davies_bouldin": float(chosen["davies_bouldin"]),
        "calinski_harabasz": float(chosen["calinski_harabasz"]),
        "dbscan_eps": eps,
        "dbscan_clusters": n_db,
        "dbscan_noise_share": float((~mask).mean()),
        "dbscan_silhouette": db_sil,
        "n_customers": int(len(out)),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }
    return {
        "customers": out,
        "profile": profile.reset_index(),
        "k_eval": ev,
        "metrics": metrics,
        "kmeans": km,
        "scaler": scaler,
    }
