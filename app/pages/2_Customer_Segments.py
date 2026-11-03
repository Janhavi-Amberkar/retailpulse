import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import P

ui.setup("Customer Segments", "👥")
ui.page_view("segments")
ui.hero(
    "Customer Segmentation",
    "RFM + behavioural features clustered with K-Means (k chosen by silhouette in the 6–8 band) "
    "and cross-checked with DBSCAN.",
    "F-02 · Unsupervised learning",
)
if not ui.need("customers", "segment_profile", "segment_k_eval"):
    st.stop()

cust = ui.parquet("customers")
prof = ui.parquet("segment_profile")
kev = ui.parquet("segment_k_eval")
sm = ui.jreport("segmentation_metrics.json")

SEG_COL = {
    s: c
    for s, c in zip(
        prof.sort_values("revenue_share", ascending=False)["segment"],
        [P["emerald"], P["violet"], P["cyan"], P["amber"], P["rose"], "#A78BFA", "#38BDF8", "#F472B6"],
    )
}

ui.kpis(
    [
        ("Customers segmented", f'{sm["n_customers"]:,}', "with ≥1 identified purchase", P["violet"]),
        ("Segments (k)", f'{sm["k"]}', "spec: 6–8", P["emerald"], "up"),
        ("Silhouette", f'{sm["silhouette"]:.3f}', "K-Means, sampled", P["cyan"]),
        ("Davies-Bouldin", f'{sm["davies_bouldin"]:.2f}', "lower is better", P["amber"]),
        (
            "DBSCAN clusters",
            f'{sm["dbscan_clusters"]}',
            f'{sm["dbscan_noise_share"] * 100:.1f}% noise · comparison model',
            P["rose"],
        ),
    ]
)

t1, t2, t3, t4 = st.tabs(["🧩 Segment profiles", "🌌 3-D cluster map", "📐 Model selection", "📋 Customer list"])

with t1:
    p = prof.sort_values("revenue_share", ascending=False)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        fig = go.Figure(
            go.Pie(
                labels=p["segment"],
                values=p["revenue_share"],
                hole=0.62,
                marker=dict(colors=[SEG_COL[s] for s in p["segment"]], line=dict(color="#070B1A", width=2)),
                textinfo="percent",
                sort=False,
            )
        )
        fig.add_annotation(text="<b>Revenue<br>share</b>", showarrow=False, font=dict(size=14, color="#fff"))
        ui.show(ui.style(fig, 340, True), key="pie")
    with c2:
        feats = ["recency", "frequency", "monetary", "avg_order_value", "n_skus", "tenure_days"]
        z = cust.groupby("segment")[feats].median()
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        z["recency"] = 1 - z["recency"]  # recent = good
        fig = go.Figure()
        for s in z.index:
            fig.add_trace(
                go.Scatterpolar(
                    r=z.loc[s].tolist() + [z.loc[s].iloc[0]],
                    theta=[
                        "Recency (recent=high)",
                        "Frequency",
                        "Monetary",
                        "Avg order",
                        "SKU breadth",
                        "Tenure",
                        "Recency (recent=high)",
                    ],
                    name=s,
                    line=dict(color=SEG_COL[s], width=2),
                    fill="toself",
                    opacity=0.35,
                )
            )
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=False),
                angularaxis=dict(gridcolor="rgba(255,255,255,.1)"),
            )
        )
        ui.show(ui.style(fig, 340), key="radar")
    ui.section("Business interpretation")
    for _, r in p.iterrows():
        pill_size = ui.pill("{:,} customers".format(int(r["size"])))
        pill_rev = ui.pill("{:.0f}% of revenue".format(r["revenue_share"] * 100), "g")
        color = SEG_COL[r["segment"]]
        seg_name, desc = r["segment"], r["description"]
        stats = "median recency {:.0f} d · {:.0f} orders · £{:,.0f} spend · £{:,.0f} avg order".format(
            r["recency"], r["frequency"], r["monetary"], r["avg_order_value"]
        )
        st.markdown(
            f'<div class="rp-card" style="border-left:4px solid {color}"><b style="font-size:1.05rem">{seg_name}</b> '
            f'{pill_size}{pill_rev}<br><span class="rp-note">{stats}</span><br>{desc}</div>',
            unsafe_allow_html=True,
        )

with t2:
    n = st.slider("Points shown", 500, 5000, 2500, 250)
    s = cust.sample(min(n, len(cust)), random_state=1)
    fig = px.scatter_3d(
        s,
        x="pc1",
        y="pc2",
        z="pc3",
        color="segment",
        color_discrete_map=SEG_COL,
        opacity=0.8,
        hover_data={"recency": True, "frequency": True, "monetary": ":.0f", "pc1": False, "pc2": False, "pc3": False},
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)"),
        )
    )
    ui.show(ui.style(fig, 560), key="3d")
    ev = sm["pca_explained_variance"]
    st.caption(
        f"PCA explains {sum(ev) * 100:.0f}% of variance in 3 components ({ev[0] * 100:.0f}% / {ev[1] * 100:.0f}% / {ev[2] * 100:.0f}%)."
    )

with t3:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(
            go.Scatter(x=kev["k"], y=kev["inertia"], mode="lines+markers", line=dict(color=P["violet"], width=3))
        )
        ui.show(ui.style(fig, 320, False, "Elbow — inertia vs k"), key="elbow")
    with c2:
        fig = make_fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=kev["k"],
                y=kev["silhouette"],
                marker_color=[P["emerald"] if k == sm["k"] else "#3b4478" for k in kev["k"]],
                name="silhouette",
            )
        )
        fig.add_vrect(x0=5.5, x1=8.5, fillcolor="rgba(52,211,153,.07)", line_width=0, annotation_text="spec band 6–8")
        ui.show(ui.style(fig, 320, False, "Silhouette by k (chosen k highlighted)"), key="sil")
    st.markdown(
        f'<div class="rp-card"><b>K-Means vs DBSCAN.</b> DBSCAN (eps={sm["dbscan_eps"]:.2f}, min_samples=10) finds only '
        f'<b>{sm["dbscan_clusters"]} dense clusters</b> and labels {sm["dbscan_noise_share"] * 100:.1f}% of customers as noise: retail RFM data is '
        f"a continuum rather than separated blobs, so it cannot deliver the 6–8 actionable segments the business needs. "
        f"K-Means is the production model; DBSCAN is kept for outlier (wholesale / anomalous) detection.</div>",
        unsafe_allow_html=True,
    )
    ui.df(kev.style.format("{:.3f}"), hide_index=True)

with t4:
    c = st.columns([1, 1, 2])
    segs = c[0].multiselect("Segment", sorted(cust["segment"].unique()), default=[])
    sort = c[1].selectbox("Sort by", ["monetary", "frequency", "recency", "avg_order_value"])
    d = cust if not segs else cust[cust["segment"].isin(segs)]
    d = d.sort_values(sort, ascending=sort == "recency").head(500).copy()
    d["customer_key"] = ui.mask_key(d["customer_key"])
    cols = [
        "customer_key",
        "segment",
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "n_skus",
        "mean_gap",
        "tenure_days",
    ]
    ui.df(d[cols].round(1), hide_index=True, height=420)
    ui.csv_button(
        cust[["customer_key", "segment"] + cols[2:]].round(2),
        "⬇ Download all segments CSV",
        "retailpulse_segments.csv",
        "segments",
    )
