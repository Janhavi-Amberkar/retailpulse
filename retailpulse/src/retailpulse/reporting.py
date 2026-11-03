"""Exportable executive PDF (reportlab) built from the live artefacts."""

from __future__ import annotations

import io
import json
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from retailpulse import config

VIOLET, CYAN, DARK = colors.HexColor("#7C5CFF"), colors.HexColor("#22D3EE"), colors.HexColor("#0B0F1F")


def _j(name: str) -> dict:
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def _table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7cbe0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        st += [
            ("BACKGROUND", (0, 0), (-1, 0), VIOLET),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ]
    t.setStyle(TableStyle(st))
    return t


def build_executive_pdf() -> bytes:
    daily = pd.read_parquet(config.DATA_PROC / "daily_sales.parquet")
    etl, fm, cm, im = (
        _j("etl_report.json"),
        _j("forecast_metrics.json"),
        _j("churn_metrics.json"),
        _j("inventory_metrics.json"),
    )
    sm, dr = _j("segmentation_metrics.json"), _j("drift_summary.json")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="RetailPulse Executive Report",
        author="RetailPulse",
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=24, textColor=DARK, alignment=0, spaceAfter=4
    )
    h2 = ParagraphStyle(
        "h2",
        parent=ss["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=VIOLET,
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555a75"))
    S = [
        Paragraph("RetailPulse – Executive Report", h1),
        Paragraph(f"AI-Powered Customer Analytics &amp; Demand Forecasting · generated {date.today():%d %b %Y}", small),
        Spacer(1, 6),
    ]

    rev, inv = daily["revenue"].sum(), daily["invoices"].sum()
    S += [
        Paragraph("1. Business snapshot", h2),
        _table(
            [
                ["Metric", "Value"],
                ["Period", f'{etl.get("date_min")} → {etl.get("date_max")}'],
                ["Revenue (cleaned)", f"£{rev:,.0f}"],
                ["Orders", f"{inv:,.0f}"],
                ["Customers (pseudonymised)", f'{etl.get("customers", 0):,}'],
                ["Active SKUs", f'{etl.get("skus", 0):,}'],
                ["Clean rows / raw rows", f'{etl.get("rows_clean", 0):,} / {etl.get("rows_raw", 0):,}'],
            ],
            [70 * mm, 90 * mm],
        ),
    ]
    if fm:
        e, c = fm["test"]["ensemble"], fm["cv"]["ensemble"]
        S += [
            Paragraph("2. Demand forecasting (Prophet + LSTM ensemble)", h2),
            _table(
                [
                    ["Split", "Daily MAPE %", "Weekly MAPE %", "30-day total error %"],
                    [
                        "Cross-validation (7 folds)",
                        f'{c["mape_daily"]:.1f}',
                        f'{c["mape_weekly"]:.1f}',
                        f'{c["ape_30d_total"]:.1f}',
                    ],
                    [
                        "Hold-out (peak season)",
                        f'{e["mape_daily"]:.1f}',
                        f'{e["mape_weekly"]:.1f}',
                        f'{e["ape_30d_total"]:.1f}',
                    ],
                ],
                [60 * mm, 35 * mm, 35 * mm, 40 * mm],
            ),
        ]
        fut = pd.read_parquet(config.DATA_PROC / "forecast_future.parquet")
        S += [
            Spacer(1, 4),
            Paragraph(
                f"Next-30-day forecast: <b>{fut['ensemble'].sum():,.0f} units</b> over {len(fut)} trading days.", body
            ),
        ]
    if cm:
        f = cm["final"]
        S += [
            Paragraph("3. Churn prediction (XGBoost + SHAP)", h2),
            _table(
                [
                    ["AUC-ROC", "PR-AUC", "Precision@20%", "Lift@20%", "F1", "Recall"],
                    [
                        f'{f["auc_roc"]:.3f}',
                        f'{f["auc_pr"]:.3f}',
                        f'{f["precision_top20"]:.3f}',
                        f'{f["lift_top20"]:.2f}x',
                        f'{f["f1"]:.3f}',
                        f'{f["recall"]:.3f}',
                    ],
                ]
            ),
        ]
        sc = pd.read_parquet(config.DATA_PROC / "churn_scores.parquet")
        crit = sc[sc["risk_tier"] == "Critical"]
        S += [
            Spacer(1, 4),
            Paragraph(
                f"{len(crit):,} customers are in the Critical tier, representing ≈ £{crit['expected_loss'].sum():,.0f} expected revenue at risk.",
                body,
            ),
        ]
    if sm:
        prof = pd.read_parquet(config.DATA_PROC / "segment_profile.parquet").sort_values(
            "revenue_share", ascending=False
        )
        S += [
            Paragraph("4. Customer segments", h2),
            _table(
                [["Segment", "Customers", "Revenue share", "Median recency (d)", "Median orders"]]
                + [
                    [
                        r["segment"],
                        f'{int(r["size"]):,}',
                        f'{r["revenue_share"] * 100:.0f}%',
                        f'{r["recency"]:.0f}',
                        f'{r["frequency"]:.0f}',
                    ]
                    for _, r in prof.iterrows()
                ]
            ),
        ]
    S.append(PageBreak())
    if im:
        rec = pd.read_parquet(config.DATA_PROC / "inventory_recommendations.parquet")
        S += [
            Paragraph("5. Inventory optimisation (stock levels simulated)", h2),
            Paragraph(f"Status mix: {', '.join(f'{k}: {v}' for k, v in im['status_counts'].items())}.", body),
            Spacer(1, 4),
        ]
        top = rec.sort_values("order_value", ascending=False).head(15)
        S += [
            _table(
                [["SKU", "Description", "On hand", "30d fcst", "ROP", "Order qty", "Status"]]
                + [
                    [
                        r["stock_code"],
                        r["description"].title()[:28],
                        f'{r["on_hand"]:,}',
                        f'{r["forecast_30d"]:,.0f}',
                        f'{r["reorder_point"]:,.0f}',
                        f'{r["recommended_order"]:,.0f}',
                        r["status"],
                    ]
                    for _, r in top.iterrows()
                ],
                [18 * mm, 55 * mm, 18 * mm, 18 * mm, 16 * mm, 20 * mm, 24 * mm],
            )
        ]
        pol = pd.DataFrame(im["policy_summary"])
        S += [
            Spacer(1, 6),
            Paragraph("Policy back-test (real demand)", h2),
            _table(
                [["Policy", "Fill rate %", "Overstock %", "Total cost £"]]
                + [
                    [
                        r["policy"],
                        f'{r["fill_rate_pct"]:.1f}',
                        f'{r["overstock_pct_of_demand"]:.1f}',
                        f'{r["total_cost"]:,.0f}',
                    ]
                    for _, r in pol.iterrows()
                ],
                [80 * mm, 25 * mm, 30 * mm, 30 * mm],
            ),
        ]
    if dr:
        d = dr["decision"]
        S += [
            Paragraph("6. Model monitoring", h2),
            Paragraph(
                f"Drift check: {dr['current']['n_alert']}/{dr['current']['n_features']} features in ALERT (PSI ≥ 0.2). "
                f"Retrain recommended: <b>{'YES' if d['retrain'] else 'no'}</b>. "
                + ("; ".join(d["reasons"]) if d["reasons"] else ""),
                body,
            ),
        ]
    S += [
        Spacer(1, 10),
        Paragraph(
            "Source: UCI Online Retail II (CC BY 4.0). Customer identifiers are salted-hash pseudonyms. "
            "Inventory on-hand levels and lead times are simulated.",
            small,
        ),
    ]
    doc.build(S)
    return buf.getvalue()
