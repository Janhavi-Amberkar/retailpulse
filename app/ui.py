"""Shared UI layer: theme/CSS, Plotly template, data loaders, role-based access, widgets."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from retailpulse import config, monitoring  # noqa: E402

P = config.PALETTE
ROLES = {"Viewer": 0, "Analyst": 1, "Admin": 2}
# Demo passcodes (override with Streamlit secrets / env vars in production).
DEFAULT_PINS = {"Analyst": "analyst", "Admin": "admin"}

FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');
"""

CSS = FONT_CSS + """
html, body, [class*="css"], .stApp, .stMarkdown, p, label, li { font-family: 'Inter', system-ui, sans-serif !important; }
h1, h2, h3, h4, .rp-title, .rp-section { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.02em; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -12%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(34,211,238,.16), transparent 55%),
    radial-gradient(800px 600px at 50% 120%, rgba(52,211,153,.10), transparent 60%),
    #070B1A;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1320px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e1433 0%, #080c20 100%);
  border-right: 1px solid rgba(124,92,255,.22);
}
.rp-brand { display:flex; align-items:center; gap:.6rem; padding:.4rem 0 1rem 0; }
.rp-logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-weight:700;
  background: linear-gradient(135deg,#7C5CFF,#22D3EE); color:#fff; box-shadow:0 6px 24px rgba(124,92,255,.5); }
.rp-brand b { font-family:'Space Grotesk'; font-size:1.18rem; color:#fff; }
.rp-brand small { display:block; color:#8E97B8; font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; margin-top:-2px; }

/* ---------- hero ---------- */
.rp-hero {
  position:relative; overflow:hidden; padding: 1.7rem 1.9rem; border-radius: 22px; margin-bottom: 1.1rem;
  background: linear-gradient(135deg, rgba(124,92,255,.22), rgba(34,211,238,.10) 55%, rgba(52,211,153,.10));
  border: 1px solid rgba(124,92,255,.35); box-shadow: 0 20px 60px rgba(7,11,26,.6), inset 0 1px 0 rgba(255,255,255,.06);
}
.rp-hero::after { content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px; border-radius:50%;
  background: radial-gradient(circle, rgba(34,211,238,.28), transparent 65%); }
.rp-badge { display:inline-block; font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; padding:.28rem .7rem;
  border-radius:999px; background:rgba(124,92,255,.2); border:1px solid rgba(124,92,255,.5); color:#c7bdff; margin-bottom:.7rem; }
.rp-title { font-size: 2.35rem; line-height:1.1; font-weight:700; margin:0;
  background: linear-gradient(92deg,#ffffff 0%,#b8b0ff 42%,#22D3EE 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.rp-sub { color:#aab3d4; margin-top:.55rem; font-size:1.02rem; max-width: 820px; }
.rp-section { font-size:1.25rem; font-weight:600; color:#fff; margin: 1.4rem 0 .5rem 0; display:flex; align-items:center; gap:.55rem; }
.rp-section::before { content:""; width:6px; height:22px; border-radius:4px; background:linear-gradient(180deg,#7C5CFF,#22D3EE); }

/* ---------- KPI cards ---------- */
.rp-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(176px, 1fr)); gap:.85rem; margin:.4rem 0 1rem 0; }
.rp-kpi { position:relative; padding:1rem 1.1rem; border-radius:16px; background: rgba(19,26,51,.72); backdrop-filter: blur(8px);
  border:1px solid rgba(124,92,255,.22); overflow:hidden; transition: transform .18s ease, border-color .18s ease; }
.rp-kpi:hover { transform: translateY(-3px); border-color: rgba(34,211,238,.6); }
.rp-kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--accent, #7C5CFF); box-shadow: 0 0 18px var(--accent, #7C5CFF); }
.rp-kpi .lab { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:#8E97B8; }
.rp-kpi .val { font-family:'JetBrains Mono', monospace; font-size:1.65rem; font-weight:600; color:#fff; margin:.25rem 0 .1rem 0; }
.rp-kpi .sub { font-size:.78rem; color:#8E97B8; }
.rp-kpi .up { color:#34D399; } .rp-kpi .down { color:#FB7185; } .rp-kpi .warn { color:#FBBF24; }

/* ---------- cards / callouts ---------- */
.rp-card { padding:1rem 1.2rem; border-radius:16px; background:rgba(19,26,51,.7); border:1px solid rgba(124,92,255,.2); margin-bottom:.8rem; }
.rp-alert { padding:.7rem 1rem; border-radius:12px; margin:.35rem 0; border:1px solid; font-size:.92rem; }
.rp-alert.crit { background:rgba(251,113,133,.10); border-color:rgba(251,113,133,.5); }
.rp-alert.warn { background:rgba(251,191,36,.09); border-color:rgba(251,191,36,.45); }
.rp-alert.ok { background:rgba(52,211,153,.09); border-color:rgba(52,211,153,.4); }
.rp-pill { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:600; margin-right:.3rem; }
.rp-pill.v { background:rgba(124,92,255,.22); color:#cfc7ff; } .rp-pill.g { background:rgba(52,211,153,.2); color:#86efc5; }
.rp-pill.r { background:rgba(251,113,133,.2); color:#fda4af; } .rp-pill.a { background:rgba(251,191,36,.2); color:#fde68a; }
.rp-note { color:#8E97B8; font-size:.82rem; }

/* ---------- streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid rgba(124,92,255,.2); }
.stTabs [data-baseweb="tab"] { height:42px; padding:0 1rem; border-radius:10px 10px 0 0; color:#aab3d4; background:transparent; }
.stTabs [aria-selected="true"] { color:#fff !important; background:linear-gradient(180deg, rgba(124,92,255,.28), rgba(124,92,255,.05)); }
.stButton>button, .stDownloadButton>button { border-radius:12px; border:1px solid rgba(124,92,255,.5); color:#fff;
  background:linear-gradient(135deg,#7C5CFF,#5b8def); font-weight:600; transition: all .15s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform: translateY(-1px); box-shadow:0 8px 24px rgba(124,92,255,.45); border-color:#22D3EE; color:#fff; }
div[data-testid="stDataFrame"] { border:1px solid rgba(124,92,255,.2); border-radius:12px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid rgba(124,92,255,.2); border-radius:12px; background:rgba(19,26,51,.5); }
hr { border-color: rgba(124,92,255,.18); }
a { color:#22D3EE; }

@media (max-width: 720px) {
  .rp-title { font-size:1.7rem; } .rp-hero { padding:1.2rem; } .rp-kpi .val { font-size:1.3rem; }
  .block-container { padding-left:.8rem; padding-right:.8rem; }
}
"""


def setup(title: str, icon: str = "📊", layout: str = "wide") -> None:
    st.set_page_config(
        page_title=f"{title} · RetailPulse", page_icon=icon, layout=layout, initial_sidebar_state="expanded"
    )
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    monitoring.start_metrics_server(int(os.getenv("METRICS_PORT", "8000")))
    sidebar()


# ---------------------------------------------------------------- roles -----
def _pin(role: str) -> str:
    key = f"{role.upper()}_PIN"
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(f"RETAILPULSE_{key}", DEFAULT_PINS[role])


def role() -> str:
    return st.session_state.get("role", "Viewer")


def has(min_role: str) -> bool:
    return ROLES[role()] >= ROLES[min_role]


def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="rp-brand"><div class="rp-logo">RP</div><div><b>RetailPulse</b>'
            "<small>Zidio · Data Science</small></div></div>",
            unsafe_allow_html=True,
        )
        st.session_state.setdefault("role", "Viewer")
        st.session_state.setdefault("user", "guest")
        with st.expander(f"🔐 Access: {role()}", expanded=False):
            sel = st.selectbox("Role", list(ROLES), index=list(ROLES).index(role()), key="role_sel")
            if sel != role():
                if sel == "Viewer":
                    st.session_state.role = "Viewer"
                    monitoring.audit(st.session_state.user, "Viewer", "role_change", "to Viewer")
                    st.rerun()
                pin = st.text_input("Passcode", type="password", key="pin_in", help="Demo: analyst / admin")
                if st.button("Unlock", key="unlock"):
                    if pin == _pin(sel):
                        st.session_state.role = sel
                        st.session_state.user = sel.lower()
                        monitoring.audit(sel.lower(), sel, "login", "passcode ok")
                        st.rerun()
                    else:
                        monitoring.audit("unknown", "Viewer", "login_failed", f"requested {sel}")
                        st.error("Wrong passcode")
            st.caption("Viewer: dashboards · Analyst: + exports & customer lists · Admin: + MLOps / audit log")
        st.markdown("---")
        st.markdown(
            '<span class="rp-note">Dataset · UCI Online Retail II (2009-12 → 2011-12)<br>'
            "Customer IDs are salted-hash pseudonyms.</span>",
            unsafe_allow_html=True,
        )


def gate(min_role: str, what: str = "this section") -> bool:
    if has(min_role):
        return True
    st.markdown(
        f'<div class="rp-alert warn">🔒 <b>{min_role}</b> access required for {what}. '
        f"Open <i>Access</i> in the sidebar and enter the demo passcode "
        f'(<code>{DEFAULT_PINS.get(min_role, "")}</code>).</div>',
        unsafe_allow_html=True,
    )
    return False


def mask_key(k: pd.Series) -> pd.Series:
    return k if has("Analyst") else k.str.slice(0, 4) + "••••••"


# --------------------------------------------------------------- widgets ----
def hero(title: str, subtitle: str, badge: str = "RetailPulse") -> None:
    st.markdown(
        f'<div class="rp-hero"><span class="rp-badge">{html.escape(badge)}</span>'
        f'<h1 class="rp-title">{html.escape(title)}</h1><div class="rp-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def section(text: str) -> None:
    st.markdown(f'<div class="rp-section">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items: (label, value, sub, accent_hex[, sub_class])"""
    cells = []
    for it in items:
        lab, val, sub, acc = it[:4]
        cls = it[4] if len(it) > 4 else ""
        cells.append(
            f'<div class="rp-kpi" style="--accent:{acc}"><div class="lab">{html.escape(lab)}</div>'
            f'<div class="val">{val}</div><div class="sub {cls}">{sub}</div></div>'
        )
    st.markdown(f'<div class="rp-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def alert(text: str, level: str = "ok") -> None:
    icon = {"crit": "🚨", "warn": "⚠️", "ok": "✅"}[level]
    st.markdown(f'<div class="rp-alert {level}">{icon} {text}</div>', unsafe_allow_html=True)


def pill(text: str, color: str = "v") -> str:
    return f'<span class="rp-pill {color}">{html.escape(text)}</span>'


def money(x: float, cur: str = "£") -> str:
    ax = abs(x)
    if ax >= 1e6:
        return f"{cur}{x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"{cur}{x / 1e3:,.1f}K"
    return f"{cur}{x:,.0f}"


def num(x: float) -> str:
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:,.1f}K"
    return f"{x:,.0f}"


# ---------------------------------------------------------------- plotly ----
COLORWAY = [P["violet"], P["cyan"], P["emerald"], P["amber"], P["rose"], "#A78BFA", "#38BDF8", "#F472B6"]


def style(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=P["text"], size=12),
        margin=dict(l=10, r=10, t=44 if title else 14, b=10),
        showlegend=legend,
        title=dict(text=title, font=dict(family="Space Grotesk", size=16), x=0.01) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#101733", bordercolor=P["violet"], font=dict(family="Inter")),
        xaxis=dict(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.12)"),
    )
    return fig


def break_gaps(df: pd.DataFrame, col: str = "date", days: int = 3) -> pd.DataFrame:
    """Insert NaN rows where the date jumps by more than `days` so line charts show closures as gaps."""
    d = df.sort_values(col).reset_index(drop=True)
    gap = d[col].diff().dt.days > days
    if not gap.any():
        return d
    blanks = d.loc[gap].copy()
    blanks[col] = blanks[col] - pd.Timedelta(days=1)
    for c in blanks.columns:
        if c != col:
            blanks[c] = np.nan
    return pd.concat([d, blanks], ignore_index=True).sort_values(col).reset_index(drop=True)


def _stretch(fn, *a, **kw):
    """Full-width widgets on every Streamlit version (`width="stretch"` is new; older use use_container_width)."""
    try:
        return fn(*a, width="stretch", **kw)
    except Exception:  # older Streamlit: width must be an int -> fall back
        return fn(*a, use_container_width=True, **kw)


def df(*a, **kw):
    return _stretch(st.dataframe, *a, **kw)


def image(*a, **kw):
    return _stretch(st.image, *a, **kw)


def show(fig: go.Figure, key: str | None = None, **kw) -> None:
    _stretch(st.plotly_chart, fig, key=key, config={"displaylogo": False}, **kw)


# ------------------------------------------------------------------ data ----
@st.cache_data(show_spinner=False)
def parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_PROC / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def jreport(name: str) -> dict:
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def have(name: str) -> bool:
    return (config.DATA_PROC / f"{name}.parquet").exists()


def csv_button(df: pd.DataFrame, label: str, fname: str, kind: str, min_role: str = "Analyst") -> None:
    """Export gated by role + audit-logged (security highlight)."""
    if not has(min_role):
        st.caption(f"🔒 CSV export needs {min_role} access")
        return
    if st.download_button(label, df.to_csv(index=False).encode(), fname, "text/csv", key=f"dl_{fname}"):
        monitoring.audit(st.session_state.get("user", "?"), role(), "export_csv", fname)
        if monitoring.HAVE_PROM:
            monitoring.EXPORTS.labels(kind).inc()


def page_view(name: str) -> None:
    if monitoring.HAVE_PROM:
        monitoring.PAGE_VIEWS.labels(name, role()).inc()
    key = f"viewed_{name}"
    if not st.session_state.get(key):
        st.session_state[key] = True
        monitoring.audit(st.session_state.get("user", "guest"), role(), "page_view", name)


def need(*names: str) -> bool:
    missing = [n for n in names if not have(n)]
    if missing:
        st.error(f"Missing artefacts: {', '.join(missing)}. Run `make pipeline` first.")
        return False
    return True


COUNTRY_FIX = {
    "EIRE": "Ireland",
    "RSA": "South Africa",
    "USA": "United States",
    "Channel Islands": "Jersey",
    "European Community": None,
    "Unspecified": None,
    "West Indies": None,
    "Hong Kong": "Hong Kong",
}
