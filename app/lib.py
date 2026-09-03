"""Shared helpers and theming for the Streamlit dashboard.

The dashboard is a reader. Every table it shows is produced offline by the
scripts in scripts/; recomputing a 500-ticker ranking on page load would make it
unusable. Anything missing is reported as a prompt to run the relevant script
rather than silently rendering an empty chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Streamlit puts the entry script's folder on sys.path, not the project root,
# so `import src...` needs help.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# --- Palette ---------------------------------------------------------------

INK = "#17132b"          # page background
SURFACE = "#221c3d"      # cards and sidebar
SURFACE_2 = "#2b2450"    # raised elements
LINE = "#372f5e"
TEXT = "#ede9fe"
MUTED = "#a79fd4"
ACCENT = "#a78bfa"

# Card gradients, in the order they should be used.
GRADIENTS = [
    ("#f72585", "#b5179e"),   # magenta
    ("#7209b7", "#560bad"),   # violet
    ("#4361ee", "#4cc9f0"),   # blue
    ("#06d6a0", "#118ab2"),   # teal
]

# Categorical colours for series, tuned to read on the dark background.
SERIES = ["#a78bfa", "#f72585", "#4cc9f0", "#06d6a0", "#ffb703", "#ff6b6b"]

PALETTE = {
    "A": "#a78bfa",
    "B": "#f72585",
    "C": "#4cc9f0",
    "actual": "#ede9fe",
    "predicted": "#a78bfa",
    "baseline": "#f72585",
}


# --- Plotly template -------------------------------------------------------
# Registered once at import. Every px/go figure in the app inherits it, so no
# page needs to restyle its charts by hand.

pio.templates["sp555"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        title=dict(font=dict(color=TEXT, size=17)),
        colorway=SERIES,
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=SURFACE_2, font=dict(color=TEXT),
                        bordercolor=LINE),
        margin=dict(l=0, r=0, t=30, b=0),
    )
)
pio.templates.default = "sp555"


# --- Theme -----------------------------------------------------------------

_CSS = f"""
<style>
  .stApp {{
      background:
        radial-gradient(900px 500px at 12% -8%, #3a2a7a 0%, transparent 60%),
        radial-gradient(800px 500px at 92% 4%, #5b3a8e 0%, transparent 55%),
        {INK};
  }}

  [data-testid="stHeader"] {{ background: transparent; }}

  .block-container {{ padding-top: 2.6rem; max-width: 1360px; }}

  h1 {{ font-weight: 800; letter-spacing: -0.02em; }}
  h2, h3 {{ font-weight: 700; letter-spacing: -0.01em; }}

  /* Sidebar reads as a panel rather than a slab */
  [data-testid="stSidebar"] > div:first-child {{
      background: {SURFACE};
      border-right: 1px solid {LINE};
  }}

  /* Metric cards */
  [data-testid="stMetric"] {{
      background: linear-gradient(155deg, {SURFACE_2} 0%, {SURFACE} 100%);
      border: 1px solid {LINE};
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.28);
  }}
  [data-testid="stMetricLabel"] p {{
      color: {MUTED} !important;
      font-size: 0.82rem !important;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }}
  [data-testid="stMetricValue"] {{
      font-size: 1.9rem !important;
      font-weight: 700;
  }}

  /* Gradient hero cards */
  .grad-card {{
      border-radius: 20px;
      padding: 20px 22px;
      color: #fff;
      min-height: 132px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      box-shadow: 0 14px 34px rgba(0,0,0,0.35);
  }}
  .grad-card .gc-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      opacity: 0.85;
  }}
  .grad-card .gc-value {{
      font-size: 2.1rem;
      font-weight: 800;
      line-height: 1.1;
      margin: 6px 0 2px;
  }}
  .grad-card .gc-note {{ font-size: 0.82rem; opacity: 0.9; }}

  /* Tables, expanders, tabs */
  [data-testid="stDataFrame"], [data-testid="stTable"] {{
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid {LINE};
  }}
  [data-testid="stExpander"] {{
      border: 1px solid {LINE};
      border-radius: 14px;
      background: {SURFACE};
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{
      background: {SURFACE};
      border-radius: 10px 10px 0 0;
      padding: 8px 16px;
  }}

  /* Callouts */
  [data-testid="stAlert"] {{ border-radius: 14px; border: 1px solid {LINE}; }}

  /* Charts sit on the page, not on a grey block */
  [data-testid="stPlotlyChart"] {{
      background: {SURFACE};
      border: 1px solid {LINE};
      border-radius: 18px;
      padding: 14px 14px 6px;
  }}

  video {{ border-radius: 18px; }}

  code {{ color: {ACCENT}; }}
</style>
"""


def page_config(title: str) -> None:
    """Set page options and inject the theme. Call first on every page."""
    st.set_page_config(
        page_title=f"S&P 555 — {title}",
        page_icon="📈",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


# --- Data access -----------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_table(path_str: str) -> pd.DataFrame | None:
    """Read a parquet/CSV artifact, returning None when it has not been built."""
    path = Path(path_str)
    candidates = [path, path.with_suffix(".csv"), path.with_suffix(".parquet")]

    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)
    return None


def processed(name: str) -> pd.DataFrame | None:
    return read_table(str(PROCESSED_DIR / name))


def report(name: str) -> pd.DataFrame | None:
    return read_table(str(REPORTS_DIR / name))


def asset(name: str) -> Path | None:
    path = ASSETS_DIR / name
    return path if path.exists() else None


def require(df: pd.DataFrame | None, missing_hint: str) -> bool:
    """Show a build instruction instead of an empty page when data is absent."""
    if df is None or df.empty:
        st.info(f"Not built yet. Run `{missing_hint}` from the project root.")
        return False
    return True


# --- Layout helpers --------------------------------------------------------


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric cards from (label, value, help) tuples."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items):
        column.metric(label, value, help=helptext)


def gradient_row(items: list[tuple[str, str, str]]) -> None:
    """Render headline figures as gradient cards from (label, value, note)."""
    columns = st.columns(len(items))
    for i, (column, (label, value, note)) in enumerate(zip(columns, items)):
        start, end = GRADIENTS[i % len(GRADIENTS)]
        column.markdown(
            f"""
            <div class="grad-card"
                 style="background: linear-gradient(140deg, {start} 0%, {end} 100%);">
              <div class="gc-label">{label}</div>
              <div>
                <div class="gc-value">{value}</div>
                <div class="gc-note">{note}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
