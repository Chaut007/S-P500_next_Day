"""Shared helpers and theming for the Streamlit dashboard.

The dashboard is a reader. Every table it shows is produced offline by the
scripts in scripts/; recomputing a 500-ticker ranking on page load would make it
unusable. Anything missing is reported as a prompt to run the relevant script
rather than silently rendering an empty chart.

The theme aims for a desktop-application feel rather than a web page: an ambient
lit background, a floating glass panel for the content, and a sidebar that reads
as application chrome. All of it is CSS injected once per page, because
Streamlit exposes no theming hook beyond config.toml's four colours.
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

from src.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# --- Palette ---------------------------------------------------------------

INK = "#0B1020"          # page background
INK_2 = "#11172A"        # raised background
SURFACE = "rgba(23, 30, 54, 0.72)"    # glass panels
SURFACE_2 = "rgba(255, 255, 255, 0.05)"
LINE = "rgba(255, 255, 255, 0.075)"
LINE_2 = "rgba(255, 255, 255, 0.13)"

TEXT = "#EDF0F7"
MUTED = "#9BA4BE"
NOTE = "#7E8AA6"        # dimmest text that still clears 4.5:1 on the panels
FAINT = "#667089"       # non-text use only: rules, disabled marks

PURPLE = "#7C5CFC"
BLUE = "#3B82F6"
PINK = "#EC4899"
CYAN = "#06B6D4"

ACCENT = PURPLE

# Accent ramps for the headline cards. These colour a 2px rule and a faint
# corner glow only -- never the card surface. Filling four cards with four
# saturated gradients makes the numbers harder to read and the page look like a
# template rather than a study.
GRADIENTS = [
    ("#A78BFA", "#7C5CFC"),   # violet
    ("#60A5FA", "#3B82F6"),   # blue
    ("#22D3EE", "#06B6D4"),   # cyan
    ("#F472B6", "#EC4899"),   # pink
]

# Categorical colours for series, tuned to read on the dark background.
SERIES = ["#A78BFA", "#F472B6", "#22D3EE", "#34D399", "#FBBF24", "#FB7185"]

# One colour per rank slot, warm at the top of the leaderboard and cool at the
# bottom. Rank is a ten-value ordinal, so a continuous colour bar asks the eye
# to interpolate a scale it cannot read precisely; ten fixed steps let a reader
# name the rank from the colour.
RANK_COLORS = [
    "#FDE68A",  # 1
    "#FCD34D",  # 2
    "#FB923C",  # 3
    "#F472B6",  # 4
    "#EC4899",  # 5
    "#C084FC",  # 6
    "#A78BFA",  # 7
    "#818CF8",  # 8
    "#60A5FA",  # 9
    "#38BDF8",  # 10
]


# Sector colours for the leaderboard. Colouring the bars by sector rather than
# giving each company an arbitrary hue is what makes the animation argue its
# point: the energy and industrial names leaving the block, and the violet of
# technology taking it over, are visible without reading a single label.
SECTOR_COLORS = {
    "Information Technology": "#A78BFA",
    "Communication Services": "#60A5FA",
    "Consumer Discretionary": "#22D3EE",
    "Financials": "#34D399",
    "Health Care": "#F472B6",
    "Consumer Staples": "#FBBF24",
    "Energy": "#FB923C",
    "Industrials": "#FB7185",
}
SECTOR_FALLBACK = "#8B96B4"


def stepped_scale(colors: list[str]) -> list[list]:
    """Turn a colour list into a Plotly colorscale with hard edges.

    Repeating each stop at both ends of its band stops Plotly interpolating
    between them, which is what keeps a rank band one flat colour.
    """
    n = len(colors)
    scale: list[list] = []
    for i, color in enumerate(colors):
        scale.append([i / n, color])
        scale.append([(i + 1) / n, color])
    return scale

PALETTE = {
    "A": "#A78BFA",
    "B": "#F472B6",
    "C": "#22D3EE",
    "actual": "#EDF0F7",
    "predicted": "#A78BFA",
    "baseline": "#F472B6",
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
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)",
                   linecolor="rgba(255,255,255,0.12)", tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)",
                   linecolor="rgba(255,255,255,0.12)", tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#1B2340", font=dict(color=TEXT),
                        bordercolor="rgba(255,255,255,0.16)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
)
pio.templates.default = "sp555"


# --- Theme -----------------------------------------------------------------
# Only this block interpolates Python values. Everything below it is a plain
# string using the custom properties, which keeps the CSS free of the doubled
# braces an f-string would otherwise require.

_TOKENS = f"""
:root {{
  --ink: {INK};
  --ink-2: {INK_2};
  --surface: {SURFACE};
  --surface-2: {SURFACE_2};
  --line: {LINE};
  --line-2: {LINE_2};
  --text: {TEXT};
  --muted: {MUTED};
  --note: {NOTE};
  --faint: {FAINT};
  --purple: {PURPLE};
  --blue: {BLUE};
  --pink: {PINK};
  --cyan: {CYAN};
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
}}
"""

_CSS_BODY = """
/* --- Ambient background -------------------------------------------------- */

[data-testid="stApp"] {
  background:
    radial-gradient(920px 580px at 8% -12%, rgba(124, 92, 252, 0.46) 0%, transparent 62%),
    radial-gradient(820px 540px at 94% -4%, rgba(59, 130, 246, 0.34) 0%, transparent 58%),
    radial-gradient(760px 520px at 52% 112%, rgba(236, 72, 153, 0.24) 0%, transparent 64%),
    var(--ink);
  background-repeat: no-repeat;
}

[data-testid="stHeader"] { background: transparent; }

/* The header keeps its default absolute position and sits above the scrolling
   content, so clearing its background let table headers and cards scroll
   underneath the toolbar and print through it -- "Deploy" landed on top of a
   column heading. config.toml turns the toolbar off, which is the actual fix;
   this gives it a surface of its own for the cases where it comes back, such
   as a run started outside the project root, so that it masks what passes
   beneath rather than laying a band across the floating panel. */
[data-testid="stToolbar"] {
  right: 12px;
  top: 8px;
  padding: 3px 5px;
  border-radius: 12px;
  background: rgba(17, 23, 42, 0.82);
  backdrop-filter: blur(10px) saturate(130%);
  -webkit-backdrop-filter: blur(10px) saturate(130%);
  border: 1px solid var(--line);
  box-shadow: 0 10px 26px -18px rgba(0, 0, 0, 0.9);
}

/* --- Floating application window ---------------------------------------- */

[data-testid="stMainBlockContainer"] {
  max-width: 1240px;
  padding: 2.5rem 2.75rem 3.25rem;
  margin-top: 1.1rem;
  margin-bottom: 2.5rem;
  background: linear-gradient(158deg, rgba(26, 34, 60, 0.82) 0%, rgba(12, 17, 34, 0.89) 100%);
  backdrop-filter: blur(18px) saturate(135%);
  -webkit-backdrop-filter: blur(18px) saturate(135%);
  border: 1px solid var(--line);
  border-radius: 26px;
  box-shadow: 0 60px 130px -44px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  animation: sp-rise 0.7s var(--ease) both;
}

@keyframes sp-rise {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: none; }
}

/* On a phone the panel runs edge to edge, so its 44px of side padding is
   spending a quarter of the screen on nothing and the rounded corners fall
   outside the viewport. Charts are the tightest thing on the page and get the
   width back. */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] {
    padding: 1.5rem 1.1rem 2rem;
    margin-top: 0.5rem;
    border-radius: 0;
    border-left: none;
    border-right: none;
  }
}

/* --- Sidebar as application chrome --------------------------------------- */

[data-testid="stSidebar"] { background: transparent; }

[data-testid="stSidebarContent"] {
  margin: 1.1rem 0 1.1rem 0.85rem;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.018) 100%);
  backdrop-filter: blur(14px) saturate(135%);
  -webkit-backdrop-filter: blur(14px) saturate(135%);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 40px 90px -40px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebarNavItems"] { padding-top: 0.35rem; }

[data-testid="stSidebarNavLink"] {
  position: relative;
  border-radius: 11px;
  margin: 2px 8px;
  padding: 9px 12px;
  color: var(--muted);
  font-weight: 600;
  transition: color 0.25s var(--ease), background 0.25s var(--ease), transform 0.25s var(--ease);
}
[data-testid="stSidebarNavLink"]:hover {
  color: var(--text);
  background: var(--surface-2);
  transform: translateX(2px);
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
  color: #fff;
  background: linear-gradient(100deg, rgba(124, 92, 252, 0.34), rgba(59, 130, 246, 0.15));
  box-shadow: inset 0 0 0 1px rgba(148, 122, 255, 0.3), 0 10px 26px -16px rgba(124, 92, 252, 0.95);
}
[data-testid="stSidebarNavLink"][aria-current="page"]::before {
  content: "";
  position: absolute; left: -8px; top: 50%; transform: translateY(-50%);
  width: 3px; height: 20px; border-radius: 0 3px 3px 0;
  background: linear-gradient(var(--purple), var(--pink));
  box-shadow: 0 0 14px 1px rgba(124, 92, 252, 0.9);
}

/* --- Typography ---------------------------------------------------------- */

/* Streamlit styles its own headings with a more specific selector, so these
   have to be scoped to the block container to win the cascade. The gradient
   fill additionally needs -webkit-text-fill-color; setting `color` alone is
   overridden and the text renders solid. */

[data-testid="stMainBlockContainer"] h1 {
  font-weight: 800;
  letter-spacing: -0.033em;
  background: linear-gradient(96deg, #ffffff 24%, #b9a8ff 80%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 {
  font-weight: 700;
  letter-spacing: -0.018em;
  color: var(--text);
}

/* --- Metric cards -------------------------------------------------------- */

[data-testid="stMetric"] {
  position: relative;
  overflow: hidden;
  background: linear-gradient(155deg, rgba(255, 255, 255, 0.07) 0%, rgba(255, 255, 255, 0.022) 100%);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 18px 38px -22px rgba(0, 0, 0, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.07);
  transition: transform 0.32s var(--ease), box-shadow 0.32s var(--ease), border-color 0.32s var(--ease);
}
[data-testid="stMetric"]:hover {
  transform: translateY(-4px);
  border-color: rgba(148, 122, 255, 0.32);
  box-shadow: 0 30px 56px -24px rgba(0, 0, 0, 0.92), 0 0 42px -22px rgba(124, 92, 252, 0.8);
}
[data-testid="stMetricLabel"] p {
  color: var(--muted) !important;
  font-size: 0.8rem !important;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
}

/* Streamlit clips the label to one line. Upper-casing it and opening the
   tracking widens it by roughly a third, which is enough to push labels of
   ordinary length past the box: "Present every single day" arrived as
   "PRESENT EVERY SI...". Let it wrap instead -- the card has room in height. */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.35;
}
[data-testid="stMetricValue"] {
  font-size: 1.95rem !important;
  font-weight: 800;
  letter-spacing: -0.02em;
}

/* --- Gradient hero cards ------------------------------------------------- */

.grad-card {
  position: relative;
  overflow: hidden;
  border-radius: 18px;
  padding: 19px 21px 17px;
  min-height: 132px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  background: linear-gradient(158deg, rgba(255, 255, 255, 0.062) 0%, rgba(255, 255, 255, 0.02) 100%);
  border: 1px solid var(--line);
  box-shadow: 0 18px 40px -24px rgba(0, 0, 0, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  transition: transform 0.32s var(--ease), box-shadow 0.32s var(--ease), border-color 0.32s var(--ease);
}

/* The only place the accent hue appears at full strength. */
.grad-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--c1), var(--c2));
}

/* A wash of the same hue, kept low enough that white text stays legible. */
.grad-card::after {
  content: "";
  position: absolute;
  top: -45%; right: -22%;
  width: 72%; height: 130%;
  background: radial-gradient(circle, var(--c1) 0%, transparent 68%);
  opacity: 0.11;
  pointer-events: none;
}

.grad-card:hover {
  transform: translateY(-4px);
  border-color: rgba(148, 122, 255, 0.3);
  box-shadow: 0 30px 58px -26px rgba(0, 0, 0, 0.92), 0 0 40px -24px var(--c1);
}
.grad-card:hover::after { opacity: 0.17; }

/* --faint is too dark for 11-13px text on this surface (about 3.4:1). The
   label and note step down through --muted and --note instead, both of which
   clear 4.5:1, and the hierarchy is carried by size and weight. */
.grad-card .gc-label {
  position: relative; z-index: 1;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.11em;
  color: var(--muted);
}
.grad-card .gc-value {
  position: relative; z-index: 1;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.028em;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  margin: 10px 0 4px;
}
.grad-card .gc-note {
  position: relative; z-index: 1;
  font-size: 0.78rem;
  color: var(--note);
}

/* --- Charts, tables, callouts -------------------------------------------- */

/* The panel goes on the element container, not on the chart itself. Plotly
   writes its measured size onto [data-testid="stPlotlyChart"], so padding
   there is taken out of the chart rather than placed around it and every
   figure hung past the panel -- which is how a legend row came to print over
   the bottom edge. Padding one level up is simply inset instead. The figure
   ends up about 30px shorter than the height it asks for, which is a fair
   trade for the panel actually containing it. */
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
  background: linear-gradient(158deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.017) 100%);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px 16px 12px;
  box-shadow: 0 20px 44px -26px rgba(0, 0, 0, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* Streamlit paints the theme's backgroundColor onto the chart's <svg> and its
   secondaryBackgroundColor into the plot-area rect, and it does so on the
   client after the figure is handed over -- neither theme=None nor setting
   paper_bgcolor on the figure prevents it. Without these two rules every chart
   lays an opaque slab over the panel above and the glass is wasted. */
[data-testid="stPlotlyChart"] .main-svg { background: transparent !important; }
[data-testid="stPlotlyChart"] .bglayer .bg { fill: transparent !important; }

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 18px 40px -26px rgba(0, 0, 0, 0.85);
}

[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--surface-2);
  overflow: hidden;
}
[data-testid="stExpander"] summary:hover { color: var(--purple); }

[data-testid="stAlert"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 16px 36px -24px rgba(0, 0, 0, 0.8);
}

[data-testid="stVideo"] video, video { border-radius: 18px; }

hr { border-color: var(--line); }

code {
  color: #C4B5FD;
  background: rgba(124, 92, 252, 0.12);
  border-radius: 6px;
  padding: 0.12em 0.38em;
}

/* --- Controls ------------------------------------------------------------ */

/* Streamlit no longer marks these up with BaseWeb attributes, so the older
   [data-baseweb="tab"] and [data-baseweb="select"] rules matched nothing and
   both controls fell back to stock styling. The test ids and ARIA roles below
   are what the current build actually emits. */

[data-testid="stTabs"] [role="tablist"] {
  gap: 6px;
  border-bottom: 1px solid var(--line);
}
[data-testid="stTab"] {
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-bottom: none;
  border-radius: 11px 11px 0 0;
  padding: 9px 17px;
  color: var(--muted);
  font-weight: 600;
  transition: color 0.25s var(--ease), background 0.25s var(--ease);
}
[data-testid="stTab"]:hover { color: var(--text); background: rgba(255, 255, 255, 0.08); }
[data-testid="stTab"][aria-selected="true"] {
  color: #fff;
  background: linear-gradient(180deg, rgba(124, 92, 252, 0.34), rgba(124, 92, 252, 0.12));
  border-color: rgba(148, 122, 255, 0.34);
}

[data-testid="stSelectbox"] div[role="group"],
[data-testid="stMultiSelect"] div[role="group"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 11px !important;
  transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease);
}
[data-testid="stSelectbox"] div[role="group"]:focus-within,
[data-testid="stMultiSelect"] div[role="group"]:focus-within {
  border-color: rgba(148, 122, 255, 0.55) !important;
  box-shadow: 0 0 0 3px rgba(124, 92, 252, 0.18);
}

.stButton > button, [data-testid="stBaseButton-secondary"] {
  border-radius: 11px;
  border: 1px solid var(--line-2);
  background: var(--surface-2);
  font-weight: 600;
  transition: all 0.25s var(--ease);
}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: rgba(148, 122, 255, 0.5);
  box-shadow: 0 10px 24px -14px rgba(124, 92, 252, 1);
  transform: translateY(-1px);
}

/* --- Scrollbar ----------------------------------------------------------- */

::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  border: 3px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.22);
  background-clip: content-box;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
"""

_CSS = f"<style>{_TOKENS}{_CSS_BODY}</style>"


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


def require(df: pd.DataFrame | None, missing_hint: str) -> bool:
    """Show a build instruction instead of an empty page when data is absent."""
    if df is None or df.empty:
        st.info(f"Not built yet. Run `{missing_hint}` from the project root.")
        return False
    return True


# --- Rendering -------------------------------------------------------------


def chart(fig, **kwargs) -> None:
    """Render a Plotly figure full width on the app's own template.

    Two things have to be undone. st.plotly_chart defaults to theme="streamlit",
    which merges Streamlit's own theme over the figure and wins on everything
    but the colorway, so theme=None is what actually puts sp555 in charge.

    The backgrounds cannot be won here at all: Streamlit's frontend writes
    config.toml's backgroundColor and secondaryBackgroundColor onto the rendered
    chart whatever the figure or the template asks for, which left an opaque
    #0B1020 rectangle inside every translucent chart panel. Clearing that is the
    one job left to CSS -- see the stPlotlyChart rules in the theme.
    """
    st.plotly_chart(fig, theme=None, width="stretch", **kwargs)


def table(data, **kwargs) -> None:
    """Render a dataframe full width.

    Exists so the width argument is written once: use_container_width is
    deprecated and due for removal.
    """
    st.dataframe(data, width="stretch", **kwargs)


# --- Layout helpers --------------------------------------------------------


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric cards from (label, value, help) tuples."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items):
        column.metric(label, value, help=helptext)


def gradient_row(items: list[tuple[str, str, str]]) -> None:
    """Render headline figures as accent-topped cards from (label, value, note).

    The accent hue is handed to the CSS as custom properties rather than being
    painted onto the card, so the surface stays consistent with every other
    panel and the figure itself carries the emphasis.
    """
    columns = st.columns(len(items))
    for i, (column, (label, value, note)) in enumerate(zip(columns, items)):
        start, end = GRADIENTS[i % len(GRADIENTS)]
        column.markdown(
            f"""
            <div class="grad-card" style="--c1: {start}; --c2: {end};">
              <div class="gc-label">{label}</div>
              <div>
                <div class="gc-value">{value}</div>
                <div class="gc-note">{note}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
