"""Shared helpers for the Streamlit dashboard.

The dashboard is a reader. Every table it shows is produced offline by the
scripts in scripts/; recomputing a 500-ticker ranking on page load would make it
unusable. Anything missing is reported as a prompt to run the relevant script
rather than silently rendering an empty chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit puts the entry script's folder on sys.path, not the project root,
# so `import src...` needs help.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

PALETTE = {
    "A": "#4C78A8",
    "B": "#F58518",
    "C": "#54A24B",
    "actual": "#333333",
    "predicted": "#4C78A8",
    "baseline": "#E45756",
}


def page_config(title: str) -> None:
    st.set_page_config(
        page_title=f"S&P 555 — {title}",
        page_icon="📈",
        layout="wide",
    )


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


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of st.metric cards from (label, value, help) tuples."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items):
        column.metric(label, value, help=helptext)
