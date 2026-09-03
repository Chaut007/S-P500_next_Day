"""Assemble the modelling table and the three ablation feature sets.

Layout of one row:

    date | x1 .. xN | trend .. | macro .. | target
           ^ market cap of the k-th largest firm on `date`
                                            ^ index close on the next trading day

The target is the only place in the project where a forward shift is allowed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)

TARGET_COL = "target"
DATE_COL = "date"

# Index level at t. Held out of every feature set on purpose -- it is carried
# through the pipeline only so evaluation can score the naive "tomorrow equals
# today" forecast. Never add it to a feature list.
BASELINE_COL = "baseline_naive"


def make_target(
    index_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: dict[str, Any] | None = None,
) -> pd.Series:
    """Index close shifted `horizon` trading days back, indexed by the calendar.

    Row t therefore carries the index level at t + horizon, which is the value
    the features at t are asked to predict. The final `horizon` rows become NaN
    and are dropped downstream.
    """
    cfg = cfg or load_config()
    horizon = cfg["target"]["horizon"]

    closes = (
        index_df.set_index(DATE_COL)["close"]
        .reindex(calendar)
        .astype("float64")
    )
    target = closes.shift(-horizon)
    target.name = TARGET_COL
    target.index.name = DATE_COL

    log.info("Target built: index close at t+%d (%d non-null rows)",
             horizon, int(target.notna().sum()))
    return target


def make_baseline(
    index_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Index close at t, the naive forecast for t+1."""
    baseline = (
        index_df.set_index(DATE_COL)["close"]
        .reindex(calendar)
        .astype("float64")
    )
    baseline.name = BASELINE_COL
    baseline.index.name = DATE_COL
    return baseline


def mcap_feature_columns(top_n: int) -> list[str]:
    return [f"x{i}" for i in range(1, top_n + 1)]


def assemble(
    mcap_values: pd.DataFrame,
    trend_features: pd.DataFrame,
    macro_features: pd.DataFrame,
    target: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    """Join every block on the trading calendar into a single wide frame."""
    table = (
        mcap_values
        .join(trend_features, how="left")
        .join(macro_features, how="left")
        .join(target, how="left")
        .join(baseline, how="left")
    )
    table.index.name = DATE_COL
    return table.reset_index()


def select_feature_set(
    table: pd.DataFrame,
    set_name: str,
    top_n: int,
    trend_cols: list[str],
    macro_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve an ablation set name (A / B / C) into a list of column names."""
    cfg = cfg or load_config()
    blocks = cfg["features"]["sets"][set_name]

    columns: list[str] = []
    if "mcap" in blocks:
        columns += mcap_feature_columns(top_n)
    if "trend" in blocks:
        columns += trend_cols
    if "macro" in blocks:
        columns += macro_cols

    leaked = {TARGET_COL, BASELINE_COL} & set(columns)
    if leaked:
        raise ValueError(f"Feature set {set_name} would leak the target: {leaked}")

    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(f"Feature set {set_name} refers to missing columns: {missing}")

    return columns


def finalise(
    table: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Drop incomplete rows and split into (X, y, dates, baseline).

    Rows are lost at both ends: the first ~20 because the rolling windows are not
    yet full, and the last `horizon` because the future index level is unknown.
    Neither may be imputed.
    """
    needed = feature_cols + [TARGET_COL, BASELINE_COL, DATE_COL]
    before = len(table)

    clean = table[needed].dropna().sort_values(DATE_COL).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No complete rows remain; check the feature pipeline")

    log.info("Dataset finalised: %d -> %d rows, %d features (%s to %s)",
             before, len(clean), len(feature_cols),
             clean[DATE_COL].min().date(), clean[DATE_COL].max().date())

    return (
        clean[feature_cols],
        clean[TARGET_COL],
        clean[DATE_COL],
        clean[BASELINE_COL],
    )
