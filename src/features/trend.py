"""Trend features derived from the top-N block.

Every function here is causal: the value at row t uses observations up to and
including t and nothing later. Forbidden anywhere in this file:

    rolling(..., center=True)   -- pulls future observations into the window
    bfill() / interpolate()     -- fills a gap with data that had not happened
    shift(-n)                   -- only legal when constructing the target

These features are ratios, so unlike the raw market cap levels they are roughly
stationary and remain usable by tree models on unseen date ranges.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)


def top_n_total(values: pd.DataFrame) -> pd.Series:
    """S(t): combined market cap of the top-N block."""
    return values.sum(axis=1, skipna=True)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Percentage change of S(t) against its value `window` trading days ago."""
    return series.pct_change(periods=window)


def ma_ratio(series: pd.Series, window: int) -> pd.Series:
    """Position of S(t) relative to its own trailing moving average."""
    ma = series.rolling(window=window, min_periods=window).mean()
    return series / ma - 1.0


def build_trend_features(
    values: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Momentum, moving-average position and within-block concentration.

    `values` is the x1..xN frame produced by marketcap.rank_top_n.
    """
    cfg = cfg or load_config()
    trend_cfg = cfg["features"]["trend"]

    total = top_n_total(values)
    out = pd.DataFrame(index=values.index)

    for window in trend_cfg["momentum_windows"]:
        out[f"mom_{window}"] = momentum(total, window)

    ma_window = trend_cfg["ma_window"]
    out[f"ma_ratio_{ma_window}"] = ma_ratio(total, ma_window)

    # Concentration inside the block. Both are scale free, so they stay
    # comparable across a decade in which the block itself grew several fold.
    arr = values.to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = arr / np.nansum(arr, axis=1)[:, None]

    out["share_1"] = weights[:, 0]
    out["hhi"] = np.nansum(weights**2, axis=1)

    log.info("Trend features built: %s", ", ".join(out.columns))
    return out
