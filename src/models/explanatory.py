"""Explanatory power, measured separately from forecasting skill.

The two questions are not the same and the project needs both.

*Forecasting* asks whether the top ten predict tomorrow's index level better
than assuming no change. They do not, and they cannot: the naive forecast is
handed today's index level, which is by far the best single predictor of
tomorrow's, while the model is denied it by design. Losing that comparison says
almost nothing about whether the constituents matter.

*Explanation* asks how much of the index the top ten account for at all. Fitting
inside a period and scoring inside the same period answers that directly, and it
is the question the research is actually about.

The gap between the two is the interesting part: strong explanatory power that
does not survive being projected forward means the relationship itself is
drifting, and that drift is measurable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.logger import get_logger

log = get_logger(__name__)


def within_period_fit(
    table: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    date_col: str = "date",
    freq: str = "YE",
) -> pd.DataFrame:
    """Fit and score a linear model inside each period.

    This is in-sample by construction, which is the point: it measures how much
    of the index the constituents account for, not how well the relationship
    projects into a future it has not seen.
    """
    df = table.dropna(subset=feature_cols + [target_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    rows: list[dict[str, Any]] = []
    for period, group in df.groupby(pd.Grouper(key=date_col, freq=freq)):
        if len(group) < len(feature_cols) + 2:
            continue

        X = group[feature_cols]
        y = group[target_col]

        model = LinearRegression().fit(X, y)
        predicted = model.predict(X)

        block_total = X.sum(axis=1)
        rows.append(
            {
                "period": period,
                "year": period.year,
                "n": len(group),
                "r2": float(r2_score(y, predicted)),
                "mape": float(np.mean(np.abs(predicted - y) / y) * 100.0),
                "index_over_block": float((y / block_total).mean()),
                "block_total": float(block_total.mean()),
                "index_level": float(y.mean()),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        log.info("Within-period R² ranges %.4f to %.4f across %d periods",
                 result["r2"].min(), result["r2"].max(), len(result))
    return result


def coefficient_drift(periods: pd.DataFrame) -> dict[str, float]:
    """Quantify how far the index-to-block relationship moves over the window."""
    if periods.empty:
        return {}

    ratio = periods["index_over_block"]
    first, last = ratio.iloc[0], ratio.iloc[-1]

    stats = {
        "ratio_first": float(first),
        "ratio_last": float(last),
        "ratio_drift_multiple": float(first / last) if last else np.nan,
        "block_growth": float(periods["block_total"].iloc[-1]
                              / periods["block_total"].iloc[0]),
        "index_growth": float(periods["index_level"].iloc[-1]
                              / periods["index_level"].iloc[0]),
        "r2_mean": float(periods["r2"].mean()),
        "r2_min": float(periods["r2"].min()),
        "mape_mean": float(periods["mape"].mean()),
    }
    stats["relative_growth"] = stats["block_growth"] / stats["index_growth"]

    log.info("Index / block ratio drifted %.4f -> %.4f (%.2fx)",
             stats["ratio_first"], stats["ratio_last"],
             stats["ratio_drift_multiple"])
    log.info("Block grew %.2fx against the index's %.2fx, a factor of %.2f",
             stats["block_growth"], stats["index_growth"],
             stats["relative_growth"])
    return stats


def rolling_explanatory_power(
    table: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    date_col: str = "date",
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """R² of a linear fit over a rolling window, for a continuous view.

    Yearly buckets are convenient to report but hide when the relationship
    weakens mid-year, which a rolling window shows.
    """
    df = table.dropna(subset=feature_cols + [target_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for end in range(window, len(df) + 1, step):
        chunk = df.iloc[end - window : end]
        X = chunk[feature_cols]
        y = chunk[target_col]

        model = LinearRegression().fit(X, y)
        rows.append(
            {
                "date": chunk[date_col].iloc[-1],
                "r2": float(r2_score(y, model.predict(X))),
                "index_over_block": float((y / X.sum(axis=1)).mean()),
            }
        )

    result = pd.DataFrame(rows)
    log.info("Rolling explanatory power computed at %d points", len(result))
    return result
