"""Regression metrics, always reported next to a naive baseline.

Predicting an index *level* flatters every metric. The S&P 500 moves about 0.7%
on an average day, while its level roughly tripled across the study window, so
the variance the model is scored against is dominated by the trend rather than by
anything it actually predicted. R-squared near 0.99 and MAPE under 1% are the
expected result of a model that has learned almost nothing useful.

The only honest reference is the naive forecast "tomorrow equals today". Every
function in this module reports it alongside the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.logger import get_logger

log = get_logger(__name__)

EPSILON = 1e-12


def mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSPE as a percentage. Not provided by scikit-learn or AutoGluon."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denominator = np.where(np.abs(y_true) < EPSILON, np.nan, y_true)
    return float(np.nanmean(((y_true - y_pred) / denominator) ** 2) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The five metrics the study reports, plus RMSE for readability."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        # scikit-learn returns a fraction; the study reports percent.
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "mspe": mean_squared_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_with_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> dict[str, float]:
    """Model metrics, baseline metrics, and the skill ratio between them.

    `skill_mae` below 1 means the model beats the naive forecast. Above 1 means
    a constant "no change" prediction would have been more accurate.
    """
    model = regression_metrics(y_true, y_pred)
    baseline = regression_metrics(y_true, baseline_pred)

    merged: dict[str, float] = dict(model)
    merged.update({f"baseline_{k}": v for k, v in baseline.items()})

    merged["skill_mae"] = model["mae"] / baseline["mae"] if baseline["mae"] else np.nan
    merged["skill_rmse"] = model["rmse"] / baseline["rmse"] if baseline["rmse"] else np.nan
    merged["beats_baseline"] = float(model["mae"] < baseline["mae"])

    return merged


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous: np.ndarray,
) -> float:
    """Share of days where the predicted direction of change is correct.

    `previous` is the index level at t, so the sign compared is the move from
    today to tomorrow rather than the level itself.
    """
    true_move = np.sign(np.asarray(y_true, dtype="float64") - previous)
    pred_move = np.sign(np.asarray(y_pred, dtype="float64") - previous)
    return float(np.mean(true_move == pred_move))


def log_metrics(metrics: dict[str, float], title: str) -> None:
    """Print a metric dictionary in a readable block."""
    log.info("--- %s ---", title)
    for key in ("mae", "mse", "rmse", "mape", "mspe", "r2"):
        if key in metrics:
            log.info("  %-16s %14.4f", key, metrics[key])
    for key in ("baseline_mae", "baseline_rmse", "skill_mae", "directional_accuracy"):
        if key in metrics:
            log.info("  %-16s %14.4f", key, metrics[key])

    if metrics.get("beats_baseline", 1.0) == 0.0:
        log.warning("Model does not beat the naive forecast; the headline "
                    "R-squared is carried by the trend, not by prediction")


def summarise_folds(rows: list[dict[str, float]]) -> pd.DataFrame:
    """Mean and standard deviation across folds.

    The standard deviation matters as much as the mean: a model that is strong on
    the 2022 bear market and weak whenever the index makes new highs is showing
    the tree-extrapolation failure, not random noise.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric = df.select_dtypes("number")

    summary = pd.concat(
        [numeric.mean().rename("mean"), numeric.std().rename("std")],
        axis=1,
    ).reset_index(names="metric")
    return summary
