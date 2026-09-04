"""Grid search for SVR, XGBoost and the LSTM.

Everything here runs on the training block alone, split further by
`TimeSeriesSplit`. The test block is never touched: tuning on it would pick the
settings that happen to suit the 20% being reported, and the reported score
would then be the best of many rather than an estimate of anything.

The two scikit-learn families go through `GridSearchCV` on the estimators built
in `zoo.py`, so the thing being searched is the thing finally trained. The LSTM
has no scikit-learn interface, so its grid is walked by hand over the same
folds, with the scalers refitted inside each one.

Scoring is MAE, matching the metric the study reports.
"""

from __future__ import annotations

import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.logger import get_logger
from src.models.evaluate import regression_metrics
from src.models.zoo import build_svr, build_xgboost, fit_lstm

log = get_logger(__name__)


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _sklearn_search(
    name: str,
    estimator: Any,
    grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run GridSearchCV and return (best params, every combination scored)."""
    combos = _grid_size(grid)
    log.info("%s | %d combinations x %d folds = %d fits",
             name, combos, n_splits, combos * n_splits)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
        refit=False,
    )

    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "model": name,
        "params": results["params"].astype(str),
        "cv_mae": -results["mean_test_score"],
        "cv_mae_std": results["std_test_score"],
        "rank": results["rank_test_score"],
    }).sort_values("cv_mae").reset_index(drop=True)

    # GridSearchCV strips the pipeline prefixes only on the estimator, not on
    # the reported params, so they are cleaned here for the report.
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}

    log.info("%s | best CV MAE %.3f in %.0fs | %s",
             name, table["cv_mae"].iloc[0], elapsed, best)
    return best, table


def tune_svr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Search C, gamma and epsilon for the RBF kernel."""
    tune_cfg = cfg["tuning"]["svr"]
    # The estimator is a TransformedTargetRegressor wrapping a Pipeline, so the
    # search keys have to address the SVR through both layers.
    grid = {f"regressor__svr__{k}": v for k, v in tune_cfg["grid"].items()}
    return _sklearn_search(
        "SVR", build_svr({}), grid, X_train, y_train, tune_cfg["n_splits"]
    )


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Search depth, learning rate, trees and subsampling."""
    tune_cfg = cfg["tuning"]["xgboost"]
    seed = cfg["project"]["random_state"]
    return _sklearn_search(
        "XGBoost", build_xgboost({}, seed), tune_cfg["grid"],
        X_train, y_train, tune_cfg["n_splits"],
    )


def tune_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Walk the LSTM grid by hand over the same expanding folds.

    Each candidate is refitted from scratch on every fold, including its
    scalers and its early stopping, so no fold informs another.
    """
    tune_cfg = cfg["tuning"]["lstm"]
    grid = tune_cfg["grid"]
    n_splits = tune_cfg["n_splits"]

    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in product(*grid.values())]
    log.info("LSTM | %d combinations x %d folds = %d fits",
             len(combos), n_splits, len(combos) * n_splits)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = list(splitter.split(X_train))

    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        started = time.perf_counter()
        fold_maes: list[float] = []

        for train_idx, valid_idx in folds:
            fitted = fit_lstm(
                X_train.iloc[train_idx], y_train.iloc[train_idx], cfg, params
            )
            preds = fitted.predict(X_train.iloc[valid_idx])
            truth = y_train.iloc[valid_idx].to_numpy(dtype="float64")
            fold_maes.append(regression_metrics(truth, preds)["mae"])

        rows.append({
            "model": "LSTM",
            "params": str(params),
            "cv_mae": float(np.mean(fold_maes)),
            "cv_mae_std": float(np.std(fold_maes)),
        })
        log.info("LSTM | %d/%d %s -> CV MAE %.3f (%.0fs)",
                 i, len(combos), params, rows[-1]["cv_mae"],
                 time.perf_counter() - started)

    table = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    best = combos[int(np.argmin([r["cv_mae"] for r in rows]))]
    log.info("LSTM | best CV MAE %.3f | %s", table["cv_mae"].iloc[0], best)
    return best, table


TUNERS = {
    "SVR": tune_svr,
    "XGBoost": tune_xgboost,
    "LSTM": tune_lstm,
}
