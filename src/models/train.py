"""AutoGluon training across expanding-window folds.

A linear learner is forced into the model list on purpose. Gradient boosted trees
and random forests predict by averaging the training targets that fall in a leaf,
so they cannot return a value above the highest target they were trained on. The
index roughly triples over the study window, which means every fold whose
validation year sets new highs is asking the tree models for a number they are
structurally incapable of producing. A linear model has no such ceiling, and it
also happens to be the theoretically correct form here: the index is a weighted
sum of constituent market caps.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, load_config
from src.logger import get_logger
from src.models.evaluate import (
    directional_accuracy,
    evaluate_with_baseline,
    log_metrics,
)
from src.models.split import Fold, iter_folds
from src.utils import set_seed

log = get_logger(__name__)

LABEL = "target"


def build_hyperparameters(cfg: dict[str, Any] | None = None) -> dict[str, dict]:
    """Explicit model list so LR can never be dropped by a preset."""
    cfg = cfg or load_config()
    included = cfg["model"]["included_model_types"]

    if "LR" not in included:
        raise ValueError(
            "included_model_types must contain 'LR'. Tree models cannot "
            "extrapolate beyond the training target range, which every "
            "new-high validation year requires."
        )

    return {name: {} for name in included}


def _chronological_tuning_split(
    train: pd.DataFrame,
    holdout_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the most recent slice of the training block for AutoGluon tuning.

    AutoGluon would otherwise carve out its internal validation set at random,
    which scatters future rows into the fitting data. Handing it an explicit,
    time-ordered tuning set keeps the ordering intact.
    """
    cut = int(len(train) * (1.0 - holdout_ratio))
    cut = max(cut, 1)
    return train.iloc[:cut], train.iloc[cut:]


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    fold_name: str,
    set_name: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray]:
    """Fit one fold and return (predictor, validation predictions)."""
    from autogluon.tabular import TabularPredictor

    cfg = cfg or load_config()
    model_cfg = cfg["model"]
    set_seed(cfg["project"]["random_state"])

    train = X_train.copy()
    train[LABEL] = y_train.to_numpy()

    fit_data, tuning_data = _chronological_tuning_split(train)

    model_path = MODELS_DIR / "autogluon" / f"{set_name}_{fold_name}"
    if model_path.exists():
        shutil.rmtree(model_path, ignore_errors=True)

    predictor = TabularPredictor(
        label=LABEL,
        problem_type="regression",
        eval_metric=model_cfg["eval_metric"],
        path=str(model_path),
        verbosity=1,
    )
    predictor.fit(
        train_data=fit_data,
        tuning_data=tuning_data,
        hyperparameters=build_hyperparameters(cfg),
        time_limit=model_cfg["time_limit"],
    )

    predictions = predictor.predict(X_valid).to_numpy()
    return predictor, predictions


def evaluate_all_models(
    predictor: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    baseline: np.ndarray,
) -> pd.DataFrame:
    """Score every individual model on the validation fold, not just the ensemble.

    This is the experiment the design predicted. AutoGluon selects its ensemble
    weights on a tuning split carved from the end of the training block, and
    that split sits inside the range the trees have already seen. Trees
    therefore look strongest exactly where the choice is made, then hit their
    ceiling on a validation year that sets new index highs. Comparing the
    linear model against the trees on the real validation window is the only
    way to see it.
    """
    truth = y_valid.to_numpy(dtype="float64")
    rows: list[dict[str, Any]] = []

    for model_name in predictor.model_names():
        try:
            preds = predictor.predict(X_valid, model=model_name).to_numpy()
        except Exception as exc:  # noqa: BLE001 - a failed model must not stop the run
            log.warning("Could not score %s: %s", model_name, exc)
            continue

        metrics = evaluate_with_baseline(truth, preds, baseline)
        metrics["model"] = model_name
        metrics["pred_max"] = float(np.max(preds))
        metrics["actual_max"] = float(np.max(truth))
        rows.append(metrics)

    return pd.DataFrame(rows)


def run_ablation_set(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    baseline: pd.Series,
    folds: list[Fold],
    set_name: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train and evaluate one feature set across every fold.

    Returns (metrics, predictions, leaderboard, per_model).
    """
    cfg = cfg or load_config()

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    leaderboard_frames: list[pd.DataFrame] = []
    per_model_frames: list[pd.DataFrame] = []

    for fold, X_train, y_train, X_valid, y_valid in iter_folds(X, y, folds):
        log.info("=== Feature set %s | %s | %d train / %d valid rows ===",
                 set_name, fold.name, len(X_train), len(X_valid))

        predictor, preds = train_fold(
            X_train, y_train, X_valid, fold.name, set_name, cfg
        )

        base = baseline.iloc[fold.valid_idx].to_numpy(dtype="float64")
        truth = y_valid.to_numpy(dtype="float64")

        metrics = evaluate_with_baseline(truth, preds, base)
        metrics["directional_accuracy"] = directional_accuracy(truth, preds, base)
        metrics["feature_set"] = set_name
        metrics["fold"] = fold.name
        metrics["year"] = fold.year
        metrics["n_train"] = len(X_train)
        metrics["n_valid"] = len(X_valid)
        metrics["best_model"] = predictor.model_best

        log_metrics(metrics, f"{set_name} / {fold.name}")
        metric_rows.append(metrics)

        prediction_frames.append(
            pd.DataFrame({
                "date": dates.iloc[fold.valid_idx].to_numpy(),
                "feature_set": set_name,
                "fold": fold.name,
                "actual": truth,
                "predicted": preds,
                "baseline": base,
            })
        )

        per_model = evaluate_all_models(predictor, X_valid, y_valid, base)
        if not per_model.empty:
            per_model["feature_set"] = set_name
            per_model["fold"] = fold.name
            per_model["year"] = fold.year
            per_model_frames.append(per_model)

            ranked = per_model.sort_values("mae")[["model", "mae", "r2", "skill_mae"]]
            log.info("Per-model validation scores for %s:\n%s",
                     fold.name, ranked.to_string(index=False))

        try:
            board = predictor.leaderboard(silent=True).copy()
            board["feature_set"] = set_name
            board["fold"] = fold.name
            leaderboard_frames.append(board)
        except Exception as exc:  # noqa: BLE001 - leaderboard is a nice-to-have
            log.warning("Leaderboard unavailable for %s: %s", fold.name, exc)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    leaderboard_df = (
        pd.concat(leaderboard_frames, ignore_index=True)
        if leaderboard_frames else pd.DataFrame()
    )
    per_model_df = (
        pd.concat(per_model_frames, ignore_index=True)
        if per_model_frames else pd.DataFrame()
    )

    return metrics_df, predictions_df, leaderboard_df, per_model_df
