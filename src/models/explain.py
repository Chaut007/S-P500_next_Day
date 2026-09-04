"""Feature importance and SHAP values for the fitted zoo.

Two different questions, kept apart on purpose:

*Permutation importance* asks how much a model's test error grows when one
column is shuffled. It works for any model, including the LSTM, because it only
needs predictions. It is the only importance measure here that is comparable
across the four families.

*SHAP* asks how each feature moved each individual prediction, which
permutation importance cannot answer. Exact TreeSHAP is used on XGBoost; the
other three families would need the sampling explainer, which on this many rows
costs far more than it tells us. The report therefore shows SHAP for the tree
model and permutation importance for everything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.models.evaluate import regression_metrics

log = get_logger(__name__)


def permutation_importance(
    fitted: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Increase in test MAE when each column is shuffled, averaged over repeats.

    Reported as a share of the baseline MAE so the four models -- whose error
    scales differ by an order of magnitude -- can be plotted together.
    """
    rng = np.random.default_rng(seed)
    truth = y_test.to_numpy(dtype="float64")

    baseline = regression_metrics(truth, fitted.predict(X_test))["mae"]

    rows: list[dict[str, Any]] = []
    for column in X_test.columns:
        deltas = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            deltas.append(regression_metrics(truth, fitted.predict(shuffled))["mae"])

        mean_mae = float(np.mean(deltas))
        rows.append({
            "model": fitted.name,
            "feature": column,
            "baseline_mae": baseline,
            "permuted_mae": mean_mae,
            "importance": mean_mae - baseline,
            "importance_pct": (mean_mae - baseline) / baseline * 100.0 if baseline else np.nan,
            "std": float(np.std(deltas)),
        })

    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    log.info("%s | top features by permutation importance: %s",
             fitted.name, ", ".join(out["feature"].head(5)))
    return out


def shap_values(
    fitted: Any,
    X_test: pd.DataFrame,
    sample: int = 400,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact TreeSHAP for the XGBoost model.

    Returns (long_values, summary):
        long_values -- one row per (row, feature) with its SHAP contribution
        summary     -- mean absolute SHAP per feature, the usual ranking
    """
    import shap

    if fitted.name != "XGBoost":
        raise ValueError(f"TreeSHAP expects the tree model, got {fitted.name}")

    frame = X_test
    if sample and len(frame) > sample:
        frame = frame.sample(n=sample, random_state=seed).sort_index()

    explainer = shap.TreeExplainer(fitted.estimator)
    values = explainer.shap_values(frame)

    long = pd.DataFrame(values, columns=frame.columns, index=frame.index)
    long = (
        long.reset_index(names="row")
        .melt(id_vars="row", var_name="feature", value_name="shap")
    )
    long["value"] = (
        frame.reset_index(names="row")
        .melt(id_vars="row", var_name="feature", value_name="value")["value"]
    )

    summary = (
        long.groupby("feature")["shap"]
        .agg(mean_abs_shap=lambda s: s.abs().mean(), mean_shap="mean")
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary["base_value"] = float(np.ravel(explainer.expected_value)[0])

    log.info("SHAP computed on %d rows | top: %s",
             len(frame), ", ".join(summary["feature"].head(5)))
    return long, summary
