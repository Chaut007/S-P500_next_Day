"""Market-cap weight versus learned feature importance.

The second research question. Index weight is mechanical: a firm contributes in
proportion to its market cap. Learned importance is not -- it measures how much
the prediction degrades when a column is shuffled. The two are not expected to
line up, and the gap between them is the finding.

Spearman rather than Pearson, because the claim under test is that the
relationship is not linear; rank correlation stays valid either way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.logger import get_logger

log = get_logger(__name__)


def permutation_importance(
    predictor: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    num_shuffle_sets: int = 5,
) -> pd.DataFrame:
    """Permutation importance from a fitted AutoGluon predictor."""
    data = X_valid.copy()
    data["target"] = y_valid.to_numpy()

    raw = predictor.feature_importance(
        data=data,
        num_shuffle_sets=num_shuffle_sets,
        silent=True,
    )
    out = raw.reset_index(names="feature")[["feature", "importance"]]
    return out


def slot_weights(mcap_values: pd.DataFrame, valid_index: np.ndarray) -> pd.DataFrame:
    """Mean market-cap weight of each rank slot over the validation window.

    Weights are normalised inside the top-N block, matching the definition used
    everywhere else in the project.
    """
    block = mcap_values.iloc[valid_index]
    arr = block.to_numpy(dtype="float64")

    with np.errstate(invalid="ignore", divide="ignore"):
        weights = arr / np.nansum(arr, axis=1)[:, None]

    return pd.DataFrame(
        {
            "feature": list(block.columns),
            "weight": np.nanmean(weights, axis=0),
        }
    )


def compare_weight_and_importance(
    weights: pd.DataFrame,
    importances: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join the two series, normalise both, and measure rank agreement.

    Importance is rescaled to sum to one so the two columns are on a comparable
    footing; `gap` is then positive for slots that matter more to the model than
    their size alone would suggest.
    """
    merged = weights.merge(importances, on="feature", how="inner")
    if merged.empty:
        raise ValueError("No overlap between weight slots and importance features")

    total = merged["importance"].clip(lower=0).sum()
    merged["importance_norm"] = (
        merged["importance"].clip(lower=0) / total if total > 0 else np.nan
    )
    merged["gap"] = merged["importance_norm"] - merged["weight"]
    merged = merged.sort_values("weight", ascending=False).reset_index(drop=True)

    stats: dict[str, float] = {}
    if len(merged) >= 3:
        rho, p_value = spearmanr(merged["weight"], merged["importance_norm"])
        pearson = merged["weight"].corr(merged["importance_norm"], method="pearson")
        stats = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "pearson_r": float(pearson) if pd.notna(pearson) else np.nan,
            "n_slots": float(len(merged)),
        }
        log.info("Weight vs importance: Spearman rho=%.3f (p=%.4f), Pearson r=%.3f",
                 stats["spearman_rho"], stats["spearman_p"], stats["pearson_r"])

    return merged, stats
