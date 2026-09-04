"""Expanding-window splitting.

This module deliberately exposes no random splitter. With ordered data a shuffled
split lets rows from the future train the model, which cannot happen at
prediction time, and the resulting scores are meaningless.

Folds grow rather than slide: every fold trains on all history before its
validation year, which is what "expanding window" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold."""

    year: int
    train_idx: np.ndarray
    valid_idx: np.ndarray

    @property
    def name(self) -> str:
        return f"val_{self.year}"


def expanding_year_folds(
    dates: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> list[Fold]:
    """Build one fold per validation year.

    A `gap` of `horizon` rows is removed from the end of each training block so
    the last training row's target cannot overlap the validation period.
    """
    cfg = cfg or load_config()
    years = cfg["split"]["validation_years"]
    gap = cfg["split"]["gap"]

    dates = pd.to_datetime(dates).reset_index(drop=True)
    year_of = dates.dt.year.to_numpy()
    positions = np.arange(len(dates))

    folds: list[Fold] = []
    for year in years:
        train_mask = year_of < year
        valid_mask = year_of == year

        if not train_mask.any() or not valid_mask.any():
            log.warning("Skipping validation year %d: no data on one side", year)
            continue

        train_idx = positions[train_mask]
        if gap:
            train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx[:0]

        if len(train_idx) == 0:
            log.warning("Skipping validation year %d: gap consumed the training set", year)
            continue

        valid_idx = positions[valid_mask]
        folds.append(Fold(year=year, train_idx=train_idx, valid_idx=valid_idx))

        log.info("Fold %s | train %d rows (%s to %s) | valid %d rows (%s to %s)",
                 f"val_{year}",
                 len(train_idx), dates.iloc[train_idx[0]].date(), dates.iloc[train_idx[-1]].date(),
                 len(valid_idx), dates.iloc[valid_idx[0]].date(), dates.iloc[valid_idx[-1]].date())

    if not folds:
        raise ValueError("No usable folds; check validation_years against the data range")

    return folds


def chronological_split(
    dates: pd.Series,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ordered rows into a leading train block and a trailing test block.

    Returns (train_idx, test_idx) as positions into `dates`, which is assumed
    already sorted. The cut is by position rather than by date so the ratio
    holds regardless of how many rows the feature warm-up dropped.

    There is no shuffling here for the same reason there is no random splitter
    above: the tail block has to be the future, or the score means nothing.
    """
    cfg = cfg or load_config()
    ratio = cfg["split"]["test_ratio"]

    if not 0.0 < ratio < 1.0:
        raise ValueError(f"split.test_ratio must lie in (0, 1), got {ratio}")

    dates = pd.to_datetime(dates).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted before splitting")

    n = len(dates)
    cut = int(n * (1.0 - ratio))
    if cut < 1 or cut >= n:
        raise ValueError(f"test_ratio {ratio} leaves an empty block for {n} rows")

    positions = np.arange(n)
    train_idx, test_idx = positions[:cut], positions[cut:]

    log.info("Chronological split | train %d rows (%s to %s) | test %d rows (%s to %s)",
             len(train_idx), dates.iloc[0].date(), dates.iloc[cut - 1].date(),
             len(test_idx), dates.iloc[cut].date(), dates.iloc[-1].date())

    return train_idx, test_idx


def iter_folds(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
) -> Iterator[tuple[Fold, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Yield (fold, X_train, y_train, X_valid, y_valid) for each fold."""
    for fold in folds:
        yield (
            fold,
            X.iloc[fold.train_idx],
            y.iloc[fold.train_idx],
            X.iloc[fold.valid_idx],
            y.iloc[fold.valid_idx],
        )
