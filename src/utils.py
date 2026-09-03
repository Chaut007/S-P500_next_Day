"""Small shared helpers: seeding, artifact IO, dataframe persistence."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.logger import get_logger

log = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Pin the random and numpy seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    log.debug("Random seed set to %d", seed)


def timestamp() -> str:
    """Timestamp string used to name artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_artifact(obj: Any, name: str, directory: Path = MODELS_DIR) -> Path:
    """Persist an object with a timestamp so earlier runs are not overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{timestamp()}.joblib"
    joblib.dump(obj, path)
    log.info("Saved artifact: %s", path)
    return path


def load_artifact(path: str | Path) -> Any:
    """Load an object written by save_artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    log.info("Loaded artifact: %s", path)
    return joblib.load(path)


def latest_artifact(pattern: str = "*.joblib", directory: Path = MODELS_DIR) -> Path:
    """Return the most recently modified artifact matching a glob pattern."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")
    return files[-1]


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a dataframe to parquet, falling back to CSV when pyarrow is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=index)
    except (ImportError, ValueError) as exc:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=index, encoding="utf-8")
        log.warning("Parquet unavailable (%s); wrote CSV instead", exc)
    log.info("Wrote %s (%d rows x %d cols)", path.name, len(df), df.shape[1])
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a table written by save_table, trying parquet then CSV."""
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    alternative = path.with_suffix(".csv" if path.suffix == ".parquet" else ".parquet")
    if alternative.exists():
        return load_table(alternative)

    raise FileNotFoundError(f"Table not found: {path}")
