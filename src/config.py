"""Project paths and configuration loading.

Every module resolves paths through this file. Relative paths break as soon as
the working directory changes, so nothing else should build paths by hand.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = PROJECT_ROOT / "assets"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_env(key: str, default: str | None = None) -> str | None:
    """Read a value from .env / the environment."""
    return os.getenv(key, default)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into. Idempotent."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        LOGS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        ASSETS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
