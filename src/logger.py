"""Project-wide logging: stdout plus a rotating file in logs/.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Loaded %d rows", len(df))
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR, get_env, load_config

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Guards against adding duplicate handlers when get_logger is called repeatedly
# with the same name -- the classic cause of every log line appearing twice.
_configured: set[str] = set()


def _ensure_utf8_stdout() -> None:
    """Force stdout to UTF-8.

    Windows consoles default to a legacy code page, which mangles any non-ASCII
    output. The file handler already sets its own encoding.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass  # stdout was redirected somewhere that does not support this


def _resolve_level() -> int:
    """Priority: LOG_LEVEL in .env > config.yaml > INFO."""
    level_name = get_env("LOG_LEVEL")
    if not level_name:
        try:
            level_name = load_config()["logging"]["level"]
        except (FileNotFoundError, KeyError, TypeError):
            level_name = "INFO"
    return getattr(logging, str(level_name).upper(), logging.INFO)


def get_logger(name: str = "sp555", level: int | None = None) -> logging.Logger:
    """Return a configured logger writing to logs/run_YYYY-MM-DD.log."""
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(level if level is not None else _resolve_level())
    logger.propagate = False  # stop records bubbling up to root and printing twice

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    _ensure_utf8_stdout()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_cfg = load_config().get("logging", {})
        file_handler = RotatingFileHandler(
            LOGS_DIR / f"run_{date.today():%Y-%m-%d}.log",
            maxBytes=log_cfg.get("max_bytes", 5 * 1024 * 1024),
            backupCount=log_cfg.get("backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:  # still want console logging if the file is unwritable
        logger.warning("File logging disabled (%s); using console only", exc)

    _configured.add(name)
    return logger
