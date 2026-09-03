"""Calendar alignment and cleaning.

One rule governs this module: gaps may only ever be filled forward. Back-filling
copies a future observation into the past, which is look-ahead bias that no
later validation step can detect.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)


def build_calendar(index_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading days of the index. This is the master calendar for everything."""
    calendar = pd.DatetimeIndex(pd.to_datetime(index_df["date"]).sort_values().unique())
    log.info("Trading calendar: %d days (%s to %s)",
             len(calendar), calendar.min().date(), calendar.max().date())
    return calendar


def clean_index(index_df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated index closes with no missing values."""
    df = index_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    before = len(df)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .dropna(subset=["date", "close"])
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("Index series is empty after cleaning; the cache may be corrupt")

    log.info("Index cleaned: %d -> %d rows", before, len(df))
    return df


def clean_constituent_prices(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: dict[str, Any] | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Align constituent closes to the trading calendar.

    Leading NaNs are preserved on purpose: a company that had not listed yet must
    stay absent from the ranking rather than inherit a later price.
    """
    cfg = cfg or load_config()
    max_missing = cfg["data"]["max_missing_ratio"]

    df = prices.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.reindex(calendar)

    # Forward fill short halts only. limit= stops a delisted ticker from being
    # carried flat for years.
    df = df.ffill(limit=ffill_limit)

    missing_ratio = df.isna().mean()
    keep = missing_ratio[missing_ratio <= max_missing].index
    dropped = df.shape[1] - len(keep)
    if dropped:
        log.warning("Dropped %d tickers with more than %.0f%% missing data",
                    dropped, max_missing * 100)

    df = df[keep]
    log.info("Constituent prices: %d dates x %d tickers", len(df), df.shape[1])
    return df


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    date_col: str = "date",
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Reindex a date-keyed frame onto the trading calendar, forward filling only.

    The bond market observes holidays the equity market does not, so FRED series
    have genuine gaps on days the index trades.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.tz_localize(None)
    out = (
        out.sort_values(date_col)
        .drop_duplicates(subset=date_col, keep="last")
        .set_index(date_col)
        .reindex(calendar)
        .ffill(limit=ffill_limit)
    )
    out.index.name = date_col
    return out.reset_index()
