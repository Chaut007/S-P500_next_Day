"""Download and cache every raw input: equity prices, the index, gold, macro."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RAW_DIR, load_config
from src.logger import get_logger

log = get_logger(__name__)

PRICES_PATH = RAW_DIR / "constituent_prices.parquet"
INDEX_PATH = RAW_DIR / "index_prices.csv"
GOLD_PATH = RAW_DIR / "gold_prices.csv"
MACRO_PATH = RAW_DIR / "macro_fred.csv"


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _exclusive_end(end_date: str) -> str:
    """yfinance treats `end` as exclusive; shift a day so config dates are inclusive.

    Without this, an end_date of 2025-12-31 silently drops 31 December.
    """
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def download_constituent_prices(
    tickers: list[str],
    cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return a date x ticker frame of closes for the whole universe.

    `auto_adjust=False` keeps the split-adjusted, dividend-unadjusted `Close`,
    which is the series market cap must be built from.
    """
    import yfinance as yf

    cfg = cfg or load_config()
    data_cfg = cfg["data"]
    price_col = data_cfg["price_col"]

    if PRICES_PATH.exists() and not force:
        log.info("Reading cached constituent prices: %s", PRICES_PATH)
        return pd.read_parquet(PRICES_PATH)

    frames: list[pd.DataFrame] = []
    batches = _chunks(sorted(tickers), data_cfg["download_batch_size"])

    for i, batch in enumerate(batches, start=1):
        log.info("Downloading prices, batch %d/%d (%d tickers)", i, len(batches), len(batch))
        raw = yf.download(
            batch,
            start=data_cfg["start_date"],
            end=_exclusive_end(data_cfg["end_date"]),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            log.warning("Batch %d returned no data", i)
            continue

        # With several tickers yfinance returns (field, ticker) columns; with a
        # single surviving ticker it collapses to plain field names.
        if isinstance(raw.columns, pd.MultiIndex):
            if price_col not in raw.columns.get_level_values(0):
                log.warning("Batch %d has no '%s' level", i, price_col)
                continue
            closes = raw[price_col]
        else:
            closes = raw[[price_col]]
            closes.columns = batch[:1]

        frames.append(closes)

    if not frames:
        raise ValueError("No price data downloaded for any batch")

    prices = pd.concat(frames, axis=1)
    prices.index.name = "date"
    prices = prices.sort_index()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PATH)
    log.info("Saved prices: %d dates x %d tickers -> %s",
             len(prices), prices.shape[1], PRICES_PATH)
    return prices


def _download_single(
    ticker: str,
    path: Path,
    cfg: dict[str, Any],
    force: bool,
    label: str,
) -> pd.DataFrame:
    """Download one symbol into a two-column date/close frame."""
    import yfinance as yf

    if path.exists() and not force:
        log.info("Reading cached %s: %s", label, path)
        return pd.read_csv(path, parse_dates=["date"])

    price_col = cfg["data"]["price_col"]
    log.info("Downloading %s (%s)", label, ticker)
    raw = yf.download(
        ticker,
        start=cfg["data"]["start_date"],
        end=_exclusive_end(cfg["data"]["end_date"]),
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned nothing for {ticker}")

    # Flatten before writing: a MultiIndex header becomes two CSV header rows,
    # and reading that back turns the ticker row into data, making every column
    # a string.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df = raw.reset_index()[["Date", price_col]]
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Saved %s: %d rows -> %s", label, len(df), path)
    return df


def download_index(cfg: dict[str, Any] | None = None, force: bool = False) -> pd.DataFrame:
    """S&P 500 index close -- the prediction target."""
    cfg = cfg or load_config()
    return _download_single(cfg["data"]["index_ticker"], INDEX_PATH, cfg, force, "index")


def download_gold(cfg: dict[str, Any] | None = None, force: bool = False) -> pd.DataFrame:
    """Gold ETF close. GLD trades on the US equity calendar, unlike GC=F."""
    cfg = cfg or load_config()
    return _download_single(cfg["data"]["gold_ticker"], GOLD_PATH, cfg, force, "gold")


def download_macro(cfg: dict[str, Any] | None = None, force: bool = False) -> pd.DataFrame:
    """Daily Treasury series from FRED.

    These are published without revision, so using them as-of date t carries no
    look-ahead risk.
    """
    cfg = cfg or load_config()

    if MACRO_PATH.exists() and not force:
        log.info("Reading cached macro: %s", MACRO_PATH)
        return pd.read_csv(MACRO_PATH, parse_dates=["date"])

    from pandas_datareader import data as pdr

    series = cfg["data"]["fred_series"]
    log.info("Downloading FRED series: %s", ", ".join(series))
    raw = pdr.DataReader(
        series,
        "fred",
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
    )

    df = raw.reset_index()
    df.columns = ["date"] + list(raw.columns)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MACRO_PATH, index=False, encoding="utf-8")
    log.info("Saved macro: %d rows -> %s", len(df), MACRO_PATH)
    return df
