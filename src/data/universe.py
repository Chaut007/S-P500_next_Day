"""S&P 500 constituent list and current shares outstanding.

Shares outstanding are only available as a current snapshot, so historical market
cap is approximated as `shares_now * split_adjusted_close(t)`. That is consistent
because both sides are expressed on today's share basis, but it understates
companies that have bought back a large fraction of their float over the study
window. See the limitations section of README.md.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.config import RAW_DIR, load_config
from src.logger import get_logger

log = get_logger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

CONSTITUENTS_PATH = RAW_DIR / "sp500_constituents.csv"
SHARES_PATH = RAW_DIR / "sp500_shares.csv"


def _to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia writes share classes as BRK.B; Yahoo expects BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def _read_wikipedia_table() -> pd.DataFrame:
    """Fetch the constituent table with a browser User-Agent.

    pandas.read_html goes through urllib, whose default User-Agent Wikipedia
    rejects with HTTP 403. Fetching the HTML ourselves and parsing the string
    avoids that.
    """
    import io

    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
    }
    response = requests.get(WIKI_URL, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    for table in tables:
        if {"Symbol", "Security"}.issubset(table.columns):
            return table

    raise ValueError("Could not find the constituent table on the Wikipedia page")


def fetch_constituents(force: bool = False) -> pd.DataFrame:
    """Return the current S&P 500 membership as ticker / name / sector.

    This is the *current* list applied to the whole history, which introduces
    survivorship bias: companies that dropped out of the index before today are
    never considered for the top-N ranking.
    """
    if CONSTITUENTS_PATH.exists() and not force:
        log.info("Reading cached constituents: %s", CONSTITUENTS_PATH)
        return pd.read_csv(CONSTITUENTS_PATH)

    log.info("Downloading constituent table from Wikipedia")
    raw = _read_wikipedia_table()

    df = pd.DataFrame(
        {
            "ticker": raw["Symbol"].map(_to_yahoo_symbol),
            "name": raw["Security"].astype(str).str.strip(),
            "sector": raw["GICS Sector"].astype(str).str.strip(),
        }
    ).drop_duplicates(subset="ticker")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSTITUENTS_PATH, index=False, encoding="utf-8")
    log.info("Saved %d constituents to %s", len(df), CONSTITUENTS_PATH)
    return df


def fetch_shares_outstanding(
    tickers: list[str],
    force: bool = False,
    pause: float = 0.0,
) -> pd.DataFrame:
    """Return current shares outstanding per ticker.

    Uses `fast_info` first because it is a single lightweight request; `info` is
    only consulted when fast_info comes back empty. Tickers that fail are
    reported and dropped rather than aborting the run.
    """
    import yfinance as yf

    if SHARES_PATH.exists() and not force:
        cached = pd.read_csv(SHARES_PATH)
        missing = set(tickers) - set(cached["ticker"])
        if not missing:
            log.info("Reading cached shares outstanding: %s", SHARES_PATH)
            return cached
        log.info("Cache is missing %d tickers; refetching those", len(missing))
        tickers = sorted(missing)
        previous = cached
    else:
        previous = pd.DataFrame(columns=["ticker", "shares"])

    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        shares = None
        try:
            fast = yf.Ticker(ticker).fast_info
            shares = fast.get("shares") if hasattr(fast, "get") else None
        except Exception:  # noqa: BLE001 - yfinance raises a wide variety of errors
            shares = None

        if not shares:
            try:
                shares = yf.Ticker(ticker).info.get("sharesOutstanding")
            except Exception:  # noqa: BLE001
                shares = None

        if shares:
            records.append({"ticker": ticker, "shares": float(shares)})
        else:
            failures.append(ticker)

        if i % 50 == 0:
            log.info("Shares outstanding: %d/%d fetched", i, len(tickers))
        if pause:
            time.sleep(pause)

    fresh = pd.DataFrame(records)
    result = (
        pd.concat([previous, fresh], ignore_index=True)
        .drop_duplicates(subset="ticker", keep="last")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SHARES_PATH, index=False, encoding="utf-8")

    if failures:
        log.warning("No shares outstanding for %d tickers: %s",
                    len(failures), ", ".join(failures[:15]))
    log.info("Shares outstanding available for %d tickers", len(result))
    return result


def get_universe(force: bool = False, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return ticker / name / sector / shares for every usable constituent."""
    cfg = cfg or load_config()
    constituents = fetch_constituents(force=force)
    shares = fetch_shares_outstanding(constituents["ticker"].tolist(), force=force)

    universe = constituents.merge(shares, on="ticker", how="inner")
    universe = universe[universe["shares"] > 0].reset_index(drop=True)

    log.info("Universe ready: %d tickers", len(universe))
    return universe
