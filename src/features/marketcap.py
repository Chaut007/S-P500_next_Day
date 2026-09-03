"""Market capitalisation and the daily top-N ranking.

The feature columns are rank slots, not companies: x1 is whatever firm is
largest on that date. That is what makes the design robust to constituents
dropping out of the top of the index, and it is why the ranking is recomputed
every single day.

Using market cap rather than share price for the feature values also removes a
discontinuity. Ranked by market cap but valued by price, two firms swapping
places would jump the column from, say, $150 to $600 for no economic reason.
Valued by market cap, a swap changes the column by almost nothing, because the
two firms are near-identical in size at the moment they cross.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.logger import get_logger

log = get_logger(__name__)

BILLION = 1e9


def compute_market_caps(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Return a date x ticker frame of market caps in USD billions.

    mcap(t) = shares_now * close(t), where close is split-adjusted but not
    dividend-adjusted, so both factors sit on today's share basis.
    """
    shares = universe.set_index("ticker")["shares"]
    common = [t for t in prices.columns if t in shares.index]

    missing = prices.shape[1] - len(common)
    if missing:
        log.warning("Ignoring %d priced tickers with no shares outstanding", missing)

    mcaps = prices[common].multiply(shares.loc[common], axis=1) / BILLION
    log.info("Market caps: %d dates x %d tickers (USD bn)", len(mcaps), mcaps.shape[1])
    return mcaps


def rank_top_n(
    mcaps: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank every date by market cap and return the top N.

    Returns (values, names):
        values -- columns x1..xN holding market caps in USD billions
        names  -- columns name_1..name_N holding the ticker occupying each slot

    `names` is not used by the model. It exists so the dashboard can show which
    company held each slot; recomputing it later would mean re-running the whole
    ranking step.
    """
    matrix = mcaps.to_numpy(dtype="float64")
    tickers = np.asarray(mcaps.columns)

    # argsort on the negated matrix gives descending order and pushes NaN last,
    # so tickers with no price on that date fall out of contention naturally.
    order = np.argsort(-matrix, axis=1, kind="stable")[:, :top_n]

    rows = np.arange(matrix.shape[0])[:, None]
    top_values = matrix[rows, order]
    top_names = tickers[order].astype(object)

    # Where a slot has no valid market cap, blank the name as well.
    invalid = np.isnan(top_values)
    top_names[invalid] = None

    value_cols = [f"x{i}" for i in range(1, top_n + 1)]
    name_cols = [f"name_{i}" for i in range(1, top_n + 1)]

    values = pd.DataFrame(top_values, index=mcaps.index, columns=value_cols)
    names = pd.DataFrame(top_names, index=mcaps.index, columns=name_cols)

    complete = (~invalid).all(axis=1).sum()
    log.info("Top-%d ranking: %d/%d dates have all slots filled",
             top_n, complete, len(values))
    return values, names


def build_top_n_table(
    mcaps: pd.DataFrame,
    top_n: int,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Wide table of x1..xN plus name_1..name_N, indexed by date.

    When `universe` is supplied the company names are attached as well, which is
    what the bar chart race labels the bars with.
    """
    values, names = rank_top_n(mcaps, top_n)
    table = values.join(names)

    if universe is not None:
        lookup = universe.set_index("ticker")["name"].to_dict()
        for i in range(1, top_n + 1):
            table[f"company_{i}"] = table[f"name_{i}"].map(lookup)

    table.index.name = "date"
    return table.reset_index()


def concentration_metrics(
    mcaps: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Daily concentration statistics computed inside the top N only.

    Everything here uses the top-N constituents alone, so it stays within the
    scope of the research question. A ratio against the whole 500 would be more
    powerful but would import information about the other 490 firms.
    """
    values, _ = rank_top_n(mcaps, top_n)
    arr = values.to_numpy(dtype="float64")

    total = np.nansum(arr, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = arr / total[:, None]

    out = pd.DataFrame(
        {
            "top_n_total": total,
            "share_1": weights[:, 0],
            "share_top3": np.nansum(weights[:, :3], axis=1),
            "hhi": np.nansum(weights**2, axis=1),
        },
        index=values.index,
    )
    out.index.name = "date"
    log.info("Concentration metrics computed for %d dates", len(out))
    return out.reset_index()
