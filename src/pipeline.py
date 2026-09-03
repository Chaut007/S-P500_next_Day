"""End-to-end dataset construction, shared by every entry point.

Keeping this in one place means `run_data`, `run_train` and `run_experiments`
cannot drift into building subtly different tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import load_config
from src.data.load import (
    download_constituent_prices,
    download_gold,
    download_index,
    download_macro,
)
from src.data.preprocess import build_calendar, clean_constituent_prices, clean_index
from src.data.universe import get_universe
from src.features.build_features import assemble, make_baseline, make_target
from src.features.macro import build_macro_features
from src.features.marketcap import (
    build_top_n_table,
    compute_market_caps,
    concentration_metrics,
    rank_top_n,
)
from src.features.trend import build_trend_features
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class Dataset:
    """Everything downstream steps need, built once."""

    table: pd.DataFrame
    mcap_values: pd.DataFrame
    top_n_table: pd.DataFrame
    concentration: pd.DataFrame
    market_caps: pd.DataFrame
    universe: pd.DataFrame
    calendar: pd.DatetimeIndex
    trend_cols: list[str] = field(default_factory=list)
    macro_cols: list[str] = field(default_factory=list)
    top_n: int = 10


def build_dataset(
    top_n: int | None = None,
    force_download: bool = False,
    cfg: dict[str, Any] | None = None,
) -> Dataset:
    """Download, clean and assemble the full modelling table."""
    cfg = cfg or load_config()
    top_n = top_n or cfg["features"]["top_n"]

    log.info("Building dataset | top_n=%d | %s to %s",
             top_n, cfg["data"]["start_date"], cfg["data"]["end_date"])

    # 1. Index first: its trading days define the master calendar.
    index_clean = clean_index(download_index(cfg, force=force_download))
    calendar = build_calendar(index_clean)

    # 2. Universe and constituent prices.
    universe = get_universe(force=force_download, cfg=cfg)
    prices_raw = download_constituent_prices(
        universe["ticker"].tolist(), cfg=cfg, force=force_download
    )
    prices = clean_constituent_prices(prices_raw, calendar, cfg=cfg)

    # 3. Market caps and the daily ranking.
    market_caps = compute_market_caps(prices, universe)
    mcap_values, _ = rank_top_n(market_caps, top_n)
    top_n_table = build_top_n_table(market_caps, top_n, universe)
    concentration = concentration_metrics(market_caps, top_n)

    # 4. Feature blocks.
    trend = build_trend_features(mcap_values, cfg=cfg)
    macro = build_macro_features(
        download_macro(cfg, force=force_download),
        download_gold(cfg, force=force_download),
        calendar,
        cfg=cfg,
    )

    # 5. Target and the naive reference.
    target = make_target(index_clean, calendar, cfg=cfg)
    baseline = make_baseline(index_clean, calendar)

    table = assemble(mcap_values, trend, macro, target, baseline)

    log.info("Dataset table: %d rows x %d columns", len(table), table.shape[1])

    return Dataset(
        table=table,
        mcap_values=mcap_values,
        top_n_table=top_n_table,
        concentration=concentration,
        market_caps=market_caps,
        universe=universe,
        calendar=calendar,
        trend_cols=list(trend.columns),
        macro_cols=list(macro.columns),
        top_n=top_n,
    )
