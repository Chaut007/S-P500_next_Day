"""Phase 1 -- download everything and build the modelling table.

Run from the project root:
    python -m scripts.run_data
    python -m scripts.run_data --force-download

Outputs land in data/processed/ and are the only thing the dashboard reads.
"""

from __future__ import annotations

import argparse
import sys

from src.config import PROCESSED_DIR, ensure_dirs, load_config
from src.logger import get_logger
from src.pipeline import build_dataset
from src.utils import save_table

log = get_logger("run_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the S&P 555 dataset")
    parser.add_argument("--force-download", action="store_true",
                        help="ignore caches and refetch every source")
    parser.add_argument("--top-n", type=int, default=None,
                        help="override features.top_n from the config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    dataset = build_dataset(
        top_n=args.top_n,
        force_download=args.force_download,
        cfg=cfg,
    )

    save_table(dataset.table, PROCESSED_DIR / "dataset.parquet")
    save_table(dataset.top_n_table, PROCESSED_DIR / "top10_daily.parquet")
    save_table(dataset.concentration, PROCESSED_DIR / "concentration.parquet")
    save_table(dataset.universe, PROCESSED_DIR / "universe.parquet")

    # Market caps for the whole universe drive the bar chart race, so they are
    # written with the date as a column rather than an index.
    caps = dataset.market_caps.copy()
    caps.index.name = "date"
    save_table(caps.reset_index(), PROCESSED_DIR / "mcap_daily.parquet")

    log.info("Dataset ready: %d rows, top_n=%d", len(dataset.table), dataset.top_n)
    log.info("Trend columns: %s", ", ".join(dataset.trend_cols))
    log.info("Macro columns: %s", ", ".join(dataset.macro_cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
