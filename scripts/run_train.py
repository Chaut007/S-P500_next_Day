"""Phase 4 -- expanding-window training and evaluation of feature sets A/B/C.

Run from the project root:
    python -m scripts.run_train
    python -m scripts.run_train --sets A B
    python -m scripts.run_train --rebuild

Writes per-fold metrics, predictions and leaderboards into reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_config
from src.features.build_features import finalise, select_feature_set
from src.logger import get_logger
from src.models.evaluate import summarise_folds
from src.models.split import expanding_year_folds
from src.models.train import run_ablation_set
from src.pipeline import build_dataset
from src.utils import load_table, save_table

log = get_logger("run_train")

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the ablation sets")
    parser.add_argument("--sets", nargs="+", default=None,
                        help="feature sets to run (default: all in the config)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the dataset instead of reading data/processed")
    parser.add_argument("--force-download", action="store_true",
                        help="implies --rebuild and refetches every source")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """Return (table, trend_cols, macro_cols, top_n) from cache or a fresh build."""
    top_n = cfg["features"]["top_n"]

    if args.rebuild or args.force_download or not Path(DATASET_PATH).exists():
        dataset = build_dataset(force_download=args.force_download, cfg=cfg)
        save_table(dataset.table, DATASET_PATH)
        return dataset.table, dataset.trend_cols, dataset.macro_cols, dataset.top_n

    log.info("Reading cached dataset: %s", DATASET_PATH)
    table = load_table(DATASET_PATH)

    # Recover the block membership from the column names so the cached table can
    # be used without rebuilding every source.
    mcap_cols = {f"x{i}" for i in range(1, top_n + 1)}
    macro_cols = [c for c in table.columns
                  if c in set(cfg["data"]["fred_series"]) | {"gold_return"}]
    reserved = mcap_cols | set(macro_cols) | {"date", "target", "baseline_naive"}
    trend_cols = [c for c in table.columns if c not in reserved]

    return table, trend_cols, macro_cols, top_n


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    table, trend_cols, macro_cols, top_n = load_dataset(args, cfg)
    set_names = args.sets or list(cfg["features"]["sets"].keys())

    log.info("Feature sets to run: %s", ", ".join(set_names))
    log.info("Trend columns: %s", ", ".join(trend_cols))
    log.info("Macro columns: %s", ", ".join(macro_cols))

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_leaderboards: list[pd.DataFrame] = []
    all_per_model: list[pd.DataFrame] = []

    for set_name in set_names:
        feature_cols = select_feature_set(
            table, set_name, top_n, trend_cols, macro_cols, cfg
        )
        X, y, dates, baseline = finalise(table, feature_cols)
        folds = expanding_year_folds(dates, cfg)

        metrics, predictions, leaderboard, per_model = run_ablation_set(
            X, y, dates, baseline, folds, set_name, cfg
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)
        if not leaderboard.empty:
            all_leaderboards.append(leaderboard)
        if not per_model.empty:
            all_per_model.append(per_model)

        summary = summarise_folds(metrics.to_dict("records"))
        summary.insert(0, "feature_set", set_name)
        log.info("Set %s mean across folds:\n%s", set_name,
                 summary[summary["metric"].isin(
                     ["mae", "mape", "mspe", "r2", "skill_mae", "directional_accuracy"]
                 )].to_string(index=False))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    save_table(metrics_df, REPORTS_DIR / "cv_results.parquet")
    save_table(pd.concat(all_predictions, ignore_index=True),
               REPORTS_DIR / "predictions.parquet")
    if all_leaderboards:
        save_table(pd.concat(all_leaderboards, ignore_index=True),
                   REPORTS_DIR / "leaderboard.parquet")

    if all_per_model:
        per_model_df = pd.concat(all_per_model, ignore_index=True)
        save_table(per_model_df, REPORTS_DIR / "model_comparison.parquet")

        # Linear versus trees is the design question this project set out to
        # test, so it gets its own summary rather than being buried in a table.
        by_model = (
            per_model_df.groupby("model")[["mae", "r2", "skill_mae"]]
            .mean()
            .sort_values("mae")
            .reset_index()
        )
        log.info("Mean validation score by model (all sets, all folds):\n%s",
                 by_model.to_string(index=False))

    # Cross-set comparison, which is the headline table of the ablation.
    comparison = (
        metrics_df
        .groupby("feature_set")[["mae", "mape", "mspe", "r2", "skill_mae",
                                 "directional_accuracy"]]
        .mean()
        .reset_index()
    )
    save_table(comparison, REPORTS_DIR / "ablation_summary.parquet")
    log.info("Ablation summary:\n%s", comparison.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
