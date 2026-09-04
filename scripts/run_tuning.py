"""Phase 7 -- grid search for SVR, XGBoost and the LSTM.

Searches each grid on the training block, split by TimeSeriesSplit, and writes
the winning settings to reports/best_params.json. The test block is not touched
here; `run_models.py --tuned` picks the file up afterwards.

Run from the project root:
    python -m scripts.run_tuning
    python -m scripts.run_tuning --models SVR XGBoost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_config
from src.features.build_features import finalise, select_feature_set
from src.logger import get_logger
from src.models.split import chronological_split
from src.models.tuning import TUNERS
from src.utils import load_table, save_table

log = get_logger("run_tuning")

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
BEST_PARAMS_PATH = REPORTS_DIR / "best_params.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search the tunable models")
    parser.add_argument("--models", nargs="+", default=None, choices=list(TUNERS),
                        help="subset to tune (default: all three)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    if not Path(DATASET_PATH).exists():
        log.error("No dataset at %s; run `python -m scripts.run_data` first",
                  DATASET_PATH)
        return 1

    top_n = cfg["features"]["top_n"]
    set_name = cfg["features"]["best_set"]

    table = load_table(DATASET_PATH)
    macro_cols = [c for c in table.columns
                  if c in set(cfg["data"]["fred_series"]) | {"gold_return"}]
    mcap_cols = {f"x{i}" for i in range(1, top_n + 1)}
    reserved = mcap_cols | set(macro_cols) | {"date", "target", "baseline_naive"}
    trend_cols = [c for c in table.columns if c not in reserved]

    feature_cols = select_feature_set(table, set_name, top_n, trend_cols, macro_cols, cfg)
    X, y, dates, _ = finalise(table, feature_cols)

    train_idx, _ = chronological_split(dates, cfg)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    log.info("Tuning on the training block only: %d rows, %d features",
             len(X_train), len(feature_cols))

    names = args.models or list(TUNERS)
    best_params: dict[str, dict] = {}
    tables: list[pd.DataFrame] = []

    if BEST_PARAMS_PATH.exists():
        best_params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))

    for name in names:
        log.info("=== %s ===", name)
        best, results = TUNERS[name](X_train, y_train, cfg)
        best_params[name] = best
        tables.append(results)

    # Merge with whatever is already on disk, dropping only the models just
    # searched. Writing `tables` alone would erase the results of an earlier
    # run that tuned a different subset, exactly as --models LSTM once did.
    if tables:
        fresh = pd.concat(tables, ignore_index=True)
        results_path = REPORTS_DIR / "tuning_results.parquet"
        if results_path.exists():
            previous = load_table(results_path)
            kept = previous[~previous["model"].isin(fresh["model"].unique())]
            fresh = pd.concat([kept, fresh], ignore_index=True)
        save_table(fresh.sort_values(["model", "cv_mae"]).reset_index(drop=True),
                   results_path)

    BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    log.info("Wrote %s", BEST_PARAMS_PATH)

    for name, params in best_params.items():
        log.info("  %-9s %s", name, params)

    return 0


if __name__ == "__main__":
    sys.exit(main())
