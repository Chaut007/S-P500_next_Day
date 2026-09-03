"""Phase 5 -- the two supporting research questions.

1. How many stocks are enough? Sweep N over features.top_n_sweep and plot the
   R-squared curve to find where extra constituents stop paying for themselves.
2. Does index weight predict learned importance? Compare market-cap weight
   against permutation importance with Spearman rank correlation.

Run from the project root:
    python -m scripts.run_experiments
    python -m scripts.run_experiments --skip-sweep
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import REPORTS_DIR, ensure_dirs, load_config
from src.features.build_features import finalise, select_feature_set
from src.logger import get_logger
from src.models.importance import (
    compare_weight_and_importance,
    permutation_importance,
    slot_weights,
)
from src.models.split import expanding_year_folds
from src.models.train import train_fold
from src.pipeline import build_dataset
from src.utils import save_table

log = get_logger("run_experiments")

# The N sweep answers "why ten?", so it uses the plain market-cap block only.
# Adding trend or macro columns would blur what the extra constituents buy.
SWEEP_SET = "A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supporting experiments")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="skip the N = 5/10/20 comparison")
    parser.add_argument("--skip-importance", action="store_true",
                        help="skip the weight vs importance analysis")
    parser.add_argument("--skip-explanatory", action="store_true",
                        help="skip the within-period explanatory power analysis")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def run_explanatory_analysis(cfg: dict, force_download: bool):
    """Measure how much of the index the top ten account for, period by period.

    Kept separate from the forecasting results because it answers the actual
    research question. A model that loses to the naive forecast can still
    explain almost all of the index level; those are different claims.
    """
    from src.features.build_features import mcap_feature_columns
    from src.models.explanatory import (
        coefficient_drift,
        rolling_explanatory_power,
        within_period_fit,
    )

    top_n = cfg["features"]["top_n"]
    dataset = build_dataset(top_n=top_n, force_download=force_download, cfg=cfg)
    feature_cols = mcap_feature_columns(top_n)

    periods = within_period_fit(dataset.table, feature_cols)
    stats = coefficient_drift(periods)
    rolling = rolling_explanatory_power(dataset.table, feature_cols)

    log.info("Within-year explanatory power:\n%s",
             periods[["year", "n", "r2", "mape", "index_over_block"]]
             .to_string(index=False))

    return periods, rolling, pd.DataFrame([stats]) if stats else pd.DataFrame()


def run_top_n_sweep(cfg: dict, force_download: bool) -> pd.DataFrame:
    """Train feature set A at each N and record accuracy per fold."""
    rows: list[dict] = []

    for top_n in cfg["features"]["top_n_sweep"]:
        log.info("### Top-N sweep: N = %d ###", top_n)
        dataset = build_dataset(top_n=top_n, force_download=force_download, cfg=cfg)

        feature_cols = select_feature_set(
            dataset.table, SWEEP_SET, top_n, dataset.trend_cols, dataset.macro_cols, cfg
        )
        X, y, dates, baseline = finalise(dataset.table, feature_cols)
        folds = expanding_year_folds(dates, cfg)

        from src.models.evaluate import evaluate_with_baseline
        from src.models.split import iter_folds

        for fold, X_train, y_train, X_valid, y_valid in iter_folds(X, y, folds):
            _, preds = train_fold(
                X_train, y_train, X_valid, fold.name, f"N{top_n}", cfg
            )
            metrics = evaluate_with_baseline(
                y_valid.to_numpy(dtype="float64"),
                preds,
                baseline.iloc[fold.valid_idx].to_numpy(dtype="float64"),
            )
            metrics.update({"top_n": top_n, "fold": fold.name, "year": fold.year})
            rows.append(metrics)
            log.info("N=%d %s -> r2=%.5f mae=%.2f skill=%.3f",
                     top_n, fold.name, metrics["r2"], metrics["mae"],
                     metrics["skill_mae"])

    results = pd.DataFrame(rows)
    curve = (
        results.groupby("top_n")[["r2", "mae", "mape", "mspe", "skill_mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    curve.columns = ["_".join(c).strip("_") for c in curve.columns]

    log.info("R-squared vs N:\n%s", curve.to_string(index=False))
    return results, curve


def run_importance_analysis(cfg: dict, force_download: bool) -> pd.DataFrame:
    """Compare market-cap weight against permutation importance, per fold."""
    top_n = cfg["features"]["top_n"]
    dataset = build_dataset(top_n=top_n, force_download=force_download, cfg=cfg)

    feature_cols = select_feature_set(
        dataset.table, "A", top_n, dataset.trend_cols, dataset.macro_cols, cfg
    )
    X, y, dates, _ = finalise(dataset.table, feature_cols)
    folds = expanding_year_folds(dates, cfg)

    from src.models.split import iter_folds

    frames: list[pd.DataFrame] = []
    stat_rows: list[dict] = []

    for fold, X_train, y_train, X_valid, y_valid in iter_folds(X, y, folds):
        log.info("### Importance: %s ###", fold.name)
        predictor, _ = train_fold(
            X_train, y_train, X_valid, fold.name, "importance", cfg
        )

        importances = permutation_importance(predictor, X_valid, y_valid)
        weights = slot_weights(X, fold.valid_idx)
        merged, stats = compare_weight_and_importance(weights, importances)

        merged["fold"] = fold.name
        merged["year"] = fold.year
        frames.append(merged)

        if stats:
            stats.update({"fold": fold.name, "year": fold.year})
            stat_rows.append(stats)

    return pd.concat(frames, ignore_index=True), pd.DataFrame(stat_rows)


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    if not args.skip_sweep:
        results, curve = run_top_n_sweep(cfg, args.force_download)
        save_table(results, REPORTS_DIR / "rsq_vs_n_folds.parquet")
        save_table(curve, REPORTS_DIR / "rsq_vs_n.parquet")

    if not args.skip_explanatory:
        periods, rolling, drift = run_explanatory_analysis(cfg, args.force_download)
        save_table(periods, REPORTS_DIR / "explanatory_by_period.parquet")
        save_table(rolling, REPORTS_DIR / "explanatory_rolling.parquet")
        if not drift.empty:
            save_table(drift, REPORTS_DIR / "explanatory_drift.parquet")

    if not args.skip_importance:
        importance, stats = run_importance_analysis(cfg, args.force_download)
        save_table(importance, REPORTS_DIR / "importance.parquet")
        if not stats.empty:
            save_table(stats, REPORTS_DIR / "importance_stats.parquet")
            log.info("Spearman by fold:\n%s", stats.to_string(index=False))

    log.info("Experiments complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
