"""Phase 6 -- the four-model comparison on a chronological 80/20 split.

AutoGluon, XGBoost, SVR and an LSTM are trained on the leading 80% of the
trading days and scored on the trailing 20%, all on feature set C at N = 10.
Permutation importance is computed for every model and TreeSHAP for XGBoost.

Run from the project root:
    python -m scripts.run_models
    python -m scripts.run_models --models XGBoost SVR
    python -m scripts.run_models --skip-explain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_config
from src.features.build_features import finalise, select_feature_set
from src.logger import get_logger
from src.models.evaluate import directional_accuracy, evaluate_with_baseline, log_metrics
from src.models.explain import permutation_importance, shap_values
from src.models.split import chronological_split
from src.models.zoo import BUILDERS, fit_all
from src.pipeline import build_dataset
from src.utils import load_table, save_table

log = get_logger("run_models")

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the four model families")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(BUILDERS), help="subset to run (default: all)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the dataset instead of reading data/processed")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-explain", action="store_true",
                        help="skip permutation importance and SHAP")
    return parser.parse_args()


def load_frame(args: argparse.Namespace, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (table, trend_cols, macro_cols), rebuilding only when asked."""
    if args.rebuild or args.force_download or not Path(DATASET_PATH).exists():
        dataset = build_dataset(force_download=args.force_download, cfg=cfg)
        save_table(dataset.table, DATASET_PATH)
        return dataset.table, dataset.trend_cols, dataset.macro_cols

    log.info("Reading cached dataset: %s", DATASET_PATH)
    table = load_table(DATASET_PATH)

    macro_cols = [c for c in table.columns
                  if c in set(cfg["data"]["fred_series"]) | {"gold_return"}]
    mcap_cols = {f"x{i}" for i in range(1, cfg["features"]["top_n"] + 1)}
    reserved = mcap_cols | set(macro_cols) | {"date", "target", "baseline_naive"}
    trend_cols = [c for c in table.columns if c not in reserved]
    return table, trend_cols, macro_cols


def main() -> int:
    args = parse_args()
    cfg = load_config()
    ensure_dirs()

    top_n = cfg["features"]["top_n"]
    set_name = cfg["features"]["best_set"]

    table, trend_cols, macro_cols = load_frame(args, cfg)
    feature_cols = select_feature_set(table, set_name, top_n, trend_cols, macro_cols, cfg)
    X, y, dates, baseline = finalise(table, feature_cols)

    train_idx, test_idx = chronological_split(dates, cfg)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    base = baseline.iloc[test_idx].to_numpy(dtype="float64")
    truth = y_test.to_numpy(dtype="float64")

    above = float((truth > y_train.max()).mean() * 100.0)
    log.info("Feature set %s | %d features | test rows above the training max: %.1f%%",
             set_name, len(feature_cols), above)

    fitted = fit_all(X_train, y_train, cfg, only=args.models)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for model in fitted:
        preds = model.predict(X_test)

        metrics = evaluate_with_baseline(truth, preds, base)
        metrics["directional_accuracy"] = directional_accuracy(truth, preds, base)
        metrics["model"] = model.name
        metrics["feature_set"] = set_name
        metrics["top_n"] = top_n
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        metrics["train_max"] = model.train_max
        metrics["pred_max"] = float(preds.max())
        metrics["actual_max"] = float(truth.max())
        # Positive means the model never reached the actual high: the ceiling.
        metrics["shortfall"] = metrics["actual_max"] - metrics["pred_max"]
        metrics["pct_test_above_train_max"] = above

        log_metrics(metrics, model.name)
        metric_rows.append(metrics)

        prediction_frames.append(pd.DataFrame({
            "date": dates.iloc[test_idx].to_numpy(),
            "model": model.name,
            "actual": truth,
            "predicted": preds,
            "baseline": base,
        }))

    scores = pd.DataFrame(metric_rows).sort_values("mae").reset_index(drop=True)
    save_table(scores, REPORTS_DIR / "model_scores.parquet")
    save_table(pd.concat(prediction_frames, ignore_index=True),
               REPORTS_DIR / "model_predictions.parquet")

    log.info("Test scores (best MAE first):\n%s",
             scores[["model", "mae", "rmse", "mape", "r2", "skill_mae",
                     "directional_accuracy", "shortfall"]].to_string(index=False))

    if args.skip_explain:
        log.info("Explainability skipped")
        return 0

    explain_cfg = cfg["explain"]
    seed = cfg["project"]["random_state"]

    importances = [
        permutation_importance(model, X_test, y_test,
                               repeats=explain_cfg["permutation_repeats"], seed=seed)
        for model in fitted
    ]
    if importances:
        save_table(pd.concat(importances, ignore_index=True),
                   REPORTS_DIR / "feature_importance.parquet")

    tree = next((m for m in fitted if m.name == "XGBoost"), None)
    if tree is None:
        log.warning("XGBoost was not run, so no SHAP values were computed")
        return 0

    long, summary = shap_values(tree, X_test, sample=explain_cfg["shap_sample"], seed=seed)
    save_table(long, REPORTS_DIR / "shap_values.parquet")
    save_table(summary, REPORTS_DIR / "shap_summary.parquet")

    log.info("SHAP ranking:\n%s", summary.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
