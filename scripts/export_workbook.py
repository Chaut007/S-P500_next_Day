"""Phase 8 -- collect every table into one Excel workbook.

The study's tables live in two places and two formats: the CSV exports under
data/processed and the parquet artifacts under reports. That is convenient for
the pipeline and inconvenient for anyone who just wants to read the results, so
this writes all of them into a single workbook with a contents sheet.

Every sheet is whatever the pipeline already wrote, so the workbook cannot
disagree with the dashboard. The one exception is `dataset_daily`, which joins
the modelling table to the ticker that held each rank slot -- a join, not a
recomputation, and the only place the two are visible side by side.

The daily and the long views are both here on purpose. `dataset_daily` has one
row per trading day and is the shape the models actually see;
`top10_ranking_daily` has one row per company per day and is the shape you need
to follow a single firm. Neither substitutes for the other.

Run from the project root:
    python -m scripts.export_workbook
    python -m scripts.export_workbook --output reports/for_submission.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.logger import get_logger
from src.utils import load_table

log = get_logger("export_workbook")

DEFAULT_OUTPUT = REPORTS_DIR / "sp555_all_tables.xlsx"

# Sheet name -> (path, what it holds). Ordered so the workbook reads like the
# study: source data, then the split, then results, then explanation.
SHEETS: dict[str, tuple[Path, str]] = {
    "top10_ranking_daily": (  # noqa: E501 - see build_dataset_daily for the per-day view
        PROCESSED_DIR / "top10_ranking_daily.csv",
        "One row per company per day: rank, close price and market cap",
    ),
    "close_prices_top10": (
        PROCESSED_DIR / "close_prices_top10.csv",
        "Daily close for the 23 tickers that ever held a top-ten slot",
    ),
    "index_close": (
        PROCESSED_DIR / "index_close.csv",
        "S&P 500 close, the series being predicted",
    ),
    "model_scores": (
        REPORTS_DIR / "model_scores.parquet",
        "Four models scored on the 80/20 test block, default settings",
    ),
    "model_scores_tuned": (
        REPORTS_DIR / "model_scores_tuned.parquet",
        "The same models after grid search",
    ),
    "model_predictions": (
        REPORTS_DIR / "model_predictions.parquet",
        "Per-day predictions on the test block, with the naive baseline",
    ),
    "model_predictions_tuned": (
        REPORTS_DIR / "model_predictions_tuned.parquet",
        "Per-day predictions from the tuned models",
    ),
    "tuning_results": (
        REPORTS_DIR / "tuning_results.parquet",
        "Every grid search combination and its cross-validated MAE",
    ),
    "feature_importance": (
        REPORTS_DIR / "feature_importance.parquet",
        "Permutation importance for all four models",
    ),
    "shap_summary": (
        REPORTS_DIR / "shap_summary.parquet",
        "Mean absolute SHAP contribution per feature (XGBoost)",
    ),
    "shap_values": (
        REPORTS_DIR / "shap_values.parquet",
        "Per-row SHAP contributions behind the summary",
    ),
    "explanatory_by_period": (
        REPORTS_DIR / "explanatory_by_period.parquet",
        "Within-year R-squared of the index on the ten market caps",
    ),
    "explanatory_drift": (
        REPORTS_DIR / "explanatory_drift.parquet",
        "How far the index-to-block ratio moves across the decade",
    ),
    "explanatory_rolling": (
        REPORTS_DIR / "explanatory_rolling.parquet",
        "The same relationship on a rolling one-year window",
    ),
    "cv_results": (
        REPORTS_DIR / "cv_results.parquet",
        "Expanding-window folds for feature sets A/B/C",
    ),
    "ablation_summary": (
        REPORTS_DIR / "ablation_summary.parquet",
        "Feature set comparison that selected set C",
    ),
    "model_comparison": (
        REPORTS_DIR / "model_comparison.parquet",
        "Per-model scores inside AutoGluon, by fold",
    ),
    "importance_weight_vs": (
        REPORTS_DIR / "importance.parquet",
        "Market-cap weight against learned importance, per slot",
    ),
    "importance_stats": (
        REPORTS_DIR / "importance_stats.parquet",
        "Spearman correlation between weight and importance, per fold",
    ),
}

# Excel refuses sheet names longer than this.
MAX_SHEET_NAME = 31


def build_dataset_daily() -> pd.DataFrame | None:
    """One row per trading day: the modelling table plus who held each slot.

    dataset.parquet carries x1..x10 as market caps but not the tickers behind
    them, because the models never see a ticker. For reading, the two belong
    together -- otherwise a row says the largest firm was worth 464bn without
    saying which firm that was.
    """
    try:
        dataset = load_table(PROCESSED_DIR / "dataset.parquet")
        top10 = load_table(PROCESSED_DIR / "top10_daily.parquet")
    except FileNotFoundError:
        return None

    dataset["date"] = pd.to_datetime(dataset["date"])
    top10["date"] = pd.to_datetime(top10["date"])

    name_cols = [c for c in top10.columns if c.startswith("name_")]
    merged = dataset.merge(
        top10[["date"] + name_cols], on="date", how="left", validate="one_to_one"
    )

    # Interleave each slot's ticker with its value so the pairs read together.
    ordered = ["date"]
    for i in range(1, len(name_cols) + 1):
        ordered += [f"name_{i}", f"x{i}"]
    ordered += [c for c in merged.columns if c not in ordered]

    return merged[ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write every table to one workbook")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    too_long = [name for name in SHEETS if len(name) > MAX_SHEET_NAME]
    if too_long:
        raise ValueError(f"Sheet names over {MAX_SHEET_NAME} chars: {too_long}")

    loaded: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    descriptions: dict[str, str] = {name: text for name, (_, text) in SHEETS.items()}
    sources: dict[str, str] = {name: path.name for name, (path, _) in SHEETS.items()}

    # The per-day view leads the workbook: it is the shape the models see.
    daily = build_dataset_daily()
    if daily is None:
        missing.append("dataset_daily")
        log.warning("Skipping dataset_daily: dataset.parquet has not been built")
    else:
        loaded["dataset_daily"] = daily
        descriptions["dataset_daily"] = (
            "One row per trading day: the 19 model features, the target, "
            "and the ticker holding each rank slot"
        )
        sources["dataset_daily"] = "dataset.parquet + top10_daily.parquet"

    for name, (path, _) in SHEETS.items():
        try:
            frame = load_table(path)
        except FileNotFoundError:
            missing.append(name)
            log.warning("Skipping %s: %s has not been built", name, path.name)
            continue
        loaded[name] = frame

    if not loaded:
        log.error("No tables found. Run the pipeline scripts first.")
        return 1

    contents = pd.DataFrame(
        [
            {
                "sheet": name,
                "rows": len(frame),
                "columns": frame.shape[1],
                "description": descriptions[name],
                "source": sources[name],
            }
            for name, frame in loaded.items()
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        contents.to_excel(writer, sheet_name="contents", index=False)
        for name, frame in loaded.items():
            # Excel has no timezone-aware datetime type, and every date in this
            # project is naive already, but a stray tz would abort the write.
            out = frame.copy()
            for column in out.select_dtypes(include=["datetimetz"]).columns:
                out[column] = out[column].dt.tz_localize(None)
            out.to_excel(writer, sheet_name=name, index=False)

    size_mb = args.output.stat().st_size / 1e6
    log.info("Wrote %s | %d sheets | %s rows | %.2f MB",
             args.output.name, len(loaded) + 1,
             f"{int(contents['rows'].sum()):,}", size_mb)

    if missing:
        log.warning("%d table(s) not built: %s", len(missing), ", ".join(missing))

    log.info("Contents:\n%s", contents[["sheet", "rows", "columns"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
