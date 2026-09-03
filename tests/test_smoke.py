"""Tests for the failure modes that stay silent.

Leakage and misalignment do not raise exceptions. They produce better-looking
numbers, which is exactly why they need to be asserted against rather than
eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.features.build_features import (
    BASELINE_COL,
    TARGET_COL,
    assemble,
    make_baseline,
    make_target,
    select_feature_set,
)
from src.features.marketcap import compute_market_caps, rank_top_n
from src.features.trend import build_trend_features
from src.models.evaluate import mean_squared_percentage_error, regression_metrics
from src.models.split import expanding_year_folds


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range("2016-01-04", "2025-12-31"))


@pytest.fixture
def fake_prices(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Twelve tickers on independent random walks, so ranks genuinely change."""
    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(12)]

    data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0002 + i * 0.00004
        steps = rng.normal(drift, 0.012, len(calendar))
        data[ticker] = (50 + 10 * i) * np.exp(np.cumsum(steps))

    return pd.DataFrame(data, index=calendar)


@pytest.fixture
def fake_universe() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    tickers = [f"T{i:02d}" for i in range(12)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": [f"Company {t}" for t in tickers],
        "sector": ["Tech"] * len(tickers),
        "shares": rng.uniform(1e9, 8e9, len(tickers)),
    })


@pytest.fixture
def fake_index(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    level = 2000 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(calendar))))
    return pd.DataFrame({"date": calendar, "close": level})


# --- Layout and configuration ----------------------------------------------


def test_project_layout():
    for sub in ("src", "config", "data", "logs", "scripts", "app", "tests"):
        assert (PROJECT_ROOT / sub).is_dir(), f"Missing directory: {sub}"


def test_config_has_required_sections(cfg):
    for key in ("project", "data", "features", "target", "split", "model"):
        assert key in cfg, f"config.yaml is missing the '{key}' section"


def test_price_column_is_not_dividend_adjusted(cfg):
    """Adj Close would silently corrupt every market cap in the study."""
    assert cfg["data"]["price_col"] == "Close", (
        "market cap must be built from the split-adjusted, dividend-unadjusted "
        "close; Adj Close depresses historical prices by the cumulative yield"
    )


def test_linear_model_is_always_available(cfg):
    """Tree models cannot predict above their training target range."""
    assert "LR" in cfg["model"]["included_model_types"]


# --- Ranking ----------------------------------------------------------------


def test_ranking_is_monotonic_across_slots(fake_prices, fake_universe):
    """x1 >= x2 >= ... by construction, which is why swaps are smooth."""
    mcaps = compute_market_caps(fake_prices, fake_universe)
    values, _ = rank_top_n(mcaps, 10)

    arr = values.to_numpy()
    differences = np.diff(arr, axis=1)
    assert np.all(differences <= 1e-9), "slots are not sorted descending"


def test_ranking_names_match_values(fake_prices, fake_universe):
    mcaps = compute_market_caps(fake_prices, fake_universe)
    values, names = rank_top_n(mcaps, 5)

    row = 500
    for slot in range(5):
        ticker = names.iloc[row, slot]
        expected = mcaps.iloc[row][ticker]
        assert values.iloc[row, slot] == pytest.approx(expected)


# --- Causality --------------------------------------------------------------


def test_trend_features_are_causal(fake_prices, fake_universe, cfg):
    """Corrupting the future must not move a single past feature value."""
    mcaps = compute_market_caps(fake_prices, fake_universe)
    values, _ = rank_top_n(mcaps, 10)

    cut = 800
    tampered = values.copy()
    tampered.iloc[cut:] *= 5.0

    before = build_trend_features(values, cfg).iloc[:cut]
    after = build_trend_features(tampered, cfg).iloc[:cut]

    pd.testing.assert_frame_equal(before, after, obj="trend features leak the future")


# --- Target alignment -------------------------------------------------------


def test_target_is_the_next_days_index(fake_index, calendar, cfg):
    """target(t) must equal the index close at t+horizon, nothing else."""
    horizon = cfg["target"]["horizon"]
    target = make_target(fake_index, calendar, cfg)
    closes = fake_index.set_index("date")["close"].reindex(calendar)

    assert target.iloc[0] == pytest.approx(closes.iloc[horizon])
    assert target.iloc[-horizon:].isna().all(), "the tail must have no target"


def test_baseline_leads_the_target_by_one_row(fake_index, calendar, cfg):
    """baseline(t+1) == target(t): the naive forecast is simply today's level."""
    target = make_target(fake_index, calendar, cfg)
    baseline = make_baseline(fake_index, calendar)
    horizon = cfg["target"]["horizon"]

    aligned = pd.DataFrame({
        "target": target.to_numpy()[:-horizon],
        "next_baseline": baseline.to_numpy()[horizon:],
    })
    assert np.allclose(aligned["target"], aligned["next_baseline"])


# --- Feature-set integrity --------------------------------------------------


def test_feature_sets_never_include_the_target(
    fake_prices, fake_universe, fake_index, calendar, cfg
):
    mcaps = compute_market_caps(fake_prices, fake_universe)
    values, _ = rank_top_n(mcaps, 10)
    trend = build_trend_features(values, cfg)
    macro = pd.DataFrame(index=values.index)

    table = assemble(
        values, trend, macro,
        make_target(fake_index, calendar, cfg),
        make_baseline(fake_index, calendar),
    )

    for set_name in cfg["features"]["sets"]:
        columns = select_feature_set(
            table, set_name, 10, list(trend.columns), [], cfg
        )
        assert TARGET_COL not in columns
        assert BASELINE_COL not in columns
        assert "date" not in columns


# --- Splitting --------------------------------------------------------------


def test_folds_expand_and_never_overlap(calendar, cfg):
    dates = pd.Series(calendar)
    folds = expanding_year_folds(dates, cfg)

    assert len(folds) == len(cfg["split"]["validation_years"])

    previous_train_size = 0
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(fold.valid_idx), "train/valid overlap"
        assert fold.train_idx.max() < fold.valid_idx.min(), "training data sits after validation"
        assert len(fold.train_idx) > previous_train_size, "window is not expanding"
        previous_train_size = len(fold.train_idx)

        valid_years = dates.iloc[fold.valid_idx].dt.year.unique()
        assert list(valid_years) == [fold.year]


def test_gap_separates_train_from_validation(calendar, cfg):
    """The last training row's target must not reach into the validation block."""
    dates = pd.Series(calendar)
    gap = cfg["split"]["gap"]
    folds = expanding_year_folds(dates, cfg)

    for fold in folds:
        assert fold.valid_idx.min() - fold.train_idx.max() > gap


# --- Metrics ----------------------------------------------------------------


def test_mspe_matches_its_definition():
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = np.array([110.0, 180.0, 440.0])

    expected = np.mean([(0.1) ** 2, (-0.1) ** 2, (0.1) ** 2]) * 100
    assert mean_squared_percentage_error(y_true, y_pred) == pytest.approx(expected)


def test_perfect_prediction_scores_perfectly():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx(0.0)
    assert metrics["mspe"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_all_five_reported_metrics_exist():
    rng = np.random.default_rng(3)
    y_true = rng.uniform(3000, 5000, 200)
    y_pred = y_true + rng.normal(0, 30, 200)

    metrics = regression_metrics(y_true, y_pred)
    for name in ("mae", "mse", "mape", "mspe", "r2"):
        assert name in metrics and np.isfinite(metrics[name])
