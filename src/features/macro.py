"""Macro control variables: Treasury yields and gold.

These are controls, not the subject of the study. Including them lets the
conclusion be stated more strongly -- the top ten explain the index even after
accounting for the rate environment -- but they are only present in feature set
C so their contribution can be isolated.

Rates enter as levels because they stay inside a 0-5% band across the window and
carry no accumulating trend. Gold enters as a return: its price roughly doubled
over the same period, and adding a second trending series would manufacture
correlation with an index that also trended up.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config
from src.data.preprocess import align_to_calendar
from src.logger import get_logger

log = get_logger(__name__)


def build_macro_features(
    macro_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return macro controls indexed by the trading calendar.

    Both inputs are aligned with forward fill only. The bond market closes on a
    few days the equity market does not, and back-filling those gaps would leak
    a future yield into the past.
    """
    cfg = cfg or load_config()
    series = cfg["data"]["fred_series"]

    rates = align_to_calendar(macro_df, calendar).set_index("date")
    available = [s for s in series if s in rates.columns]
    if len(available) < len(series):
        log.warning("FRED series missing from download: %s",
                    set(series) - set(available))

    out = rates[available].apply(pd.to_numeric, errors="coerce")

    gold = align_to_calendar(gold_df, calendar).set_index("date")
    gold_close = pd.to_numeric(gold["close"], errors="coerce")
    out["gold_return"] = gold_close.pct_change()

    out.index.name = "date"
    log.info("Macro features built: %s", ", ".join(out.columns))
    return out
