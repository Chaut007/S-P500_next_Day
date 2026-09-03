"""Dashboard home page.

Launch from the project root:
    streamlit run app/Home.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import asset, metric_row, page_config, processed, report

page_config("Home")

st.title("Do the ten largest stocks move the S&P 500?")

st.markdown(
    """
Every trading day from **1 January 2016 to 31 December 2025**, the S&P 500
constituents are ranked by market capitalisation and the top ten are taken as
the independent variables. The target is the index close on the **following**
trading day.

The columns are rank slots, not companies: `x1` is whatever firm is largest on
that date. That is what keeps the design honest when a company falls out of the
top ten, which over ten years happens repeatedly.
"""
)

# --- Headline numbers -------------------------------------------------------

dataset = processed("dataset.parquet")
concentration = processed("concentration.parquet")
ablation = report("ablation_summary.parquet")

rows = f"{len(dataset):,}" if dataset is not None else "—"

span = "—"
if dataset is not None and "date" in dataset:
    dates = pd.to_datetime(dataset["date"])
    span = f"{dates.min():%b %Y} – {dates.max():%b %Y}"

drift = report("explanatory_drift.parquet")

explanatory_r2 = "—"
ratio_drift = "—"
if drift is not None and not drift.empty:
    explanatory_r2 = f"{drift.iloc[0]['r2_mean']:.3f}"
    ratio_drift = f"{drift.iloc[0]['ratio_drift_multiple']:.2f}×"

metric_row(
    [
        ("Modelling rows", rows, "Trading days with a complete feature vector"),
        ("Study window", span, None),
        ("Within-year R²", explanatory_r2,
         "Index level regressed on the ten market caps, inside each year"),
        ("Relationship drift", ratio_drift,
         "How far index ÷ top-ten market cap moved across the decade"),
    ]
)

st.success(
    "**The top ten explain the index, but the relationship does not hold "
    "still.** Within any single year they account for well over 90% of the "
    "index level. Across the decade the index-to-block ratio falls by roughly "
    "a factor of two and a half, which is what defeats every out-of-sample "
    "forecast here. See **Explanatory Power** for the evidence.",
    icon="✅",
)

st.divider()

# --- Animation --------------------------------------------------------------

st.subheader("Ten years of the leaderboard")

video = asset("top10_race.mp4")
if video is not None:
    st.video(str(video))
    st.caption(
        "Month-end market capitalisation, interpolated between frames. "
        "Watch the energy and industrial names slide out while the bars at the "
        "top pull further away from the field."
    )
else:
    st.info("Animation not built yet. Run `python -m scripts.build_race` "
            "after `python -m scripts.run_data`.")

st.divider()

# --- Method summary ---------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("How it is built")
    st.markdown(
        """
- **Market cap** = current shares outstanding × split-adjusted close.
  `Close`, never `Adj Close` — the dividend adjustment would push historical
  prices down by roughly a fifth over ten years and corrupt the ranking.
- **Ranking** is recomputed every trading day.
- **Validation** is an expanding window: each fold trains on all history before
  its validation year.
- **A linear learner is always in the model pool.** Tree models cannot predict
  above the highest target they were trained on, and the index roughly triples
  across the window.
"""
    )

with right:
    st.subheader("How to read the results")
    st.markdown(
        """
Two different questions live in this project and they have different answers.

**Explaining** — how much of the index do the ten largest members account for?
Measured inside a period. The answer is: almost all of it.

**Forecasting** — do they beat the naive "tomorrow equals today" rule? Measured
across expanding folds by `skill_mae`, the model's error divided by the naive
error. The answer is no, and it is not close.

That gap is not a modelling failure to be tuned away. The naive rule starts from
today's index level, which this design deliberately withholds from the model,
and the mapping the model must learn drifts steadily over the decade.
"""
    )

st.divider()

st.subheader("Pipeline")
st.code(
    """python -m scripts.run_data          # download, rank, assemble
python -m scripts.build_race        # render the animation above
python -m scripts.run_train         # expanding-window ablation A / B / C
python -m scripts.run_experiments   # N sweep and weight vs importance""",
    language="bash",
)
