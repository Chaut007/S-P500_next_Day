"""The answer to the research question, separated from forecasting skill."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import metric_row, page_config, report, require

page_config("Explanatory power")

st.title("Do the top ten explain the index?")

st.markdown(
    """
This page answers the research question. The forecasting pages answer a
different one, and conflating them would give the wrong conclusion.

**Explaining** asks how much of the index level the ten largest constituents
account for. It is measured by fitting inside a period and scoring inside the
same period.

**Forecasting** asks whether they beat the naive "tomorrow equals today" rule.
They cannot, and the reason is structural rather than interesting: the naive
rule is handed today's index level, which the model is denied by design.
"""
)

periods = report("explanatory_by_period.parquet")
if not require(periods, "python -m scripts.run_experiments"):
    st.stop()

drift = report("explanatory_drift.parquet")
rolling = report("explanatory_rolling.parquet")

# --- Headline ---------------------------------------------------------------

if drift is not None and not drift.empty:
    row = drift.iloc[0]
    metric_row(
        [
            ("Mean within-year R²", f"{row['r2_mean']:.3f}",
             "Index level regressed on the ten market caps, inside each year"),
            ("Mean within-year MAPE", f"{row['mape_mean']:.2f}%", None),
            ("Ratio drift", f"{row['ratio_drift_multiple']:.2f}×",
             "How far index ÷ block moved from the first year to the last"),
            ("Block vs index growth", f"{row['relative_growth']:.2f}×",
             "The top ten grew this much faster than the index itself"),
        ]
    )

st.success(
    "**Yes — decisively.** Within any single year the ten largest constituents "
    "account for well over 90% of the variation in the index level, with a mean "
    "absolute percentage error around 1%. The relationship is strong.",
    icon="✅",
)

st.warning(
    "**But it does not hold still.** The index-to-block ratio falls steadily "
    "across the decade, so coefficients fitted on early years are simply wrong "
    "for later ones. That drift, not noise, is what breaks every out-of-sample "
    "forecast in this study.",
    icon="⚠️",
)

st.divider()

# --- Within-year fit --------------------------------------------------------

st.subheader("Explanatory power year by year")

left, right = st.columns([3, 2])

with left:
    fig = px.bar(
        periods, x="year", y="r2",
        labels={"r2": "R² within the year", "year": ""},
        height=380,
    )
    fig.update_yaxes(range=[0.8, 1.0])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Axis starts at 0.8 so the differences between years are visible.")

with right:
    st.dataframe(
        periods[["year", "n", "r2", "mape", "index_over_block"]].rename(columns={
            "year": "Year", "n": "Days", "r2": "R²",
            "mape": "MAPE %", "index_over_block": "Index ÷ block",
        }).style.format({
            "R²": "{:.4f}", "MAPE %": "{:.3f}", "Index ÷ block": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

st.divider()

# --- The drift --------------------------------------------------------------

st.subheader("The relationship is moving")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=periods["year"], y=periods["index_over_block"],
    mode="lines+markers", line=dict(width=3), marker=dict(size=10),
    name="Index ÷ top-ten market cap",
))
fig.update_layout(
    height=400,
    xaxis_title="",
    yaxis_title="Index level ÷ block market cap",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
Read this as the coefficient a model has to learn. It falls by roughly a factor
of two and a half across the window, and it falls **monotonically** — this is a
trend, not noise a longer sample would average away.

A model fitted on the early years therefore expects far more index per unit of
top-ten market cap than the later years deliver, and over-predicts. That is
exactly the error direction observed on every expanding-window fold apart from
2022.
"""
)

if drift is not None and not drift.empty:
    row = drift.iloc[0]
    st.info(
        f"Over the study window the top-ten block grew "
        f"**{row['block_growth']:.2f}×** while the index grew "
        f"**{row['index_growth']:.2f}×**. The concentration of the index into "
        f"its largest members is the mechanism behind the drift.",
        icon="📈",
    )

# --- Rolling ----------------------------------------------------------------

if rolling is not None and not rolling.empty:
    st.divider()
    st.subheader("Rolling one-year window")

    rolling = rolling.copy()
    rolling["date"] = pd.to_datetime(rolling["date"])

    fig = px.line(
        rolling, x="date", y="r2",
        labels={"r2": "R² over the trailing year", "date": ""},
        height=360,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    trough = rolling.loc[rolling["r2"].idxmin()]
    st.caption(
        "Calendar years are a reporting convenience; a rolling window shows "
        "when the relationship genuinely weakens. The trailing-year R² stays "
        f"between {rolling['r2'].min():.2f} and {rolling['r2'].max():.2f} "
        f"throughout, with its deepest trough in "
        f"{trough['date']:%B %Y}."
    )
