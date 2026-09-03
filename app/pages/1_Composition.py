"""Who occupied the top ten, and how concentrated the block became."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import page_config, processed, require

page_config("Composition")

st.title("Composition of the top ten")

st.markdown(
    "The professor's objection to fixing a handful of tickers was that they drop "
    "out. This page is the evidence: membership churns steadily, and the block "
    "grows more top-heavy every year."
)

top10 = processed("top10_daily.parquet")
if not require(top10, "python -m scripts.run_data"):
    st.stop()

top10 = top10.copy()
top10["date"] = pd.to_datetime(top10["date"])

# --- Membership over time ---------------------------------------------------

st.subheader("Rank slot occupancy")

slot_cols = [c for c in top10.columns if c.startswith("name_")]
long = top10.melt(
    id_vars="date", value_vars=slot_cols,
    var_name="slot", value_name="ticker",
).dropna(subset=["ticker"])
long["rank"] = long["slot"].str.removeprefix("name_").astype(int)

# Weekly sampling keeps the scatter readable; daily would be ~25,000 points of
# which most are visually identical.
weekly = long[long["date"].dt.dayofweek == 2]

order = (
    weekly.groupby("ticker")["rank"].mean().sort_values().index.tolist()
)

fig = px.scatter(
    weekly,
    x="date",
    y="ticker",
    color="rank",
    color_continuous_scale="Viridis_r",
    category_orders={"ticker": order},
    labels={"rank": "Rank", "date": "", "ticker": ""},
    height=max(420, 22 * len(order)),
)
fig.update_traces(marker=dict(size=5, opacity=0.85))
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Each row is a company; a mark means it held a top-ten slot that week. "
    "Darker means a higher rank. Gaps are the churn the design is built to absorb."
)

# --- Entries and exits ------------------------------------------------------

first_last = (
    long.groupby("ticker")["date"]
    .agg(first_seen="min", last_seen="max", days="count")
    .reset_index()
    .sort_values("days", ascending=False)
)
first_last["first_seen"] = first_last["first_seen"].dt.date
first_last["last_seen"] = first_last["last_seen"].dt.date

left, right = st.columns([2, 1])

with left:
    st.subheader("Time spent in the top ten")
    st.dataframe(
        first_last.rename(columns={
            "ticker": "Ticker", "first_seen": "First seen",
            "last_seen": "Last seen", "days": "Days in top 10",
        }),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

with right:
    st.subheader("Churn")
    total = len(first_last)
    permanent = int((first_last["days"] == len(top10)).sum())
    st.metric("Distinct companies", total,
              help="Companies that held a top-ten slot at any point")
    st.metric("Present every single day", permanent)
    st.markdown(
        f"""
Across the window **{total}** different companies passed through a block that is
only ten wide. Fixing the list to whichever names lead today would have quietly
rewritten the early years.
"""
    )

st.divider()

# --- Concentration ----------------------------------------------------------

st.subheader("Concentration inside the block")

concentration = processed("concentration.parquet")
if require(concentration, "python -m scripts.run_data"):
    concentration = concentration.copy()
    concentration["date"] = pd.to_datetime(concentration["date"])

    tab1, tab2 = st.tabs(["Share of the leader", "Combined size"])

    with tab1:
        fig = px.line(
            concentration, x="date", y=["share_1", "share_top3", "hhi"],
            labels={"value": "", "date": "", "variable": ""},
            height=420,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`share_1` is the leader's weight inside the top ten, `hhi` the "
            "Herfindahl index of the block. Both are computed from the top ten "
            "alone, so they stay inside the scope of the research question."
        )

    with tab2:
        fig = px.area(
            concentration, x="date", y="top_n_total",
            labels={"top_n_total": "USD billions", "date": ""},
            height=420,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Combined market cap of the top ten. This series is the `S(t)` that "
            "the momentum and moving-average features in set B are built from."
        )
