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

# --- Market cap composition -------------------------------------------------

st.subheader("Market cap composition")

st.markdown(
    "Slots are rank positions, not companies, so a period's composition has to "
    "be built by company: every day each of the ten slots contributes its market "
    "cap to whichever firm held it. The shares below are each company's total "
    "over the period divided by the block's total over the same period."
)

value_cols = [f"x{i}" for i in range(1, len(slot_cols) + 1)]
company_cols = [f"company_{i}" for i in range(1, len(slot_cols) + 1)]
has_names = all(c in top10.columns for c in company_cols)

# Melt slots into (date, company, market cap) so a company that moved between
# slots is still counted once per day.
label_cols = company_cols if has_names else slot_cols
holdings = pd.concat(
    [
        pd.DataFrame({
            "date": top10["date"],
            "company": top10[label],
            "mcap": top10[value],
        })
        for label, value in zip(label_cols, value_cols)
    ],
    ignore_index=True,
).dropna(subset=["company", "mcap"])

held_out = sorted(int(y) for y in top10["date"].dt.year.unique())
first_held_out = 2020 if 2020 in held_out else held_out[0]

periods = {
    f"Held-out years ({first_held_out}–{held_out[-1]})":
        holdings["date"].dt.year >= first_held_out,
    "Whole window": pd.Series(True, index=holdings.index),
    **{
        str(year): holdings["date"].dt.year == year
        for year in reversed(held_out)
    },
}

choice = st.selectbox("Period", list(periods), index=0)
block = holdings[periods[choice]]

shares = (
    block.groupby("company")["mcap"].sum()
    .sort_values(ascending=False)
    .reset_index()
)
shares["share"] = shares["mcap"] / shares["mcap"].sum()

# Anything under 2% becomes an "Other" wedge; a pie with twenty slivers is
# unreadable and the tail is not what the chart is for.
major = shares[shares["share"] >= 0.02].copy()
tail = shares[shares["share"] < 0.02]
if not tail.empty:
    major = pd.concat(
        [
            major,
            pd.DataFrame([{
                "company": f"Other ({len(tail)} companies)",
                "mcap": tail["mcap"].sum(),
                "share": tail["share"].sum(),
            }]),
        ],
        ignore_index=True,
    )

pie_left, pie_right = st.columns([3, 2])

with pie_left:
    fig = px.pie(
        major, values="share", names="company", hole=0.52,
        height=440,
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{percent:.1%}",
        hovertemplate="%{label}<br>%{percent:.2%} of the block<br>"
                      "%{customdata:,.0f} bn USD-days<extra></extra>",
        customdata=major["mcap"],
        marker=dict(line=dict(color="rgba(0,0,0,0.35)", width=1)),
        sort=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

with pie_right:
    st.dataframe(
        shares.assign(share=shares["share"] * 100).rename(columns={
            "company": "Company", "share": "Share %", "mcap": "USD bn-days",
        })[["Company", "Share %", "USD bn-days"]].style.format({
            "Share %": "{:.2f}", "USD bn-days": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=440,
    )

leader = shares.iloc[0]
st.caption(
    f"Over **{choice.lower()}**, {leader['company']} alone accounts for "
    f"**{leader['share']:.1%}** of the top-ten block, and the "
    f"{len(shares)} companies that appear are far from equally weighted. "
    "Shares are market-cap-days, so a company counts more both for being larger "
    "and for staying in the block longer."
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
