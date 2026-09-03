"""Index weight against learned importance -- the second research question."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from lib import page_config, report, require

page_config("Weight vs importance")

st.title("Weight against importance")

st.markdown(
    """
Index weight is mechanical: a company contributes in proportion to its market
cap. Learned importance is not — it measures how much the prediction degrades
when a column is shuffled.

If the two lined up, a model would be telling us nothing a weighting scheme does
not already say. The gap between them is the finding.
"""
)

importance = report("importance.parquet")
if not require(importance, "python -m scripts.run_experiments"):
    st.stop()

stats = report("importance_stats.parquet")

# --- Rank agreement ---------------------------------------------------------

if stats is not None and not stats.empty:
    st.subheader("Rank agreement per fold")

    left, right = st.columns([1, 1])

    with left:
        display = stats.rename(columns={
            "fold": "Fold", "spearman_rho": "Spearman ρ",
            "spearman_p": "p", "pearson_r": "Pearson r", "n_slots": "Slots",
        })
        st.dataframe(
            display.style.format({
                "Spearman ρ": "{:.3f}", "p": "{:.4f}",
                "Pearson r": "{:.3f}", "Slots": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        mean_rho = stats["spearman_rho"].mean()
        mean_r = stats["pearson_r"].mean()
        st.metric("Mean Spearman ρ", f"{mean_rho:.3f}",
                  help="Rank correlation between weight and importance")
        st.metric("Mean Pearson r", f"{mean_r:.3f}",
                  help="Linear correlation, shown for contrast")
        st.markdown(
            "Spearman is the honest statistic here: the claim under test is "
            "that the relationship is **not** linear, and rank correlation "
            "stays valid whether it is or not."
        )

    st.divider()

# --- Scatter ----------------------------------------------------------------

st.subheader("Weight against importance by rank slot")

folds = sorted(importance["fold"].unique())
chosen = st.multiselect("Folds", folds, default=folds)

subset = importance[importance["fold"].isin(chosen)]
if subset.empty:
    st.info("Select at least one fold.")
    st.stop()

fig = px.scatter(
    subset,
    x="weight", y="importance_norm", color="fold", text="feature",
    labels={"weight": "Market-cap weight inside the top ten",
            "importance_norm": "Normalised permutation importance",
            "fold": "Fold"},
    height=520,
)
fig.update_traces(textposition="top center", marker=dict(size=11))

# A 45-degree guide: points on it would mean importance simply tracks weight.
limit = float(max(subset["weight"].max(), subset["importance_norm"].max())) * 1.05
fig.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit,
              line=dict(dash="dash", color="#999999"))
fig.add_annotation(x=limit * 0.82, y=limit * 0.9,
                   text="importance = weight", showarrow=False,
                   font=dict(color="#999999"))
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Points above the dashed line matter more to the model than their size "
    "alone would justify; points below matter less."
)

st.divider()

# --- Gap --------------------------------------------------------------------

st.subheader("Where importance departs from weight")

gap = (
    subset.groupby("feature")[["weight", "importance_norm", "gap"]]
    .mean()
    .reset_index()
    .sort_values("gap")
)

fig = px.bar(
    gap, x="gap", y="feature", orientation="h",
    color="gap", color_continuous_scale="RdBu", color_continuous_midpoint=0,
    labels={"gap": "Importance − weight", "feature": ""},
    height=420,
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Underlying numbers"):
    st.dataframe(
        gap.rename(columns={
            "feature": "Slot", "weight": "Mean weight",
            "importance_norm": "Mean importance", "gap": "Gap",
        }).style.format({
            "Mean weight": "{:.4f}", "Mean importance": "{:.4f}", "Gap": "{:+.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
