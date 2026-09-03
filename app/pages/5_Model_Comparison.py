"""Linear against trees -- why the model pool must contain a linear learner."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from lib import page_config, report, require

page_config("Model comparison")

st.title("Linear against trees")

st.markdown(
    """
Gradient boosted trees and random forests predict by averaging the training
targets that land in a leaf, so they **cannot return a value above the highest
target they were trained on**. The S&P 500 roughly triples across this window,
which means most folds ask them for a number they cannot produce.

The design assumed a linear learner would therefore win: the index is a weighted
sum of constituent market caps, and a linear model has no ceiling. `LR` was
pinned into the model pool for exactly that reason, and this page scores every
model individually on the real validation window rather than on the tuning split
AutoGluon uses to choose its ensemble.
"""
)

st.error(
    "**The assumption was wrong.** `LinearModel` finishes second from last on "
    "mean MAE, ahead of only `CatBoost` and behind the other three tree models. "
    "A single fitted slope cannot span a relationship that drifts by a factor "
    "of 2.2 across the decade (see **Explanatory Power**), so the linear model "
    "projects the wrong slope indefinitely while the trees at least clamp to "
    "their training range — which bounds the error rather than compounding it. "
    "The ceiling is real; escaping it is not sufficient.",
    icon="✖️",
)

comparison = report("model_comparison.parquet")
if not require(comparison, "python -m scripts.run_train"):
    st.stop()

# --- Overall ranking --------------------------------------------------------

st.subheader("Mean validation error by model")

overall = (
    comparison.groupby("model")[["mae", "r2", "skill_mae"]]
    .mean()
    .sort_values("mae")
    .reset_index()
)

left, right = st.columns([1, 1])

with left:
    st.dataframe(
        overall.rename(columns={
            "model": "Model", "mae": "Mean MAE",
            "r2": "Mean R²", "skill_mae": "Skill (MAE)",
        }).style.format({
            "Mean MAE": "{:.1f}", "Mean R²": "{:.3f}", "Skill (MAE)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with right:
    fig = px.bar(
        overall, x="mae", y="model", orientation="h",
        labels={"mae": "Mean MAE (index points)", "model": ""},
        height=320,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(categoryorder="total descending"))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- The ceiling ------------------------------------------------------------

st.subheader("The prediction ceiling")

st.markdown(
    "If a model cannot extrapolate, its largest prediction is pinned near the "
    "largest target it saw in training, while the actual index climbs past it. "
    "The gap below is that ceiling, measured directly."
)

ceiling = comparison.copy()
ceiling["shortfall"] = ceiling["actual_max"] - ceiling["pred_max"]

fig = px.bar(
    ceiling.groupby(["fold", "model"])["shortfall"].mean().reset_index(),
    x="fold", y="shortfall", color="model", barmode="group",
    labels={"shortfall": "Actual max − predicted max (points)",
            "fold": "", "model": "Model"},
    height=440,
)
fig.add_hline(y=0, line_color="#999999")
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "A tall positive bar means the model never got near the year's actual high. "
    "Bars close to zero mean the model tracked the index into new territory. "
    "Every tree model is positive in every year — the ceiling is universal — "
    "while `LinearModel` is negative in every year: it escapes the ceiling by "
    "overshooting the high instead, which is not the same as being right."
)

st.divider()

# --- Per fold ---------------------------------------------------------------

st.subheader("By fold")

metric = st.selectbox("Metric", ["mae", "r2", "skill_mae", "mape", "mspe"])

folds = sorted(comparison["fold"].unique())
chosen_sets = st.multiselect(
    "Feature sets", sorted(comparison["feature_set"].unique()),
    default=sorted(comparison["feature_set"].unique()),
)

subset = comparison[comparison["feature_set"].isin(chosen_sets)]

fig = px.line(
    subset.sort_values("year"),
    x="year", y=metric, color="model", markers=True,
    facet_col="feature_set" if len(chosen_sets) > 1 else None,
    labels={metric: metric.upper(), "year": "Validation year", "model": "Model"},
    height=440,
)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**Two folds are the control: 2022 and 2023.** They are the only validation years
where the index stayed inside the range the models had already seen, so the
ceiling is idle in both — and they split. 2022 is the best fold in the study.
2023 is not, and the ranking inside it inverts: the three tree models take the
top three places on MAE and `LinearModel` comes last. With no new highs to reach
for, the only thing left to punish is the drifted slope, and the linear model is
the one that commits to a slope.

Everywhere else the index leaves that range and the other failure mode takes
over: the trees flatten against their ceiling while the linear model overshoots
it. That 2023 is an in-range year and still a bad one is the reason the ceiling
cannot carry the whole explanation on its own.

The ensemble wins overall because blending the two failure modes is less bad
than committing to either.
"""
)

with st.expander("Full per-model results"):
    columns = [c for c in ["feature_set", "fold", "year", "model", "mae", "mape",
                           "mspe", "r2", "skill_mae", "pred_max", "actual_max"]
               if c in comparison.columns]
    st.dataframe(comparison[columns], use_container_width=True, hide_index=True)
