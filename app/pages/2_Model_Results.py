"""Expanding-window results for feature sets A, B and C."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import PALETTE, page_config, report, require

page_config("Model results")

st.title("Model results")

st.markdown(
    "Three nested feature sets, six expanding-window folds each. "
    "**A** is the ten market caps alone, **B** adds trend features built from "
    "the same block, **C** adds Treasury yields and gold as controls."
)

metrics = report("cv_results.parquet")
if not require(metrics, "python -m scripts.run_train"):
    st.stop()

summary = report("ablation_summary.parquet")

# --- Ablation headline ------------------------------------------------------

st.subheader("Ablation summary")

if summary is not None and not summary.empty:
    display = summary.rename(columns={
        "feature_set": "Set", "mae": "MAE", "mape": "MAPE %",
        "mspe": "MSPE %", "r2": "R²", "skill_mae": "Skill (MAE)",
        "directional_accuracy": "Directional acc.",
    })
    st.dataframe(
        display.style.format({
            "MAE": "{:.2f}", "MAPE %": "{:.3f}", "MSPE %": "{:.5f}",
            "R²": "{:.5f}", "Skill (MAE)": "{:.3f}",
            "Directional acc.": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.warning(
    "**Skill below 1.0 is the only result that means anything here.** "
    "R² is inflated by the index trend: the level roughly triples across the "
    "window, so the variance being explained is mostly drift, not prediction. "
    "Skill compares the model against the naive forecast that tomorrow equals "
    "today.",
    icon="⚠️",
)

st.divider()

# --- Skill by fold ----------------------------------------------------------

st.subheader("Skill against the naive forecast, by fold")

fig = px.bar(
    metrics, x="fold", y="skill_mae", color="feature_set", barmode="group",
    color_discrete_map=PALETTE,
    labels={"skill_mae": "MAE ÷ baseline MAE", "fold": "", "feature_set": "Set"},
    height=420,
)
fig.add_hline(y=1.0, line_dash="dash", line_color="#E45756",
              annotation_text="naive forecast", annotation_position="top left")
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Bars below the dashed line beat doing nothing. Folds validating years that "
    "set new index highs are where tree models hit their ceiling. Two folds stay "
    "inside the range already seen in training — 2022, the one bear market in "
    "the window, and 2023 — and only 2022 scores well. The ceiling is therefore "
    "not the whole explanation; see **Model Comparison**."
)

st.divider()

# --- Metric explorer --------------------------------------------------------

st.subheader("All metrics by fold")

metric_choice = st.selectbox(
    "Metric",
    ["mae", "mape", "mspe", "r2", "rmse", "skill_mae", "directional_accuracy"],
    index=0,
)

pivot = metrics.pivot_table(
    index="fold", columns="feature_set", values=metric_choice, aggfunc="mean"
)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.dataframe(pivot.style.format("{:.5f}"), use_container_width=True)

with col_right:
    fig = px.line(
        metrics.sort_values("year"),
        x="year", y=metric_choice, color="feature_set", markers=True,
        color_discrete_map=PALETTE,
        labels={metric_choice: metric_choice.upper(), "year": "Validation year",
                "feature_set": "Set"},
        height=330,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Per-fold detail, including the winning model"):
    columns = [c for c in ["feature_set", "fold", "year", "n_train", "n_valid",
                           "best_model", "mae", "mape", "mspe", "r2",
                           "baseline_mae", "skill_mae", "directional_accuracy"]
               if c in metrics.columns]
    st.dataframe(metrics[columns], use_container_width=True, hide_index=True)

st.divider()

# --- Predictions ------------------------------------------------------------

st.subheader("Predicted against actual")

predictions = report("predictions.parquet")
if require(predictions, "python -m scripts.run_train"):
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    choose_left, choose_right = st.columns(2)
    with choose_left:
        chosen_set = st.selectbox("Feature set",
                                  sorted(predictions["feature_set"].unique()))
    with choose_right:
        chosen_fold = st.selectbox("Fold",
                                   sorted(predictions["fold"].unique()))

    subset = predictions[
        (predictions["feature_set"] == chosen_set)
        & (predictions["fold"] == chosen_fold)
    ].sort_values("date")

    if subset.empty:
        st.info("No predictions for that combination.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["actual"], name="Actual",
            line=dict(color=PALETTE["actual"], width=2)))
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["predicted"], name="Predicted",
            line=dict(color=PALETTE["predicted"], width=2)))
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["baseline"], name="Naive baseline",
            line=dict(color=PALETTE["baseline"], width=1, dash="dot")))
        fig.update_layout(
            height=460, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Index level",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        error = subset["predicted"] - subset["actual"]
        st.caption(
            f"Mean error {error.mean():+,.1f} points, "
            f"mean absolute error {error.abs().mean():,.1f} points. "
            "A prediction line that flattens while the actual keeps climbing is "
            "the tree ceiling showing itself."
        )
