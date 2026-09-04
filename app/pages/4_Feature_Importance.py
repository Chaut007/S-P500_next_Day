"""Which features the models lean on -- SHAP and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import chart, page_config, report, require, table

page_config("Feature importance")

st.title("What the models actually use")

st.markdown(
    """
Two measures, because they answer different questions.

**SHAP** decomposes each individual prediction into a contribution per feature,
so it shows both how much a feature matters and in which direction. Exact
TreeSHAP is used on **XGBoost**.

**Permutation importance** asks how much the test error grows when one column is
shuffled. It needs only predictions, so it works for all four families and is
the only measure here comparable across them.
"""
)

shap_summary = report("shap_summary.parquet")
shap_long = report("shap_values.parquet")
importance = report("feature_importance.parquet")

if not require(shap_summary, "python -m scripts.run_models"):
    st.stop()

# --- SHAP ranking -----------------------------------------------------------

st.subheader("SHAP — how much each feature moves a prediction")

base_value = float(shap_summary["base_value"].iloc[0])

left, right = st.columns([3, 2])

with left:
    ranked = shap_summary.sort_values("mean_abs_shap", ascending=True)
    fig = px.bar(
        ranked, x="mean_abs_shap", y="feature", orientation="h",
        labels={"mean_abs_shap": "Mean |SHAP| (index points)", "feature": ""},
        height=460,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    chart(fig)

with right:
    table(
        shap_summary[["feature", "mean_abs_shap", "mean_shap"]].rename(columns={
            "feature": "Feature", "mean_abs_shap": "Mean |SHAP|",
            "mean_shap": "Mean SHAP",
        }).style.format({"Mean |SHAP|": "{:,.1f}", "Mean SHAP": "{:+,.1f}"}),
        hide_index=True,
        height=460,
    )

st.caption(
    f"Base value {base_value:,.0f} index points — the model's average output "
    "before any feature is taken into account. Every contribution below is "
    "measured against it."
)

st.info(
    "**The largest constituent is not the most useful one.** The ranking is led "
    "by middle slots rather than `x1`, which is the same result the "
    "**Weight vs Importance** page reaches from the other direction: how much a "
    "company weighs in the index says little about how much the model leans on "
    "its slot.",
    icon="📊",
)

st.divider()

# --- Beeswarm ---------------------------------------------------------------

if shap_long is not None and not shap_long.empty:
    st.subheader("Where the contributions come from")

    order = shap_summary["feature"].tolist()
    swarm = shap_long[shap_long["feature"].isin(order[:12])].copy()

    # Colour by the feature's own value, normalised inside each feature so the
    # scale is comparable across columns that differ by orders of magnitude.
    swarm["scaled"] = swarm.groupby("feature")["value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    )

    # px.strip cannot take a continuous colour scale, so the swarm is drawn as
    # a scatter: each feature gets a row index and the dots are jittered around
    # it by hand.
    shown = order[:12][::-1]
    positions = {name: i for i, name in enumerate(shown)}
    swarm = swarm[swarm["feature"].isin(shown)]

    rng = np.random.default_rng(0)
    y_jittered = (
        swarm["feature"].map(positions).to_numpy(dtype="float64")
        + rng.uniform(-0.34, 0.34, len(swarm))
    )

    fig = go.Figure(go.Scatter(
        x=swarm["shap"], y=y_jittered,
        mode="markers",
        marker=dict(
            size=5, opacity=0.55,
            color=swarm["scaled"],
            colorscale=[[0.0, "#3B82F6"], [0.5, "#A78BFA"], [1.0, "#F472B6"]],
            colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                          ticktext=["low", "high"]),
        ),
        hovertext=swarm["feature"],
        hovertemplate="%{hovertext}<br>SHAP %{x:,.1f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#9BA4BE")
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="SHAP contribution (index points)",
        yaxis=dict(tickvals=list(positions.values()),
                   ticktext=list(positions.keys()), title=""),
    )
    chart(fig)

    st.caption(
        "Each dot is one test day. Pink means that day had a high value for the "
        "feature, blue a low one. A band sitting entirely to the right of zero "
        "means the feature only ever pushed the prediction upward."
    )

    st.divider()

    # --- Dependence ---------------------------------------------------------

    st.subheader("One feature in detail")

    feature = st.selectbox("Feature", order, index=0)
    block = shap_long[shap_long["feature"] == feature]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=block["value"], y=block["shap"], mode="markers",
        marker=dict(size=6, color="#A78BFA", opacity=0.6),
        name=feature,
    ))
    fig.add_hline(y=0, line_color="#9BA4BE")
    fig.update_layout(
        height=380,
        xaxis_title=f"{feature} (market cap, USD bn)" if feature.startswith("x") else feature,
        yaxis_title="SHAP contribution (index points)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    chart(fig)

    st.caption(
        "A rising line means the model reads a larger value for this feature as "
        "a higher index level."
    )

st.divider()

# --- Permutation importance -------------------------------------------------

st.subheader("Permutation importance across all four models")

if not require(importance, "python -m scripts.run_models"):
    st.stop()

worst = importance["importance_pct"].max()

st.warning(
    f"**This measure says almost nothing on this split, and that is the "
    f"result.** The largest effect any single feature has on any model is "
    f"**{worst:.1f}%** of its baseline error. Shuffling a column barely moves "
    "the error because the error is not made of feature signal — it is made of "
    "the gap between where the models can reach and where the index actually "
    "went. When a model fails structurally, permutation importance measures "
    "nothing worth reading.",
    icon="⚠️",
)

pivot = importance.pivot_table(
    index="feature", columns="model", values="importance_pct", aggfunc="mean"
)
ordering = pivot.mean(axis=1).sort_values(ascending=True).index
pivot = pivot.loc[ordering]

fig = px.bar(
    pivot.reset_index().melt(id_vars="feature", var_name="Model",
                             value_name="importance_pct"),
    x="importance_pct", y="feature", color="Model", orientation="h",
    barmode="group",
    labels={"importance_pct": "Increase in test MAE when shuffled (%)",
            "feature": ""},
    height=560,
)
fig.add_vline(x=0, line_color="#9BA4BE")
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
chart(fig)

st.caption(
    "SVR is the only model with a visible response, and only because its RBF "
    "kernel decays toward the training mean as inputs move away from the support "
    "vectors, which makes it more sensitive to the inputs than the models that "
    "simply flatten."
)

with st.expander("Full permutation results"):
    table(
        importance[["model", "feature", "baseline_mae", "permuted_mae",
                    "importance", "importance_pct", "std"]].rename(columns={
            "model": "Model", "feature": "Feature", "baseline_mae": "Baseline MAE",
            "permuted_mae": "Shuffled MAE", "importance": "Δ MAE",
            "importance_pct": "Δ %", "std": "SD",
        }).style.format({
            "Baseline MAE": "{:,.1f}", "Shuffled MAE": "{:,.1f}",
            "Δ MAE": "{:+,.2f}", "Δ %": "{:+.2f}", "SD": "{:,.2f}",
        }),
        hide_index=True,
    )
