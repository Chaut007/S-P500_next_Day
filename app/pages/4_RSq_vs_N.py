"""How many constituents are enough? The R-squared curve against N."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import page_config, report, require

page_config("How many stocks")

st.title("How many stocks are enough?")

st.markdown(
    """
Ten is a round number, not an answer. This page sweeps N over 5, 10 and 20 using
the market-cap block alone, and reads off where adding constituents stops paying
for itself.
"""
)

curve = report("rsq_vs_n.parquet")
if not require(curve, "python -m scripts.run_experiments"):
    st.stop()

folds = report("rsq_vs_n_folds.parquet")

# --- The curve --------------------------------------------------------------

st.subheader("Accuracy against N")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=curve["top_n"],
        y=curve["r2_mean"],
        error_y=dict(type="data", array=curve.get("r2_std"), visible=True),
        mode="lines+markers",
        marker=dict(size=11),
        line=dict(width=3),
        name="R²",
    )
)
fig.update_layout(
    height=440,
    xaxis_title="Number of constituents (N)",
    yaxis_title="Mean R² across folds",
    margin=dict(l=0, r=0, t=10, b=0),
)
fig.update_xaxes(tickvals=curve["top_n"].tolist())
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Error bars are the standard deviation across the six expanding-window "
    "folds. A wide bar means the result depends heavily on which year is being "
    "predicted, which matters more than the mean."
)

# --- Marginal gain ----------------------------------------------------------

st.subheader("What each extra step buys")

table = curve[["top_n", "r2_mean", "r2_std", "mae_mean", "skill_mae_mean"]].copy()
table["r2_gain"] = table["r2_mean"].diff()
table["mae_gain"] = -table["mae_mean"].diff()

st.dataframe(
    table.rename(columns={
        "top_n": "N", "r2_mean": "Mean R²", "r2_std": "SD",
        "mae_mean": "Mean MAE", "skill_mae_mean": "Skill (MAE)",
        "r2_gain": "Δ R²", "mae_gain": "MAE improvement",
    }).style.format({
        "Mean R²": "{:.5f}", "SD": "{:.5f}", "Mean MAE": "{:.2f}",
        "Skill (MAE)": "{:.3f}", "Δ R²": "{:+.5f}", "MAE improvement": "{:+.2f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
The expected shape was a knee: a large gain from five to ten, a small one from
ten to twenty, justifying a stop at ten. **That is not what the data shows.**

Accuracy improves as N grows, and the standard deviation across folds is larger
than the differences between values of N. Which year is being predicted matters
more than how many constituents are used.

That result is consistent with the drift on the Explanatory Power page: a wider
block captures more of the index, so less of the movement is left to the
constituents outside it, and the coefficient has less distance to travel. On
this evidence, ten is a reasonable reporting choice but not an optimum the data
picks out on its own.
"""
)

st.info(
    "`Skill (MAE)` stays above 1.0 at every N, meaning no choice of N beats the "
    "naive forecast. That comparison is unfavourable by construction — the naive "
    "rule is given today's index level and the model is not — so read this curve "
    "as explanatory reach, not forecasting ability.",
    icon="ℹ️",
)

# --- Per fold ---------------------------------------------------------------

if folds is not None and not folds.empty:
    st.divider()
    st.subheader("Per fold")

    metric = st.selectbox("Metric", ["r2", "mae", "mape", "mspe", "skill_mae"])

    fig = px.line(
        folds.sort_values("year"),
        x="year", y=metric, color="top_n", markers=True,
        labels={metric: metric.upper(), "year": "Validation year", "top_n": "N"},
        height=420,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw fold results"):
        st.dataframe(folds, use_container_width=True, hide_index=True)
