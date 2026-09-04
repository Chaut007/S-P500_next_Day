"""Four model families on one chronological 80/20 split."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import chart, gradient_row, page_config, report, require, table

page_config("Model comparison")

st.title("Four models, one split")

st.markdown(
    """
AutoGluon, XGBoost, SVR and an LSTM are trained on the leading **80%** of
trading days and scored on the trailing **20%**, all on feature set **C** at
N = 10. No model sees a test row while fitting, scaling or early stopping.
"""
)

scores = report("model_scores.parquet")
if not require(scores, "python -m scripts.run_models"):
    st.stop()

scores = scores.sort_values("mae").reset_index(drop=True)
best = scores.iloc[0]
above = float(scores["pct_test_above_train_max"].iloc[0])

# --- Headline ---------------------------------------------------------------

gradient_row(
    [
        ("Best model", str(best["model"]), f"MAE {best['mae']:,.0f} index points"),
        ("Test above training range", f"{above:.1f}%",
         "of the 20% test block sets new highs"),
        ("Best skill vs naive", f"{scores['skill_mae'].min():.1f}×",
         "1.0 would match doing nothing"),
        ("Smallest shortfall", f"{scores['shortfall'].min():,.0f}",
         "points short of the actual high"),
    ]
)

st.error(
    f"**Every model loses to the naive forecast, by {scores['skill_mae'].min():.0f}× "
    f"to {scores['skill_mae'].max():.0f}×.** This split is a harder test than it "
    f"looks: **{above:.1f}%** of the test block sits above the highest index level "
    "in training. Three of these four families cannot return a value above their "
    "training range at all, so what is being measured here is the extrapolation "
    "ceiling rather than fit quality.",
    icon="✖️",
)

st.divider()

# --- Scoreboard -------------------------------------------------------------

st.subheader("Test scores")

columns = ["model", "mae", "rmse", "mape", "r2", "skill_mae",
           "directional_accuracy", "shortfall"]
display = scores[[c for c in columns if c in scores.columns]].rename(columns={
    "model": "Model", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE %",
    "r2": "R²", "skill_mae": "Skill (MAE)",
    "directional_accuracy": "Directional acc.", "shortfall": "Shortfall",
})

table(
    display.style.format({
        "MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MAPE %": "{:.2f}",
        "R²": "{:.3f}", "Skill (MAE)": "{:.1f}",
        "Directional acc.": "{:.3f}", "Shortfall": "{:,.0f}",
    }),
    hide_index=True,
)

st.caption(
    "`Skill (MAE)` is the model's error divided by the naive \"tomorrow equals "
    "today\" error, so below 1.0 beats doing nothing. `Shortfall` is the year's "
    "actual high minus the model's highest prediction: a large positive value "
    "means the model never came close to the top of the range."
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        scores, x="mae", y="model", orientation="h",
        labels={"mae": "Test MAE (index points)", "model": ""},
        height=320,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(categoryorder="total descending"))
    chart(fig)

with right:
    fig = px.bar(
        scores, x="shortfall", y="model", orientation="h",
        labels={"shortfall": "Actual high − highest prediction", "model": ""},
        height=320,
    )
    fig.add_vline(x=0, line_color="#9BA4BE")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(categoryorder="total descending"))
    chart(fig)

st.divider()

# --- Predictions ------------------------------------------------------------

st.subheader("Predicted against actual")

predictions = report("model_predictions.parquet")
if require(predictions, "python -m scripts.run_models"):
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    chosen = st.multiselect(
        "Models",
        sorted(predictions["model"].unique()),
        default=sorted(predictions["model"].unique()),
    )

    subset = predictions[predictions["model"].isin(chosen)]
    if subset.empty:
        st.info("Select at least one model.")
    else:
        first = subset[subset["model"] == chosen[0]].sort_values("date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=first["date"], y=first["actual"], name="Actual",
            line=dict(color="#EDF0F7", width=2.5)))
        fig.add_trace(go.Scatter(
            x=first["date"], y=first["baseline"], name="Naive baseline",
            line=dict(color="#7E8AA6", width=1, dash="dot")))

        for model in chosen:
            block = subset[subset["model"] == model].sort_values("date")
            fig.add_trace(go.Scatter(
                x=block["date"], y=block["predicted"], name=model,
                line=dict(width=2)))

        train_max = float(scores["train_max"].iloc[0])
        fig.add_hline(
            y=train_max, line_dash="dash", line_color="#F472B6",
            annotation_text="highest level seen in training",
            annotation_position="bottom left",
        )

        fig.update_layout(
            height=480, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Index level",
            legend=dict(orientation="h", y=1.1),
        )
        chart(fig)

        st.caption(
            "The dashed pink line is the ceiling. Everything the tree and kernel "
            "models predict stays underneath it while the actual index climbs "
            "away above."
        )

st.divider()

# --- Tuning -----------------------------------------------------------------

st.subheader("Does tuning change any of this?")

tuning = report("tuning_results.parquet")
tuned_scores = report("model_scores_tuned.parquet")

if tuning is None or tuning.empty:
    st.info("Not built yet. Run `python -m scripts.run_tuning` from the project root.")
else:
    searched = sorted(tuning["model"].unique())
    st.markdown(
        f"Grid search over **{len(tuning)}** parameter combinations for "
        f"**{', '.join(searched)}**, scored by MAE on `TimeSeriesSplit` folds "
        "**inside the training block**. The test block takes no part in the "
        "search — tuning on it would pick whatever suits the 20% being reported."
    )

    best_rows = (
        tuning.sort_values("cv_mae").groupby("model", as_index=False).first()
        .rename(columns={"model": "Model", "params": "Best parameters",
                         "cv_mae": "CV MAE", "cv_mae_std": "CV SD"})
    )
    table(
        best_rows[["Model", "Best parameters", "CV MAE", "CV SD"]].style.format(
            {"CV MAE": "{:,.1f}", "CV SD": "{:,.1f}"}
        ),
        hide_index=True,
    )

    if tuned_scores is not None and not tuned_scores.empty:
        merged = (
            scores[["model", "mae", "shortfall"]]
            .merge(tuned_scores[["model", "mae", "shortfall"]],
                   on="model", suffixes=("_default", "_tuned"))
        )
        merged["Δ MAE"] = merged["mae_tuned"] - merged["mae_default"]
        merged["Δ %"] = merged["Δ MAE"] / merged["mae_default"] * 100

        st.markdown("**Test MAE before and after tuning**")
        table(
            merged.rename(columns={
                "model": "Model", "mae_default": "Default",
                "mae_tuned": "Tuned", "shortfall_tuned": "Shortfall (tuned)",
            })[["Model", "Default", "Tuned", "Δ MAE", "Δ %", "Shortfall (tuned)"]]
            .style.format({
                "Default": "{:,.1f}", "Tuned": "{:,.1f}",
                "Δ MAE": "{:+,.1f}", "Δ %": "{:+.1f}",
                "Shortfall (tuned)": "{:,.0f}",
            }),
            hide_index=True,
        )

        improved = int((merged["Δ MAE"] < 0).sum())
        best_tuned = float(tuned_scores["skill_mae"].min())
        st.warning(
            f"Tuning improved {improved} of {len(merged)} models on the test "
            f"block, and the best skill is still **{best_tuned:.0f}×** worse "
            "than the naive forecast. No setting of `max_depth` lets a tree "
            "predict above the largest target it was trained on, and no kernel "
            "width stops an RBF decaying toward its training mean. The ceiling "
            "is a property of the model family, not of its hyperparameters.",
            icon="⚠️",
        )
    else:
        st.caption(
            "Run `python -m scripts.run_models --tuned` to score these settings "
            "on the test block and fill in the before/after comparison."
        )

    with st.expander("Every combination searched"):
        table(
            tuning.rename(columns={
                "model": "Model", "params": "Parameters",
                "cv_mae": "CV MAE", "cv_mae_std": "CV SD", "rank": "Rank",
            }).style.format({"CV MAE": "{:,.1f}", "CV SD": "{:,.1f}"}),
            hide_index=True,
        )

st.divider()

st.markdown(
    """
**What the ranking is actually measuring.** On a test block where almost every
row is a new high, a model's score is set by how it fails once it leaves the
range it was fitted on:

- **XGBoost** averages the training targets inside a leaf, so its largest
  possible prediction is the largest target it saw. It flattens.
- **SVR** with an RBF kernel decays toward its bias term as the inputs move away
  from the support vectors, so it does not merely flatten — it drifts back
  toward the training mean.
- **LSTM** has a linear output layer and a standardised target, so nothing caps
  it arithmetically. It extrapolates further than the other two, which is why
  its shortfall is the smallest of the three.
- **AutoGluon** carries a linear member in its ensemble, which is the only part
  of it with no ceiling at all.

None of this makes any of them useful here. The naive forecast is handed today's
index level and the models are not, and no amount of model capacity closes that
gap — see **Explanatory Power** for the half of the question these scores do not
answer.
"""
)
