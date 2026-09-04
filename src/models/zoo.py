"""The four model families, behind one interface.

Each family gets the same (X_train, y_train) and must return predictions for
X_test in index points, so their errors are directly comparable.

The comparison is deliberately unfair in one direction and the report has to
say so: 98% of the test block sits above the highest index level in training.
Three of these four families cannot return a value above their training range
at all, so the split is a direct test of the extrapolation ceiling rather than
of fit quality.

    AutoGluon   ensemble over LR/GBM/CAT/XGB/RF -- the linear member is the
                only part of it that can leave the training range
    XGBoost     gradient boosted trees, hard ceiling at the training maximum
    SVR         RBF kernel, decays to the bias term away from the training data
    LSTM        sequence model; the ceiling depends entirely on how the target
                is scaled, which is why it is standardised and not min-maxed
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, load_config
from src.logger import get_logger
from src.utils import set_seed

log = get_logger(__name__)

LABEL = "target"


@dataclass
class Fitted:
    """A trained model plus everything the report needs from it."""

    name: str
    predict: Any                       # callable: DataFrame -> np.ndarray
    train_max: float
    estimator: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


# --- Estimator builders -----------------------------------------------------
# Shared with src/models/tuning.py. Defining an estimator in one place and
# searching over it in another is how a grid search ends up tuning a model that
# is not the one finally trained.


def build_svr(params: dict[str, Any]):
    """SVR wrapped so that both scalers are fitted inside the caller's split.

    A plain StandardScaler fitted once outside cross-validation would let every
    fold see the mean and variance of the folds after it. Putting the feature
    scaler in a Pipeline and the target scaler in TransformedTargetRegressor
    makes scikit-learn refit both on each training fold.
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("svr", SVR(**params))]),
        transformer=StandardScaler(),
    )


def build_xgboost(params: dict[str, Any], seed: int):
    """Gradient boosted trees. No scaling: trees are invariant to it."""
    from xgboost import XGBRegressor

    return XGBRegressor(**params, random_state=seed, n_jobs=-1, tree_method="hist")


# --- AutoGluon --------------------------------------------------------------


def fit_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> Fitted:
    """AutoML ensemble. Its tuning split is carved from the end of the block."""
    from autogluon.tabular import TabularPredictor

    model_cfg = cfg["model"]
    train = X_train.copy()
    train[LABEL] = y_train.to_numpy()

    # Hold out the most recent slice for AutoGluon's own model selection, so it
    # never chooses weights using rows that sit after the ones it fits on.
    cut = max(int(len(train) * 0.85), 1)
    fit_data, tuning_data = train.iloc[:cut], train.iloc[cut:]

    path = MODELS_DIR / "zoo" / "autogluon"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

    predictor = TabularPredictor(
        label=LABEL,
        problem_type="regression",
        eval_metric=model_cfg["eval_metric"],
        path=str(path),
        verbosity=1,
    )
    predictor.fit(
        train_data=fit_data,
        tuning_data=tuning_data,
        hyperparameters={name: {} for name in model_cfg["included_model_types"]},
        time_limit=model_cfg["time_limit"],
    )

    return Fitted(
        name="AutoGluon",
        predict=lambda X: predictor.predict(X).to_numpy(dtype="float64"),
        train_max=float(y_train.max()),
        estimator=predictor,
        extras={"best_model": predictor.model_best},
    )


# --- XGBoost ----------------------------------------------------------------


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Fitted:
    """Gradient boosted trees on the raw features; no scaling needed."""
    params = dict(params or cfg["zoo"]["xgboost"])
    model = build_xgboost(params, cfg["project"]["random_state"])
    model.fit(X_train, y_train)

    return Fitted(
        name="XGBoost",
        predict=lambda X: model.predict(X).astype("float64"),
        train_max=float(y_train.max()),
        estimator=model,
    )


# --- SVR --------------------------------------------------------------------


def fit_svr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Fitted:
    """RBF support vector regression.

    Both the features and the target are standardised on the training block
    only. Without scaling the market-cap columns run to hundreds while the
    trend ratios sit near zero, and the kernel is dominated by whichever column
    happens to be largest.
    """
    params = dict(params or cfg["zoo"]["svr"])

    model = build_svr(params)
    model.fit(X_train, y_train)

    inner = model.regressor_.named_steps["svr"]
    return Fitted(
        name="SVR",
        predict=lambda X: model.predict(X).astype("float64"),
        train_max=float(y_train.max()),
        estimator=model,
        extras={"n_support": int(inner.n_support_.sum())},
    )


# --- LSTM -------------------------------------------------------------------


def _make_sequences(
    features: np.ndarray,
    target: np.ndarray | None,
    window: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Turn rows into overlapping windows ending at each row.

    Row t becomes the window [t-window+1 .. t], so the sequence never contains
    an observation later than the row it is predicting for.
    """
    if len(features) < window:
        raise ValueError(f"need at least {window} rows to build a sequence")

    idx = np.arange(window - 1, len(features))
    windows = np.stack([features[i - window + 1 : i + 1] for i in idx])
    aligned = target[idx] if target is not None else None
    return windows, aligned


def fit_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Fitted:
    """A two-layer LSTM predicting the index level directly.

    The target is standardised (mean/sd of the training block), never scaled to
    [0, 1]. A min-max target would cap the output layer at the training maximum,
    which on this split is a hard ceiling under 98% of the test rows -- the
    network would be unable to be right, and the result would say more about the
    preprocessing than about the model.
    """
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    lstm_cfg = {**cfg["zoo"]["lstm"], **(params or {})}
    window = lstm_cfg["window"]
    seed = cfg["project"]["random_state"]

    set_seed(seed)
    torch.manual_seed(seed)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))

    xs = x_scaler.transform(X_train).astype("float32")
    ys = y_scaler.transform(y_train.to_numpy().reshape(-1, 1)).ravel().astype("float32")

    seq, aligned = _make_sequences(xs, ys, window)

    # Early stopping watches the tail of the training block, so no test row is
    # ever seen while choosing when to stop.
    cut = max(int(len(seq) * 0.85), 1)
    tr_x, tr_y = seq[:cut], aligned[:cut]
    va_x, va_y = seq[cut:], aligned[cut:]

    class Net(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=lstm_cfg["hidden_size"],
                num_layers=lstm_cfg["num_layers"],
                dropout=lstm_cfg["dropout"],
                batch_first=True,
            )
            # Linear output, deliberately unbounded.
            self.head = nn.Linear(lstm_cfg["hidden_size"], 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    net = Net(seq.shape[2]).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lstm_cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    tr_x_t = torch.tensor(tr_x, device=device)
    tr_y_t = torch.tensor(tr_y, device=device)
    va_x_t = torch.tensor(va_x, device=device) if len(va_x) else None
    va_y_t = torch.tensor(va_y, device=device) if len(va_y) else None

    batch = lstm_cfg["batch_size"]
    best_loss, best_state, waited = float("inf"), None, 0

    for epoch in range(lstm_cfg["epochs"]):
        net.train()
        order = torch.randperm(len(tr_x_t))
        for start in range(0, len(order), batch):
            sel = order[start : start + batch]
            optimiser.zero_grad()
            loss = loss_fn(net(tr_x_t[sel]), tr_y_t[sel])
            loss.backward()
            optimiser.step()

        if va_x_t is None or not len(va_x_t):
            continue

        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(va_x_t), va_y_t))

        if val_loss < best_loss - 1e-6:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= lstm_cfg["patience"]:
                log.info("LSTM early stop at epoch %d (val MSE %.5f)", epoch + 1, best_loss)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # The first `window - 1` test rows need the tail of the training block to
    # form a full sequence, so the two are stitched before windowing.
    tail = x_scaler.transform(X_train.iloc[-(window - 1):]).astype("float32")

    def predict(X: pd.DataFrame) -> np.ndarray:
        scaled = x_scaler.transform(X).astype("float32")
        joined = np.concatenate([tail, scaled], axis=0)
        windows, _ = _make_sequences(joined, None, window)
        with torch.no_grad():
            out = net(torch.tensor(windows, device=device)).cpu().numpy()
        return y_scaler.inverse_transform(out.reshape(-1, 1)).ravel().astype("float64")

    return Fitted(
        name="LSTM",
        predict=predict,
        train_max=float(y_train.max()),
        estimator=net,
        extras={"window": window, "best_val_mse": best_loss},
    )


BUILDERS = {
    "AutoGluon": fit_autogluon,
    "XGBoost": fit_xgboost,
    "SVR": fit_svr,
    "LSTM": fit_lstm,
}


def fit_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any] | None = None,
    only: list[str] | None = None,
    params: dict[str, dict[str, Any]] | None = None,
) -> list[Fitted]:
    """Fit every family and return them in a stable order.

    `params` maps a model name to a hyperparameter override, as written by
    scripts/run_tuning.py. AutoGluon takes no override: it searches its own
    model pool inside the time budget it is given.
    """
    cfg = cfg or load_config()
    names = only or list(BUILDERS)
    params = params or {}

    fitted: list[Fitted] = []
    for name in names:
        override = params.get(name)
        log.info("=== Fitting %s on %d rows%s ===", name, len(X_train),
                 f" | tuned: {override}" if override else "")
        if name == "AutoGluon":
            fitted.append(BUILDERS[name](X_train, y_train, cfg))
        else:
            fitted.append(BUILDERS[name](X_train, y_train, cfg, override))
    return fitted
