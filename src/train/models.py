from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

# from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from typing import Protocol, List


class Model(Protocol):
    # def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Model:
    model = LinearRegression()
    model.fit(x, y)
    return model


def fit_lasso_regression(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.1
) -> Model:
    model = MultiTaskLasso(alpha=alpha, max_iter=1_000)
    model.fit(x, y)
    return model


def fit_decision_tree_regression(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 16,
    min_samples_split: int = 30,
    min_samples_leaf: int = 4,
    random_state: int = 42,
) -> Model:
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def fit_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 128,
    random_state: int = 42,
) -> Model:
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(x, y)
    return model


def fit_gbm_regression(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 4,
    learning_rate: float = 0.12,
    max_depth: int = 5,
    random_state: int = 42,
) -> Model:
    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    multi_gbr = MultiOutputRegressor(gbr)
    model = multi_gbr.fit(x, y)
    return model


def metrics_rsquared(model: Model, x: np.ndarray, y: np.ndarray):
    rsquared = model.score(x, y)
    print(f"R Squared: {rsquared:.4f}")
    return rsquared


def metrics_relative(model: Model, x: np.ndarray, y: np.ndarray):
    y_pred = model.predict(x)
    mae_rel = np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y))
    print(f"Relative MAE: {mae_rel:.4f}")


def metrics_effective_delta_rsquared(
    model: Model,
    x: np.ndarray,
    y_t: np.ndarray,
    delta_y: np.ndarray,
) -> float:
    """
    Compute an 'effective' R^2 on the delta implied by a level model.

    Uses:
        R^2_level = model.score(x, y_t)
        v = Var(y_t)
        c = Var(delta_y)

    and returns:
        R^2_delta_eff = 1 - (1 - R^2_level) * v / c
    """
    # Level R^2 from the model
    r2_level = model.score(x, y_t)

    # Variances
    v = np.var(y_t, ddof=0)  # Var(x_t)
    c = np.var(delta_y, ddof=0)  # Var(x_t - x_{t-1})

    print(f"Variance for y_t: {v}")
    print(f"Variacne for del y_t: {c}")

    if c == 0:
        # No variance in delta: can't define an R^2 on the delta sensibly
        print("Effective delta R^2: c (Var(delta)) == 0, returning NaN")
        return float("nan")

    r2_delta_eff = float(1 - (1 - r2_level) * (v / c))

    print(f"Level R^2: {r2_level:.4f}")
    print(f"Var(level): {v:.6f}, Var(delta): {c:.6f}")
    print(f"Effective R^2 on delta (scaled): {r2_delta_eff:.4f}")

    return r2_delta_eff


def plot_delta_scatter(
    out_file: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    measures: List[str],
):
    n_targets = y_true.shape[1]
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))
    _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= n_targets:
            ax.axis("off")
            continue
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        min_val, max_val = y_true[:, i].min(), y_true[:, i].max()
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=1
        )
        ax.set_title(measures[i])
        ax.set_xlabel("Δy true")
        ax.set_ylabel("Δy predicted")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def z_transform(df: pd.DataFrame, col: str):
    """
    Z-transform a column in a DataFrame.
    """
    mean = df[col].mean()
    std = df[col].std(ddof=0)  # or ddof=1 if you prefer

    if std == 0 or np.isnan(std):
        raise Exception
        z = (df[col] - mean) * 0  # all zeros
    else:
        z = (df[col] - mean) / std

    df[col] = z
