from __future__ import annotations
import sqlite3
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATABASE = "source.db"


def load_data() -> pd.DataFrame:
    """Load IPFQR facility quality measures."""
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql_query(
            """SELECT * FROM ipfqr_quality_measures_facility
                WHERE 
                    hbips_2_overall_rate_per_1000 IS NOT NULL
                    AND hbips_3_overall_rate_per_1000 IS NOT NULL
                    AND smd_percent IS NOT NULL
                    AND sub_2_percent IS NOT NULL
                    AND sub_3_percent IS NOT NULL
                    AND tob_3_percent IS NOT NULL
                    AND tob_3a_percent IS NOT NULL
                    AND tr_1_percent IS NOT NULL
                    AND imm_2_percent IS NOT NULL
                    AND readm_30_ipf_rate IS NOT NULL
            ;""",
            conn,
        )
        # print(df.head)
        return df


def structure_ipfqr_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and delta targets for all _percent metrics,
    using previous-year autoregressive features.
    """
    granularity = ["submission_year", "facility_id"]
    target_cols = [
        "hbips_2_overall_rate_per_1000",
        "hbips_3_overall_rate_per_1000",
        "smd_percent",
        "sub_2_percent",
        "sub_3_percent",
        "tob_3_percent",
        "tob_3a_percent",
        "tr_1_percent",
        "imm_2_percent",
        "readm_30_ipf_rate",
    ]

    df_perf = df[granularity + target_cols]
    # Shift forward to create previous-year features
    df_prev = df_perf.copy()
    df_prev["submission_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in target_cols}
    )
    print("df_prev")
    print(df_prev.shape)
    print("----------------------------------")

    # Merge to get previous-year features
    df_merged = pd.merge(
        df_perf,
        df_prev,
        on=["facility_id", "submission_year"],
        how="inner",
    )
    # Drop rows with NaNs in targets or previous-year targets
    cols_to_check = target_cols + [f"{c}_prev" for c in target_cols]
    df_clean = df_merged.dropna(subset=cols_to_check)
    # Features: previous-year percent columns
    feature_cols = [f"{c}_prev" for c in target_cols]

    x = df_clean[feature_cols].values
    y = df_clean[target_cols].values
    delta_y = y - df_clean[[f"{c}_prev" for c in target_cols]].values

    return x, delta_y, target_cols


def fit_decision_tree_regression(
    x: np.ndarray, y: np.ndarray
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(x, y)
    return model


def metrics(model, x: np.ndarray, y: np.ndarray):
    y_pred = model.predict(x)
    print(f"R² (model score): {model.score(x, y):.4f}")
    print(f"R² (manual r2_score): {r2_score(y, y_pred):.4f}")


def metrics_relative(model, x: np.ndarray, y: np.ndarray):
    y_pred = model.predict(x)
    mae_rel = np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y))
    print(f"Relative MAE: {mae_rel:.4f}")


def plot_delta_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]
):
    n_targets = y_true.shape[1]
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows)
    )
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
        ax.set_title(target_names[i])
        ax.set_xlabel("Δy true")
        ax.set_ylabel("Δy predicted")
    plt.tight_layout()
    plt.savefig(f"./metrics/psychiatric", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    print(df.columns)
    print(df.shape)
    x, delta_y, targets = structure_ipfqr_data(df)
    print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x, delta_y, test_size=0.2, random_state=42
    )

    model = fit_decision_tree_regression(x_train, y_train)

    print("Metrics on training set:")
    metrics(model, x_train, y_train)
    print("\nMetrics on test set:")
    metrics(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    delta_pred = model.predict(x_test)
    plot_delta_scatter(y_test, delta_pred, targets)
