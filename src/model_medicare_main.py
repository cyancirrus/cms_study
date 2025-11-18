from __future__ import annotations
import sqlite3
from typing import List, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    MultiTaskLasso,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATABASE = "source.db"


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...


def load_data() -> pd.DataFrame:
    """Load all hospital clinical outcomes."""
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query(
            """
            SELECT *
                FROM hvbp_clinical_outcomes
        ;""",
            conn,
        )
        # state = "IL"


def load_readmissions_scaled() -> pd.DataFrame:
    """Load hospital readmissions data and scale rates."""
    query = """
        SELECT
            facility_id,
            submission_year, 
            predicted_readmission_rate,
            expected_readmission_rate
        FROM fy_hospital_readmissions_reduction_program_hospital
        WHERE
            measure_name = "READM-30-COPD-HRRP"
    ;
     """
    # measure_name = "READM-30-AMI-HRRP"
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql_query(query, conn)

    df = df.rename(columns={"submission_year": "fiscal_year"})
    df["predicted_readmission_rate"] /= 1_000
    df["expected_readmission_rate"] /= 1_000
    return df


def structure_data_multivar_with_readmissions(
    df: pd.DataFrame, df_read: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix X and target delta_y without leakage."""
    granularity = ["fiscal_year", "facility_id"]

    # All columns to consider as features
    dimensions = [
        # New metrics
        "mort_30_ami_achievement_threshold",
        "mort_30_ami_benchmark",
        "mort_30_hf_achievement_threshold",
        "mort_30_hf_benchmark",
        "mort_30_pn_achievement_threshold",
        "mort_30_pn_benchmark",
        "mort_30_copd_achievement_threshold",
        "mort_30_copd_benchmark",
        "mort_30_cabg_achievement_threshold",
        "mort_30_cabg_benchmark",
        "comp_hip_knee_achievement_threshold",
        "comp_hip_knee_benchmark",
        # Prior metrics
        "mort_30_ami_baseline_rate",
        "mort_30_ami_performance_rate",
        "mort_30_hf_baseline_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_baseline_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_baseline_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_baseline_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_baseline_rate",
        "comp_hip_knee_performance_rate",
    ]

    target = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    # Merge with readmissions data
    df_merged = pd.merge(
        df,
        df_read,
        on=["facility_id", "fiscal_year"],
        how="left",
    )

    # Shift FORWARD to get previous year's data
    df_prev = df_merged.copy()
    df_prev["fiscal_year"] -= 1

    prev_cols = dimensions + list(
        df_read.columns.difference(["facility_id", "fiscal_year"])
    )
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    # Now current year row gets actual previous year's features
    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "fiscal_year"],
        how="inner",
    )

    # Drop rows with NaNs in any required columns
    cols_to_check = target + [f"{c}_prev" for c in prev_cols]
    df_final = df_final.dropna(subset=cols_to_check)

    # Features: exclude previous-year target columns to prevent leakage
    # prev_feature_cols = [c for c in prev_cols if c not in target]
    prev_feature_cols = prev_cols
    feature_cols = [f"{c}_prev" for c in prev_feature_cols]
    # Prepare arrays
    x = df_final[feature_cols].values
    y = df_final[target].values
    delta_y = y - df_final[[f"{c}_prev" for c in target]].values

    return x, delta_y


def structure_data_multivar_with_readmissions_and_demographics(
    df: pd.DataFrame, df_read: pd.DataFrame, db_path: str = DATABASE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and target delta_y without leakage,
    including facility-level zip demographics.
    """
    granularity = ["fiscal_year", "facility_id"]

    # All columns to consider as features (same as your original)
    dimensions = [
        # New metrics
        "mort_30_ami_achievement_threshold",
        "mort_30_ami_benchmark",
        "mort_30_hf_achievement_threshold",
        "mort_30_hf_benchmark",
        "mort_30_pn_achievement_threshold",
        "mort_30_pn_benchmark",
        "mort_30_copd_achievement_threshold",
        "mort_30_copd_benchmark",
        "mort_30_cabg_achievement_threshold",
        "mort_30_cabg_benchmark",
        "comp_hip_knee_achievement_threshold",
        "comp_hip_knee_benchmark",
        # Prior metrics
        "mort_30_ami_baseline_rate",
        "mort_30_ami_performance_rate",
        "mort_30_hf_baseline_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_baseline_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_baseline_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_baseline_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_baseline_rate",
        "comp_hip_knee_performance_rate",
    ]

    target = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    # --- load facility_zip_code + zip_demographics ---
    with sqlite3.connect(db_path) as conn:
        df_fac_zip = pd.read_sql_query(
            """
            SELECT
                submission_year AS fiscal_year,
                facility_id,
                zip_code
            FROM facility_zip_code;
            """,
            conn,
        )
        df_zip_demo = pd.read_sql_query(
            """
            SELECT
                zip_code,
                log(msa_personal_income_k) as msa_personal_income_k,
                log(msa_population_density)  as msa_population_density,
                log(msa_per_capita_income)  as msa_per_capita_income
            FROM zip_demographics;
            """,
            conn,
        )
    for col in [
        "msa_personal_income_k",
        "msa_population_density",
        "msa_per_capita_income",
    ]:
        mean = df_zip_demo[col].mean()
        std = df_zip_demo[col].std()
        df_zip_demo[col] = (df_zip_demo[col] - mean) / std

    # normalize dtypes for join keys
    df["facility_id"] = df["facility_id"]
    df["fiscal_year"] = df["fiscal_year"].astype("int64")

    df_read["facility_id"] = df_read["facility_id"]
    df_read["fiscal_year"] = df_read["fiscal_year"].astype("int64")

    df_fac_zip["facility_id"] = df_fac_zip["facility_id"]
    df_fac_zip["fiscal_year"] = df_fac_zip["fiscal_year"].astype(
        "int64"
    )
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)

    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)

    # facility-year -> zip-year -> demographics
    df_zip = pd.merge(
        df_fac_zip, df_zip_demo, on="zip_code", how="left"
    )

    # --- Merge with readmissions data (same as original) ---
    df_merged = pd.merge(
        df,
        df_read,
        on=["facility_id", "fiscal_year"],
        how="left",
    )

    # --- Add current-year demographics (these will get shifted to prev) ---
    df_merged = pd.merge(
        df_merged,
        df_zip[
            [
                "facility_id",
                "fiscal_year",
                "msa_personal_income_k",
                "msa_population_density",
                "msa_per_capita_income",
            ]
        ],
        on=["facility_id", "fiscal_year"],
        how="left",
    )

    # Shift FORWARD to get previous year's data
    df_prev = df_merged.copy()
    df_prev["fiscal_year"] -= 1

    # prev_cols = clinical dims + readmission cols + demo cols
    prev_cols = (
        dimensions
        + list(
            df_read.columns.difference(["facility_id", "fiscal_year"])
        )
        + [
            "msa_personal_income_k",
            "msa_population_density",
            "msa_per_capita_income",
        ]
    )

    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    # Now current year row gets actual previous year's features
    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "fiscal_year"],
        how="inner",
    )

    # Drop rows with NaNs in any required columns
    cols_to_check = target + [f"{c}_prev" for c in prev_cols]
    df_final = df_final.dropna(subset=cols_to_check)

    # Features: all previous-year cols (clinical + readm + demos)
    prev_feature_cols = prev_cols
    feature_cols = [f"{c}_prev" for c in prev_feature_cols]

    x = df_final[feature_cols].values
    y = df_final[target].values
    delta_y = y - df_final[[f"{c}_prev" for c in target]].values

    return x, delta_y


def structure_data_multivar(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare multivariate features and delta targets without leakage."""
    granularity = ["fiscal_year", "facility_id"]

    dimensions = [
        # New metrics
        "mort_30_ami_achievement_threshold",
        "mort_30_ami_benchmark",
        "mort_30_hf_achievement_threshold",
        "mort_30_hf_benchmark",
        "mort_30_pn_achievement_threshold",
        "mort_30_pn_benchmark",
        "mort_30_copd_achievement_threshold",
        "mort_30_copd_benchmark",
        "mort_30_cabg_achievement_threshold",
        "mort_30_cabg_benchmark",
        "comp_hip_knee_achievement_threshold",
        "comp_hip_knee_benchmark",
        # Prior metrics
        "mort_30_ami_baseline_rate",
        "mort_30_ami_performance_rate",
        "mort_30_hf_baseline_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_baseline_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_baseline_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_baseline_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_baseline_rate",
        "comp_hip_knee_performance_rate",
    ]

    target = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    df_perf = df[granularity + dimensions]

    df_prev = df_perf.copy()
    df_prev["fiscal_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in dimensions}
    )
    df_merged = pd.merge(
        df_perf,
        df_prev,
        on=["facility_id", "fiscal_year"],
        how="inner",
    )

    prev_cols = dimensions

    # Drop NaNs
    cols_to_check = target + [f"{c}_prev" for c in prev_cols]
    df_final = df_merged.dropna(subset=cols_to_check)

    # Features: exclude previous-year targets to avoid leakage
    feature_cols = [f"{c}_prev" for c in prev_cols if c not in target]

    x = df_final[feature_cols].values
    y = df_final[target].values
    delta_y = y - df_final[[f"{c}_prev" for c in target]].values

    return x, delta_y


def structure_data_ar(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare autoregressive features for a single target (AMI)."""
    performance_cols = [
        c for c in df.columns if "performance_rate" in c
    ]
    df_perf = df[["facility_id", "fiscal_year"] + performance_cols]

    df_prev = df_perf.copy()
    df_prev["fiscal_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in performance_cols}
    )

    df_merged = pd.merge(
        df_perf,
        df_prev,
        on=["facility_id", "fiscal_year"],
        how="inner",
    )

    cols_to_check = [
        "mort_30_ami_performance_rate",
        "mort_30_ami_performance_rate_prev",
    ]
    df_clean = df_merged.dropna(subset=cols_to_check)

    x = df_clean[["mort_30_ami_performance_rate_prev"]].values
    y = df_clean["mort_30_ami_performance_rate"].values
    return x, y


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Model:
    model = LinearRegression()
    model.fit(x, y)
    return model


def fit_lasso_regression(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> MultiTaskLasso:
    model = MultiTaskLasso(alpha=alpha, max_iter=1_000)
    model.fit(x, y)
    return model


def fit_decision_tree_regression(
    x: np.ndarray, y: np.ndarray
) -> DecisionTreeRegressor:
    # model = DecisionTreeRegressor()
    # model = DecisionTreeRegressor(
    #     max_depth=24,
    #     min_samples_split=2,
    #     min_samples_leaf=16,
    #     random_state=42,
    # )
    # model = DecisionTreeRegressor()
    model = RandomForestRegressor(1_000)
    # model = GradientBoostingRegressor()
    model.fit(x, y)
    return model


def metrics(model: Model, x: np.ndarray, y: np.ndarray) -> None:
    y_pred = model.predict(x)
    print(f"R² (model score): {model.score(x, y):.4f}")
    print(f"R² (manual r2_score): {r2_score(y, y_pred):.4f}")


def metrics_relative(
    model: Model, x: np.ndarray, y: np.ndarray
) -> None:
    y_pred = model.predict(x)
    mae_rel = np.mean(np.abs(y - y_pred)) / np.mean(np.abs(y))
    print(f"R² (model score): {model.score(x, y):.4f}")
    print(f"R² (manual r2_score): {r2_score(y, y_pred):.4f}")
    print(f"Relative MAE: {mae_rel:.4f}")


def plot_delta_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
) -> None:
    n_targets = y_true.shape[1]
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_targets:
            ax.axis("off")
            continue
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        min_val, max_val = (
            y_true[:, i].min(),
            y_true[:, i].max(),
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=1,
        )
        ax.set_title(target_names[i])
        ax.set_xlabel("Δy true")
        ax.set_ylabel("Δy predicted")

    plt.tight_layout()
    plt.savefig(f"./metrics/medicare", dpi=150)
    # plt.show()
    plt.close()


# 0.2841
if __name__ == "__main__":
    df = load_data()
    df_read = load_readmissions_scaled()
    # x, y = structure_data_multivar_with_readmissions(df, df_read)
    x, y = structure_data_multivar_with_readmissions_and_demographics(
        df, df_read
    )
    # x, y = structure_data_multivar(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    # BEST MODEL CURRENTLY
    model = fit_linear_regression(x_train, y_train)
    # model = fit_decision_tree_regression(x_train, y_train)
    # model = fit_lasso_regression(x_train, y_train, 1e-4)

    print("Metrics on training set:")
    metrics(model, x_train, y_train)

    print("\nMetrics on test set:")
    metrics(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    delta_pred = model.predict(x_test)
    delta_true = y_test

    print(
        "Δy true min/max/mean:",
        delta_true.min(),
        delta_true.max(),
        delta_true.mean(),
    )
    print(
        "Δy predicted min/max/mean:",
        delta_pred.min(),
        delta_pred.max(),
        delta_pred.mean(),
    )

    target_names = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    plot_delta_scatter(delta_true, delta_pred, target_names)
