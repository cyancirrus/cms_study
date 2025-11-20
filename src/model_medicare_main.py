from __future__ import annotations
import sqlite3
from typing import List, Protocol, Tuple
from initialize_environment import (
    RANDOM_STATE,
    ENGINE,
    GENERATE_PREDICTIONS,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tables import CmsSchema
from train.models import (
    plot_delta_scatter,
    metrics_rsquared,
    metrics_relative,
    fit_linear_regression,
    # fit_lasso_regression,
    # fit_decision_tree_regression,
    # fit_gbm_regression,
    # fit_random_forest_regression
)


def load_data() -> pd.DataFrame:
    """Load all hospital clinical outcomes."""
    return ENGINE.read(CmsSchema.hvbp_clinical_outcomes)


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
    df = ENGINE.exec(query)

    df = df.rename(columns={"submission_year": "submission_year"})
    df["predicted_readmission_rate"] /= 1_000
    df["expected_readmission_rate"] /= 1_000
    return df


def structure_data_ar(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare autoregressive features for a single target (AMI)."""
    performance_cols = [
        c for c in df.columns if "performance_rate" in c
    ]
    df_perf = df[["facility_id", "submission_year"] + performance_cols]

    df_prev = df_perf.copy()
    df_prev["submission_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in performance_cols}
    )

    df_merged = pd.merge(
        df_perf,
        df_prev,
        on=["facility_id", "submission_year"],
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


def structure_data_multivar_with_readmissions(
    df: pd.DataFrame, df_read: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix X and target delta_y without leakage."""
    granularity = ["submission_year", "facility_id"]

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
        on=["facility_id", "submission_year"],
        how="left",
    )

    # Shift FORWARD to get previous year's data
    df_prev = df_merged.copy()
    df_prev["submission_year"] += 1

    prev_cols = dimensions + list(
        df_read.columns.difference(["facility_id", "submission_year"])
    )
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    # Now current year row gets actual previous year's features
    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "submission_year"],
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


def structure_data_multivar(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare multivariate features and delta targets without leakage."""
    granularity = ["submission_year", "facility_id"]

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
    df_prev["submission_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in dimensions}
    )
    df_merged = pd.merge(
        df_perf,
        df_prev,
        on=["facility_id", "submission_year"],
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


def structure_data_multivar_with_readmissions_and_demographics_full(
    df: pd.DataFrame,
    df_read: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full panel df_final with current-year + previous-year
    clinical/readmission/demo features, plus targets.
    """
    granularity = ["submission_year", "facility_id"]

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
    df_fac_zip = ENGINE.exec(
        """
        SELECT
            submission_year AS submission_year,
            facility_id,
            zip_code
        FROM facility_zip_code;
        """,
    )
    df_zip_demo = ENGINE.exec(
        """
        SELECT
            zip_code,
            log(msa_personal_income_k) as msa_personal_income_k,
            log(msa_population_density)  as msa_population_density,
            log(msa_per_capita_income)  as msa_per_capita_income
        FROM zip_demographics;
        """,
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
    df["submission_year"] = df["submission_year"].astype("int64")

    df_read["facility_id"] = df_read["facility_id"]
    df_read["submission_year"] = df_read["submission_year"].astype(
        "int64"
    )

    df_fac_zip["facility_id"] = df_fac_zip["facility_id"]
    df_fac_zip["submission_year"] = df_fac_zip[
        "submission_year"
    ].astype("int64")
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)

    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)

    # facility-year -> zip-year -> demographics
    df_zip = pd.merge(
        df_fac_zip, df_zip_demo, on="zip_code", how="left"
    )

    # --- Merge with readmissions data ---
    df_merged = pd.merge(
        df,
        df_read,
        on=["facility_id", "submission_year"],
        how="left",
    )

    # --- Add current-year demographics (these will get shifted to prev) ---
    df_merged = pd.merge(
        df_merged,
        df_zip[
            [
                "facility_id",
                "submission_year",
                "msa_personal_income_k",
                "msa_population_density",
                "msa_per_capita_income",
            ]
        ],
        on=["facility_id", "submission_year"],
        how="left",
    )

    # ---------- IMPORTANT: define what “prev” means ----------

    # We want df_final row at year t to contain:
    #   - current targets y_t
    #   - prev features from year t-1
    #
    # So we take df_merged (year t-1) and make its submission_year = t
    # so that it joins onto df_merged at t.
    df_prev = df_merged.copy()
    df_prev["submission_year"] += 1  # t-1 -> t

    prev_cols = (
        dimensions
        + list(
            df_read.columns.difference(
                ["facility_id", "submission_year"]
            )
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

    # For each (facility, t), attach previous-year features (t-1)
    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "submission_year"],
        how="inner",
    )

    # Drop rows with NaNs in any required columns
    cols_to_check = target + [f"{c}_prev" for c in prev_cols]
    df_final = df_final.dropna(subset=cols_to_check)

    return df_final, target


def structure_data_multivar_with_readmissions_and_demographics(
    df: pd.DataFrame,
    df_read: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df_final, target = (
        structure_data_multivar_with_readmissions_and_demographics_full(
            df, df_read
        )
    )
    prev_cols = (
        [
            # same dimensions as before
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
        + list(
            df_read.columns.difference(
                ["facility_id", "submission_year"]
            )
        )
        + [
            "msa_personal_income_k",
            "msa_population_density",
            "msa_per_capita_income",
        ]
    )

    feature_cols = [f"{c}_prev" for c in prev_cols]

    X = df_final[feature_cols].values
    y = df_final[target].values
    y_prev = df_final[[f"{c}_prev" for c in target]].values
    delta_y = y - y_prev

    return X, delta_y, target


def predict_next_period_for_latest_year(
    model: Any,
    df: pd.DataFrame,
    df_read: pd.DataFrame,
) -> pd.DataFrame:
    """
    Using trained delta model, predict next-period targets (T+1) for the
    latest observed submission_year T in df.
    """
    df_final, target = (
        structure_data_multivar_with_readmissions_and_demographics_full(
            df, df_read
        )
    )

    # Latest observed year T
    T = df_final["submission_year"].max()
    df_T = df_final[df_final["submission_year"] == T].copy()

    # Features at "prev" time = info at T for step T->T+1
    prev_cols = (
        [
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
        + list(
            df_read.columns.difference(
                ["facility_id", "submission_year"]
            )
        )
        + [
            "msa_personal_income_k",
            "msa_population_density",
            "msa_per_capita_income",
        ]
    )

    feature_cols = [f"{c}_prev" for c in prev_cols]
    X_prev = df_T[feature_cols].values

    # Predict deltas (Δy_{T+1})
    delta_pred = model.predict(
        X_prev
    )  # shape (n_facilities, len(target))

    # Base = current-year targets y_T
    y_T = df_T[target].values

    # y_{T+1} = y_T + Δŷ_{T+1}
    y_T_plus_1_pred = y_T + delta_pred

    # Package
    df_pred = df_T[["facility_id"]].copy()
    df_pred["submission_year"] = T + 1

    for i, col in enumerate(target):
        df_pred[f"{col}"] = y_T_plus_1_pred[:, i]

    return df_pred


# 0.2841
if __name__ == "__main__":
    df = load_data()
    df_read = load_readmissions_scaled()
    # x, y = structure_data_multivar_with_readmissions(df, df_read)
    x, delta_y, target = (
        structure_data_multivar_with_readmissions_and_demographics(
            df, df_read
        )
    )
    # x, y = structure_data_multivar(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, delta_y, test_size=0.2, random_state=RANDOM_STATE
    )
    # BEST MODEL CURRENTLY
    model = fit_linear_regression(x_train, y_train)
    # model = fit_decision_tree_regression(x_train, y_train)
    # model = fit_lasso_regression(x_train, y_train, 1e-4)

    print("Metrics on training set:")
    metrics_rsquared(model, x_train, y_train)

    print("\nMetrics on test set:")
    metrics_rsquared(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    delta_pred = model.predict(x_test)
    delta_true = y_test
    target_names = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    plot_delta_scatter(
        "metrics/medicare.png", delta_true, delta_pred, target_names
    )

    if GENERATE_PREDICTIONS:
        predictions = predict_next_period_for_latest_year(
            model, df, df_read
        )
        ENGINE.write(
            predictions, CmsSchema.prediction_hvbp_clinical_outcomes
        )
