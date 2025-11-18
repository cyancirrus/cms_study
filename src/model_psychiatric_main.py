from __future__ import annotations
from initialize_environment import RANDOM_STATE, ENGINE
import sqlite3
from typing import List, Tuple, Final
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from train.search import gbm_grid_search
import matplotlib.pyplot as plt

from train.models import (
    plot_delta_scatter,
    metrics_rsquared,
    metrics_relative,
    fit_linear_regression,
    fit_lasso_regression,
    fit_decision_tree_regression,
    fit_gbm_regression,
    # fit_random_forest_regression
)

DATABASE = "source.db"
# hbips_2_overall_rate_per_1000 is not NULL
# and hbips_3_overall_rate_per_1000 is not NULL
QUERY_IPFQR_MEASURES: Final[
    str
] = """
SELECT * FROM ipfqr_quality_measures_facility
    WHERE 
        smd_percent IS NOT NULL
        AND sub_2_percent IS NOT NULL
        AND sub_3_percent IS NOT NULL
        AND tob_3_percent IS NOT NULL
        AND tob_3a_percent IS NOT NULL
        AND tr_1_percent IS NOT NULL
        AND imm_2_percent IS NOT NULL
        AND readm_30_ipf_rate IS NOT NULL

;"""


def load_data() -> pd.DataFrame:
    """Load IPFQR facility quality measures."""
    df = ENGINE.exec(QUERY_IPFQR_MEASURES)
    return df


def structure_ipfqr_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and delta targets for all _percent metrics,
    using previous-year autoregressive features.
    """
    granularity = ["submission_year", "facility_id"]
    # Identify all percent columns as targets
    target_cols = [
        # "hbips_2_overall_rate_per_1000",
        # "hbips_3_overall_rate_per_1000",
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


def structure_ipfqr_with_demographics(
    df: pd.DataFrame, db_path: str = DATABASE
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix X and target delta_y for IPFQR data,
    using previous-year autoregressive features plus zip demographics.

    - Targets: IPF quality/readmission metrics (current year).
    - Features: previous-year targets + previous-year demographics.
    """

    granularity = ["submission_year", "facility_id"]

    # Core metrics you know are non-null (aligned with SQL WHERE)
    target_cols = [
        # "hbips_2_overall_rate_per_1000",
        # "hbips_3_overall_rate_per_1000",
        "smd_percent",
        "sub_2_percent",
        "sub_3_percent",
        "tob_3_percent",
        "tob_3a_percent",
        "tr_1_percent",
        "imm_2_percent",
        "readm_30_ipf_rate",
    ]

    # --- load facility_zip_code + zip_demographics ---
    with sqlite3.connect(db_path) as conn:
        df_fac_zip = pd.read_sql_query(
            """
            SELECT
                submission_year AS submission_year,
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
                log(msa_personal_income_k / 100_000) as msa_personal_income_k,
                log(msa_population_density / 1000_000)  as msa_population_density,
                log(msa_per_capita_income / 100_000)  as msa_per_capita_income
            FROM zip_demographics;
            """,
            conn,
        )

    # Z-score normalize demos
    for col in [
        "msa_personal_income_k",
        "msa_population_density",
        # "msa_per_capita_income",
    ]:
        mean = df_zip_demo[col].mean()
        std = df_zip_demo[col].std()
        df_zip_demo[col] = (df_zip_demo[col] - mean) / std

    # normalize dtypes for join keys
    df["facility_id"] = df["facility_id"]
    df["submission_year"] = df["submission_year"].astype("int64")

    df_fac_zip["facility_id"] = df_fac_zip["facility_id"]
    df_fac_zip["submission_year"] = df_fac_zip[
        "submission_year"
    ].astype("int64")
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)

    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)

    # facility-year -> zip -> demographics
    df_zip = pd.merge(
        df_fac_zip,
        df_zip_demo,
        on="zip_code",
        how="left",
    )

    # --- base perf data (only targets + keys) ---
    df_perf = df[granularity + target_cols]

    # --- attach current-year demographics (will become "prev" after shift) ---
    df_merged = pd.merge(
        df_perf,
        df_zip[
            [
                "facility_id",
                "submission_year",
                "msa_personal_income_k",
                "msa_population_density",
                # "msa_per_capita_income",
            ]
        ],
        on=["facility_id", "submission_year"],
        how="left",
    )

    # --- shift to build previous-year features ---
    df_prev = df_merged.copy()
    # so that row (facility, year=t) will pick up year t-1's features
    df_prev["submission_year"] += 1

    prev_cols = target_cols + [
        "msa_personal_income_k",
        "msa_population_density",
        # "msa_per_capita_income",
    ]

    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    # Merge current year with previous-year features
    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "submission_year"],
        how="inner",
    )

    # Drop rows with NaNs in targets or all prev features
    cols_to_check = (
        target_cols
        + [f"{c}_prev" for c in target_cols]  # prev targets (for delta)
        + [
            "msa_personal_income_k_prev",
            "msa_population_density_prev",
            # "msa_per_capita_income_prev",
        ]
    )
    df_final = df_final.dropna(subset=cols_to_check)

    # Features: previous-year targets + previous-year demos
    feature_cols = [f"{c}_prev" for c in prev_cols]

    x = df_final[feature_cols].values
    y = df_final[target_cols].values
    # deltas relative to previous-year target values
    delta_y = y - df_final[[f"{c}_prev" for c in target_cols]].values

    return x, delta_y, target_cols


# 25 km
# R Squared: 0.1335


if __name__ == "__main__":
    df = load_data()
    # x, delta_y, targets = structure_ipfqr_data(df)
    x, delta_y, targets = structure_ipfqr_with_demographics(df)

    print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x, delta_y, test_size=0.2, random_state=RANDOM_STATE
    )
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # model = fit_lasso_regression(x_train, y_train, alpha = 0.6);

    # model = fit_decision_tree_regression(
    #     x_train,
    #     y_train,
    #     max_depth=16,
    #     min_samples_split=30,
    #     min_samples_leaf=4,
    #     random_state=RANDOM_STATE,
    # )

    # 25 km, 3knn R^2 = 0.1335
    ## with demographics
    model = fit_gbm_regression(
        x_train,
        y_train,
        n_estimators=64,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    ## without demographics
    # model = fit_gbm_regression(
    #     x_train,
    #     y_train,

    #     n_estimators = 40,
    #     learning_rate = 0.125,
    #     max_depth = 2,
    #     random_state = RANDOM_STATE,
    # )

    print("Metrics on training set:")
    metrics_rsquared(model, x_train, y_train)
    print("\nMetrics on test set:")
    metrics_rsquared(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    delta_pred = model.predict(x_test)
    plot_delta_scatter(
        "./metrics/psychiatric", y_test, delta_pred, targets
    )

    # ## for the scructure base ipfqr
    # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    #     n_estimators_range = [40, 44, 48, 64, 128, 256],
    #     learning_rate_range = np.linspace(0.05, 0.15, 5),
    #     max_depth_range = [2, 3],
    #     # max_depth_range = [2, 5, 6],
    # )

    ## for the scructure ipfqr with demographics
    # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    #     n_estimators_range = [36, 38, 40, 42, 44],
    #     learning_rate_range = np.linspace(0.02, 0.10, 5),
    #     max_depth_range = [2, 3, 4, 5, 6],
    # )

# )
