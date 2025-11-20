from __future__ import annotations

from initialize_environment import (
    RANDOM_STATE,
    ENGINE,
    GENERATE_PREDICTIONS,
)
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
from tables import CmsSchema

from train.models import (
    plot_delta_scatter,
    metrics_rsquared,
    metrics_relative,
    fit_linear_regression,
    z_transform,
    # fit_lasso_regression,
    # fit_decision_tree_regression,
    # fit_gbm_regression,
    # fit_random_forest_regression
)

DATABASE = "source.db"
# hbips_2_overall_rate_per_1000 is not NULL
# and hbips_3_overall_rate_per_1000 is not NULL
QUERY_IPFQR_MEASURES: Final[
    str
] = """
SELECT * FROM hvbp_tps

    WHERE 
        unweighted_normalized_clinical_outcomes_domain_score IS NOT NULL
        AND weighted_normalized_clinical_outcomes_domain_score IS NOT NULL
        AND unweighted_person_and_community_engagement_domain_score IS NOT NULL
        AND weighted_person_and_community_engagement_domain_score IS NOT NULL
        AND unweighted_normalized_safety_domain_score IS NOT NULL
        AND weighted_safety_domain_score IS NOT NULL
        AND unweighted_normalized_efficiency_and_cost_reduction_domain_score IS NOT NULL
        AND weighted_efficiency_and_cost_reduction_domain_score IS NOT NULL
        AND total_performance_score IS NOT NULL

;"""


def load_data() -> pd.DataFrame:
    """Load IPFQR facility quality measures."""
    df = ENGINE.exec(QUERY_IPFQR_MEASURES)
    # z_transform(
    #     df, "unweighted_normalized_clinical_outcomes_domain_score"
    # )
    # z_transform(
    #     df, "weighted_normalized_clinical_outcomes_domain_score"
    # )
    # z_transform(
    #     df, "unweighted_person_and_community_engagement_domain_score"
    # )
    # z_transform(
    #     df, "weighted_person_and_community_engagement_domain_score"
    # )
    # z_transform(df, "unweighted_normalized_safety_domain_score")
    # z_transform(df, "weighted_safety_domain_score")
    # z_transform(
    #     df,
    #     "unweighted_normalized_efficiency_and_cost_reduction_domain_score",
    # )
    # z_transform(
    #     df, "weighted_efficiency_and_cost_reduction_domain_score"
    # )
    # z_transform(df, "total_performance_score")
    return df


def structure_hvbp_tps(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and delta targets for all _percent metrics,
    using previous-year autoregressive features.
    """
    granularity = ["submission_year", "facility_id"]
    # Identify all percent columns as targets
    target_cols = [
        "unweighted_normalized_clinical_outcomes_domain_score",
        "weighted_normalized_clinical_outcomes_domain_score",
        "unweighted_person_and_community_engagement_domain_score",
        "weighted_person_and_community_engagement_domain_score",
        "unweighted_normalized_safety_domain_score",
        "weighted_safety_domain_score",
        "unweighted_normalized_efficiency_and_cost_reduction_domain_score",
        "weighted_efficiency_and_cost_reduction_domain_score",
        "total_performance_score",
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


def structure_hvbp_tps_with_demographics_full(
    df: pd.DataFrame, db_path: str = DATABASE
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Same as structure_hvbp_tps_with_demographics, but returns df_final
    with all features + targets, instead of (x, delta_y).
    """

    granularity = ["submission_year", "facility_id"]

    target_cols = [
        "unweighted_normalized_clinical_outcomes_domain_score",
        "weighted_normalized_clinical_outcomes_domain_score",
        "unweighted_person_and_community_engagement_domain_score",
        "weighted_person_and_community_engagement_domain_score",
        "unweighted_normalized_safety_domain_score",
        "weighted_safety_domain_score",
        "unweighted_normalized_efficiency_and_cost_reduction_domain_score",
        "weighted_efficiency_and_cost_reduction_domain_score",
        "total_performance_score",
    ]

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

    df["facility_id"] = df["facility_id"]
    df["submission_year"] = df["submission_year"].astype("int64")

    df_fac_zip["facility_id"] = df_fac_zip["facility_id"]
    df_fac_zip["submission_year"] = df_fac_zip[
        "submission_year"
    ].astype("int64")
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)

    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)

    df_zip = pd.merge(
        df_fac_zip,
        df_zip_demo,
        on="zip_code",
        how="left",
    )

    df_perf = df[granularity + target_cols]

    df_merged = pd.merge(
        df_perf,
        df_zip[
            [
                "facility_id",
                "submission_year",
                "msa_personal_income_k",
                "msa_population_density",
            ]
        ],
        on=["facility_id", "submission_year"],
        how="left",
    )

    df_prev = df_merged.copy()
    df_prev["submission_year"] += 1

    prev_cols = target_cols + [
        "msa_personal_income_k",
        "msa_population_density",
    ]

    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "submission_year"],
        how="inner",
    )

    cols_to_check = (
        target_cols
        + [f"{c}_prev" for c in target_cols]
        + [
            "msa_personal_income_k_prev",
            "msa_population_density_prev",
        ]
    )
    df_final = df_final.dropna(subset=cols_to_check)

    return df_final, target_cols


def structure_hvbp_tps_with_demographics(
    df: pd.DataFrame, db_path: str = DATABASE
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df_final, target_cols = structure_hvbp_tps_with_demographics_full(
        df, db_path
    )

    prev_cols = target_cols + [
        "msa_personal_income_k",
        "msa_population_density",
    ]
    feature_cols = [f"{c}_prev" for c in prev_cols]

    x = df_final[feature_cols].values
    y = df_final[target_cols].values
    delta_y = y - df_final[[f"{c}_prev" for c in target_cols]].values

    return x, y, delta_y, target_cols


def predict_next_period_for_latest_year(
    model: Any,
    df: pd.DataFrame,
    db_path: str = DATABASE,
) -> pd.DataFrame:
    """
    Using trained delta model, predict next-period targets (t+1) for the
    latest observed submission_year in df.

    Returns a DataFrame with:
      facility_id, submission_year, <target_cols>_pred
    suitable for writing via EngineProtocol.write.
    """
    df_final, target_cols = structure_hvbp_tps_with_demographics_full(
        df, db_path
    )

    # 1. identify latest year we *have* as "current" (call it T)
    latest_year = df_final["submission_year"].max()

    # 2. subset to that year
    df_latest = df_final[
        df_final["submission_year"] == latest_year
    ].copy()

    # 3. build feature matrix X_prev (same as training features)
    prev_cols = target_cols + [
        "msa_personal_income_k",
        "msa_population_density",
    ]
    feature_cols = [f"{c}_prev" for c in prev_cols]

    X_prev = df_latest[
        feature_cols
    ].values  # features at year T (prev for T+1)

    # 4. predict *deltas* for next year (T+1)
    delta_pred = model.predict(
        X_prev
    )  # shape: (n_facilities, len(target_cols))

    # 5. get the "previous" target values that deltas are *relative to*
    # these are the *_prev columns, i.e. the y_T for step T -> T+1
    y_prev = df_latest[[f"{c}_prev" for c in target_cols]].values
    # 6. reconstruct y_{T+1} = y_T + Δŷ_{T+1} if normalizing values need to do something differenct
    y_T_plus_1_pred = y_prev + delta_pred
    # 7. package into a nice DataFrame
    df_pred = df_latest[["facility_id"]].copy()
    df_pred["submission_year"] = latest_year + 1  # this is T+1

    for i, col in enumerate(target_cols):
        df_pred[f"{col}_pred"] = y_T_plus_1_pred[:, i]

    return df_pred


if __name__ == "__main__":
    df = load_data()
    x, y, delta_y, targets = structure_hvbp_tps_with_demographics(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, delta_y, test_size=0.2, random_state=RANDOM_STATE
    )
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    model = fit_linear_regression(x_train, y_train)

    print("Metrics on training set:")
    metrics_rsquared(model, x_train, y_train)
    print("\nMetrics on test set:")
    metrics_rsquared(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    delta_pred = model.predict(x_test)
    plot_delta_scatter("./metrics/general", y_test, delta_pred, targets)

    # Compute variances
    var_y = np.var(y, ddof=1)  # sample variance
    var_Dy = np.var(delta_y, ddof=1)

    # Print nicely
    print(f"Variance y := {var_y:.4f}")
    print(f"Variance Δy := {var_Dy:.4f}")

    # ## for the scructure base hvbp_tps
    # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    #     n_estimators_range = [40, 44, 48, 64, 128, 256],
    #     learning_rate_range = np.linspace(0.05, 0.15, 5),
    #     max_depth_range = [2, 3],
    # )

    # ## for the scructure hvbp_tps with demographics
    # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    #     # n_estimators_range = [256, 300, 400, 512, ],
    #     n_estimators_range = [512,  624],
    #     learning_rate_range = np.linspace(0.75, 0.125, 4),
    #     max_depth_range = [3],
    # )
    if GENERATE_PREDICTIONS:
        model_predict = fit_linear_regression(x, delta_y)
        predictions = predict_next_period_for_latest_year(model, df)
        ENGINE.write(predictions, CmsSchema.prediction_hvbp_tps)
