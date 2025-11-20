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
from tables import CmsSchema
import matplotlib.pyplot as plt
from train.models import (
    plot_delta_scatter,
    metrics_rsquared,
    metrics_relative,
    fit_linear_regression,
    z_transform,
    # metrics_effective_delta_rsquared,
    # fit_lasso_regression,
    # fit_decision_tree_regression,
    # fit_gbm_regression,
    # fit_random_forest_regression
)

DATABASE = "source.db"
QUERY_IPFQR_MEASURES: Final[
    str
] = """
SELECT
        submission_year,
        facility_id,
        log(1 + hbips_2_overall_rate_per_1000) as hbips_2_overall_rate_per_1000,
        log(1 + hbips_3_overall_rate_per_1000) as hbips_3_overall_rate_per_1000,
        log(smd_percent) as smd_percent,
        log(sub_2_percent) as sub_2_percent,
        log(sub_3_percent) as sub_3_percent,
        log(tob_3_percent) as tob_3_percent,
        log(tob_3a_percent) as tob_3a_percent,
        log(tr_1_percent) as tr_1_percent,
        log(imm_2_percent) as imm_2_percent,
        log(readm_30_ipf_rate) as readm_30_ipf_rate,
        hbips_2_overall_rate_per_1000 as hbips_2_overall_rate_per_1000_raw,
        hbips_3_overall_rate_per_1000 as hbips_3_overall_rate_per_1000_raw,
        smd_percent as smd_percent_raw,
        sub_2_percent as sub_2_percent_raw,
        sub_3_percent as sub_3_percent_raw,
        tob_3_percent as tob_3_percent_raw,
        tob_3a_percent as tob_3a_percent_raw,
        tr_1_percent as tr_1_percent_raw,
        imm_2_percent as imm_2_percent_raw,
        readm_30_ipf_rate as readm_30_ipf_rate_raw
    
    FROM
        ipfqr_quality_measures_facility
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
;"""
# submission_year,
# facility_id,
# hbips_2_overall_rate_per_1000,
# hbips_3_overall_rate_per_1000,
# smd_percent as smd_percent,
# sub_2_percent as sub_2_percent,
# sub_3_percent as sub_3_percent,
# tob_3_percent as tob_3_percent,
# tob_3a_percent as tob_3a_percent,
# tr_1_percent as tr_1_percent,
# imm_2_percent as imm_2_percent,
# readm_30_ipf_rate as readm_30_ipf_rate


def load_data() -> pd.DataFrame:
    """Load IPFQR facility quality measures."""
    df = ENGINE.exec(QUERY_IPFQR_MEASURES)
    z_transform(df, "hbips_2_overall_rate_per_1000")
    z_transform(df, "hbips_3_overall_rate_per_1000")
    z_transform(df, "smd_percent")
    z_transform(df, "sub_2_percent")
    z_transform(df, "sub_3_percent")
    z_transform(df, "tob_3_percent")
    z_transform(df, "tob_3a_percent")
    z_transform(df, "tr_1_percent")
    z_transform(df, "readm_30_ipf_rate")
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


def structure_ipfqr_double_diff(
    df: pd.DataFrame, db_path: str = DATABASE
):
    """
    Prepare features for double-difference model:
        Δy_{t+1} = f(Δy_t, prev_demographics)
    Returns:
        x: previous delta + demographics features
        y: current delta target
        target_delta_cols: list of target delta column names
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

    # --- load demographics ---
    with sqlite3.connect(db_path) as conn:
        df_fac_zip = pd.read_sql_query(
            "SELECT submission_year, facility_id, zip_code FROM facility_zip_code;",
            conn,
        )
        df_zip_demo = pd.read_sql_query(
            """
            SELECT zip_code,
                log(msa_personal_income_k) as msa_personal_income_k,
                log(msa_population_density) as msa_population_density,
                log(msa_per_capita_income) as msa_per_capita_income
            FROM zip_demographics;
            """,
            conn,
        )

    # z-score normalize
    for col in [
        "msa_personal_income_k",
        "msa_population_density",
        "msa_per_capita_income",
    ]:
        df_zip_demo[col] = (
            df_zip_demo[col] - df_zip_demo[col].mean()
        ) / df_zip_demo[col].std()

    # merge facility -> zip -> demo
    df_fac_zip["submission_year"] = df_fac_zip[
        "submission_year"
    ].astype(int)
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)
    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)
    df_zip = pd.merge(
        df_fac_zip, df_zip_demo, on="zip_code", how="left"
    )

    # --- merge performance data with demographics ---
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

    # --- compute first difference delta_y_t = y_t - y_{t-1} ---
    df_prev = df_merged.copy()
    df_prev["submission_year"] += 1
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in target_cols}
    )

    df_delta = pd.merge(
        df_merged,
        df_prev[
            ["facility_id", "submission_year"]
            + [f"{c}_prev" for c in target_cols]
        ],
        on=["facility_id", "submission_year"],
        how="inner",
    )

    for c in target_cols:
        df_delta[f"{c}_delta"] = df_delta[c] - df_delta[f"{c}_prev"]

    # --- compute double difference: previous delta features ---
    delta_cols = [f"{c}_delta" for c in target_cols]
    df_prev_delta = df_delta[
        ["facility_id", "submission_year"]
        + delta_cols
        + ["msa_personal_income_k", "msa_population_density"]
    ].copy()
    df_prev_delta["submission_year"] += 1

    # rename previous delta columns for features
    df_prev_delta = df_prev_delta.rename(
        columns={c: f"{c}_prev" for c in delta_cols}
    )

    # merge back to get x (prev delta + current demo) and y (current delta)
    df_final = pd.merge(
        df_delta,
        df_prev_delta,
        on=[
            "facility_id",
            "submission_year",
            "msa_personal_income_k",
            "msa_population_density",
        ],
        how="inner",
    )

    # drop any remaining NaNs
    feature_cols = [f"{c}_prev" for c in target_cols] + [
        "msa_personal_income_k",
        "msa_population_density",
    ]
    target_delta_cols = [f"{c}_delta" for c in target_cols]
    df_final = df_final.dropna(subset=feature_cols + target_delta_cols)

    x = df_final[feature_cols].values
    y = df_final[target_delta_cols].values

    return x, y, target_delta_cols


def structure_ipfqr_with_demographics(df: pd.DataFrame):
    """
    Prepare feature matrix X and target delta_y for IPFQR data,
    using previous-year autoregressive features plus zip demographics.

    - Targets: IPF quality/readmission metrics (current year).
    - Features: previous-year targets + previous-year demographics.
    """

    granularity = ["submission_year", "facility_id"]

    # Core metrics you know are non-null (aligned with SQL WHERE)
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

    # Z-score normalize demos
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

    return x, y, target_cols, delta_y


def predict_next_ipfqr_period(
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict next-period IPFQR metrics (T+1) from the latest year in df.
    """
    # Rebuild full panel with prev-year features
    base_values = [
        "hbips_2_overall_rate_per_1000_raw",
        "hbips_3_overall_rate_per_1000_raw",
        "smd_percent_raw",
        "sub_2_percent_raw",
        "sub_3_percent_raw",
        "tob_3_percent_raw",
        "tob_3a_percent_raw",
        "tr_1_percent_raw",
        "imm_2_percent_raw",
        "readm_30_ipf_rate_raw",
    ]

    _, y, target_cols, _ = structure_ipfqr_with_demographics(df)

    # Get latest year
    latest_year = df["submission_year"].max()

    # Subset to latest year
    df_latest = df[df["submission_year"] == latest_year].copy()

    # Build prev-feature columns (these are the inputs for delta model)
    prev_cols = target_cols + [
        "msa_personal_income_k",
        "msa_population_density",
    ]
    feature_cols = [f"{c}_prev" for c in prev_cols]

    # Rebuild panel for latest year to get *_prev columns
    df_fac_zip = ENGINE.exec(
        "SELECT submission_year, facility_id, zip_code FROM facility_zip_code;"
    )
    df_zip_demo = ENGINE.exec(
        """
        SELECT
            zip_code,
            log(msa_personal_income_k) as msa_personal_income_k,
            log(msa_population_density) as msa_population_density,
            log(msa_per_capita_income) as msa_per_capita_income
        FROM zip_demographics;
        """
    )
    # Z-score normalize
    for col in [
        "msa_personal_income_k",
        "msa_population_density",
        "msa_per_capita_income",
    ]:
        df_zip_demo[col] = (
            df_zip_demo[col] - df_zip_demo[col].mean()
        ) / df_zip_demo[col].std()

    df_fac_zip["submission_year"] = df_fac_zip[
        "submission_year"
    ].astype("int64")
    df_latest["submission_year"] = df_latest["submission_year"].astype(
        "int64"
    )
    df_fac_zip["zip_code"] = df_fac_zip["zip_code"].astype(str)
    df_zip_demo["zip_code"] = df_zip_demo["zip_code"].astype(str)

    df_zip = pd.merge(
        df_fac_zip, df_zip_demo, on="zip_code", how="left"
    )
    df_merged = pd.merge(
        df_latest,
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

    # Build prev columns for prediction: row at T sees features at T as prev for T+1
    df_prev = df_merged.copy()
    df_prev = df_prev.rename(
        columns={c: f"{c}_prev" for c in prev_cols}
    )

    df_final = pd.merge(
        df_merged,
        df_prev,
        on=["facility_id", "submission_year"],
        how="left",
    )

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")
    X_prev = imputer.fit_transform(df_final[feature_cols])

    # # Features for prediction
    # X_prev = df_final[feature_cols].values

    # Predict deltas
    delta_pred = model.predict(X_prev)

    # Add deltas to current-year raw targets to get next-period predictions
    y_prev = df_latest[base_values].values
    y_T_plus_1_pred = y_prev + delta_pred

    # Build DataFrame to write
    df_pred = df_latest[["facility_id"]].copy()
    df_pred["submission_year"] = latest_year + 1
    for i, col in enumerate(base_values):
        orig_name = col.replace("_raw", "")  # strip the _raw suffix
        df_pred[orig_name] = y_T_plus_1_pred[:, i]

    return df_pred


# 25 km
# R Squared: 0.1335


if __name__ == "__main__":
    df = load_data()

    # x, delta_y, targets = structure_ipfqr_data(df)

    x, y, targets, delta_y = structure_ipfqr_with_demographics(df)
    # x, delta_y, targets = structure_ipfqr_double_diff(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, delta_y, test_size=0.2, random_state=RANDOM_STATE
    )
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    model = fit_linear_regression(x_train, y_train)
    # model = fit_lasso_regression(x_train, y_train, alpha = 0.06);

    # model = fit_decision_tree_regression(
    #     x_train,
    #     y_train,
    #     max_depth=16,
    #     min_samples_split=30,
    #     min_samples_leaf=4,
    #     random_state=RANDOM_STATE,
    # )

    # model = fit_gbm_regression(
    #     x_train,
    #     y_train,
    #     n_estimators=96,
    #     learning_rate=0.075,
    #     max_depth=3,
    #     random_state=RANDOM_STATE,
    # )
    # # n_estimators=64, learning_rate=0.094, max_depth=3

    # ## without demographics
    # # model = fit_gbm_regression(
    # #     x_train,
    # #     y_train,

    # #     n_estimators = 40,
    # #     learning_rate = 0.125,
    # #     max_depth = 2,
    # #     random_state = RANDOM_STATE,
    # # )

    print("Metrics on training set:")
    metrics_rsquared(model, x_train, y_train)
    print("\nMetrics on test set:")
    metrics_rsquared(model, x_test, y_test)
    metrics_relative(model, x_test, y_test)

    # # NOTE: must use this if not going to delta y the model
    # metrics_effective_delta_rsquared(model, x_test, y_test, delta_y)

    delta_pred = model.predict(x_test)
    plot_delta_scatter(
        "./metrics/psychiatric", y_test, delta_pred, targets
    )

    # Compute variances
    var_y = np.var(y, ddof=1)  # sample variance
    var_Dy = np.var(delta_y, ddof=1)

    # Print nicely
    print(f"Variance y := {var_y:.4f}")
    print(f"Variance Δy := {var_Dy:.4f}")
    # # # ## for the scructure ipfqr with demographics
    # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    #     n_estimators_range = [24, 48, 64, 128],
    #     learning_rate_range = np.linspace(0.05, 0.15, 4),
    #     max_depth_range = [3],
    # )

    # ## for the scructure base ipfqr
    # # gbm_grid_search(RANDOM_STATE, x_train, y_train, x_test, y_test,
    # #     n_estimators_range = [16, 32, 42, 64],
    # #     learning_rate_range = np.linspace(0.02, 0.10, 4),
    # #     max_depth_range = [2, 3],
    # # )
    if GENERATE_PREDICTIONS:
        predictions = predict_next_ipfqr_period(model, df)

        ENGINE.write(
            predictions,
            CmsSchema.prediction_ipfqr_quality_measures_facility,
        )
