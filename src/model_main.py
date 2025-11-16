from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
from typing import (
    List,
    Protocol,
    Tuple,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATABASE = "source.db"


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query("SELECT * FROM hvbp_clinical_outcomes;", conn)


def structure_data_multivar(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Input measures
    granularity: List[str] = [
        "fiscal_year",
        "facility_id",
    ]

    dimensions: List[str] = [
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
    target: List[str] = [
        "mort_30_ami_performance_rate",
        "mort_30_hf_performance_rate",
        "mort_30_pn_performance_rate",
        "mort_30_copd_performance_rate",
        "mort_30_cabg_performance_rate",
        "comp_hip_knee_performance_rate",
    ]

    # --- Step 3: Focus on relevant columns ---
    df_perf = df[granularity + dimensions]

    # --- Step 4: Shift previous year ---
    df_prev = df_perf.copy()
    df_prev["fiscal_year"] += 1
    df_prev = df_prev.rename(columns={c: c + "_prev" for c in dimensions})

    # Merge current year with previous year
    df_merged = pd.merge(
        df_perf, df_prev, on=["facility_id", "fiscal_year"], how="inner"
    )

    # Drop rows with NaNs in features or targets
    cols_to_check = target + [c + "_prev" for c in dimensions]
    df_clean = df_merged.dropna(subset=cols_to_check)

    # Prepare X (previous year features) and y (current year targets)
    x = df_clean[[c + "_prev" for c in dimensions]].values
    y = df_clean[target].values

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    return x, y


def structure_data_ar(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # --- Step 3: Focus on performance columns ---
    performance_cols = [c for c in df.columns if "performance_rate" in c]
    df_perf = df[["facility_id", "fiscal_year"] + performance_cols]

    # --- Step 4: Shift previous year ---
    df_prev = df_perf.copy()
    df_prev["fiscal_year"] += 1
    df_prev = df_prev.rename(columns={c: c + "_prev" for c in performance_cols})

    df_merged = pd.merge(
        df_perf, df_prev, on=["facility_id", "fiscal_year"], how="inner"
    )

    # --- Step 5: Drop NaNs for target metric ---
    cols_to_check = [
        "mort_30_ami_performance_rate",
        "mort_30_ami_performance_rate_prev",
    ]
    df_clean = df_merged.dropna(subset=cols_to_check)

    # --- Step 6: Prepare data for regression ---
    x = df_clean[["mort_30_ami_performance_rate_prev"]].values
    y = df_clean["mort_30_ami_performance_rate"].values
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    return (x, y)


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Model:
    model = LinearRegression()
    model.fit(x, y)
    return model


def metrics(model, x: np.ndarray, y: np.ndarray):
    y_pred = model.predict(x)
    r2 = model.score(x, y)
    r2_manual = r2_score(y, y_pred)

    print("Coefficient (A):", model.coef_[0])
    print("Intercept:", model.intercept_)
    print(f"R² (explained variance): {r2:.4f}")
    print(f"R² (manual check): {r2_manual:.4f}")


if __name__ == "__main__":
    data = load_data()
    # x, y = structure_data_ar(data)
    x, y = structure_data_multivar(data)
    model = fit_linear_regression(x, y)
    metrics(model, x, y)
