from typing import Dict
import numpy as np
import pandas as pd
import sqlite3


def column_filter_summary(
    database: str,
    year: int,
    table: str,
    column: str,
    measure: str,
    condition: str,
) -> pd.DataFrame:
    with sqlite3.connect(database) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT
                {column},
                {measure}
            FROM {table}
            WHERE
                submission_year = {year}
                and {column} = \"{condition}\"
            """,
            conn,
        )
        df = df.convert_dtypes().infer_objects()
        return df


def column_summary(database: str, year: int, table: str, column: str) -> pd.DataFrame:
    with sqlite3.connect(database) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT
                {column}
            FROM {table}
            WHERE
                submission_year = {year}
            """,
            conn,
        )
        df = df.convert_dtypes().infer_objects()
        return df


def zeroized_score_by_group(
    database: str,
    year: int,
    table: str,
    label: str,
    count_better_col: str,
    count_worse_col: str,
    count_total_col: str,
    group: str,
) -> pd.DataFrame:
    query = f"""
    SELECT 
        {group},
        SUM({count_better_col}) as total_better,
        SUM({count_worse_col}) as total_worse,
        SUM({count_total_col}) as total_measures
    FROM {table}
    WHERE
        submission_year = {year}
    GROUP BY
        {group}
    """

    with sqlite3.connect(database) as conn:
        df = pd.read_sql_query(query, conn)

    # compute zeroized score
    df[f"{label}_zeroized_score"] = (df["total_better"] - df["total_worse"]) / df[
        "total_measures"
    ]
    return df[[group, f"{label}_zeroized_score"]]


def histogram_summary(
    df: pd.DataFrame, column: str, bin_count: int
) -> Dict[str, any] | None:
    """Compute histogram data for a column in a table."""
    if column not in df.columns:
        raise ValueError(f"Column: {column} not found in table")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) == 0:
        print(f"skipping column: {column}, no data")
        return None

    # Compute bins
    min_val = series.min()
    max_val = series.max()
    bins = np.linspace(min_val, max_val, bin_count + 1)

    # Return histogram data instead of plotting
    counts, _ = np.histogram(df[column], bins=bins)
    return {
        "data": df[column].values,
        "bins": bins,
        "counts": counts,
        "title": f"Distribution of {column}",
    }
