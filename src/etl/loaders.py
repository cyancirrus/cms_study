from initialize_environment import DATABASE
import pandas as pd
import re
import sqlite3


def clean_column_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        na_values=[
            "Not Available",
            "Not Applicable",
            "Too Few to Report",
            "N/A",
            "NA",
            "",
        ],
    )
    df.columns = [clean_column_name(c) for c in df.columns]
    try:
        result = df.apply(pd.to_numeric)
        df = result
    except Exception:
        pass
    assert isinstance(df, pd.DataFrame)
    return df


def load_and_enrich_region_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        na_values=[
            "Not Available",
            "Not Applicable",
            "Too Few to Report",
            "N/A",
            "NA",
            "",
        ],
    )
    df.columns = [clean_column_name(c) for c in df.columns]
    df["fips_state_code"] = pd.to_numeric(df["fips_state_code"], errors="coerce")
    df["region_id"], _ = pd.factorize(df["region"])

    # division_id: integers 0..n_divisions-1 in order of first appearance
    df["division_id"], _ = pd.factorize(df["division"])
    return df


def load_and_clean_and_append_year_csv(path: str, year: int) -> pd.DataFrame:
    # Set low_memory = false in pd.read_csv if error in a heterogeneous column but no current errors
    df = pd.read_csv(
        path,
        na_values=[
            "Not Available",
            "Not Applicable",
            "Too Few to Report",
            "N/A",
            "NA",
            "",
        ],
    )
    df.columns = [clean_column_name(c) for c in df.columns]
    df["submission_year"] = year
    try:
        result = df.apply(pd.to_numeric)
        df = result
    except Exception:
        pass
    assert isinstance(df, pd.DataFrame)
    return df
