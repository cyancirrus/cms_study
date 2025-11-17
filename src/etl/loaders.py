import pandas as pd
import re


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


def load_and_enrich_region_csv(path: str, year:int) -> pd.DataFrame:
    # Areas like midwest atlantic data isfrom 2017 but shoudld be stable
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
    df["year"] = year

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


def load_and_map_msa_statistics(path: str, year: int) -> pd.DataFrame:
    # 2023 unsure of the data refresh schedule
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
    df = df.rename(
        columns={
            "metropolitan_statistical_area_geofips": "cbsafp",
            str(year): "metric",
        }
    )
    df["year"] = year
    df = df[["year", "cbsafp", "linecode", "description", "metric"]]
    df["cbsafp"] = df["cbsafp"].astype(int)
    df["linecode"] = df["linecode"].astype(int)
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    assert isinstance(df, pd.DataFrame)
    return df


def load_and_map_msa_dim(path: str, year: int) -> pd.DataFrame:
    # 2015 but dim codes are stable might be 2025 documentation is sparse need to verify
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
    df = df.rename(
        columns={
            "cbsa_code": "cbsafp",
            "state": "state_abbreviation",
        }
    )
    df["year"] = year
    df = df[["year", "cbsafp", "cbsa_title", "state_abbreviation"]]
    assert isinstance(df, pd.DataFrame)
    return df


def load_and_map_msa_centroids(path: str, year: int) -> pd.DataFrame:
    # 2025 as most recently updated
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
    df = df.rename(
        columns={
            "name": "msa_title",
            "intptlat": "latitude",
            "intptlon": "longitude",
        }
    )
    df["year"] = year
    df = df[
        ["year", "cbsafp", "msa_title", "state_abbreviation", "latitude", "longitude"]
    ]
    df["cbsafp"] = df["cbsafp"].astype(int)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    assert isinstance(df, pd.DataFrame)
    return df


def load_and_map_msa_zip(path: str, year: int) -> pd.DataFrame:
    # 2015 but dim codes are stable might be 2025 documentation is sparse need to verify
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
    df = df.rename(
        columns={
            "zip": "zip_code",
            "lat": "latitude",
            "lng": "longitude",
        }
    )
    df["year"] = year
    df = df[["zip_code", "latitude", "longitude"]]
    df["zip_code"] = df["zip_code"].astype(int)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    assert isinstance(df, pd.DataFrame)
    return df
