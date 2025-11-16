from dotenv import load_dotenv
from typing import Final
from initialize_environment import DATABASE
import os
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


def insert_into_existing_table(df: pd.DataFrame, db_path: str, table_name: str):
    # Insert the expected schema ie the intersection of CSV cols and table cols
    with sqlite3.connect(db_path) as conn:
        existing_cols = pd.read_sql(f"PRAGMA table_info({table_name});", conn)[
            "name"
        ].tolist()
        available_cols = [c for c in df.columns if c in existing_cols]
        df[available_cols].to_sql(table_name, conn, if_exists="append", index=False)


def process_table(data_path: str, table_name: str):
    print(f"loading table: {table_name}")
    df = load_and_clean_csv(data_path)
    insert_into_existing_table(df, DATABASE, table_name)


def process_table_append_year(data_path: str, table_name: str, year: int):
    print(f"loading table: {table_name}")
    df = load_and_clean_and_append_year_csv(data_path, year)
    insert_into_existing_table(df, DATABASE, table_name)


def process_cms_data(directory: str, year: int):
    print("----------------------------------------")
    print(f"        Loading year {year}            ")
    print("----------------------------------------")
    process_table_append_year(
        f"./{directory}/Hospital_General_Information.csv",
        "hospital_general_information",
        year,
    )
    process_table_append_year(
        f"./{directory}/Complications_and_Deaths-Hospital.csv",
        "complications_and_deaths_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/FY_{year}_HAC_Reduction_Program_Hospital.csv",
        "fy_hac_reduction_program_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/FY_{year}_Hospital_Readmissions_Reduction_Program_Hospital.csv",
        "fy_hospital_readmissions_reduction_program_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/HCAHPS-Hospital.csv", "hcahps_hospital", year
    )
    process_table_append_year(
        f"./{directory}/IPFQR_QualityMeasures_Facility.csv",
        "ipfqr_quality_measures_facility",
        year,
    )
    process_table_append_year(
        f"./{directory}/Medicare_Hospital_Spending_Per_Patient-Hospital.csv",
        "medicare_hospital_spending_per_patient_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/PCH_HCAHPS_HOSPITAL.csv", "pch_hcahps_hospital", year
    )
    process_table_append_year(
        f"./{directory}/Timely_and_Effective_Care-Hospital.csv",
        "timely_and_effective_care_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/Unplanned_Hospital_Visits-Hospital.csv",
        "unplanned_hospital_visits_hospital",
        year,
    )
    process_table_append_year(
        f"./{directory}/hvbp_clinical_outcomes.csv", "hvbp_clinical_outcomes", year
    )
    process_table_append_year(
        f"./{directory}/hvbp_efficiency_and_cost_reduction.csv",
        "hvbp_efficiency_and_cost_reduction",
        year,
    )
    process_table_append_year(
        f"./{directory}/hvbp_person_and_community_engagement.csv",
        "hvbp_person_and_community_engagement",
        year,
    )
    process_table_append_year(f"./{directory}/hvbp_safety.csv", "hvbp_safety", year)
    process_table_append_year(f"./{directory}/hvbp_tps.csv", "hvbp_tps", year)
    process_table_append_year(f"./{directory}/hvbp_tps.csv", "hvbp_tps", year)
    print()


def process_augmented_tables():
    # TODO: Set up definitions for msa want to check if like we can get historical data in
    process_table("./data/augmented/zip_lat_long.csv", "zip_lat_long")
    process_table("./data/augmented/msa_centers.csv", "msa_centers")
    process_table("./data/augmented/msa_id.csv", "msa_id")
    process_table("./data/augmented/msa_statistics.csv", "msa_statistics")
    process_table("./data/augmented/zip_lat_long.csv", "zip_lat_long")


def process_all_years_cms():
    # Historical data is only consistent back to 2021
    process_cms_data("./data/source/", 2025)
    process_cms_data("./data/historical/2024", 2024)
    process_cms_data("./data/historical/2023", 2023)
    process_cms_data("./data/historical/2022", 2022)
    process_cms_data("./data/historical/2021", 2021)


if __name__ == "__main__":
    process_all_years_cms()
