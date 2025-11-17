from etl.process import (
    process_table_method,
    process_table_append_year,
    process_table_region,
    process_table,
)
from etl.loaders import (
    load_and_enrich_region_csv,
    load_and_map_msa_statistics,
    load_and_map_msa_dim,
    load_and_map_msa_centroids,
    load_and_map_msa_zip,
    load_and_clean_and_append_year_csv,
)


def extract_cms_data(
    database: str, directory: str, year: int
):
    print("----------------------------------------")
    print(f"        Loading year {year}            ")
    print("----------------------------------------")
    process_table_append_year(
        database,
        f"./{directory}/Hospital_General_Information.csv",
        "hospital_general_information",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/Complications_and_Deaths-Hospital.csv",
        "complications_and_deaths_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/FY_{year}_HAC_Reduction_Program_Hospital.csv",
        "fy_hac_reduction_program_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/FY_{year}_Hospital_Readmissions_Reduction_Program_Hospital.csv",
        "fy_hospital_readmissions_reduction_program_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/HCAHPS-Hospital.csv",
        "hcahps_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/IPFQR_QualityMeasures_Facility.csv",
        "ipfqr_quality_measures_facility",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/Medicare_Hospital_Spending_Per_Patient-Hospital.csv",
        "medicare_hospital_spending_per_patient_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/PCH_HCAHPS_HOSPITAL.csv",
        "pch_hcahps_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/Timely_and_Effective_Care-Hospital.csv",
        "timely_and_effective_care_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/Unplanned_Hospital_Visits-Hospital.csv",
        "unplanned_hospital_visits_hospital",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_clinical_outcomes.csv",
        "hvbp_clinical_outcomes",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_efficiency_and_cost_reduction.csv",
        "hvbp_efficiency_and_cost_reduction",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_person_and_community_engagement.csv",
        "hvbp_person_and_community_engagement",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_safety.csv",
        "hvbp_safety",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_tps.csv",
        "hvbp_tps",
        year,
    )
    process_table_append_year(
        database,
        f"./{directory}/hvbp_tps.csv",
        "hvbp_tps",
        year,
    )
    print()


def extract_augmented_tables(database: str):
    # TODO: Set up definitions for msa want to check if like we can get historical data in
    process_table_method(
        load_and_enrich_region_csv,
        database,
        "./data/augmented/region/state_region.csv",
        "state_region",
        2025,
    )
    process_table_method(
        load_and_map_msa_zip,
        database,
        "./data/augmented/msa/zip_lat_long.csv",
        "zip_lat_long",
        2015,
    )
    process_table_method(
        load_and_map_msa_centroids,
        database,
        "./data/augmented/msa/msa_centroids.csv",
        "msa_centroids",
        2025,
    )
    process_table_method(
        load_and_map_msa_dim,
        database,
        "./data/augmented/msa_dim.csv",
        "msa_dim",
        2015,
    )
    process_table_method(
        load_and_map_msa_statistics,
        database,
        "./data/augmented/msa/msa_statistics.csv",
        "msa_statistics",
        2023,
    )
    process_table_method(
        load_and_map_msa_zip,
        database,
        "./data/augmented/zip_lat_long.csv",
        "zip_lat_long",
        2015,
    )


def extract_all_years_cms(database: str):
    # Historical data is only consistent back to 2021
    extract_cms_data(database, "./data/source/", 2025)
    extract_cms_data(
        database, "./data/historical/2024", 2024
    )
    extract_cms_data(
        database, "./data/historical/2023", 2023
    )
    extract_cms_data(
        database, "./data/historical/2022", 2022
    )
    extract_cms_data(
        database, "./data/historical/2021", 2021
    )
