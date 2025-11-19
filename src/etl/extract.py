from database.bridge import EngineProtocol
from tables import CmsSchema
from etl.process import (
    process_table_method,
    process_table_append_year,
)
from etl.loaders import (
    load_and_enrich_region_csv,
    load_and_map_msa_statistics,
    load_and_map_msa_dim,
    load_and_map_msa_centroids,
    load_and_map_msa_zip,
)


def extract_cms_data(engine: EngineProtocol, directory: str, year: int):
    print("----------------------------------------")
    print(f"        Loading year {year}            ")
    print("----------------------------------------")
    process_table_append_year(
        engine,
        f"./{directory}/Hospital_General_Information.csv",
        CmsSchema.hospital_general_information,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/Complications_and_Deaths-Hospital.csv",
        CmsSchema.complications_and_deaths_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/FY_{year}_HAC_Reduction_Program_Hospital.csv",
        CmsSchema.fy_hac_reduction_program_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/FY_{year}_Hospital_Readmissions_Reduction_Program_Hospital.csv",
        CmsSchema.fy_hospital_readmissions_reduction_program_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/HCAHPS-Hospital.csv",
        CmsSchema.hcahps_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/IPFQR_QualityMeasures_Facility.csv",
        CmsSchema.ipfqr_quality_measures_facility,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/Medicare_Hospital_Spending_Per_Patient-Hospital.csv",
        CmsSchema.medicare_hospital_spending_per_patient_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/PCH_HCAHPS_HOSPITAL.csv",
        CmsSchema.pch_hcahps_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/Timely_and_Effective_Care-Hospital.csv",
        CmsSchema.timely_and_effective_care_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/Unplanned_Hospital_Visits-Hospital.csv",
        CmsSchema.unplanned_hospital_visits_hospital,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/hvbp_clinical_outcomes.csv",
        CmsSchema.hvbp_clinical_outcomes,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/hvbp_efficiency_and_cost_reduction.csv",
        CmsSchema.hvbp_efficiency_and_cost_reduction,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/hvbp_person_and_community_engagement.csv",
        CmsSchema.hvbp_person_and_community_engagement,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/hvbp_safety.csv",
        CmsSchema.hvbp_safety,
        year,
    )
    process_table_append_year(
        engine,
        f"./{directory}/hvbp_tps.csv",
        CmsSchema.hvbp_tps,
        year,
    )
    print()


def extract_augmented_tables(engine: EngineProtocol):
    # TODO: Set up definitions for msa want to check if like we can get historical data in
    process_table_method(
        load_and_enrich_region_csv,
        engine,
        "./data/augmented/region/state_region.csv",
        CmsSchema.state_region,
        2025,
    )
    process_table_method(
        load_and_map_msa_centroids,
        engine,
        "./data/augmented/msa/msa_centroids.csv",
        CmsSchema.msa_centroid,
        2025,
    )
    process_table_method(
        load_and_map_msa_dim,
        engine,
        "./data/augmented/msa/msa_dim.csv",
        CmsSchema.msa_dim,
        2015,
    )
    process_table_method(
        load_and_map_msa_statistics,
        engine,
        "./data/augmented/msa/msa_statistics.csv",
        CmsSchema.msa_statistics,
        2023,
    )
    process_table_method(
        load_and_map_msa_zip,
        engine,
        "./data/augmented/msa/zip_lat_long.csv",
        CmsSchema.zip_lat_long,
        2015,
    )
    print()


def extract_all_years_cms(engine: EngineProtocol):
    # Historical data is only consistent back to 2021
    extract_cms_data(engine, "./data/source/", 2025)
    extract_cms_data(engine, "./data/historical/2024", 2024)
    extract_cms_data(engine, "./data/historical/2023", 2023)
    extract_cms_data(engine, "./data/historical/2022", 2022)
    extract_cms_data(engine, "./data/historical/2021", 2021)
