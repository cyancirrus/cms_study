from visualization.draw import (
    draw_summary,
    draw_filter_summary,
    draw_zeroized_state_summary,
    draw_zeroized_facility_summary,
    draw_zeroized_type_summary,
)


def log_report_start(report_name: str, year: int):
    print("----------------------------------------")
    print(f"Processing {report_name} for year {year}")
    print("----------------------------------------")


def report_hvbp_tps(database: str, year: int):
    """Draw report histograms for HVBP TPS measures."""
    output_file: str = "./visualizations/hvbp_tps"

    measures: list[str] = [
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
    draw_summary(database, year, output_file, "hvbp_tps", measures)


def report_fy_hospital_readmissions_reduction_program_hospital(
    database: str, year: int
):
    """Draw report histograms for HVBP TPS measures."""
    output_file: str = (
        "./visualizations/fy_hospital_readmissions_reduction_program_hospital"
    )

    measures: list[str] = [
        "number_of_discharges",
        "excess_readmission_ratio",
        "predicted_readmission_rate",
        "expected_readmission_rate",
        "number_of_readmissions",
    ]
    draw_summary(
        database,
        year,
        output_file,
        "fy_hospital_readmissions_reduction_program_hospital",
        measures,
    )


def report_hospital_general_information(database: str, year: int):
    output_file: str = "./visualizations/hospital_general_information"
    measures: list[str] = [
        # # -- mortality
        "mort_group_measure_count",
        "count_of_facility_mort_measures",
        "count_of_mort_measures_better",
        "count_of_mort_measures_no_different",
        "count_of_mort_measures_worse",
        # "mort_group_footnote",
        # # -- safety
        "safety_group_measure_count",
        "count_of_facility_safety_measures",
        "count_of_safety_measures_better",
        "count_of_safety_measures_no_different",
        "count_of_safety_measures_worse",
        # "safety_group_footnote",
        # # -- readmission
        "readm_group_measure_count",
        "count_of_facility_readm_measures",
        "count_of_readm_measures_better",
        "count_of_readm_measures_no_different",
        "count_of_readm_measures_worse",
        # "readm_group_footnote",
        # # -- person and community engagement (patient experience)
        "pt_exp_group_measure_count",
        "count_of_facility_pt_exp_measures",
        # "pt_exp_group_footnote",
        # # -- total expense, medicare spending / beneficiary
        "te_group_measure_count",
        "count_of_facility_te_measures",
        # "te_group_footnote",
    ]
    draw_summary(
        database,
        year,
        output_file,
        "hospital_general_information",
        measures,
    )


def report_fy_hac_reduction_program_hospital(database: str, year: int):
    output_file: str = (
        "./visualizations/fy_hac_reduction_program_hospital"
    )
    measures: list[str] = [
        # "psi90_composite_value", this is null
        # "psi90_w_z_score",
        "clabsi_sir",
        "clabsi_w_z_score",
        "cauti_sir",
        "cauti_w_z_score",
        "ssi_sir",
        "ssi_w_z_score",
        "cdi_sir",
        "cdi_w_z_score",
        "mrsa_sir",
        "mrsa_w_z_score",
        "total_hac_score",
        # "payment_reduction",
    ]
    draw_summary(
        database,
        year,
        output_file,
        "fy_hac_reduction_program_hospital",
        measures,
    )


def report_filter_complications_and_deaths_hospital(
    database: str, year: int
):
    output_file: str = (
        "./visualizations/complications_and_deaths_hospital"
    )
    conditions: list[str] = [
        "Rate of complications for hip/knee replacement patients",
        "Hybrid Hospital-Wide All-Cause Risk Standardized Mortality Rate",
        "Death rate for heart attack patients",
        "Death rate for CABG surgery patients",
        "Death rate for COPD patients",
        "Death rate for heart failure patients",
        "Death rate for pneumonia patients",
        "Death rate for stroke patients",
        "Pressure ulcer rate",
        # these are per 1000 so need to divide it by 1000
        "Death rate among surgical inpatients with serious treatable complications",
        "Iatrogenic pneumothorax rate",
        "In-hospital fall-associated fracture rate",
        "Postoperative hemorrhage or hematoma rate",
        "Postoperative acute kidney injury requiring dialysis rate",
        "Postoperative respiratory failure rate",
        "Perioperative pulmonary embolism or deep vein thrombosis rate",
        "Postoperative sepsis rate",
        "Postoperative wound dehiscence rate",
        "Abdominopelvic accidental puncture or laceration rate",
        "CMS Medicare PSI 90: Patient safety and adverse events composite",
    ]
    draw_filter_summary(
        database,
        year,
        output_file,
        "complications_and_deaths_hospital",
        "measure_name",
        "score",
        conditions,
    )


def report_zeroized_general_hospital_by_state(database: str, year: int):
    output_file: str = (
        "./visualizations/zeroized_hospital_general_information_state"
    )
    labels: list[str] = [
        "mort",
        "safety",
        "readmission",
    ]
    better: list[str] = [
        "count_of_mort_measures_better",
        "count_of_safety_measures_better",
        "count_of_readm_measures_better",
    ]
    worse: list[str] = [
        "count_of_mort_measures_worse",
        "count_of_safety_measures_worse",
        "count_of_readm_measures_worse",
    ]
    count: list[str] = [
        "mort_group_measure_count",
        "safety_group_measure_count",
        "readm_group_measure_count",
    ]
    draw_zeroized_state_summary(
        database,
        year,
        output_file,
        "hospital_general_information",
        labels,
        better,
        worse,
        count,
    )


def report_zeroized_general_hospital_by_type(database: str, year: int):
    output_file: str = (
        "./visualizations/zeroized_hospital_general_information_type"
    )
    labels: list[str] = [
        "mort",
        "safety",
        "readmission",
    ]
    better: list[str] = [
        "count_of_mort_measures_better",
        "count_of_safety_measures_better",
        "count_of_readm_measures_better",
    ]
    worse: list[str] = [
        "count_of_mort_measures_worse",
        "count_of_safety_measures_worse",
        "count_of_readm_measures_worse",
    ]
    count: list[str] = [
        "mort_group_measure_count",
        "safety_group_measure_count",
        "readm_group_measure_count",
    ]
    draw_zeroized_type_summary(
        database,
        year,
        output_file,
        "hospital_general_information",
        labels,
        better,
        worse,
        count,
    )


def report_medicare_hospital_spending_per_patient_hospital(
    database: str, year: int
):
    output_file: str = (
        "./visualizations/medicare_hospital_spending_per_patient_hospital"
    )

    measures: list[str] = [
        "score",
    ]
    draw_summary(
        database,
        year,
        output_file,
        "medicare_hospital_spending_per_patient_hospital",
        measures,
    )


def report_zeroized_general_hospital_by_facility(
    database: str, year: int
):
    output_file: str = (
        "./visualizations/zeroized_hospital_general_information_facility"
    )
    labels: list[str] = [
        "mort",
        "safety",
        "readmission",
    ]
    better: list[str] = [
        "count_of_mort_measures_better",
        "count_of_safety_measures_better",
        "count_of_readm_measures_better",
    ]
    worse: list[str] = [
        "count_of_mort_measures_worse",
        "count_of_safety_measures_worse",
        "count_of_readm_measures_worse",
    ]
    count: list[str] = [
        "mort_group_measure_count",
        "safety_group_measure_count",
        "readm_group_measure_count",
    ]
    draw_zeroized_facility_summary(
        database,
        year,
        output_file,
        "hospital_general_information",
        labels,
        better,
        worse,
        count,
    )
