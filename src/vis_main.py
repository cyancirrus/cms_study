from typing import Final
from dotenv import load_dotenv
from initialize_environment import DATABASE
from visualization.report import (
    log_report_start,
    report_hvbp_tps,
    report_fy_hospital_readmissions_reduction_program_hospital,
    report_hospital_general_information,
    report_fy_hac_reduction_program_hospital,
    report_filter_complications_and_deaths_hospital,
    report_medicare_hospital_spending_per_patient_hospital,
    # report_zeroized_general_hospital_by_state,
    # report_zeroized_general_hospital_by_type,
    # report_zeroized_general_hospital_by_facility,
)
import os
import sys

# TODO: - Create exclusion column for "too little data" if it's a rate

YEAR: Final[int] = int(sys.argv[1])
load_dotenv()
DATABASE: Final[str] = str(os.getenv("DATABASE"))

if __name__ == "__main__":

    log_report_start("HVBP TPS", YEAR)
    report_hvbp_tps(DATABASE, YEAR)

    log_report_start("Hospital General Information", YEAR)
    report_hospital_general_information(DATABASE, YEAR)

    log_report_start("FY HAC Reduction Program", YEAR)
    report_fy_hac_reduction_program_hospital(DATABASE, YEAR)

    log_report_start(
        "FY Hospital Readmissions Reduction Program", YEAR
    )
    report_fy_hospital_readmissions_reduction_program_hospital(
        DATABASE, YEAR
    )

    log_report_start(
        "Medicare Hospital Spending Per Patient", YEAR
    )
    report_medicare_hospital_spending_per_patient_hospital(
        DATABASE, YEAR
    )

    log_report_start(
        "Complications and Deaths Hospital", YEAR
    )
    report_filter_complications_and_deaths_hospital(
        DATABASE, YEAR
    )

    # report_zeroized_general_hospital_by_state(YEAR);
    # report_zeroized_general_hospital_by_type(YEAR);
    # report_zeroized_general_hospital_by_facility(YEAR);
