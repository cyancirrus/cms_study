# fmt: off
from enum import Enum


class CmsSchema(Enum):
    complications_and_deaths_hospital                  = "complications_and_deaths_hospital"
    fy_hac_reduction_program_hospital                  = "fy_hac_reduction_program_hospital"
    fy_hospital_readmissions_reduction_program_hospital= "fy_hospital_readmissions_reduction_program_hospital"
    hcahps_hospital                                    = "hcahps_hospital"
    hospital_general_information                       = "hospital_general_information"
    hvbp_clinical_outcomes                             = "hvbp_clinical_outcomes"
    hvbp_efficiency_and_cost_reduction                 = "hvbp_efficiency_and_cost_reduction"
    hvbp_person_and_community_engagement               = "hvbp_person_and_community_engagement"
    hvbp_safety                                        = "hvbp_safety"
    hvbp_tps                                           = "hvbp_tps"
    ipfqr_quality_measures_facility                    = "ipfqr_quality_measures_facility"
    medicare_hospital_spending_per_patient_hospital    = "medicare_hospital_spending_per_patient_hospital"
    pch_hcahps_hospital                                = "pch_hcahps_hospital"
    state_region                                       = "state_region"
    timely_and_effective_care_hospital                 = "timely_and_effective_care_hospital"
    unplanned_hospital_visits_hospital                 = "unplanned_hospital_visits_hospital"
    # augmented
    facility_zip_code                                  = "facility_zip_code"
    msa_centroid                                       = "msa_centroid"
    msa_dim                                            = "msa_dim"
    msa_statistics                                     = "msa_statistics"
    zip_lat_long                                       = "zip_lat_long"
    zip_demographics                                   = "zip_demographics"
    # predictions
    prediction_hvbp_tps                                = "prediction_hvbp_tps"
    prediction_hvbp_clinical_outcomes                  = "prediction_hvbp_clinical_outcomes"
