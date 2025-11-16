#!/bin/bash
source ./scripts/environment.sh

create_database() {
	touch "$DATABASE"
}

create_tables() {
	sqlite3 "$DATABASE" <<EOF
.read ./etl/complications_and_deaths.sql
.read ./etl/fy_hac_reduction_program_hospital.sql
.read ./etl/fy_hospital_readmissions_reduction_program_hospital.sql
.read ./etl/hcahps_hospital.sql
.read ./etl/hospital_general_information.sql
.read ./etl/hvbp_clinical_outcomes.sql
.read ./etl/hvbp_efficiency_and_cost_reduction.sql
.read ./etl/hvbp_person_and_community_engagement.sql
.read ./etl/hvbp_safety.sql
.read ./etl/hvbp_tps.sql
.read ./etl/ipfqr_quality_measures_facility.sql
.read ./etl/medicare_hospital_spending_per_patient_hospital.sql
.read ./etl/pch_hcahps_hospital.sql
.read ./etl/timely_and_effective_care_hospital.sql
.read ./etl/unplanned_hospital_visits_hospital.sql
.read ./etl/zip_lat_long.sql
EOF
	echo "All typed tables created."
}

import_data() {
	python ./src/etl_main.py
	echo "Data has been loaded."
}
initialize_environment
create_database
create_tables
import_data
