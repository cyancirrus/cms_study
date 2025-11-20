#!/bin/bash
set -e # exit immediately if any command fails

source ./scripts/environment.sh

log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

exit_with_error() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
	exit 1
}

remove_database() {
	rm "$DATABASE"
}

create_database() {
	if [ ! -f "$DATABASE" ]; then
		touch "$DATABASE" || exit_with_error "Failed to create database file $DATABASE"
		log "Database $DATABASE created."
	else
		log "Database $DATABASE already exists, skipping creation."
	fi
}

create_tables() {
	local sql_files=(
		"./etl/source/complications_and_deaths.sql"
		"./etl/source/fy_hac_reduction_program_hospital.sql"
		"./etl/source/fy_hospital_readmissions_reduction_program_hospital.sql"
		"./etl/source/hcahps_hospital.sql"
		"./etl/source/hospital_general_information.sql"
		"./etl/source/hvbp_clinical_outcomes.sql"
		"./etl/source/hvbp_efficiency_and_cost_reduction.sql"
		"./etl/source/hvbp_person_and_community_engagement.sql"
		"./etl/source/hvbp_safety.sql"
		"./etl/source/hvbp_tps.sql"
		"./etl/source/ipfqr_quality_measures_facility.sql"
		"./etl/source/medicare_hospital_spending_per_patient_hospital.sql"
		"./etl/source/pch_hcahps_hospital.sql"
		"./etl/source/timely_and_effective_care_hospital.sql"
		"./etl/source/unplanned_hospital_visits_hospital.sql"

		"./etl/augmented/facility_zip_code.sql"
		"./etl/augmented/msa_centroid.sql"
		"./etl/augmented/msa_dim.sql"
		"./etl/augmented/msa_statistics.sql"
		"./etl/augmented/state_region.sql"
		"./etl/augmented/zip_lat_long.sql"
		"./etl/augmented/zip_demographics.sql"

		"./etl/prediction/prediction_hvbp_tps.sql"
		"./etl/prediction/prediction_hvbp_clinical_outcomes.sql"
		"./etl/prediction/prediction_ipfqr_quality_measures_facility.sql"

		"./etl/recommendation/recommendation_hospital.sql"
	)
	for sql_file in "${sql_files[@]}"; do
		if [ ! -f "$sql_file" ]; then
			exit_with_error "Missing SQL file: $sql_file"
		fi
		log "Importing $sql_file..."
		sqlite3 "$DATABASE" ".read $sql_file" || exit_with_error "Failed to execute $sql_file"
	done
	log "All tables created successfully."
}

import_data() {
	log "Starting ETL import..."
	python ./src/etl_main.py || exit_with_error "ETL import failed"
	log "Data has been loaded successfully."
}

transform_data() {
	log "Starting ETL transform..."
	python ./src/transform_main.py || exit_with_error "ETL transform failed"
}

model_predictions() {
	export GENERATE_PREDICTIONS=1
	echo "-------------------------------------"
	echo "             Medicare                "
	echo "-------------------------------------"
	python ./src/model_medicare_main.py
	echo "-------------------------------------"
	echo "             Psychiatric             "
	echo "-------------------------------------"
	python ./src/model_psychiatric_main.py
	echo "-------------------------------------"
	echo "             General                 "
	echo "-------------------------------------"
	python ./src/model_general_main.py
	echo ""
}

create_recommendations() {
	log "Creating Recommendations"
	python ./src/recommendation_main.py
}

initialize_environment

# # Incase the database didn't exist prior
create_database
remove_database
create_database
create_tables
import_data
transform_data
model_predictions
create_recommendations
log "Database initialization complete."
