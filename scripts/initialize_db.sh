#!/bin/bash
set -e  # exit immediately if any command fails

source ./scripts/environment.sh

log() {
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] $1"
}

exit_with_error() {
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] ERROR: $1"
    exit 1
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
        "complications_and_deaths.sql"
        "fy_hac_reduction_program_hospital.sql"
        "fy_hospital_readmissions_reduction_program_hospital.sql"
        "hcahps_hospital.sql"
        "hospital_general_information.sql"
        "hvbp_clinical_outcomes.sql"
        "hvbp_efficiency_and_cost_reduction.sql"
        "hvbp_person_and_community_engagement.sql"
        "hvbp_safety.sql"
        "hvbp_tps.sql"
        "ipfqr_quality_measures_facility.sql"
        "medicare_hospital_spending_per_patient_hospital.sql"
        "pch_hcahps_hospital.sql"
        "timely_and_effective_care_hospital.sql"
        "unplanned_hospital_visits_hospital.sql"
        "zip_lat_long.sql"
    )

    for sql_file in "${sql_files[@]}"; do
        if [ ! -f "./etl/$sql_file" ]; then
            exit_with_error "Missing SQL file: ./etl/$sql_file"
        fi
        log "Importing $sql_file..."
        sqlite3 "$DATABASE" ".read ./etl/$sql_file" || exit_with_error "Failed to execute $sql_file"
    done
    log "All tables created successfully."
}

import_data() {
    log "Starting ETL import..."
    python ./src/etl_main.py || exit_with_error "ETL import failed"
    log "Data has been loaded successfully."
}

initialize_environment

create_database
create_tables
import_data

log "Database initialization complete."

