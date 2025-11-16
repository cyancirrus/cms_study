#!/usr/bin/env bash
source ./scripts/launch_api.sh
source ./scripts/terminate_api.sh

retrieve_recommended_hospital() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"
	params="zip_code=12345&"
	params+="hospital_type=womens&"
	params+="service_category=psychiatric"

	curl -s "${base_url}${params}"
}

launch
retrieve_recommended_hospital
shutdown
