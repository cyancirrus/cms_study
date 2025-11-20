##!/usr/bin/env bash
#source ./scripts/launch_api.sh
#source ./scripts/terminate_api.sh

# TODO: Switch to haversine distance as like euclidean approximaion is off

retrieve_recommended_hospital_general_chicago() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"

	# Chicago-ish coordinates
	user_lat="41.7"
	user_long="-87.6"

	params="user_lat=${user_lat}&"
	params+="user_long=${user_long}&"
	params+="distance_filter=1&" # ~50km radius
	params+="number_results=10&"

	curl -s "${base_url}${params}" | jq
}

retrieve_recommended_hospital_psychiatry_chicago() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"

	# Chicago-ish coordinates
	user_lat="41.7"
	user_long="-87.6"

	params="user_lat=${user_lat}&"
	params+="user_long=${user_long}&"
	params+="distance_filter=1&" # ~50km radius
	params+="number_results=10&"
	params+="service_category=psychiatry" # must match your ServiceCategory enum value

	curl -s "${base_url}${params}" | jq
}
retrieve_recommended_hospital_accute_care_chicago() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"

	# Chicago-ish coordinates
	user_lat="41.7"
	user_long="-87.6"

	params="user_lat=${user_lat}&"
	params+="user_long=${user_long}&"
	params+="distance_filter=1&" # ~50km radius
	params+="number_results=10&"
	params+="hospital_type=Acute%20Care%20Hospitals&"
	params+="service_category=psychiatry" # must match your ServiceCategory enum value

	curl -s "${base_url}${params}" | jq
}

retrieve_recommended_hospital_accute_care_kansas_city() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"

	# Chicago-ish coordinates
	user_lat="39.0997"
	user_long="-94.5786"

	params="user_lat=${user_lat}&"
	params+="user_long=${user_long}&"
	params+="distance_filter=1&" # ~50km radius
	params+="number_results=10&"
	params+="hospital_type=Acute%20Care%20Hospitals&"
	params+="service_category=psychiatry" # must match your ServiceCategory enum value

	curl -s "${base_url}${params}" | jq
}

# retrieve_recommended_hospital_general_chicago
# retrieve_recommended_hospital_psychiatry_chicago
# retrieve_recommended_hospital_accute_care_chicago
retrieve_recommended_hospital_accute_care_kansas_city
