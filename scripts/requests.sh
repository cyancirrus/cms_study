##!/usr/bin/env bash
#source ./scripts/launch_api.sh
#source ./scripts/terminate_api.sh

#retrieve_recommended_hospital() {
#	base_url="http://127.0.0.1:5678/recommend_hospital?"
#	params="zip_code=12345&"
#	params+="hospital_type=womens&"
#	params+="service_category=psychiatric"

#	curl -s "${base_url}${params}"
#}

#!/usr/bin/env bash
# source ./scripts/launch_api.sh
# source ./scripts/terminate_api.sh

#!/usr/bin/env bash

retrieve_recommended_hospital() {
	base_url="http://127.0.0.1:5678/recommend_hospital?"

	# Chicago-ish coordinates
	user_lat="41.7"
	user_long="-87.6"

	params="user_lat=${user_lat}&"
	params+="user_long=${user_long}&"
	params+="distance_filter=100&" # ~50km radius
	params+="number_results=10&"
	# params+="hospital_type=womens&"        # must match your HospitalType enum value
	# params+="service_category=psychiatric" # must match your ServiceCategory enum value

	curl -s "${base_url}${params}"
}

# launch
retrieve_recommended_hospital
# shutdown

# launch
# retrieve_recommended_hospital
# shutdown
