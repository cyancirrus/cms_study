#!/bin/bash
source ./scripts/environment.sh

model_explore() {
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

model_checkin() {
	# optionally load .env
	if [ -f ./.env ]; then
		export $(grep -v '^#' .env | xargs)
	fi

	# static log file
	METRIC_FILE="./MODEL_PERFORMANCE.txt"

	# overwrite each run
	echo "Random seed: $RANDOM_STATE" >"$METRIC_FILE"
	echo "-------------------------------------" >>"$METRIC_FILE"

	echo "Medicare model" >>"$METRIC_FILE"
	python ./src/model_medicare_main.py >>"$METRIC_FILE" 2>&1

	echo "-------------------------------------" >>"$METRIC_FILE"
	echo "Psychiatric model" >>"$METRIC_FILE"
	python ./src/model_psychiatric_main.py >>"$METRIC_FILE" 2>&1

	echo "-------------------------------------" >>"$METRIC_FILE"
	echo "General model" >>"$METRIC_FILE"
	python ./src/model_general_main.py >>"$METRIC_FILE" 2>&1

	echo "-------------------------------------" >>"$METRIC_FILE"
	echo "All models finished" >>"$METRIC_FILE"

}

model_explore
model_checkin
