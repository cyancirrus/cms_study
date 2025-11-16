#!/bin/bash
source ./scripts/environment.sh

create_summaries() {
	python ./src/vis_main.py 2025
	# python ./src/vis_main.py 2024
	# python ./src/vis_main.py 2023
	# python ./src/vis_main.py 2022
	# python ./src/vis_main.py 2021
	echo "Data summaries created"
}

create_summaries;
