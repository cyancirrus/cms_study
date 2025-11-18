#!/bin/bash
source ./scripts/environment.sh

model_explore() {
	# python ./src/model_medicare_main.py
	python ./src/model_psychiatric_main.py
	echo "Model has been ran"
}
model_explore
