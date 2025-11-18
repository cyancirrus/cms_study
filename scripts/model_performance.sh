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
	echo "Model has been ran"
}
model_explore
