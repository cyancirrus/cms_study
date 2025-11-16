#!/bin/bash
source ./scripts/environment.sh

model_explore() {
	python ./src/model_main.py
	echo "Model has been ran"
}
model_explore
