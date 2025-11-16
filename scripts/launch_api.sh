#!/usr/bin/env bash

launch() {
	# Start Uvicorn in the background
	uvicorn src.api.main:app --host 127.0.0.1 --port 5678 &
	# Give the server a second to start
	sleep 2
	# # Open Firefox to your API
	open -a "Firefox" "http://127.0.0.1:5678/docs"
}

launch
