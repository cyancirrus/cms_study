#!/usr/bin/env bash

launch_api() {
	uvicorn src.api.main:app --host 127.0.0.1 --port 5678 --reload
}

launch_api
