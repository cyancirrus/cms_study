#!/usr/bin/env bash

shutdown() {
	pkill -f "uvicorn src.api.main:app" &
}
shutdown
