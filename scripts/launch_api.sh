#!/usr/bin/env bash
launch() {
	# # Directory of this script
	# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	# # Project root is one level up from scripts/
	# PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

	# # Make sure Python sees PROJECT_ROOT (which contains src/)
	# export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

	# # Optional: just to verify in logs
	# echo "PROJECT_ROOT: $PROJECT_ROOT"
	# echo "PYTHONPATH: $PYTHONPATH"

	uvicorn src.api.main:app --host 127.0.0.1 --port 5678 &
	sleep 2
	# Open Firefox to your API
	open -a "Firefox" "http://127.0.0.1:5678/docs"
}

# launch() {
#   # Directory of this script
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # Project root is one level up from scripts/
#   PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

#   # Make sure Python sees PROJECT_ROOT (which contains src/)
#   export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

#   # Optional: just to verify in logs
#   echo "PROJECT_ROOT: $PROJECT_ROOT"
#   echo "PYTHONPATH: $PYTHONPATH"

#   uvicorn src.api.main:app --host 127.0.0.1 --port 5678 &
#   sleep 2
# }

launch

# # Start Uvicorn in the background
# uvicorn src.api.main:app --host 127.0.0.1 --port 5678 &
# # Give the server a second to start
