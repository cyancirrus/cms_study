initialize_environment() {
    ENV_FILE="./.env"  # adjust path if needed
    if [ -f "$ENV_FILE" ]; then
        # Automatically export all sourced variables
        set -a
        source "$ENV_FILE"
        set +a
        echo "Environment variables loaded from $ENV_FILE"
    else
        echo ".env file not found at $ENV_FILE!"
        exit 1
    fi
}


check_variables() {
    required_vars=(
        DATABASE
        # API_KEY
        # API_SECRET
        # HOST
        # PORT
        # ... add the rest
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "Error: $var is not set!"
            exit 1
        fi
    done
    echo "All required environment variables are set."
}

initialize_environment
check_variables

