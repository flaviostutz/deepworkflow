.PHONY: all run setup-secrets

all: run

# Load .env if it exists (non-secret model provider configuration)
-include .env
export

# Fetch API key from macOS Keychain (run 'make setup-secrets' to store it)
DEEPWORKFLOW_API_KEY := $(shell security find-generic-password -a "$(USER)" -s "deepworkflow-example-basic-cli/api-key" -w 2>/dev/null)
export DEEPWORKFLOW_API_KEY

setup-secrets:
	@read -p "Enter value for 'api-key' (leave empty to skip): " SECRET_VALUE; \
	if [ -n "$$SECRET_VALUE" ]; then \
		security add-generic-password -a "$(USER)" -s "deepworkflow-example-basic-cli/api-key" -w "$$SECRET_VALUE" -U; \
		echo "Secret 'api-key' stored in keychain under group 'deepworkflow-example-basic-cli'."; \
	fi

run:
	mise exec -- uv lock --upgrade-package deepworkflow
	mise exec -- uv sync --project . --frozen
	mise exec -- uv run --project . deepworkflow --config deepworkflow.yml
