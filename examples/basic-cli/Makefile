.PHONY: all run run-mlflow setup-secrets

all: run

# Load .env from repo root (non-secret Azure configuration)
include ../../.env
export

# Fetch API key from macOS Keychain (run 'make setup-secrets' to store it)
DEEPWORKFLOW_API_KEY := $(shell security find-generic-password -a "$(USER)" -s "azure-openai/dev-api-key" -w 2>/dev/null)
export DEEPWORKFLOW_API_KEY

setup-secrets:
	$(MAKE) -C ../.. setup-secrets

run:
	mise exec -- uv lock --upgrade-package deepworkflow
	mise exec -- uv sync --project . --frozen
	mise exec -- uv run --project . deepworkflow --config deepworkflow.yml
	@echo ""
	@echo "To inspect MLflow traces, run: make run-mlflow (opens http://localhost:5001)"

run-mlflow:
	@echo "MLflow UI starting at http://localhost:5001"
	mise exec -- mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
