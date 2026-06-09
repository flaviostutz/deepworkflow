.PHONY: all run run-mlflow setup-secrets

all: run

# Load .env from repo root (non-secret Azure configuration)
include ../../.env
export

setup-secrets:
	$(MAKE) -C ../.. setup-secrets

run:
	mise exec -- uv lock --upgrade-package deepworkflow
	mise exec -- uv sync --project . --frozen
	mise exec -- uv run --project . python main.py
	@echo ""
	@echo "To inspect MLflow traces, run: make run-mlflow (opens http://localhost:5002)"

run-mlflow:
	@echo "MLflow UI starting at http://localhost:5002"
	mise exec -- mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --host 0.0.0.0 --port 5002
