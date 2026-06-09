.PHONY: eval eval-simple dev-mlflow setup-secrets

# Load .env from repo root (non-secret Azure configuration)
include ../../.env
export

setup-secrets:
	$(MAKE) -C ../.. setup-secrets

eval: eval-simple

eval-simple:
	mise exec -- uv run --project ../../lib python eval_simple.py
	@echo ""
	@echo "To inspect MLflow traces, run: make dev-mlflow (opens http://localhost:5003)"

dev-mlflow:
	@echo "MLflow UI starting at http://localhost:5003"
	mise exec -- mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5003
