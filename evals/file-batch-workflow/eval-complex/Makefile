.PHONY: eval run dev-mlflow setup-secrets cleanup

# Load .env from repo root (non-secret Azure configuration)
include ../../../.env
export

cleanup:
	rm -rf .workspace

setup-secrets:
	$(MAKE) -C ../../.. setup-secrets

eval: cleanup
	mise exec -- uv run --project ../../../lib python eval-complex.py
	@echo ""
	@echo "To inspect MLflow traces, run: make dev-mlflow (opens http://localhost:5004)"

run: cleanup
	mise exec -- uv run --project ../../../lib python eval-complex.py || true
	@echo ""
	@echo "To inspect MLflow traces, run: make dev-mlflow (opens http://localhost:5004)"

dev-mlflow:
	@echo "MLflow UI starting at http://localhost:5004"
	mise exec -- mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5004
