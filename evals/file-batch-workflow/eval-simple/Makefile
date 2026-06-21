.PHONY: eval run dev-mlflow setup-secrets cleanup

# Load .env from repo root (non-secret Azure configuration)
include ../../../.env
export

cleanup:
	rm -rf .workspace

setup-secrets:
	$(MAKE) -C ../../.. setup-secrets

eval: cleanup
	mise exec -- uv run --project ../../../lib python eval.py
	@echo ""
	@echo "To inspect MLflow traces, run: make dev-mlflow (opens http://localhost:5100)"

run: cleanup
	mise exec -- uv run --project ../../../lib python eval.py || true
	@echo ""
	@echo "To inspect MLflow traces, run: make dev-mlflow (opens http://localhost:5100)"

dev-mlflow:
	@echo "MLflow UI starting at http://localhost:5100"
	mise exec -- mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5100
