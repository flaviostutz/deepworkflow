.PHONY: eval eval-simple setup-secrets

# Load .env for non-secret Azure endpoint configuration
-include .env
export

setup-secrets:
	@read -p "Enter value for 'dev-api-key' (leave empty to skip): " SECRET_VALUE; \
	if [ -n "$$SECRET_VALUE" ]; then \
		security add-generic-password -a "$(USER)" -s "azure-openai/dev-api-key" -w "$$SECRET_VALUE" -U; \
		echo "Secret 'dev-api-key' stored in keychain under service 'azure-openai/dev-api-key'."; \
	fi

eval: eval-simple

eval-simple:
	mise exec -- uv run --project ../../lib python eval_simple.py
