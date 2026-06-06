.PHONY: all run setup-secrets

all: run

setup-secrets:
	@read -p "Enter value for 'api-key' (leave empty to skip): " SECRET_VALUE; \
	if [ -n "$$SECRET_VALUE" ]; then \
		security add-generic-password -a "$(USER)" -s "deepworkflow-example-basic-lib/api-key" -w "$$SECRET_VALUE" -U; \
		echo "Secret 'api-key' stored in keychain under group 'deepworkflow-example-basic-lib'."; \
	fi

run:
	mise exec -- uv lock --upgrade-package deepworkflow
	mise exec -- uv sync --project . --frozen
	mise exec -- uv run --project . python main.py
