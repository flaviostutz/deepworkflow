UV_PROJECT_ENVIRONMENT ?= $(shell pwd)/.venv
export UV_PROJECT_ENVIRONMENT

# Load .env from repo root (non-secret Azure configuration). Fails if not found.
# Copy .env.example to .env and fill in your values to get started.
include .env
export

.PHONY: setup install build lint lint-fix test-unit test-examples test clean all eval run-mlflow setup-secrets

setup-secrets:
	@read -p "Enter value for 'azure-openai/dev-api-key' (leave empty to skip): " SECRET_VALUE; \
	if [ -n "$$SECRET_VALUE" ]; then \
		security add-generic-password -a "$(USER)" -s "azure-openai/dev-api-key" -w "$$SECRET_VALUE" -U; \
		echo "Secret stored in keychain under service 'azure-openai/dev-api-key'."; \
	fi

setup:
	mise install
	$(MAKE) install

install:
	$(MAKE) -C lib install

build:
	$(MAKE) -C lib build

lint:
	$(MAKE) -C lib lint

lint-fix:
	$(MAKE) -C lib lint-fix

test-unit:
	$(MAKE) -C lib test-unit

test-examples:
	$(MAKE) -C examples

test: test-unit
	@echo "All tests passed!"
	@echo "Run make test-examples to execute example scripts. Note that some examples may require additional setup or credentials."

eval:
	$(MAKE) -C evals/file_batch_workflow eval

clean:
	rm -rf .venv .cache
	$(MAKE) -C lib clean

all: build lint test

bump:
	npx -y filedist update
