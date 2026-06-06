UV_PROJECT_ENVIRONMENT ?= $(shell pwd)/.venv
export UV_PROJECT_ENVIRONMENT

.PHONY: setup install build lint lint-fix test-unit test-examples test clean all eval

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
	@for dir in examples/*/; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Running example: $$dir"; \
			mise exec -- uv sync --project "$$dir" --frozen --all-extras; \
			mise exec -- uv run --project "$$dir" python "$$dir/main.py"; \
		fi; \
	done

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
