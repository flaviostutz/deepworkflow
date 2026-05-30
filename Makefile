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

test: test-unit test-examples

eval:
	$(MAKE) -C evals/deepworkflow eval

clean:
	rm -rf .venv .cache
	$(MAKE) -C lib clean

all: build lint test

bump:
	npx -y filedist update
