UV_PROJECT_ENVIRONMENT ?= $(shell pwd)/../.venv
export UV_PROJECT_ENVIRONMENT

.PHONY: install build lint lint-fix test-unit clean all update-lockfile run dev publish

install:
	mise exec -- uv sync --project . --frozen --all-extras --dev

build:
	mise exec -- uv sync --project . --frozen --all-extras --dev
	mise exec -- uv build --project . --out-dir dist

lint:
	mise exec -- uv run --project . ruff format --check .
	mise exec -- uv run --project . ruff check .
	mise exec -- uv run --project . ty check
	mise exec -- uv run --project . pip-audit

lint-fix:
	mise exec -- uv run --project . ruff format .
	mise exec -- uv run --project . ruff check . --fix
	mise exec -- uv run --project . ty check
	mise exec -- uv run --project . pip-audit

test-unit:
	mise exec -- uv run --project . pytest --cov=src/deepworkflow --cov-branch --cov-report=term-missing --cov-fail-under=65

clean:
	rm -rf dist .cache

all: build lint test-unit

update-lockfile:
	mise exec -- uv lock --project . --upgrade

run:
	mise exec -- uv run --project . python -m deepworkflow

dev: run

publish:
	mise exec -- uv publish --project .
