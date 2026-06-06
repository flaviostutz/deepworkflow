.PHONY: all run

all: run

run:
	mise exec -- uv lock --upgrade-package deepworkflow
	mise exec -- uv sync --project . --frozen
	mise exec -- uv run --project . deepworkflow --config deepworkflow.yml
