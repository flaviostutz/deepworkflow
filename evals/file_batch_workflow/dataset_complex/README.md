# dataset_complex

A complex evaluation dataset for the `file_batch_workflow`. It exercises the full write-mode path with a 20-file data-processing library spread across two sub-folders (`parsers/` and `transformers/`), activating the convergence loop and quality-evaluation loop.

## Purpose

Measures whether the workflow can:

- Read 20 Python source files in one or more batches.
- Write a per-file analysis report to `reports/<source_filename_without_extension>.md`.
- Detect real bugs and security issues seeded into each file (see expected keywords in `expected_output.jsonl`).

## Dataset creation

Each Python file in `data/` was written by hand and intentionally contains one or more defects representative of common bugs in data-processing pipelines (e.g. XXE injection in XML parsers, `yaml.load` without a Loader, ReDoS patterns, integer division by zero, mutation of input collections). The expected output keywords in `expected_output.jsonl` were derived from the known defects in each file.

## Data quality

- All 20 source files are synthetic but representative of real-world data-processing code.
- Expected keywords are matched case-insensitively against the actual workflow output using keyword overlap (similarity score).
- The passing threshold is `EVAL_MIN_SIMILARITY = 0.7` (70% of expected keywords must appear in the report).

## Consuming the dataset

Run the eval from the `evals/file_batch_workflow/` directory:

```bash
make eval-complex
```

The eval script (`eval_complex.py`) does the following:

1. Copies `data/` to `.workspace/dataset_complex/` so the workflow can write report files without modifying the original dataset.
2. Runs `run_workflow` with write-mode enabled, passing the 20-file glob `**/*.py`.
3. Loads `expected_output.jsonl` and computes per-file keyword similarity against the written reports.
4. Logs per-file and aggregate metrics to a local MLflow experiment (`deepworkflow-complex`).
5. Raises `SystemExit` if the average similarity falls below `EVAL_MIN_SIMILARITY`.

## Schema

See [`dataset.schema.json`](dataset.schema.json) for the structure of each line in `expected_output.jsonl`:

- `input.task_instructions` — the prompt passed to `DeepWorkflowConfig`.
- `input.files` — list of file paths relative to the dataset root (e.g. `data/parsers/csv_parser.py`).
- `expected_output` — map of file stem → semicolon-separated keywords expected in the corresponding report.
