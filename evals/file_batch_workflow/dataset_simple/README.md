# dataset_simple

Minimal evaluation dataset for the `file_batch_workflow` — a single-sample read-only task used to verify that the workflow correctly identifies bugs in a small Python workspace.

## What this dataset contains

One JSONL record (`expected_output.jsonl`) pairing a task instruction with the expected output keywords. The `data/` sub-folder holds the Python source files the workflow is asked to analyse.

## Creation procedure

1. A small Python file (`data/sample.py`) was written with two known bugs:
   - A `divide` function with no `ZeroDivisionError` guard.
   - A `greet` function that uses `== None` instead of `is None`.
2. The workflow was invoked in read-only mode with `task_instructions = "Analyze each file and report any potential bugs or issues."`.
3. The expected output was defined as a comma-separated list of the bug descriptions the workflow must surface.

## Data quality

- Single sample; coverage is intentionally minimal (smoke-test level).
- Keyword similarity scoring (`EVAL_MIN_SIMILARITY = 0.5`) means at least half the expected keywords must appear in the actual output.

## How to consume

```python
import json
from pathlib import Path

record = json.loads(Path("dataset_simple/expected_output.jsonl").read_text().strip().splitlines()[0])
task_instructions = record["input"]["task_instructions"]
expected_output   = record["expected_output"]
```

Each line in `expected_output.jsonl` is a JSON object conforming to `dataset.schema.json`.
