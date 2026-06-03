"""Simple evaluation: single instruction, few files, read-only mode."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from deepworkflow import WorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

mlflow.langchain.autolog()

WORKSPACE_DIR = "dataset_simple/workspace"
EXPECTED_OUTPUT_PATH = Path("dataset_simple/expected_output.jsonl")

CONFIG = WorkflowConfig(
    workspace_dir=WORKSPACE_DIR,
    task_instructions="Analyze each file and report any potential bugs or issues.",
    task_files=["**/*.py"],
    task_files_write_option=WriteOption.READ_ONLY,
    judge_minimum=JudgeVerdict.WARNING,
    judge_max_retries=1,
    on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE,
)


def _load_expected_output() -> str:
    """Load the expected output from the JSONL dataset."""
    record = json.loads(EXPECTED_OUTPUT_PATH.read_text().strip().splitlines()[0])
    return record["expected_output"]


def _keyword_similarity(actual: str, expected: str) -> float:
    """Compute similarity as fraction of expected keywords found in actual output."""
    expected_keywords = [kw.strip().lower() for kw in expected.replace(";", ",").split(",") if kw.strip()]
    if not expected_keywords:
        return 0.0
    actual_lower = actual.lower()
    matched = sum(1 for kw in expected_keywords if kw in actual_lower)
    return matched / len(expected_keywords)


def run_eval() -> None:
    """Run the simple evaluation slice."""
    mlflow.set_experiment("deepworkflow-simple")
    expected_output = _load_expected_output()

    with mlflow.start_run(run_name="simple-eval"):
        mlflow.log_param("task_instructions", CONFIG.task_instructions)
        mlflow.log_param("judge_minimum", CONFIG.judge_minimum.name)
        mlflow.log_param("write_option", CONFIG.task_files_write_option.value)

        try:
            result = run_workflow(CONFIG)
            similarity = _keyword_similarity(result.output, expected_output)
            mlflow.log_metric("success", 1)
            mlflow.log_metric("output_length", len(result.output))
            mlflow.log_metric("similarity_score", similarity)
            mlflow.log_metric("output_matches_expected", 1 if similarity >= 0.5 else 0)
        except RuntimeError as e:
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))


if __name__ == "__main__":
    run_eval()
