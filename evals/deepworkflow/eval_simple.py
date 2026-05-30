"""Simple evaluation: single instruction, few files, read-only mode."""

from __future__ import annotations

import mlflow

from deepworkflow import WorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

WORKSPACE_DIR = "dataset_simple/workspace"

CONFIG = WorkflowConfig(
    workspace_dir=WORKSPACE_DIR,
    task_instructions=["Analyze each file and report any potential bugs or issues."],
    task_files_write_option=WriteOption.READ_ONLY,
    judge_minimum=JudgeVerdict.WARNING,
    judge_max_retries=1,
    on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE,
)


def run_eval() -> None:
    """Run the simple evaluation slice."""
    mlflow.set_experiment("deepworkflow-simple")

    with mlflow.start_run(run_name="simple-eval"):
        mlflow.log_param("task_instructions", CONFIG.task_instructions)
        mlflow.log_param("judge_minimum", CONFIG.judge_minimum.name)
        mlflow.log_param("write_option", CONFIG.task_files_write_option.value)

        try:
            output = run_workflow(CONFIG)
            mlflow.log_metric("success", 1)
            mlflow.log_param("output_length", len(output))
        except RuntimeError as e:
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))


if __name__ == "__main__":
    run_eval()
