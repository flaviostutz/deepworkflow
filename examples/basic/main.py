"""Basic example: run deepworkflow as a library."""

from __future__ import annotations

from deepworkflow import WorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def main() -> None:
    config = WorkflowConfig(
        workspace_dir=".",
        task_instructions=["List all Python files and summarize their purpose."],
        task_files_write_option=WriteOption.READ_ONLY,
        judge_minimum=JudgeVerdict.WARNING,
        judge_max_retries=1,
        on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE,
    )

    result = run_workflow(config)
    print(result)  # noqa: T201


if __name__ == "__main__":
    main()
