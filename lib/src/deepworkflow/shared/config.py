from __future__ import annotations

from dataclasses import dataclass, field

from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for a deepworkflow run."""

    workspace_dir: str
    task_instructions: str
    task_files: list[str]
    task_files_write_option: WriteOption
    judge_minimum: JudgeVerdict
    judge_max_retries: int
    on_max_retries_exceeded: OnMaxRetriesExceeded

    task_files_batch_size: int | None = None
    judge_instructions: str | None = None
    max_failure_retries: int = 0
    model: str = field(default="openai:gpt-4o")
