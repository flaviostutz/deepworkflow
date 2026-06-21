from __future__ import annotations

from typing import Any, TypedDict

from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, BatchOutput, EffortConfig, JudgeLevel, JudgeVerdict


class file_batch_workflow_state(TypedDict, total=False):  # noqa: N801
    """LangGraph state for the file_batch_workflow graph."""

    # Configuration (set once at start)
    config: DeepWorkflowConfig
    effort_config: EffortConfig

    # Map: setup  # noqa: ERA001
    map_files: list[str]

    # Map: plan  # noqa: ERA001
    map_batches: list[BatchDefinition]
    map_plan_overview: str
    reduce_instructions: str
    batch_evaluate_quality_instructions: str

    # Map: evaluate  # noqa: ERA001
    map_evaluate_level: JudgeLevel
    map_evaluate_verdict: JudgeVerdict
    map_evaluate_retry_count: int

    # Batch: tracking  # noqa: ERA001
    batch_current_index: int
    batch_quality_retry_count: int
    batch_convergence_repeat_count: int

    # Batch: per-iteration (reset each iteration)
    batch_plan: str
    batch_execute_output: str
    batch_execute_messages: list[Any]
    batch_files_read: list[str]
    batch_files_written: list[str]
    batch_evaluate_level: JudgeLevel
    batch_evaluate_verdict: JudgeVerdict
    batch_evaluate_feedbacks: list
    batch_evaluate_convergence_output: str
    batch_evaluate_convergence_verdict: JudgeVerdict

    # Batch: cumulative (across batch-repeat passes)
    batch_cumulative_files_read: list[str]
    batch_cumulative_files_written: list[str]
    batch_cumulative_output: str

    # Batch: outputs  # noqa: ERA001
    batch_results: list[BatchOutput]

    # Reduce
    reduce_output: str

    # Error state
    error: str | None
