from __future__ import annotations

from typing import Any, TypedDict

from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, BatchOutput, EffortConfig, JudgeLevel, JudgeVerdict


class file_batch_workflow_state(TypedDict, total=False):  # noqa: N801
    """LangGraph state for the file_batch_workflow graph."""

    # Configuration (set once at start)
    config: DeepWorkflowConfig
    effort_config: EffortConfig

    # Phase 1: Map
    task_files: list[str]
    task_file_batches: list[BatchDefinition]
    task_overview: str
    consolidation_instructions: str
    evaluate_quality_batch_instructions: str

    # Map evaluate
    map_evaluate_quality_verdict: JudgeLevel
    map_evaluate_judge_verdict: JudgeVerdict
    map_retry_count: int

    # Iteration tracking
    current_batch_index: int
    retry_count: int
    batch_repeat_count: int

    # Phase 2: Plan/Execute/Reflect/Judge (per-batch, reset each iteration)
    plan_output: str
    execute_output: str
    execute_messages: list[Any]
    files_read: list[str]
    files_written: list[str]
    evaluate_quality_verdict: JudgeLevel
    evaluate_quality_judge_verdict: JudgeVerdict
    batch_convergence_output: str
    batch_convergence_verdict: JudgeVerdict

    # Accumulated results across batch-repeat passes
    cumulative_files_read: list[str]
    cumulative_files_written: list[str]
    previous_execute_output: str

    # Accumulated results
    batch_outputs: list[BatchOutput]

    # Phase 3: Reduce
    workflow_output: str

    # Error state
    error: str | None
