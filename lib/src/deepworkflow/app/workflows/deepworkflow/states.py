from __future__ import annotations

from typing import Any, TypedDict

from deepworkflow.shared.config import WorkflowConfig
from deepworkflow.shared.types import BatchDefinition, BatchOutput, JudgeFeedback, JudgeVerdict


class WorkflowState(TypedDict, total=False):
    """LangGraph state for the deepworkflow graph."""

    # Configuration (set once at start)
    config: WorkflowConfig

    # Phase 1: Map
    task_files: list[str]
    task_file_batches: list[BatchDefinition]
    task_overview: str
    consolidation_instructions: str

    # Map judge
    map_judge_verdict: JudgeVerdict
    map_judge_feedbacks: list[JudgeFeedback]
    map_retry_count: int

    # Iteration tracking
    current_batch_index: int
    retry_count: int

    # Phase 2: Plan/Execute/Reflect/Judge (per-batch, reset each iteration)
    plan_output: str
    execute_output: str
    execute_messages: list[Any]
    files_read: list[str]
    files_written: list[str]
    judge_verdict: JudgeVerdict
    judge_feedbacks: list[JudgeFeedback]

    # Accumulated results
    batch_outputs: list[BatchOutput]

    # Phase 3: Reduce
    workflow_output: str

    # Error state
    error: str | None
