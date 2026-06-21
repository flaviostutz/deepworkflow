from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.shared.types import BatchDefinition

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def map_plan_step(state: file_batch_workflow_state) -> dict:
    """Split map_files into batches deterministically (no LLM).

    Files are grouped sequentially in the order they appear in ``map_files``.
    When ``max_batches=1`` all files go into a single batch regardless of
    ``max_files_per_batch``.  Otherwise files are chunked by ``max_files_per_batch``.

    The step requires ``map_files`` to be non-empty; the caller is responsible
    for ensuring this (``map_resolve_step`` must have run first and
    ``DeepWorkflowConfig.task_files`` must be set).
    """
    config = state["config"]
    effort_config = state["effort_config"]
    task_files = state.get("map_files") or []

    if not task_files:
        return {
            "error": (
                "map_plan_step: map_files is empty.  "
                "Set DeepWorkflowConfig.task_files when using map_plan_mode='static'."
            )
        }

    # Single-batch shortcut
    if effort_config.max_batches == 1:
        chunks = [task_files]
    elif effort_config.max_files_per_batch:
        chunk_size = effort_config.max_files_per_batch
        chunks = [task_files[i : i + chunk_size] for i in range(0, len(task_files), chunk_size)]
    else:
        # Fallback: everything in one batch
        chunks = [task_files]

    batches = [
        BatchDefinition(
            batch_files=chunk,
            batch_instructions=(
                "Execute the task instructions on this set of files. "
                "Other batches are processing a different set of files."
            ),
        )
        for chunk in chunks
    ]

    return {
        "map_batches": batches,
        "map_plan_overview": config.task_instructions,
        "reduce_instructions": "Combine all batch outputs into a single consolidated result.",
        "batch_evaluate_quality_instructions": effort_config.batch_evaluate_quality_instructions or "",
        "batch_current_index": 0,
        "batch_quality_retry_count": 0,
    }
