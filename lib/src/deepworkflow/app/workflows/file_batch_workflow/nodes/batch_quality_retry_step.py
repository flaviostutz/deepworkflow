from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.workflow_log import _stats_var


def batch_quality_retry_step(state: file_batch_workflow_state) -> dict:
    """Increment retry count before looping back to plan."""
    new_retry_count = state.get("batch_quality_retry_count", 0) + 1

    # Only count as a quality retry if an actual plan+execute cycle will follow.
    # batch_quality_check_retries routes to plan when retry_count <= max_retries; otherwise to
    # max_retries_exceeded.  The last call here (new_retry_count > max) must not
    # be counted.
    effort_config = state.get("effort_config")
    max_retries = effort_config.batch_evaluate_quality_max_retries if effort_config is not None else 0
    if max_retries is not None and new_retry_count <= max_retries:
        stats = _stats_var.get()
        if stats is not None:
            stats.batch_quality_retries += 1

    return {
        "batch_quality_retry_count": new_retry_count,
        "batch_convergence_repeat_count": 0,
        "batch_cumulative_files_read": [],
        "batch_cumulative_files_written": [],
    }
