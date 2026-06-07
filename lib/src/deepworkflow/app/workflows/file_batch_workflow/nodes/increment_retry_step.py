from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def increment_retry_step(state: file_batch_workflow_state) -> dict:
    """Increment retry count before looping back to plan."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "batch_repeat_count": 0,
        "cumulative_files_read": [],
        "cumulative_files_written": [],
    }
