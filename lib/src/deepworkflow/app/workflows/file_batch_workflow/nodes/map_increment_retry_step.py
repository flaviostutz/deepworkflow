from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def map_increment_retry_step(state: file_batch_workflow_state) -> dict:
    """Increment map retry count before looping back to map_batches_agent."""
    return {"map_retry_count": state.get("map_retry_count", 0) + 1}
