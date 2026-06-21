from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def map_evaluate_retry_step(state: file_batch_workflow_state) -> dict:
    """Increment map evaluate retry count before looping back to map_plan_agent."""
    return {"map_evaluate_retry_count": state.get("map_evaluate_retry_count", 0) + 1}
