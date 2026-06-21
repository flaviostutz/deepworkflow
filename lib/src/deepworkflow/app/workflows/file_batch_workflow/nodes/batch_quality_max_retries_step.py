from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def batch_quality_max_retries_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Pass-through node enabling conditional routing after max retries are exceeded."""
    return {}
