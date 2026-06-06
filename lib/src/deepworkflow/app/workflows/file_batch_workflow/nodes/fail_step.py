from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def fail_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Mark workflow as failed."""
    return {"error": "Workflow failed"}
