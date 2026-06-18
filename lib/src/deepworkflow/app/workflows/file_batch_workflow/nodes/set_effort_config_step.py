from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def set_effort_config_step(state: file_batch_workflow_state) -> dict:
    """Copy effort_config from DeepWorkflowConfig into the graph state.

    This step runs when ``effort="custom"``.  It simply propagates the
    user-supplied ``EffortConfig`` into the state so all downstream nodes
    can read it from ``state["effort_config"]``.
    """
    return {"effort_config": state["config"].effort_config}
