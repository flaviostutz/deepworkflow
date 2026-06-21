from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def map_effort_step(state: file_batch_workflow_state) -> dict:
    """Copy the fully-resolved EffortConfig from DeepWorkflowConfig into the graph state.

    This step runs when ``effort.type="static"``.  Because ``EffortConfig.__post_init__``
    already resolves all detail fields from ``level`` at construction time, no additional
    processing is needed here.
    """
    return {"effort_config": state["config"].effort}
