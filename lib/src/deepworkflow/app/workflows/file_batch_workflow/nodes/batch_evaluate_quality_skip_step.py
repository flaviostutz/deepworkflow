from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import JudgeLevel


def batch_evaluate_quality_skip_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Skip batch evaluate quality: set a passing verdict so the batch is accepted immediately."""
    return {"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []}
