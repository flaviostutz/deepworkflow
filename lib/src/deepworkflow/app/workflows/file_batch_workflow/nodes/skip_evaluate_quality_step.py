from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import JudgeLevel


def skip_evaluate_quality_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Skip evaluate quality: set a passing verdict so the batch is accepted immediately."""
    return {"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []}
