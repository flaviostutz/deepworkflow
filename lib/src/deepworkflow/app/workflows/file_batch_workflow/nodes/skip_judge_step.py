from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import JudgeVerdict


def skip_judge_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Skip judge evaluation: set a passing verdict so the batch is accepted immediately."""
    return {"judge_verdict": JudgeVerdict.OK, "judge_feedbacks": []}
