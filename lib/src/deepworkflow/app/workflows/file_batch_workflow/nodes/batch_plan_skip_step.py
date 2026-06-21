from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_PLAN_INSTRUCTION = (
    "Before executing the task instructions you MUST plan your actions step by step on how to "
    "perform the task instructions on the batch files. "
    "Use the todo tool to organize and track your progress."
)


def batch_plan_skip_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Inject a self-planning instruction into batch_plan instead of running batch_plan_agent.

    When ``effort_config.batch_skip_plan=True`` the separate planning agent is skipped.
    This step sets ``batch_plan`` to a directive that instructs the batch_execute_agent
    to plan its own actions before proceeding.
    """
    return {"batch_plan": _PLAN_INSTRUCTION}
