from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.shared.agent import create_workflow_agent
from deepworkflow.shared.prompts import workflow_role

if TYPE_CHECKING:
    from deepworkflow.app.workflows.deepworkflow.states import WorkflowState

EXECUTE_PROMPT = """{workflow_context}

Follow the plan below to accomplish the task.

Task instructions:
{task_instructions}

Task overview (broad strategy):
{task_overview}

Batch-specific instructions:
{batch_instructions}

Files to work with:
{batch_files}

Write option: {write_option}

Plan to follow:
{plan_output}

{judge_feedback_section}

Execute the plan now. If write option is 'read-only', produce analysis/report output only. \
If write option allows writing, make the necessary file changes."""


def execute_task_step(state: WorkflowState) -> dict:
    """Execute the plan for the current batch."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]
    plan_output = state["plan_output"]

    task_overview = state.get("task_overview", "")
    batch_instructions = current_batch.batch_instructions or "(none)"

    judge_feedback_section = ""
    if state.get("judge_feedbacks"):
        feedback_lines = [
            f"- [{fb.type.name}] {fb.file}: {fb.description}" + (f" | Proposal: {fb.proposal}" if fb.proposal else "")
            for fb in state["judge_feedbacks"]
        ]
        judge_feedback_section = (
            "Previous attempt was rejected. Address this feedback:\n"
            + "\n".join(feedback_lines)
            + "\n\nNote: The proposals above are optional suggestions. "
            "You do not need to follow them exactly if you find a better way to fix the issue."
        )

    prompt = EXECUTE_PROMPT.format(
        workflow_context=workflow_role("execute_task_step", "Execute the planned task on the current batch of files"),
        task_instructions=config.task_instructions,
        task_overview=task_overview,
        batch_instructions=batch_instructions,
        batch_files="\n".join(current_batch.batch_files),
        write_option=config.task_files_write_option.value,
        plan_output=plan_output,
        judge_feedback_section=judge_feedback_section,
    )

    agent = create_workflow_agent(
        model=config.model,
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=config.task_files_write_option,
    )

    result = agent.invoke({"messages": "Execute the plan now."})
    last_message = result["messages"][-1]
    execute_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Store messages for reflect_task_step to continue the conversation
    messages = result.get("messages", [])

    return {"execute_output": execute_output, "execute_messages": messages}
