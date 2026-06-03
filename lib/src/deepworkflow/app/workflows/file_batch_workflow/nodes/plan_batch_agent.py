from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_workflow_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

PLAN_PROMPT = """{workflow_context}

Given a set of files and a task instruction, produce a detailed step-by-step plan \
for how to accomplish the task.

Task instructions:
{task_instructions}

Task overview (broad strategy from the planning phase):
{task_overview}

Batch-specific instructions:
{batch_instructions}

Files to work with:
{batch_files}

Write option: {write_option}

{judge_feedback_section}

Produce a clear, actionable plan. Do not execute the plan — only describe what steps should be taken."""


def plan_batch_agent(state: file_batch_workflow_state) -> dict:
    """Plan the execution for the current batch.

    Creates a fresh agent each time (including retries):
    'spawns a brand new agent thread with only task context + judge_feedback'.
    """
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    task_overview = state.get("task_overview", "")
    batch_instructions = current_batch.batch_instructions or "(none)"

    judge_feedback_section = ""
    if state.get("judge_feedbacks"):
        feedback_lines = [
            f"- [{fb.type.name}] {fb.file}: {fb.description}" + (f" | Proposal: {fb.proposal}" if fb.proposal else "")
            for fb in state["judge_feedbacks"]
        ]
        judge_feedback_section = "Previous attempt was rejected by the judge. Address this feedback:\n" + "\n".join(
            feedback_lines
        )

    prompt = PLAN_PROMPT.format(
        workflow_context=workflow_role("plan_batch_agent", "Plan execution strategy for the current batch"),
        task_instructions=config.task_instructions,
        task_overview=task_overview,
        batch_instructions=batch_instructions,
        batch_files="\n".join(current_batch.batch_files),
        write_option=config.task_files_write_option.value,
        judge_feedback_section=judge_feedback_section,
    )

    agent = create_workflow_agent(
        model=config.model,
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Read the relevant files and produce your plan."})
    last_message = result["messages"][-1]
    plan_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {"plan_output": plan_output}
