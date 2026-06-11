from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Produce a detailed, actionable step-by-step plan for accomplishing the task on the current batch of
files. Do not execute the plan."""

_ROLE = """\
You are the `plan_batch_agent`. You are an expert at analysing a file set and designing a precise,
ordered execution plan that a separate agent will carry out."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}
- task_overview: {task_overview}

Agent-specific inputs:
- batch_instructions: {batch_instructions}
- files_to_work_with:
{batch_files}
- write_option: {write_option}
{judge_feedback_section}"""

_TOOL_GUIDANCE = f"""{TOOL_GUIDANCE_BASE}

Read the relevant files before planning so your plan is grounded in their actual content.
Do not make any changes — only analyse and plan."""

_OUTPUT_FORMAT = """\
A clear, actionable prose or numbered-step plan. Each step must describe exactly what should be done
(file path, change type, rationale). Do not include any implementation — only the plan."""


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

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            task_overview=task_overview,
            batch_instructions=batch_instructions,
            batch_files="\n".join(current_batch.batch_files),
            write_option=config.workspace_write_option.value,
            judge_feedback_section=judge_feedback_section,
        ),
        tool_guidance=_TOOL_GUIDANCE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("plan_batch_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    plan_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {"plan_output": plan_output}
