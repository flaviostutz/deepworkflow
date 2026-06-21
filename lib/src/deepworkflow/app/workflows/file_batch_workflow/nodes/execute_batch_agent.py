from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Execute the provided plan on the current batch of files and produce a summary of what was done."""

_ROLE = """\
You are the `execute_batch_agent`. You are an expert at carrying out a well-defined plan precisely,
reading and modifying files according to the task instructions and write option."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}
- task_overview: {task_overview}

Agent-specific inputs:
- batch_instructions: {batch_instructions}
- files_to_work_with:
{batch_files}
- write_option: {write_option}
- plan_to_follow:
{batch_plan}
{cumulative_files_section}{evaluate_quality_feedback_section}{cumulative_execute_output_section}"""

_GUARDRAILS = """\
- If write_option is 'read-only', produce analysis or report output only — do NOT write or modify
  any files.
- If write_option allows writing, make the necessary file changes as described in the plan.
- Address all evaluate quality feedback from the previous attempt when present.
- If cumulative_execute_output is provided, do not repeat work already captured there. Only perform
  additional work. If nothing meaningful remains to be done, output exactly:
  'The previous output is already sufficient.'"""

_OUTPUT_FORMAT = """\
A prose summary describing what was done: which files were read or modified, what changes were made,
and any notable observations or issues encountered."""


def execute_batch_agent(state: file_batch_workflow_state) -> dict:
    """Execute the plan for the current batch."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]
    batch_plan = state["batch_plan"]

    task_overview = state.get("task_overview", "")
    batch_instructions = current_batch.batch_instructions or "(none)"

    evaluate_quality_feedback_section = ""
    judge_verdict = state.get("evaluate_quality_judge_verdict")
    if judge_verdict is not None and judge_verdict.findings:
        feedback_lines = [
            f"- [{f.level.name}] {f.title}"
            + (f": {f.reason}" if f.reason else "")
            + (f" | Fix: {f.fix}" if f.fix else "")
            for f in judge_verdict.findings
        ]
        evaluate_quality_feedback_section = (
            "Previous attempt was rejected. Address this feedback:\n"
            + "\n".join(feedback_lines)
            + "\n\nNote: The proposals above are optional suggestions. "
            "You do not need to follow them exactly if you find a better way to fix the issue."
        )

    cumulative_execute_output = state.get("cumulative_execute_output", "")
    if cumulative_execute_output:
        cumulative_execute_output_section = (
            "Previous passes already completed — your work is ADDITIONAL to this:\n"
            + cumulative_execute_output
            + "\n\nIf there is nothing meaningful left to do, output exactly: "
            "'The previous output is already sufficient.'"
        )
    else:
        cumulative_execute_output_section = ""

    cumulative_files_read = state.get("cumulative_files_read", [])
    cumulative_files_written = state.get("cumulative_files_written", [])
    if cumulative_files_read or cumulative_files_written:
        cumulative_files_section = "Files already worked on in previous passes:\n"
        if cumulative_files_read:
            cumulative_files_section += "- Already read: " + ", ".join(cumulative_files_read) + "\n"
        if cumulative_files_written:
            cumulative_files_section += "- Already written: " + ", ".join(cumulative_files_written) + "\n"
    else:
        cumulative_files_section = ""

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            task_overview=task_overview,
            batch_instructions=batch_instructions,
            batch_files="\n".join(current_batch.batch_files),
            write_option=config.workspace_write_option.value,
            batch_plan=batch_plan,
            cumulative_files_section=cumulative_files_section,
            evaluate_quality_feedback_section=evaluate_quality_feedback_section,
            cumulative_execute_output_section=cumulative_execute_output_section,
        ),
        guardrails=_GUARDRAILS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("execute_batch_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=config.workspace_write_option,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    execute_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Store messages for reflect_batch_agent to continue the conversation
    messages = result.get("messages", [])

    return {"execute_output": execute_output, "execute_messages": messages}
