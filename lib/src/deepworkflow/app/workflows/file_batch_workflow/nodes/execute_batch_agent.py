from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import workflow_role

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

EXECUTE_PROMPT = """<OBJECTIVE>
Execute the provided plan on the current batch of files and produce a summary of what was done.
</OBJECTIVE>

<ROLE>
You are the `execute_batch_agent` (see WORKFLOW_CONTEXT). You are an expert at carrying out a \
well-defined plan precisely, reading and modifying files according to the task instructions and \
write option.
</ROLE>

<INPUT>
Workflow-level inputs:
- task_instructions: {task_instructions}
- task_overview: {task_overview}

Agent-specific inputs:
- batch_instructions: {batch_instructions}
- files_to_work_with:
{batch_files}
- write_option: {write_option}
- plan_to_follow:
{plan_output}
{judge_feedback_section}
</INPUT>

<GUARDRAILS>
- If write_option is 'read-only', produce analysis or report output only — do NOT write or modify
  any files.
- If write_option allows writing, make the necessary file changes as described in the plan.
- Address all judge feedback from the previous attempt when present.
</GUARDRAILS>

<OUTPUT_FORMAT>
A prose summary describing what was done: which files were read or modified, what changes were made,
and any notable observations or issues encountered.
</OUTPUT_FORMAT>

<WORKFLOW_CONTEXT>
{workflow_context}
</WORKFLOW_CONTEXT>"""


def execute_batch_agent(state: file_batch_workflow_state) -> dict:
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
        workflow_context=workflow_role("execute_batch_agent", "Execute the planned task on the current batch of files"),
        task_instructions=config.task_instructions,
        task_overview=task_overview,
        batch_instructions=batch_instructions,
        batch_files="\n".join(current_batch.batch_files),
        write_option=config.workspace_write_option.value,
        plan_output=plan_output,
        judge_feedback_section=judge_feedback_section,
    )

    agent = create_agent(
        model=config.model("execute_batch_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=config.workspace_write_option,
    )

    result = agent.invoke({"messages": "Execute the plan now."})
    last_message = result["messages"][-1]
    execute_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Store messages for reflect_batch_agent to continue the conversation
    messages = result.get("messages", [])

    return {"execute_output": execute_output, "execute_messages": messages}
