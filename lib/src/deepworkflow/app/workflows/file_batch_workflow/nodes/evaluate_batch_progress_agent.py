from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Decide whether the latest execution pass made meaningful progress toward completing the task."""

_ROLE = """\
You are the `evaluate_batch_progress_agent`. You are the progress judge: a lightweight checker who
assesses whether substantial, non-trivial work was done in the most recent pass — not whether the
final result is high quality (that is the quality judge's responsibility)."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope:
{batch_files}
- execute_output: {execute_output}
- files_read: {files_read}
- files_written: {files_written}"""

_GUARDRAILS = """\
- Do NOT evaluate the overall quality of the result — that is the quality judge's responsibility
  (`evaluate_batch_quality_agent`).
- Only assess whether this pass made meaningful, non-trivial progress (e.g. files were changed in
  a useful way, substantial work was done)."""

_OUTPUT_FORMAT = """\
PROGRESS: YES
REASON: <brief explanation>

or

PROGRESS: NO
REASON: <brief explanation>"""


def evaluate_batch_progress_agent(state: file_batch_workflow_state) -> dict:
    """Evaluate whether the latest pass made meaningful progress toward the task goal."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
            execute_output=state.get("execute_output", ""),
            files_read=", ".join(state.get("files_read", [])),
            files_written=", ".join(state.get("files_written", [])),
        ),
        guardrails=_GUARDRAILS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("evaluate_batch_progress_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    batch_progress = _parse_progress_output(content)

    return {"batch_progress": batch_progress}


def _parse_progress_output(content: str) -> bool:
    """Parse the progress agent response into a boolean."""
    for line in content.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("PROGRESS:"):
            value = stripped.removeprefix("PROGRESS:").strip()
            return value == "YES"
    # Default to False if no PROGRESS line found (treat as no progress)
    return False
