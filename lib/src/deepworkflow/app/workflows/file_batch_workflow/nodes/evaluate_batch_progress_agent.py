from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, build_agent_prompt

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

The full execution conversation and all tool call history are available in the messages preceding
this request. Inspect them to assess what was actually done."""

_GUARDRAILS = """\
- Do NOT evaluate the overall quality of the result — that is the quality judge's responsibility
  (`evaluate_batch_quality_agent`).
- Only assess whether this pass made meaningful, non-trivial progress by inspecting the tool call
  history in the preceding messages.
- Consider PROGRESS: NO if any of the following apply:
  * No files were written, or the files written contain only trivial changes (e.g. whitespace,
    formatting, or style adjustments with no semantic difference).
  * The output produced is low-signal for the task — for example, a code-review task that surfaced
    only INFO-level observations with no new WARNINGs or ERRORs is not meaningful progress.
  * The volume of new or changed content is negligible relative to the scope of the task.
- No tools are available for this agent. All reasoning is done in-context by inspecting the prior
  tool call messages only."""

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
    execute_messages = state.get("execute_messages", [])

    system_prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
        ),
        guardrails=_GUARDRAILS,
        output_format=_OUTPUT_FORMAT,
    )

    model = config.model("evaluate_batch_progress_agent")
    messages = [SystemMessage(content=system_prompt), *execute_messages, HumanMessage(content=STANDARD_USER_MESSAGE)]
    response = model.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    batch_progress = _parse_progress_output(content)

    return {"batch_progress": batch_progress, "batch_progress_output": content}


def _parse_progress_output(content: str) -> bool:
    """Parse the progress agent response into a boolean."""
    for line in content.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("PROGRESS:"):
            value = stripped.removeprefix("PROGRESS:").strip()
            return value == "YES"
    # Default to False if no PROGRESS line found (treat as no progress)
    return False
