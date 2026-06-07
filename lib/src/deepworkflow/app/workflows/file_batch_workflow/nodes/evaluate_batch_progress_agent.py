from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

PROGRESS_PROMPT = """{workflow_context}

You are the **progress judge** for a single batch execution pass.
Your role is to decide whether the LATEST pass made meaningful progress toward completing the
task — NOT to evaluate the overall quality of the result. Quality evaluation is handled by a
separate judge (evaluate_batch_quality_agent) that runs after all passes complete.

Task instructions: {task_instructions}

Files in scope: {batch_files}

Execution output:
{execute_output}

Files read: {files_read}
Files written: {files_written}

Did this pass make meaningful, non-trivial progress toward completing the task?
Consider: were files actually changed in a useful way? Was substantial work done?
Answer YES if real progress was made, NO if the pass was a no-op or made negligible changes.

Respond in exactly this format:
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

    prompt = PROGRESS_PROMPT.format(
        workflow_context=workflow_role(
            "evaluate_batch_progress_agent", "Judge whether meaningful progress was made in this execution pass"
        ),
        task_instructions=config.task_instructions,
        batch_files="\n".join(current_batch.batch_files),
        execute_output=state.get("execute_output", ""),
        files_read=", ".join(state.get("files_read", [])),
        files_written=", ".join(state.get("files_written", [])),
    )

    agent = create_agent(
        model=config.model("evaluate_batch_progress_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Evaluate whether meaningful progress was made in this pass."})
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
