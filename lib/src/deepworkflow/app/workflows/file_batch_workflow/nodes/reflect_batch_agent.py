from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

REFLECT_MESSAGE = """Now reflect on what you just did. Inspect your tool call history and identify:
1. Which files you READ during execution (list full paths, one per line)
2. Which files you WROTE/MODIFIED during execution (list full paths, one per line)

Respond in exactly this format:
FILES_READ:
<one file path per line>

FILES_WRITTEN:
<one file path per line>

If no files were read or written in a section, leave it empty after the header."""


def reflect_batch_agent(state: file_batch_workflow_state) -> dict:
    """Reflect on execution to identify files read and written.

    Continues the same agent thread from execute_batch_agent by sending
    a follow-up message to the existing conversation.
    """
    config = state["config"]
    execute_messages = state.get("execute_messages", [])

    if not execute_messages:
        # Fallback: no messages to continue, return empty
        return {"files_read": [], "files_written": []}

    # Create agent and invoke with existing messages + reflect follow-up
    agent = create_agent(
        model=config.model("reflect_batch_agent"),
        system_prompt=workflow_role(
            "reflect_batch_agent", "Identify which files were read and written during execution"
        ),
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    # Continue conversation by appending reflect message to existing messages
    messages = [*execute_messages, {"role": "user", "content": REFLECT_MESSAGE}]
    result = agent.invoke({"messages": messages})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    files_read, files_written = _parse_reflect_output(content)

    return {"files_read": files_read, "files_written": files_written}


def _parse_reflect_output(content: str) -> tuple[list[str], list[str]]:
    """Parse the structured reflection output into file lists."""
    files_read: list[str] = []
    files_written: list[str] = []

    current_section: list[str] | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("FILES_READ"):
            current_section = files_read
        elif stripped.startswith("FILES_WRITTEN"):
            current_section = files_written
        elif stripped and current_section is not None:
            current_section.append(stripped)

    return files_read, files_written
