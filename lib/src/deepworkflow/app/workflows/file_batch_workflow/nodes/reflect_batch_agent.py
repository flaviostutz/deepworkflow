from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, build_agent_prompt

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Inspect the conversation and tool call history from the previous execution phase, then report
exactly which files were read and which were written or modified.

Steps:
1. Inspect the prior tool call messages in the conversation thread.
2. Identify every file path that appears in read-tool calls → FILES_READ list.
3. Identify every file path that appears in write/modify-tool calls → FILES_WRITTEN list.
4. Respond in exactly the OUTPUT_FORMAT — nothing else."""

_ROLE = """\
You are the `reflect_batch_agent`. You are a precise observer who analyses the execution
transcript to extract file I/O facts — nothing more."""

_INPUT = """\
The execution conversation history provided as the message thread preceding this request."""

_TOOL_GUIDANCE = """\
No tools are available for this agent. All reasoning is done in-context by inspecting
the prior tool call messages only."""

_OUTPUT_FORMAT = """\
FILES_READ:
<one absolute or relative file path per line>

FILES_WRITTEN:
<one absolute or relative file path per line>

If no files were read or written in a section, leave it empty after the header."""

_SYSTEM_PROMPT = build_agent_prompt(
    objective=_OBJECTIVE,
    role=_ROLE,
    input_section=_INPUT,
    tool_guidance=_TOOL_GUIDANCE,
    output_format=_OUTPUT_FORMAT,
)


def reflect_batch_agent(state: file_batch_workflow_state) -> dict:
    """Reflect on execution to identify files read and written.

    Continues the same agent thread from execute_batch_agent by sending
    a follow-up message to the existing conversation.
    """
    config = state["config"]
    execute_messages = state.get("execute_messages", [])

    if not execute_messages:
        return {"files_read": [], "files_written": []}

    model = config.model("reflect_batch_agent")
    messages = [SystemMessage(content=_SYSTEM_PROMPT), *execute_messages, HumanMessage(content=STANDARD_USER_MESSAGE)]
    response = model.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

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
