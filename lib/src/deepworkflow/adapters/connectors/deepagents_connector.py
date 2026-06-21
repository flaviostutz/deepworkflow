from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import create_deep_agent

from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


def create_agent(
    *,
    model: BaseChatModel,
    system_prompt: str,
    workspace_dir: str,
    write_option: WriteOption = WriteOption.READ_ONLY,
) -> CompiledStateGraph:
    """Create a deepagent configured for use within the workflow.

    Accepts a pre-initialised LangChain ``BaseChatModel`` instance, which is
    obtained by calling ``config.model(agent_name)`` at the call site.  This
    allows callers to supply any LangChain-compatible LLM without relying on
    environment-variable-based provider detection.

    JSON output is enforced solely through the system prompt's ``OUTPUT_FORMAT``
    instructions, which works for all providers.
    """
    from deepagents.backends import FilesystemBackend

    permissions = _build_permissions(write_option)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=workspace_dir, virtual_mode=True),
        permissions=permissions,
    )


def _build_permissions(write_option: WriteOption) -> list | None:
    """Map WriteOption to deepagents FilesystemPermission rules."""
    from deepagents.middleware.filesystem import FilesystemPermission

    if write_option == WriteOption.READ_ONLY:
        return [
            FilesystemPermission(operations=["read"], paths=["/**"]),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ]
    if write_option == WriteOption.WRITE_ANY:
        return [
            FilesystemPermission(operations=["read", "write"], paths=["/**"]),
        ]
    # WRITE_ONLY_TASK_FILES: permissions are set per-batch at call site
    return None
