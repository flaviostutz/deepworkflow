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
    """
    from deepagents.backends import FilesystemBackend

    permissions = _build_permissions(write_option)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=workspace_dir),
        permissions=permissions,
    )


def _build_permissions(write_option: WriteOption) -> list[dict[str, str]] | None:
    """Map WriteOption to deepagents filesystem permissions."""
    if write_option == WriteOption.READ_ONLY:
        return [{"path": "**", "mode": "read"}]
    if write_option == WriteOption.WRITE_ANY:
        return [{"path": "**", "mode": "write"}]
    # WRITE_ONLY_TASK_FILES: permissions are set per-batch at call site
    return None
