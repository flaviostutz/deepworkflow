from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from deepworkflow.shared.types import WriteOption


def create_workflow_agent(
    *,
    model: str,
    system_prompt: str,
    workspace_dir: str,
    write_option: WriteOption = WriteOption.READ_ONLY,
) -> Any:
    """Create a deepagent configured for use within the workflow.

    Outbound connector that initialises a deepagents ReAct agent backed by a
    sandboxed filesystem. Supports both OpenAI and Azure OpenAI providers; the
    provider is selected via environment variables (OPENAI_API_TYPE=azure for
    Azure, omit for OpenAI).
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
