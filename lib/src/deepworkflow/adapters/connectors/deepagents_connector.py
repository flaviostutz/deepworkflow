from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import create_deep_agent

from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


def bind_json_mode(model: BaseChatModel) -> BaseChatModel:
    """Return the model with JSON output enforced, in a provider-aware way.

    For OpenAI and Azure OpenAI models, binds ``response_format={"type": "json_object"}``
    at the API level so the model is constrained to emit valid JSON.

    For other providers (e.g. Anthropic/Claude), returns the model unchanged because
    they do not accept this OpenAI-specific parameter — JSON output is enforced through
    the system prompt's ``OUTPUT_FORMAT`` instructions instead.

    .. warning::
        The returned value is a ``_ChatModelBinding`` (not a plain ``BaseChatModel``)
        when binding is applied.  It is safe to call ``.invoke()`` on directly, but
        must NOT be passed to ``create_agent`` / ``deepagents.create_deep_agent``
        which require a plain ``BaseChatModel``.
    """
    try:
        provider = model._get_ls_params().get("ls_provider", "")
    except Exception:  # noqa: BLE001
        provider = ""
    if provider in ("openai", "azure"):
        return model.bind(response_format={"type": "json_object"})  # type: ignore[return-value]
    return model


def create_agent(
    *,
    model: BaseChatModel,
    system_prompt: str,
    workspace_dir: str,
    write_option: WriteOption = WriteOption.READ_ONLY,
    json_mode: bool = False,
) -> CompiledStateGraph:
    """Create a deepagent configured for use within the workflow.

    Accepts a pre-initialised LangChain ``BaseChatModel`` instance, which is
    obtained by calling ``config.model(agent_name)`` at the call site.  This
    allows callers to supply any LangChain-compatible LLM without relying on
    environment-variable-based provider detection.

    The ``json_mode`` parameter is accepted for API compatibility but is not
    applied via model binding — JSON output is enforced through the system
    prompt's output format instructions, which the parsers already handle
    (including markdown-fenced code blocks).
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
