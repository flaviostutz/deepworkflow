from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WorkflowLogLevel, WriteOption

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

# Module-level registry mapping config_id → model factory callable.
# Keeps the non-serializable Callable out of the LangGraph checkpoint state.
_model_registry: dict[str, Callable] = {}


@dataclass(frozen=True)
class _ModelRef:
    """Serializable proxy for a model factory callable.

    Stores only a UUID key into ``_model_registry`` so LangGraph can checkpoint
    this object without ever touching the underlying non-serializable Callable.
    The actual factory is recovered from the registry on each ``__call__``.
    """

    _config_id: str

    def __call__(self, agent_name: str) -> BaseChatModel:
        factory = _model_registry.get(self._config_id)
        if factory is None:
            msg = (
                f"No model factory found for config_id={self._config_id!r}. "
                "The factory may have been lost if the process restarted."
            )
            raise RuntimeError(msg)
        return factory(agent_name)


@dataclass(frozen=True)
class DeepWorkflowConfig:
    """Configuration for a deepworkflow run."""

    workspace_dir: str
    """Workspace directory containing all files available for agents to read/write."""

    task_instructions: str
    """Instructions describing the objectives of the task, how to map/group work into
    focus batches, and how to judge whether outcomes have the desired quality."""

    model: Callable[[str], BaseChatModel]
    """Factory function that returns a LangChain ``BaseChatModel`` instance for each
    agent invocation.  Called as ``config.model(agent_name)`` where *agent_name* is the
    name of the agent being created (e.g. ``"execute_batch_agent"``).  The agent name
    can be used to return different models or configurations for different parts of the
    workflow.

    Pass any callable; it is wrapped internally in a serialization-safe proxy so
    that LangGraph can checkpoint the workflow state without errors.

    Example using a single model for all agents::

        from langchain_openai import ChatOpenAI
        model=lambda agent_name: ChatOpenAI(model="gpt-4o", api_key="...")

    Example using different models per agent::

        def my_factory(agent_name: str) -> BaseChatModel:
            if agent_name in ("execute_batch_agent", "plan_batch_agent"):
                return ChatOpenAI(model="gpt-4o", api_key="...")
            return ChatOpenAI(model="gpt-4o-mini", api_key="...")

        model=my_factory
    """

    workspace_write_option: WriteOption
    """Whether agents can write files in the workspace.  One of ``READ_ONLY``,
    ``WRITE_ANY``, or ``WRITE_ONLY_TASK_FILES``."""

    judge_max_retries: int
    """Maximum number of times a batch can be retried because it did not reach the
    minimum required judge outcome."""

    judge_on_max_retries: OnMaxRetriesExceeded
    """What to do when the maximum number of retries is reached without achieving the
    required minimum judge verdict.  Either ``FAIL`` (abort the workflow) or
    ``CONTINUE`` (record the output as-is and move on)."""

    task_files: list[str] | None = None
    """List of files or glob selectors pointing to **existing files** in the workspace
    to distribute across execution batches.  All resolved paths must exist on disk —
    the map agent will reject any path that cannot be found.  Supports line-range
    suffixes (e.g. ``README.md:34-56``).  If ``None``, the map agent will discover
    relevant files from the workspace based on ``task_instructions``."""

    task_files_exclude: list[str] | None = None
    """List of files or glob patterns for files that must always be excluded from
    processing batches.  Patterns are resolved relative to ``workspace_dir``.
    Exclusion is applied after ``task_files`` glob expansion in ``resolve_globs_step``
    and is also communicated to ``map_batches_agent`` when it discovers files itself.
    The ``evaluate_map_batches_agent`` verifies that no excluded file appears in any
    batch.  If ``None``, no files are excluded."""

    judge_min: JudgeVerdict = field(default=JudgeVerdict.WARNING)
    """Minimum judge verdict required to consider a batch execution successful.
    If the verdict is below this threshold the batch is retried with judge feedback.
    Applied during both map and batch execution phases."""

    task_files_batch_size: int | None = None
    """Number of files to include in each batch.  If ``0``, all files go into a single
    batch.  If ``None``, the map agent decides the optimal grouping based on
    ``task_instructions``."""

    judge_batch_instructions: str | None = None
    """Specific quality criteria used by the judge when evaluating a batch.  Use
    mandatory/advisory language such as MUST, REQUIRED, MANDATORY or SHOULD, COULD to
    mark findings as ERROR, WARNING, or INFO respectively.  If ``None``, a default
    quality check is applied."""

    max_failure_retries: int = 0
    """How many times to retry when a system error occurs (e.g. network or
    authentication failures)."""

    judge_skip: bool = False
    """If ``True``, no judge steps are performed during map and batch execution.
    Can save time and tokens, but the quality of the work will not be verified."""

    batch_repeat_max: int = 0
    """Maximum number of additional plan→execute→reflect passes to run per batch
    when the progress judge detects meaningful work was done.  ``0`` (default)
    disables the repeat loop entirely.  When greater than zero, after each
    reflect a lightweight progress judge asks whether the pass made meaningful
    progress; if yes and the ceiling has not been reached the loop repeats with
    a fresh agent session.  The quality judge still runs only once, after all
    repeats complete."""

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    """MLflow tracking URI used to store experiment runs.  Defaults to a local
    SQLite database (``mlflow.db``).  Set to a remote URI such as
    ``http://my-mlflow-server:5000`` to use a remote tracking server."""

    log_level: WorkflowLogLevel = field(default=WorkflowLogLevel.NONE)
    """Console log verbosity for the workflow run.  One of ``NONE`` (default), ``TRACE``,
    or ``INFO``.

    - ``NONE`` — no console output.
    - ``TRACE`` — every MLflow span printed as JSON (raw tracing).
    - ``INFO`` — agent/route headers, in/out summaries, elapsed time, and a summary block.
    """

    def __post_init__(self) -> None:
        # Wrap raw callables in a _ModelRef so the frozen-dataclass field stores
        # a serialization-safe proxy rather than a bare Callable.  _ModelRef is
        # itself a frozen dataclass (string-only fields) that LangGraph can
        # checkpoint without issues.  Re-wrapping is skipped when the value is
        # already a _ModelRef (e.g. after LangGraph deserialises the checkpoint).
        if not isinstance(self.model, _ModelRef):
            config_id = str(_uuid.uuid4())
            _model_registry[config_id] = self.model
            object.__setattr__(self, "model", _ModelRef(config_id))
