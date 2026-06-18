from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded, WorkflowLogLevel, WriteOption

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


_EFFORT_MAX_LEVEL = 10
_EFFORT_AGENT_MODE_THRESHOLD = 4
_EFFORT_SKIP_PLAN_THRESHOLD = 6


def resolveEffortConfig(level: int) -> EffortConfig:  # noqa: N802
    """Return a preset ``EffortConfig`` for the given effort level (1-10).

    - Level 1: everything static, all evaluations skipped; only the execute agent runs.
    - Level 10: everything agentic, maximum evaluation retries (10).

    Thresholds:
    - ``map_batches_mode`` / ``consolidate_mode`` flip to ``"agent"`` at level >= 4.
    - ``skip_batch_plan`` becomes ``False`` (plan agent enabled) at level >= 6.
    - Retry counts are linearly interpolated: ``0`` at level 1, ``10`` at level 10.
    """
    if not 1 <= level <= _EFFORT_MAX_LEVEL:
        msg = f"resolveEffortConfig: level must be between 1 and {_EFFORT_MAX_LEVEL}, got {level}"
        raise ValueError(msg)

    # Linear interpolation: 0 at level=1, 10 at level=10
    retries = round((level - 1) * _EFFORT_MAX_LEVEL / (_EFFORT_MAX_LEVEL - 1))

    mode: Literal["agent", "static"] = "agent" if level >= _EFFORT_AGENT_MODE_THRESHOLD else "static"
    consolidate: Literal["agent", "static"] = "agent" if level >= _EFFORT_AGENT_MODE_THRESHOLD else "static"
    skip_plan = level < _EFFORT_SKIP_PLAN_THRESHOLD

    if mode == "static" and level == 1:
        # Level 1: single-batch mode; max_files_per_batch not needed
        return EffortConfig(
            map_batches_mode="static",
            max_batches=1,
            max_files_per_batch=None,
            evaluate_map_max_retries=0,
            skip_batch_plan=True,
            evaluate_batch_convergence_max_retries=0,
            evaluate_batch_quality_max_retries=0,
            consolidate_mode="static",
        )

    if mode == "static":
        # Levels 2-3: static map but multiple batches are possible; caller must set max_files_per_batch.
        # We set a sensible default of 10 files/batch for levels 2-3.
        return EffortConfig(
            map_batches_mode="static",
            max_batches=None,
            max_files_per_batch=10,
            evaluate_map_max_retries=0,
            skip_batch_plan=skip_plan,
            evaluate_batch_convergence_max_retries=0,
            evaluate_batch_quality_max_retries=retries,
            consolidate_mode="static",
        )

    return EffortConfig(
        map_batches_mode="agent",
        max_batches=None,
        max_files_per_batch=None,
        evaluate_map_max_retries=retries,
        skip_batch_plan=skip_plan,
        evaluate_batch_convergence_max_retries=retries,
        evaluate_batch_quality_max_retries=retries,
        consolidate_mode=consolidate,
    )


@dataclass(frozen=True)
class DeepWorkflowConfig:
    """Configuration for a deepworkflow run."""

    workspace_dir: str
    """Workspace directory containing all files available for agents to read/write."""

    task_instructions: str
    """Instructions describing the objectives of the task, how to map/group work into
    focus batches, and how to evaluate whether outcomes have the desired quality."""

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

    effort: Literal["auto", "custom"] = "custom"
    """How to determine the effort configuration.

    - ``"custom"`` (default): use the ``effort_config`` field directly.
    - ``"auto"``: an ``analyze_task_effort_agent`` runs first and derives an ``EffortConfig``
      automatically based on the task instructions and files.
    """

    effort_config: EffortConfig | None = None
    """Effort configuration controlling how many LLM calls, evaluations, and retries the
    workflow performs.  Required when ``effort="custom"``; ignored when ``effort="auto"``.
    Use ``resolveEffortConfig(level)`` to construct a preset from a 1-10 scale."""

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

    evaluate_quality_on_max_retries: OnMaxRetriesExceeded = field(default=OnMaxRetriesExceeded.CONTINUE)
    """What to do when the maximum number of retries is reached without achieving the
    required minimum evaluate_quality verdict.  Either ``FAIL`` (abort the workflow) or
    ``CONTINUE`` (record the output as-is and move on).  Defaults to ``CONTINUE``."""

    evaluate_quality_min: JudgeLevel = field(default=JudgeLevel.WARNING)
    """Minimum evaluate_quality verdict required to consider a batch execution successful.
    If the verdict is below this threshold the batch is retried with evaluate quality feedback.
    Applied during both map and batch execution phases."""

    evaluate_quality_batch_instructions: str | None = None
    """Specific quality criteria used by evaluate quality when evaluating a batch.  Use
    mandatory/advisory language such as MUST, REQUIRED, MANDATORY or SHOULD, COULD to
    mark findings as ERROR, WARNING, or INFO respectively.  If ``None``, a default
    quality check is applied."""

    max_failure_retries: int = 0
    """How many times to retry when a system error occurs (e.g. network or
    authentication failures)."""

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    """MLflow tracking URI used to store experiment runs.  Defaults to a local
    SQLite database (``mlflow.db``).  Set to a remote URI such as
    ``http://my-mlflow-server:5000`` to use a remote tracking server."""

    log_level: WorkflowLogLevel = field(default=WorkflowLogLevel.NONE)
    """Console log verbosity for the workflow run.  One of ``NONE`` (default),
    ``INFO``, ``DEBUG``, or ``TRACE``.

    - ``NONE`` — no console output.
    - ``INFO`` — agent/route headers, in/out summaries, elapsed time, and a summary block.
    - ``DEBUG`` — like INFO, but prints full LLM-generated text (plans, outputs, evaluations)
      without truncation.
    - ``TRACE`` — every MLflow span printed as JSON (raw tracing).
    """

    def __post_init__(self) -> None:
        if self.effort == "custom" and self.effort_config is None:
            msg = "DeepWorkflowConfig: effort_config is required when effort='custom'."
            raise ValueError(msg)

        # Wrap raw callables in a _ModelRef so the frozen-dataclass field stores
        # a serialization-safe proxy rather than a bare Callable.  _ModelRef is
        # itself a frozen dataclass (string-only fields) that LangGraph can
        # checkpoint without issues.  Re-wrapping is skipped when the value is
        # already a _ModelRef (e.g. after LangGraph deserialises the checkpoint).
        if not isinstance(self.model, _ModelRef):
            config_id = str(_uuid.uuid4())
            _model_registry[config_id] = self.model
            object.__setattr__(self, "model", _ModelRef(config_id))
