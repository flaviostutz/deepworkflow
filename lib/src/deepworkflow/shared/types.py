from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

# Sentinel used as the default for EffortConfig fields that must be resolved from ``level``
# but whose resolved value may legitimately be ``None`` (meaning "no limit").
# Applies only to ``max_batches`` and ``max_files_per_batch``.
_UNSET: Any = object()


class JudgeLevel(IntEnum):
    """Quality verdict levels. Higher value = better quality. OK > INFO > WARNING > ERROR."""

    ERROR = 0
    WARNING = 1
    INFO = 2
    OK = 3


class WriteOption(StrEnum):
    """File write permission modes for task execution."""

    READ_ONLY = "read-only"
    WRITE_ANY = "write-any"
    WRITE_ONLY_TASK_FILES = "write-only-task-files"


class OnMaxRetriesExceeded(StrEnum):
    """Behavior when evaluate_quality retries are exhausted."""

    FAIL = "fail"
    CONTINUE = "continue"


class WorkflowLogLevel(StrEnum):
    """Console log verbosity for the workflow run."""

    NONE = "none"
    """No console output."""

    INFO = "info"
    """Print agent/route header lines, in/out summaries, elapsed time, and a summary block."""

    DEBUG = "debug"
    """Like INFO but prints full LLM-generated text (plans, outputs, evaluations) without truncation."""

    TRACE = "trace"
    """Print every MLflow span as JSON (raw tracing)."""


@dataclass(frozen=True)
class EvaluateFeedback:
    """Single feedback item from the evaluate agent."""

    file: str
    type: JudgeLevel
    description: str
    proposal: str = ""


@dataclass(frozen=True)
class JudgeFinding:
    """Single finding from a judge node"""

    level: JudgeLevel
    title: str
    reason: str = ""
    details: str = ""
    fix: str = ""


@dataclass
class JudgeVerdict:
    """Structured verdict returned by judge nodes"""

    verdict: JudgeLevel
    findings: list[JudgeFinding] = field(default_factory=list)


def _level_defaults(level: int) -> dict:
    """Return fully-resolved default field values for an effort level (1-10)."""
    _max = 10
    retries = round((level - 1) * _max / (_max - 1))
    mode: Literal["agent", "static"] = "agent" if level >= 4 else "static"
    skip_plan = level < 6

    base: dict = {
        "evaluate_quality_min": JudgeLevel.WARNING,
        "evaluate_quality_on_max_retries": OnMaxRetriesExceeded.CONTINUE,
        "evaluate_quality_batch_instructions": None,
    }

    if level == 1:
        return {
            **base,
            "map_batches_mode": "static",
            "max_batches": 1,
            "max_files_per_batch": None,
            "evaluate_map_max_retries": 0,
            "skip_batch_plan": True,
            "evaluate_batch_convergence_max_retries": 0,
            "evaluate_batch_quality_max_retries": 0,
            "consolidate_mode": "static",
        }

    if mode == "static":  # levels 2-3
        return {
            **base,
            "map_batches_mode": "static",
            "max_batches": None,
            "max_files_per_batch": 10,
            "evaluate_map_max_retries": 0,
            "skip_batch_plan": skip_plan,
            "evaluate_batch_convergence_max_retries": 0,
            "evaluate_batch_quality_max_retries": retries,
            "consolidate_mode": "static",
        }

    # levels 4-10 (agent mode)
    return {
        **base,
        "map_batches_mode": "agent",
        "max_batches": None,
        "max_files_per_batch": None,
        "evaluate_map_max_retries": retries,
        "skip_batch_plan": skip_plan,
        "evaluate_batch_convergence_max_retries": retries,
        "evaluate_batch_quality_max_retries": retries,
        "consolidate_mode": "agent",
    }


@dataclass(frozen=True)
class EffortConfig:
    """Controls how much computational effort (LLM calls, evaluations, retries) the workflow uses.

    ``level`` selects a preset from 1 (minimal, all static, no evaluations) to 10 (maximal,
    all agentic, maximum retries).  Any detail field set explicitly overrides its level-derived
    default; unset fields are resolved automatically in ``__post_init__``.

    ``type`` controls whether the resolved config is used directly (``"custom"``) or whether a
    specialized ``analyze_task_effort_agent`` derives the best configuration automatically
    (``"auto"``).  When ``type="auto"``, the agent inspects ``task_instructions`` and the
    workspace files to decide the optimal effort level and quality gates; quality-gate
    instructions written in the prompt (MUST / SHOULD / COULD language) are automatically
    picked up.  No detail fields may be set alongside ``type="auto"``.

    Examples::

        # Use a level preset (all detail fields resolved automatically)
        EffortConfig(level=5)

        # Level preset with quality overrides
        EffortConfig(level=5, evaluate_quality_min=JudgeLevel.OK)

        # Fully custom — override whatever is needed
        EffortConfig(level=3, map_batches_mode="static", max_files_per_batch=20)

        # Let an agent decide automatically
        EffortConfig(type="auto")
    """

    level: int = 3
    """Base effort preset (1-10).  Used when ``type="custom"``.

    - Level 1: single-batch, all evaluations skipped (fastest / cheapest).
    - Level 3: default when ``effort`` is omitted entirely.
    - Level 10: maximum evaluations and retries (highest quality, most expensive).
    """

    type: Literal["auto", "custom"] = "custom"
    """How to determine effort.

    - ``"custom"`` (default): resolve detail fields from ``level``, applying any explicit
      overrides.
    - ``"auto"``: a specialized ``analyze_task_effort_agent`` analyses
      ``task_instructions`` and the workspace to decide the optimal level and quality gates.
      Quality-gate instructions embedded in the prompt (MUST / SHOULD / COULD language) are
      automatically used to configure evaluation criteria.  ``level`` and all detail fields
      are ignored.
    """

    # ------------------------------------------------------------------
    # Detail fields — all default to None / _UNSET meaning
    # "not specified; resolve from level in __post_init__".
    # After construction every field is fully resolved.
    # ------------------------------------------------------------------

    map_batches_mode: Literal["agent", "static"] | None = None
    """Whether to use an LLM agent or a deterministic algorithm to split files into batches.
    Resolved from ``level`` if not set."""

    max_batches: Any = field(default=_UNSET)
    """Maximum number of batches allowed.  ``None`` means no limit.
    Resolved from ``level`` if not set (``_UNSET`` default)."""

    max_files_per_batch: Any = field(default=_UNSET)
    """Maximum files per batch.  ``None`` means no limit.  Required when
    ``map_batches_mode="static"`` unless ``max_batches=1``.
    Resolved from ``level`` if not set (``_UNSET`` default)."""

    evaluate_map_max_retries: int | None = None
    """Max retries for the map evaluation loop.  ``0`` skips LLM map evaluation entirely.
    Resolved from ``level`` if not set."""

    skip_batch_plan: bool | None = None
    """When ``True``, skip the plan_batch_agent and inject a planning instruction into the
    execute_batch_agent prompt.  Resolved from ``level`` if not set."""

    evaluate_batch_convergence_max_retries: int | None = None
    """Max additional plan→execute→reflect passes per batch.  ``0`` disables the repeat loop.
    Resolved from ``level`` if not set."""

    evaluate_batch_quality_max_retries: int | None = None
    """Max retries for the per-batch quality evaluation loop.  ``0`` skips quality evaluation.
    Resolved from ``level`` if not set."""

    consolidate_mode: Literal["agent", "static"] | None = None
    """Whether to use an LLM agent or a deterministic formatter for final consolidation.
    Resolved from ``level`` if not set."""

    evaluate_quality_min: JudgeLevel | None = None
    """Minimum evaluate_quality verdict required to accept a batch result.
    Resolved from ``level`` if not set (default: ``JudgeLevel.WARNING``)."""

    evaluate_quality_on_max_retries: OnMaxRetriesExceeded | None = None
    """Behaviour when max retries are exhausted without meeting ``evaluate_quality_min``.
    Resolved from ``level`` if not set (default: ``CONTINUE``)."""

    evaluate_quality_batch_instructions: str | None = None
    """Custom quality criteria for the evaluate quality agent.  Use MUST / SHOULD / COULD
    language to mark findings as ERROR / WARNING / INFO.  ``None`` applies the default
    quality check."""

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    _DETAIL_FIELDS = (
        "map_batches_mode",
        "max_batches",
        "max_files_per_batch",
        "evaluate_map_max_retries",
        "skip_batch_plan",
        "evaluate_batch_convergence_max_retries",
        "evaluate_batch_quality_max_retries",
        "consolidate_mode",
        "evaluate_quality_min",
        "evaluate_quality_on_max_retries",
        "evaluate_quality_batch_instructions",
    )

    def __post_init__(self) -> None:
        # --- type="auto" guard (checked BEFORE resolution) ---
        if self.type == "auto":
            explicitly_set = [
                f for f in self._DETAIL_FIELDS if getattr(self, f) is not None and getattr(self, f) is not _UNSET
            ]
            if explicitly_set:
                msg = f"EffortConfig: no options can be set when type='auto'. Found: {sorted(explicitly_set)}"
                raise ValueError(msg)
            return  # leave detail fields at None / _UNSET; agent will provide them

        # --- Validate level ---
        if not 1 <= self.level <= 10:
            msg = f"EffortConfig: level must be between 1 and 10, got {self.level}"
            raise ValueError(msg)

        # --- Resolve unset fields from level defaults ---
        defaults = _level_defaults(self.level)
        for name, default_val in defaults.items():
            current = getattr(self, name)
            if current is None or current is _UNSET:
                object.__setattr__(self, name, default_val)

        # --- Validate static mode constraint ---
        if self.map_batches_mode == "static" and self.max_files_per_batch is None and self.max_batches != 1:
            msg = (
                "EffortConfig: max_files_per_batch is required when map_batches_mode='static', "
                "unless max_batches=1 (single-batch mode)."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class BatchDefinition:
    """Defines a single batch produced by the map_batches_agent."""

    batch_files: list[str]
    batch_instructions: str | None = None


@dataclass(frozen=True)
class BatchOutput:
    """Output produced per batch execution."""

    task_files: list[str]
    evaluate_quality_verdict: JudgeLevel
    evaluate_quality_feedbacks: list[EvaluateFeedback]
    files_read: list[str]
    files_written: list[str]
    execute_output: str


@dataclass(frozen=True)
class WorkflowResult:
    """Result returned by run_workflow."""

    thread_id: str
    output: str
    status: str
