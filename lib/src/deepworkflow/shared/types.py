from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Literal


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


@dataclass(frozen=True)
class EffortConfig:
    """Controls how much computational effort (LLM calls, evaluations, retries) the workflow uses.

    Use ``resolveEffortConfig(level)`` to construct a preset from a 1-10 scale, or set each field
    manually for fine-grained control.
    """

    map_batches_mode: Literal["agent", "static"] = "agent"
    """Whether to use an LLM agent or a deterministic algorithm to split files into batches."""

    max_batches: int | None = None
    """Maximum number of batches allowed.  ``None`` means no limit."""

    max_files_per_batch: int | None = None
    """Maximum files per batch.  Required when ``map_batches_mode="static"`` unless ``max_batches=1``."""

    evaluate_map_max_retries: int = 1
    """Max retries for the map evaluation loop (both quality failures and limit violations in agent
    mode).  ``0`` skips LLM map evaluation entirely."""

    skip_batch_plan: bool = False
    """When ``True``, skip the plan_batch_agent and instead inject a planning instruction into
    the execute_batch_agent prompt."""

    evaluate_batch_convergence_max_retries: int = 0
    """Max additional plan→execute→reflect passes per batch.  ``0`` disables the repeat loop."""

    evaluate_batch_quality_max_retries: int = 1
    """Max retries for the per-batch quality evaluation loop.  ``0`` skips quality evaluation."""

    consolidate_mode: Literal["agent", "static"] = "agent"
    """Whether to use an LLM agent or a deterministic formatter for final result consolidation."""

    def __post_init__(self) -> None:
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
