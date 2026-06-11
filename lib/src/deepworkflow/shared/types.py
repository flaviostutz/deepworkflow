from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class JudgeVerdict(IntEnum):
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
    """Behavior when judge retries are exhausted."""

    FAIL = "fail"
    CONTINUE = "continue"


class WorkflowLogLevel(StrEnum):
    """Console log verbosity for the workflow run."""

    NONE = "none"
    """No console output."""

    TRACE = "trace"
    """Print every MLflow span as JSON (raw tracing)."""

    INFO = "info"
    """Print agent/route header lines, in/out summaries, elapsed time, and a summary block."""


@dataclass(frozen=True)
class JudgeFeedback:
    """Single feedback item from the judge agent."""

    file: str
    type: JudgeVerdict
    description: str
    proposal: str = ""


@dataclass(frozen=True)
class BatchDefinition:
    """Defines a single batch produced by the map_batches_agent."""

    batch_files: list[str]
    batch_instructions: str | None = None


@dataclass(frozen=True)
class BatchOutput:
    """Output produced per batch execution."""

    task_files: list[str]
    judge_verdict: JudgeVerdict
    judge_feedbacks: list[JudgeFeedback]
    files_read: list[str]
    files_written: list[str]
    execute_output: str


@dataclass(frozen=True)
class WorkflowResult:
    """Result returned by run_workflow."""

    thread_id: str
    output: str
    status: str
