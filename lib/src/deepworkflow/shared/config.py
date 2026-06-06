from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


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
