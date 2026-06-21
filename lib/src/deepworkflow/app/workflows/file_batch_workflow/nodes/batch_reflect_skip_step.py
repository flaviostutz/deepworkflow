from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def batch_reflect_skip_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Skip batch_reflect_agent when ``effort_config.batch_skip_reflect=True`` (e.g. level 0).

    Returns empty file lists so ``batch_output_record_step`` receives valid state without
    running an LLM call.
    """
    return {"batch_files_read": [], "batch_files_written": []}
