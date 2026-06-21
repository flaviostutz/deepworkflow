from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def skip_reflect_batch_step(state: file_batch_workflow_state) -> dict:  # noqa: ARG001
    """Skip reflect_batch_agent when ``effort_config.skip_reflect=True`` (e.g. level 0).

    Returns empty file lists so ``record_output_step`` receives valid state without
    running an LLM call.
    """
    return {"files_read": [], "files_written": []}
