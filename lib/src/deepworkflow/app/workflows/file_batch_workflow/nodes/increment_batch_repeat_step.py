from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def increment_batch_repeat_step(state: file_batch_workflow_state) -> dict:
    """Accumulate files from the current pass and prepare for the next repeat pass.

    Merges files_read and files_written into their cumulative accumulators,
    increments the repeat counter, and resets per-pass execution state so that
    plan_batch_agent starts a fully fresh session.
    """
    cumulative_files_read = list(state.get("cumulative_files_read", []))
    cumulative_files_written = list(state.get("cumulative_files_written", []))

    # Accumulate files from the completed pass (preserve order, allow duplicates
    # so the caller can decide to deduplicate if needed)
    cumulative_files_read.extend(state.get("files_read", []))
    cumulative_files_written.extend(state.get("files_written", []))

    return {
        "cumulative_files_read": cumulative_files_read,
        "cumulative_files_written": cumulative_files_written,
        "batch_repeat_count": state.get("batch_repeat_count", 0) + 1,
        # Reset per-pass state for the next fresh session
        "plan_output": "",
        "execute_output": "",
        "execute_messages": [],
        "files_read": [],
        "files_written": [],
    }
