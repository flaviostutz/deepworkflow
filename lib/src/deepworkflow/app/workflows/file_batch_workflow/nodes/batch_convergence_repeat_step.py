from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def batch_convergence_repeat_step(state: file_batch_workflow_state) -> dict:
    """Accumulate files from the current pass and prepare for the next repeat pass.

    Merges batch_files_read and batch_files_written into their cumulative accumulators,
    accumulates batch_execute_output into batch_cumulative_output,
    increments the repeat counter, and resets per-pass execution state so that
    batch_plan_agent starts a fully fresh session.
    """
    cumulative_files_read = list(state.get("batch_cumulative_files_read", []))
    cumulative_files_written = list(state.get("batch_cumulative_files_written", []))

    # Accumulate files from the completed pass, deduplicating while preserving order
    cumulative_files_read.extend(state.get("batch_files_read", []))
    cumulative_files_written.extend(state.get("batch_files_written", []))
    cumulative_files_read = list(dict.fromkeys(cumulative_files_read))
    cumulative_files_written = list(dict.fromkeys(cumulative_files_written))

    # Accumulate batch_execute_output so the next plan pass can see what was already done
    current_execute_output = state.get("batch_execute_output", "")
    existing_previous = state.get("batch_cumulative_output", "")
    if existing_previous and current_execute_output:
        cumulative_execute_output = existing_previous + "\n---\n" + current_execute_output
    elif current_execute_output:
        cumulative_execute_output = current_execute_output
    else:
        cumulative_execute_output = existing_previous

    return {
        "batch_cumulative_files_read": cumulative_files_read,
        "batch_cumulative_files_written": cumulative_files_written,
        "batch_cumulative_output": cumulative_execute_output,
        "batch_convergence_repeat_count": state.get("batch_convergence_repeat_count", 0) + 1,
        # Reset per-pass state for the next fresh session
        "batch_plan": "",
        "batch_execute_output": "",
        "batch_execute_messages": [],
        "batch_files_read": [],
        "batch_files_written": [],
    }
