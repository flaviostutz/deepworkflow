from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import BatchOutput


def batch_output_record_step(state: file_batch_workflow_state) -> dict:
    """Record the current batch output and advance to the next batch."""
    batch_index = state["batch_current_index"]
    current_batch = state["map_batches"][batch_index]

    # Accumulate the final pass into cumulative — same pattern as batch_convergence_repeat_step.
    # batch_cumulative_* holds all previous passes; batch_files_* / batch_execute_output hold the last pass only.
    files_read = list(state.get("batch_cumulative_files_read", [])) + list(state.get("batch_files_read", []))
    files_written = list(state.get("batch_cumulative_files_written", [])) + list(state.get("batch_files_written", []))

    cumulative_execute_output = state.get("batch_cumulative_output", "")
    current_execute_output = state.get("batch_execute_output", "")
    if cumulative_execute_output and current_execute_output:
        merged_execute_output = cumulative_execute_output + "\n---\n" + current_execute_output
    elif current_execute_output:
        merged_execute_output = current_execute_output
    else:
        merged_execute_output = cumulative_execute_output

    output = BatchOutput(
        batch_files=current_batch.batch_files,
        evaluate_level=state["batch_evaluate_level"],
        evaluate_feedbacks=state.get("batch_evaluate_feedbacks", []),
        batch_files_read=files_read,
        batch_files_written=files_written,
        batch_execute_output=merged_execute_output,
    )

    batch_results = list(state.get("batch_results", []))
    batch_results.append(output)

    return {
        "batch_results": batch_results,
        "batch_current_index": batch_index + 1,
        "batch_quality_retry_count": 0,
        "batch_evaluate_feedbacks": [],
        "batch_convergence_repeat_count": 0,
        "batch_cumulative_files_read": [],
        "batch_cumulative_files_written": [],
        "batch_cumulative_output": "",
    }
