from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import BatchOutput


def record_output_step(state: file_batch_workflow_state) -> dict:
    """Record the current batch output and advance to the next batch."""
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    # Accumulate the final pass into cumulative — same pattern as increment_batch_repeat_step.
    # cumulative_* holds all previous passes; files_* / execute_output hold the last pass only.
    files_read = list(state.get("cumulative_files_read", [])) + list(state.get("files_read", []))
    files_written = list(state.get("cumulative_files_written", [])) + list(state.get("files_written", []))

    cumulative_execute_output = state.get("cumulative_execute_output", "")
    current_execute_output = state.get("execute_output", "")
    if cumulative_execute_output and current_execute_output:
        merged_execute_output = cumulative_execute_output + "\n---\n" + current_execute_output
    elif current_execute_output:
        merged_execute_output = current_execute_output
    else:
        merged_execute_output = cumulative_execute_output

    output = BatchOutput(
        task_files=current_batch.batch_files,
        evaluate_quality_verdict=state["evaluate_quality_verdict"],
        evaluate_quality_feedbacks=state.get("evaluate_quality_feedbacks", []),
        files_read=files_read,
        files_written=files_written,
        execute_output=merged_execute_output,
    )

    batch_outputs = list(state.get("batch_outputs", []))
    batch_outputs.append(output)

    return {
        "batch_outputs": batch_outputs,
        "current_batch_index": batch_index + 1,
        "retry_count": 0,
        "evaluate_quality_feedbacks": [],
        "batch_repeat_count": 0,
        "cumulative_files_read": [],
        "cumulative_files_written": [],
        "cumulative_execute_output": "",
    }
