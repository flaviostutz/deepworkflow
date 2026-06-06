from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import BatchOutput


def record_output_step(state: file_batch_workflow_state) -> dict:
    """Record the current batch output and advance to the next batch."""
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    output = BatchOutput(
        task_files=current_batch.batch_files,
        judge_verdict=state["judge_verdict"],
        judge_feedbacks=state.get("judge_feedbacks", []),
        files_read=state.get("files_read", []),
        files_written=state.get("files_written", []),
        execute_output=state.get("execute_output", ""),
    )

    batch_outputs = list(state.get("batch_outputs", []))
    batch_outputs.append(output)

    return {
        "batch_outputs": batch_outputs,
        "current_batch_index": batch_index + 1,
        "retry_count": 0,
        "judge_feedbacks": [],
    }
