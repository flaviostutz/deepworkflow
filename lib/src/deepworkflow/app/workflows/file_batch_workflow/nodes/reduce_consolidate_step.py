from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def reduce_consolidate_step(state: file_batch_workflow_state) -> dict:
    """Consolidate all batch outputs into a final workflow output without using an LLM.

    Produces a structured markdown document with one section per batch, listing:
    - files processed
    - files read during execution
    - files written during execution
    - the execute_output from that batch
    """
    batch_outputs = state.get("batch_outputs") or []

    if not batch_outputs:
        return {"workflow_output": "(no batch outputs to consolidate)"}

    sections: list[str] = []
    for i, output in enumerate(batch_outputs):
        lines: list[str] = [f"## Batch {i + 1}"]

        if output.task_files:
            lines.append(f"**Files processed:** {', '.join(output.task_files)}")

        if output.files_read:
            lines.append(f"**Files read:** {', '.join(output.files_read)}")

        if output.files_written:
            lines.append(f"**Files written:** {', '.join(output.files_written)}")

        lines.append(f"**Quality verdict:** {output.evaluate_quality_verdict.name}")
        lines.append("")
        lines.append("**Output:**")
        lines.append(output.execute_output or "(no output)")

        sections.append("\n".join(lines))

    workflow_output = "# Consolidated Results\n\n" + "\n\n---\n\n".join(sections)
    return {"workflow_output": workflow_output}
