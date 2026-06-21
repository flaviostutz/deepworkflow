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
    - the batch_execute_output from that batch
    """
    batch_results = state.get("batch_results") or []

    if not batch_results:
        return {"reduce_output": "(no batch outputs to consolidate)"}

    sections: list[str] = []
    for i, output in enumerate(batch_results):
        lines: list[str] = [f"## Batch {i + 1}"]

        if output.batch_files:
            lines.append(f"**Files processed:** {', '.join(output.batch_files)}")

        if output.batch_files_read:
            lines.append(f"**Files read:** {', '.join(output.batch_files_read)}")

        if output.batch_files_written:
            lines.append(f"**Files written:** {', '.join(output.batch_files_written)}")

        lines.append(f"**Quality verdict:** {output.evaluate_level.name}")
        lines.append("")
        lines.append("**Output:**")
        lines.append(output.batch_execute_output or "(no output)")

        sections.append("\n".join(lines))

    reduce_output = "# Consolidated Results\n\n" + "\n\n---\n\n".join(sections)
    return {"reduce_output": reduce_output}
