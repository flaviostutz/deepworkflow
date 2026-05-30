from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.app.workflows.deepworkflow.nodes import parse_judge_output
from deepworkflow.shared.agent import create_workflow_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.deepworkflow.states import WorkflowState

EVALUATE_MAP_PROMPT = """{workflow_context}

You are a judge evaluating the quality of a batch planning step.

Task instructions: {task_instructions}

Original files to process ({file_count} total):
{task_files}

Batch size constraint: {batch_size_constraint}

Proposed batch plan:
- Task overview: {task_overview}
- Consolidation instructions: {consolidation_instructions}
- Batches ({batch_count}):
{batches_summary}

Evaluate the batch plan against these criteria:
1. **Completeness**: Every file from the original list appears in exactly one batch (no lost files, no invented files).
2. **Batch size**: Each batch respects the batch size constraint.
3. **Disjointness**: No file appears in more than one batch.
4. **Logical grouping**: Files are grouped in a way that makes sense for the task.
5. **Instructions quality**: task_overview, consolidation_instructions, and batch_instructions are clear and actionable.

Respond in JSON format:
{{
  "judge_feedbacks": [
    {{
      "file": "general",
      "type": "OK|INFO|WARNING|ERROR",
      "description": "explanation",
      "proposal": "how to fix the issue"
    }}
  ],
  "judge_verdict": "OK|INFO|WARNING|ERROR"
}}

For each feedback item with type WARNING or ERROR, include a concrete
"proposal" describing how to fix the issue. For OK or INFO items,
proposal can be empty.

The verdict should be the WORST (lowest) type across all feedbacks."""


def evaluate_map(state: WorkflowState) -> dict:
    """Judge the map_batches_agent output."""
    config = state["config"]
    task_files = state["task_files"]
    batches = state["task_file_batches"]
    task_overview = state.get("task_overview", "")
    consolidation_instructions = state.get("consolidation_instructions", "")

    batch_size_constraint = (
        f"Maximum {config.task_files_batch_size} files per batch"
        if config.task_files_batch_size
        else "All files in one batch"
    )

    batches_summary = "\n".join(
        f"  Batch {i + 1}: {len(b.batch_files)} files — {b.batch_instructions or '(no instructions)'}"
        for i, b in enumerate(batches)
    )

    prompt = EVALUATE_MAP_PROMPT.format(
        workflow_context=workflow_role("evaluate_map_agent", "Judge the quality of the batch planning step"),
        task_instructions=config.task_instructions,
        task_files="\n".join(task_files),
        file_count=len(task_files),
        batch_size_constraint=batch_size_constraint,
        task_overview=task_overview,
        consolidation_instructions=consolidation_instructions,
        batch_count=len(batches),
        batches_summary=batches_summary,
    )

    agent = create_workflow_agent(
        model=config.model,
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Evaluate the batch plan."})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    verdict, feedbacks = parse_judge_output(content)

    return {"map_judge_verdict": verdict, "map_judge_feedbacks": feedbacks}
