from __future__ import annotations

import json
from typing import TYPE_CHECKING

from deepworkflow.shared.agent import create_workflow_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import BatchDefinition, WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.deepworkflow.states import WorkflowState

MAP_BATCHES_PROMPT = """{workflow_context}

You are a batch planning agent. Your job is to analyze the task and files, then split the work into \
logical batches that will be processed separately.

Task instructions:
{task_instructions}

Files to process ({file_count} total):
{task_files}

Batch size constraint: {batch_size_constraint}

{judge_feedback_section}

Your mission:
1. Scan the workspace to understand the codebase structure and context.
2. Group the files into logical batches (respecting the batch size constraint).
3. Produce a task_overview — a high-level strategy/context that all batches should follow.
4. Produce consolidation_instructions — how to merge/consolidate results from all batches at the end.
5. For each batch, produce batch_instructions — why those files are grouped together \
AND specific instructions that the execute agent must follow when processing this batch.

IMPORTANT:
- Every file from the input list MUST appear in exactly one batch (no file lost, no file duplicated).
- If batch size constraint is "all in one batch", put ALL files in a single batch.
- Do NOT perform the actual task — only plan the batching strategy.
- If the task_instructions are too confusing, ambiguous, or insufficient for you to create meaningful \
batches or derive consolidation instructions, respond with an error.

Respond in JSON format:

On success:
{{
  "task_overview": "Overall strategy and context for all batches...",
  "consolidation_instructions": "How to consolidate/merge results from all batches...",
  "batches": [
    {{"batch_files": ["file1.py", "file2.py"], "batch_instructions": "Grouping rationale and specific execution guidance..."}},
    {{"batch_files": ["file3.py"], "batch_instructions": "Grouping rationale and specific execution guidance..."}}
  ]
}}

On failure (unclear instructions):
{{
  "error": true,
  "message": "The task instructions are insufficient because... To improve, consider..."
}}"""


def map_batches(state: WorkflowState) -> dict:
    """Plan batch definitions using a read-only agent."""
    config = state["config"]
    task_files = state["task_files"]

    batch_size_constraint = (
        f"Maximum {config.task_files_batch_size} files per batch"
        if config.task_files_batch_size
        else "All files in one batch"
    )

    judge_feedback_section = ""
    map_feedbacks = state.get("map_judge_feedbacks")
    if map_feedbacks:
        feedback_lines = [
            f"- [{fb.type.name}] {fb.file}: {fb.description}" + (f" | Proposal: {fb.proposal}" if fb.proposal else "")
            for fb in map_feedbacks
        ]
        judge_feedback_section = (
            "Previous batch plan was rejected by the judge. Address this feedback:\n" + "\n".join(feedback_lines)
        )

    prompt = MAP_BATCHES_PROMPT.format(
        workflow_context=workflow_role("map_batches_agent", "Plan how to split files into batches for processing"),
        task_instructions=config.task_instructions,
        task_files="\n".join(task_files),
        file_count=len(task_files),
        batch_size_constraint=batch_size_constraint,
        judge_feedback_section=judge_feedback_section,
    )

    agent = create_workflow_agent(
        model=config.model,
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Analyze the workspace and plan the batch strategy."})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    return _parse_map_output(content)


def _parse_map_output(content: str) -> dict:
    """Parse the map agent's JSON output."""
    # Extract JSON from potential markdown code fences
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last lines (fences)
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"map_batches_agent produced invalid JSON output: {content[:200]}"}

    # Check for error response
    if data.get("error"):
        return {"error": data.get("message", "map_batches_agent could not plan batches.")}

    # Parse success response
    task_overview = data.get("task_overview", "")
    consolidation_instructions = data.get("consolidation_instructions", "")
    raw_batches = data.get("batches", [])

    if not raw_batches:
        return {"error": "map_batches_agent produced zero batches."}

    batches = [
        BatchDefinition(
            batch_files=batch["batch_files"],
            batch_instructions=batch.get("batch_instructions"),
        )
        for batch in raw_batches
    ]

    return {
        "task_overview": task_overview,
        "consolidation_instructions": consolidation_instructions,
        "task_file_batches": batches,
        "current_batch_index": 0,
        "retry_count": 0,
        "batch_outputs": [],
    }
