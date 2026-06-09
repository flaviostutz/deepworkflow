from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import BatchDefinition, WriteOption

DEFAULT_JUDGE_BATCH_INSTRUCTIONS = (
    "- All features and requirements described in task_instructions MUST be present in the output and written files.\n"
    "- The output and written files MUST contain no factual errors or contradictions relative to the task_instructions.\n"
    "- Every file listed in the batch MUST have been addressed; no file may be silently skipped.\n"
    "- Output format or schema specified in task_instructions MUST be followed strictly.\n"
    "- The content SHOULD be complete with no missing sections, fields, or logic that a careful reader would expect.\n"
    "- Naming conventions, style guidelines, and structural patterns mentioned in task_instructions SHOULD be respected.\n"
    "- Output COULD include additional context or clarifications that improve readability, but only when they do not conflict with the instructions."
)

_EXTRACT_JUDGE_INSTRUCTIONS_PROMPT = """Extract quality evaluation criteria from the task instructions below.
Express each criterion using MUST/SHOULD/COULD severity language:
- MUST / REQUIRED / MANDATORY → for critical requirements that must be met
- SHOULD / RECOMMENDED → for important but non-critical requirements
- COULD / MAY / SUGGESTED → for nice-to-have improvements

Task instructions:
{task_instructions}

Return ONLY a bullet-point list (one criterion per line, starting with "- ").
Include only criteria specific to these task instructions — no generic rules.
If no specific quality criteria can be extracted, return an empty string."""

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

MAP_BATCHES_PROMPT = """{workflow_context}

You are a batch planning agent. Your job is to analyze the task and files, then split the work into \
logical batches that will be processed separately.

Task instructions:
{task_instructions}

{files_section}

NOTE: Every file listed above is an **existing file in the workspace**. Do not invent or assume any file that is not listed (or discovered via filesystem exploration). Only reference real, existing paths.

Batch size constraint: {batch_size_constraint}

{judge_feedback_section}

Quality language convention (used throughout this workflow):
- MUST / REQUIRED / MANDATORY → failure to comply is an ERROR
- SHOULD / RECOMMENDED → failure to comply is a WARNING
- COULD / MAY / SUGGESTED → failure to comply is an INFO

Your mission:
1. Validate the task_instructions before proceeding:
   - Are they sufficient to define a batching strategy?
   - Do they define expected quality standards using the MANDATORY/ADVISORY language above (MUST/SHOULD/COULD)? If quality criteria are present but lack this language, treat it as ambiguous and include that in the error message.
   - Is it clear how to consolidate the final result after all batches complete?
   If any of the above is missing or unclear, respond immediately with an error.
2. Scan the workspace to understand the codebase structure and context.
3. {discover_or_group}
4. Produce a task_overview — a high-level strategy/context that all batches should follow. Use MUST/SHOULD/COULD language to express quality requirements that apply to all batches.
5. Produce consolidation_instructions — how to merge/consolidate results from all batches at the end.
6. For each batch, produce batch_instructions — why those files are grouped together \
AND specific instructions that the execute agent must follow when processing this batch. Use MUST/SHOULD/COULD language to express quality requirements specific to this batch.

IMPORTANT:
- Every file selected for processing MUST appear in exactly one batch (no file lost, no file duplicated). No file may be omitted or assigned to more than one batch.
- All files in `batch_files` MUST be existing files in the workspace. Do not invent paths or reference files that do not exist.
- Each batch's instructions MUST be coherent with the task_instructions and with the specific group of files in that batch.
- If the task involves creating or modifying **output files** (files to be produced or changed as a result of the task), include the full list of expected output files either in `task_overview` (when shared across all batches) or in the relevant `batch_instructions` (when specific to a batch). This enables all batches to know upfront which output files they are responsible for and avoids conflicts.
- The task_overview MUST provide enough shared context, target outputs, or guidelines for all batches to run in parallel and still converge to a consistent result — for example: a target list of files to be created, naming conventions, or design guidelines that prevent conflicts across batches.
- Minimize write concurrency: group files that are likely to be written together into the same batch to avoid multiple batches modifying the same files simultaneously.
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
    {{"batch_files": ["file1.py", "file2.py"], "batch_instructions": "Grouping rationale..."}},
    {{"batch_files": ["file3.py"], "batch_instructions": "Grouping rationale..."}}
  ]
}}

On failure (unclear instructions):
{{
  "error": true,
  "message": "The task instructions are insufficient because... To improve, consider..."
}}"""


_MANDATORY_ADVISORY_KEYWORDS = {"must", "required", "mandatory", "should", "recommended", "could", "may", "suggested"}


def map_batches_agent(state: file_batch_workflow_state) -> dict:
    """Plan batch definitions using a read-only agent."""
    config = state["config"]
    task_files = state["task_files"]

    if config.judge_batch_instructions is not None:
        words = set(config.judge_batch_instructions.lower().split())
        if not words & _MANDATORY_ADVISORY_KEYWORDS:
            return {
                "error": (
                    "judge_batch_instructions must contain at least one MANDATORY/ADVISORY keyword "
                    "(MUST, REQUIRED, MANDATORY, SHOULD, RECOMMENDED, COULD, MAY, SUGGESTED) "
                    "to clearly express quality severity. "
                    "Update the instructions to use this language convention."
                )
            }

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
        judge_feedback_section = "Previous batch plan was rejected by the judge. Address this feedback:\n" + "\n".join(
            feedback_lines
        )

    if task_files:
        files_section = f"Files to process ({len(task_files)} total):\n" + "\n".join(task_files)
        discover_or_group = "Group the provided files into logical batches (respecting the batch size constraint)."
    else:
        files_section = (
            "No files have been pre-selected. You MUST use your filesystem tools to explore the workspace "
            "and identify all files that are relevant to the task instructions. "
            "Only include files that are meaningful to the task — do not add unrelated files."
        )
        if config.task_files_exclude:
            files_section += (
                "\n\nFiles/patterns to ALWAYS EXCLUDE — do NOT include any matching file in any batch:\n"
                + "\n".join(config.task_files_exclude)
            )
        discover_or_group = (
            "Explore the workspace using your filesystem tools, discover the relevant files, "
            "then group them into logical batches (respecting the batch size constraint)."
        )

    prompt = MAP_BATCHES_PROMPT.format(
        workflow_context=workflow_role("map_batches_agent", "Plan how to split files into batches for processing"),
        task_instructions=config.task_instructions,
        files_section=files_section,
        discover_or_group=discover_or_group,
        batch_size_constraint=batch_size_constraint,
        judge_feedback_section=judge_feedback_section,
    )

    agent = create_agent(
        model=config.model("map_batches_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Analyze the workspace and plan the batch strategy."})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    parsed = _parse_map_output(content)
    if parsed.get("error"):
        return parsed

    parsed["judge_batch_instructions"] = _derive_judge_instructions(config)
    return parsed


def _derive_judge_instructions(config) -> str:  # type: ignore[misc]
    """Return judge_batch_instructions from config or derive them via LLM."""
    if config.judge_batch_instructions is not None:
        return config.judge_batch_instructions

    model = config.model("map_batches_agent")
    messages = [
        SystemMessage(content="You extract quality evaluation criteria from instructions."),
        HumanMessage(content=_EXTRACT_JUDGE_INSTRUCTIONS_PROMPT.format(task_instructions=config.task_instructions)),
    ]
    response = model.invoke(messages)
    extracted = response.content.strip() if hasattr(response, "content") else str(response).strip()

    if extracted:
        return DEFAULT_JUDGE_BATCH_INSTRUCTIONS + "\n" + extracted
    return DEFAULT_JUDGE_BATCH_INSTRUCTIONS


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
