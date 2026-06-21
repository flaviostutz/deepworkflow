from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
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

_OBJECTIVE = """\
Analyze the task and list of files, then split the work into logical, non-overlapping batches
for parallel processing. Return a structured error response when task instructions are insufficient."""

_ROLE = """\
You are the `map_batches_agent`. You are an expert at partitioning large file sets into balanced,
directory-aware batches that minimise write conflicts and allow each batch to converge independently
to a consistent result."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files:
{files_section}
- batch_size_constraint: {batch_size_constraint}
{evaluate_quality_feedback_section}

Quality language convention (used throughout this workflow):
- MUST / REQUIRED / MANDATORY → failure to comply is an ERROR
- SHOULD / RECOMMENDED → failure to comply is a WARNING
- COULD / MAY / SUGGESTED → failure to comply is an INFO"""

_STEPS_TEMPLATE = """\
1. Validate task_instructions before proceeding:
   - Are they sufficient to define a batching strategy?
   - Do they define expected quality standards using MUST/SHOULD/COULD language? If quality criteria
     are present but lack this language, treat it as ambiguous and include that in the error message.
   - Is it clear how to consolidate the final result after all batches complete?
   If any of the above is missing or unclear, respond immediately with an error (see OUTPUT_FORMAT).
2. Scan the workspace to understand the codebase structure and context.
3. {discover_or_group}
4. Produce a `task_overview` — a high-level strategy and context that all batches should follow.
   Use MUST/SHOULD/COULD language to express quality requirements that apply to all batches.
5. Produce `consolidation_instructions` — how to merge/consolidate results from all batches at the end.
6. For each batch, produce `batch_instructions` — why those files are grouped together AND specific
   instructions the execute agent must follow when processing this batch. Use MUST/SHOULD/COULD
   language to express quality requirements specific to this batch."""

_GUARDRAILS = """\
- Every file selected for processing MUST appear in exactly one batch (no file lost, no file
  duplicated). No file may be omitted or assigned to more than one batch.
- All files in `batch_files` MUST be existing files in the workspace. Do not invent paths or
  reference files that do not exist.
- Each batch's instructions MUST be coherent with task_instructions and with the specific group
  of files in that batch.
- If the task involves creating or modifying output files, include the full list of expected output
  files in `task_overview` (shared across all batches) or in `batch_instructions` (batch-specific).
- `task_overview` MUST provide enough shared context, target outputs, or guidelines for all batches
  to run in parallel and still converge to a consistent result (e.g. a target list of files to be
  created, naming conventions, or design guidelines that prevent conflicts across batches).
- Minimise write concurrency: group files that are likely to be written together into the same
  batch to avoid multiple batches modifying the same files simultaneously.
- If batch size constraint is "all in one batch", put ALL files in a single batch.
- Do NOT perform the actual task — only plan the batching strategy."""

_OUTPUT_FORMAT = """\
On success:
{{
  "task_overview": "Overall strategy and context for all batches...",
  "consolidation_instructions": "How to consolidate/merge results from all batches...",
  "batches": [
    {{"batch_files": ["file1.py", "file2.py"], "batch_instructions": "Grouping rationale and specific instructions..."}},
    {{"batch_files": ["file3.py"], "batch_instructions": "Grouping rationale and specific instructions..."}}
  ]
}}

On failure (unclear or insufficient instructions):
{{
  "error": true,
  "message": "The task instructions are insufficient because... To improve, consider..."
}}"""

_MANDATORY_ADVISORY_KEYWORDS = {"must", "required", "mandatory", "should", "recommended", "could", "may", "suggested"}


def map_batches_agent(state: file_batch_workflow_state) -> dict:
    """Plan batch definitions using a read-only agent."""
    config = state["config"]
    effort_config = state["effort_config"]
    task_files = state["task_files"]

    if effort_config.evaluate_quality_batch_instructions is not None:
        words = set(effort_config.evaluate_quality_batch_instructions.lower().split())
        if not words & _MANDATORY_ADVISORY_KEYWORDS:
            return {
                "error": (
                    "evaluate_quality_batch_instructions must contain at least one MANDATORY/ADVISORY keyword "
                    "(MUST, REQUIRED, MANDATORY, SHOULD, RECOMMENDED, COULD, MAY, SUGGESTED) "
                    "to clearly express quality severity. "
                    "Update the instructions to use this language convention."
                )
            }

    constraints: list[str] = []
    if effort_config.max_files_per_batch:
        constraints.append(f"Maximum {effort_config.max_files_per_batch} files per batch")
    if effort_config.max_batches:
        constraints.append(f"Maximum {effort_config.max_batches} batches total")
    batch_size_constraint = "; ".join(constraints) if constraints else "No explicit size constraint"

    evaluate_quality_feedback_section = ""
    map_feedbacks = state.get("map_evaluate_quality_feedbacks")
    if map_feedbacks:
        feedback_lines = [
            f"- [{fb.type.name}] {fb.file}: {fb.description}" + (f" | Proposal: {fb.proposal}" if fb.proposal else "")
            for fb in map_feedbacks
        ]
        evaluate_quality_feedback_section = (
            "Previous batch plan was rejected by evaluate_quality. Address this feedback:\n" + "\n".join(feedback_lines)
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

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            files_section=files_section,
            batch_size_constraint=batch_size_constraint,
            evaluate_quality_feedback_section=evaluate_quality_feedback_section,
        ),
        steps=_STEPS_TEMPLATE.format(discover_or_group=discover_or_group),
        guardrails=_GUARDRAILS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("map_batches_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
        json_mode=True,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    parsed = _parse_map_output(content)
    if parsed.get("error"):
        return parsed

    parsed["evaluate_quality_batch_instructions"] = _derive_evaluate_quality_instructions(config, effort_config)
    return parsed


def _derive_evaluate_quality_instructions(config, effort_config) -> str:  # type: ignore[misc]
    """Return evaluate_quality_batch_instructions from effort_config or derive them via LLM."""
    if effort_config.evaluate_quality_batch_instructions is not None:
        return effort_config.evaluate_quality_batch_instructions

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
