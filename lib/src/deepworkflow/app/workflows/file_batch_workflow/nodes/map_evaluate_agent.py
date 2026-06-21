from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_evaluate_verdict
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
from deepworkflow.shared.types import JudgeFinding, JudgeLevel, JudgeVerdict, WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


_EVAL_MAP_OBJECTIVE = """\
Judge the quality of the batch planning step and return structured findings with a verdict."""

_EVAL_MAP_ROLE = """\
You are the `map_evaluate_agent`. You are an expert evaluator judging whether a batch plan
is coherent, well-scoped, and ready for parallel execution."""

_EVAL_MAP_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- original_files_to_process ({file_count} total — all are existing files in the workspace):
{map_files}
- batch_size_constraint: {batch_size_constraint}
- proposed_batch_plan:
  - map_plan_overview: {map_plan_overview}
  - reduce_instructions: {reduce_instructions}
  - batches ({batch_count}):
{batches_summary}

Note: file existence, task_files coverage, and batch disjointness are verified algorithmically
before this runs — do NOT re-check those.

Quality language convention: when evaluating criteria derived from task_instructions or batch
instructions, map the severity based on the language used:
- MUST / REQUIRED / MANDATORY → non-compliance is an ERROR
- SHOULD / RECOMMENDED → non-compliance is a WARNING
- COULD / MAY / SUGGESTED → non-compliance is an INFO"""

_EVAL_MAP_STEPS = """\
Evaluate the batch plan against these criteria:
1. **Batch size**: Each batch respects the batch size constraint.
2. **Logical grouping**: Files are grouped in a way that makes sense for the task.
3. **Instructions quality**: map_plan_overview, reduce_instructions, and batch_instructions are
   clear and actionable.
4. **Batch coherence**: Each batch's instructions are coherent with the task_instructions and with
   the specific group of files assigned to that batch.
5. **Parallel convergence**: The map_plan_overview provides enough shared strategy, target outputs, or
   guidelines for all batches to run independently and still converge to a consistent,
   non-conflicting result.
6. **Output files declared**: If the task involves creating or modifying output files, the expected
   output file list is declared either in map_plan_overview (for shared outputs) or in the relevant
   batch_instructions (for batch-specific outputs).
7. **Write concurrency**: The batch grouping minimises the risk of multiple batches writing to the
   same files concurrently."""

_EVAL_MAP_OUTPUT_FORMAT = """\
{{
  "verdict": "OK|INFO|WARNING|ERROR",
  "findings": [
    {{
      "level": "OK|INFO|WARNING|ERROR",
      "title": "<10 words — action-oriented label for this finding>",
      "reason": "<30 words max — why this level was assigned; required when level is not OK>",
      "details": "<up to 400 words — notes and findings; required when level is not OK>",
      "fix": "<up to 200 words — concrete fix steps; include only when directly inferrable>"
    }}
  ]
}}

The verdict MUST be the worst (lowest) level across all findings."""


def map_evaluate_agent(state: file_batch_workflow_state) -> dict:
    """Evaluate the map_plan_agent output using an LLM qualitative evaluator.

    Deterministic checks (file existence, coverage, limits) are handled by
    ``map_plan_validate_step`` which always runs before this node.
    """
    config = state["config"]
    effort_config = state["effort_config"]
    map_files = state["map_files"]
    batches = state["map_batches"]
    map_plan_overview = state.get("map_plan_overview", "")
    reduce_instructions = state.get("reduce_instructions", "")

    # ── LLM qualitative evaluator ─────────────────────────────────────────────
    constraints: list[str] = []
    if effort_config.max_files_per_batch:
        constraints.append(f"Maximum {effort_config.max_files_per_batch} files per batch")
    if effort_config.max_batches:
        constraints.append(f"Maximum {effort_config.max_batches} batches total")
    batch_size_constraint = "; ".join(constraints) if constraints else "No explicit size constraint"

    batches_summary = "\n".join(
        f"  Batch {i + 1}: {len(b.batch_files)} files — {b.batch_instructions or '(no instructions)'}"
        for i, b in enumerate(batches)
    )

    prompt = build_agent_prompt(
        objective=_EVAL_MAP_OBJECTIVE,
        role=_EVAL_MAP_ROLE,
        input_section=_EVAL_MAP_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            map_files="\n".join(map_files),
            file_count=len(map_files),
            batch_size_constraint=batch_size_constraint,
            map_plan_overview=map_plan_overview,
            reduce_instructions=reduce_instructions,
            batch_count=len(batches),
            batches_summary=batches_summary,
        ),
        steps=_EVAL_MAP_STEPS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_EVAL_MAP_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("map_evaluate_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    llm_judge = parse_evaluate_verdict(content)

    final_verdict = llm_judge.verdict
    findings = llm_judge.findings
    if not findings:
        findings = [JudgeFinding(level=JudgeLevel.OK, title="Batch plan looks good")]
        final_verdict = JudgeLevel.OK

    merged = JudgeVerdict(verdict=final_verdict, findings=findings)
    return {
        "map_evaluate_level": final_verdict,
        "map_evaluate_verdict": merged,
    }
