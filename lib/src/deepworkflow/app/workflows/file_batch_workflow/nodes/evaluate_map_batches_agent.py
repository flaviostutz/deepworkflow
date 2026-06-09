from __future__ import annotations

import fnmatch
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_judge_output
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import BatchDefinition, JudgeFeedback, JudgeVerdict, WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_LINE_RANGE_RE = re.compile(r":\d+-\d+$")


def _base_path(file_ref: str) -> str:
    """Strip optional :start-end line-range suffix from a file reference."""
    return _LINE_RANGE_RE.sub("", file_ref)


def _algorithmic_map_checks(  # noqa: C901, PLR0912
    workspace_dir: str,
    task_files: list[str],
    batches: list[BatchDefinition],
    task_files_exclude: list[str] | None = None,
) -> list[JudgeFeedback]:
    """Deterministic checks that do not rely on an LLM.

    1. Every file in every batch's ``batch_files`` must exist on disk.
    2. When ``task_files`` was provided (non-empty), every task file must appear
       in exactly one batch — no file may be missing or assigned to more than one
       batch, and no invented file may appear in a batch.
    3. No batch file may match any pattern in ``task_files_exclude``.
    """
    feedbacks: list[JudgeFeedback] = []
    workspace = Path(workspace_dir)

    all_batch_files: list[str] = [f for b in batches for f in b.batch_files]

    # ── Check 1: all batch_files must exist in the workspace ──────────────────
    for file_ref in all_batch_files:
        base = _base_path(file_ref)
        path = Path(base)
        if not path.is_absolute():
            path = workspace / base
        if not path.exists():
            feedbacks.append(
                JudgeFeedback(
                    file=file_ref,
                    type=JudgeVerdict.ERROR,
                    description=f"File '{file_ref}' does not exist in the workspace.",
                    proposal=f"Remove '{file_ref}' from batch_files or correct the path.",
                )
            )

    # ── Check 2: task_files_exclude — no excluded file may appear in any batch ──
    if task_files_exclude:
        for file_ref in all_batch_files:
            base = _base_path(file_ref)
            # Build candidate paths for matching: absolute and relative forms
            abs_path = base if Path(base).is_absolute() else str(workspace / base)
            for pattern in task_files_exclude:
                if fnmatch.fnmatch(abs_path, pattern) or fnmatch.fnmatch(base, pattern):
                    feedbacks.append(
                        JudgeFeedback(
                            file=file_ref,
                            type=JudgeVerdict.ERROR,
                            description=(
                                f"File '{file_ref}' matches exclude pattern '{pattern}' "
                                "and must not be included in any batch."
                            ),
                            proposal=f"Remove '{file_ref}' from batch_files.",
                        )
                    )
                    break

    # ── Check 3: task_files coverage & disjointness ───────────────────────────
    if task_files:
        task_bases = [_base_path(f) for f in task_files]
        batch_bases = [_base_path(f) for f in all_batch_files]

        task_set = set(task_bases)
        batch_counter: Counter[str] = Counter(batch_bases)

        # Missing: task file not assigned to any batch
        for tf in task_bases:
            if batch_counter[tf] == 0:
                feedbacks.append(
                    JudgeFeedback(
                        file=tf,
                        type=JudgeVerdict.ERROR,
                        description=f"File '{tf}' from task_files is not assigned to any batch.",
                        proposal=f"Add '{tf}' to exactly one batch.",
                    )
                )

        # Duplicated: task file assigned to more than one batch
        for bf, count in batch_counter.items():
            if count > 1 and bf in task_set:
                feedbacks.append(
                    JudgeFeedback(
                        file=bf,
                        type=JudgeVerdict.ERROR,
                        description=f"File '{bf}' appears in {count} batches (must be in exactly one).",
                        proposal=f"Remove '{bf}' from all but one batch.",
                    )
                )

        # Invented: batch file not in the original task_files list
        for bf in batch_counter:
            if bf not in task_set:
                feedbacks.append(
                    JudgeFeedback(
                        file=bf,
                        type=JudgeVerdict.ERROR,
                        description=(
                            f"File '{bf}' is assigned to a batch but was not in the original task_files list."
                        ),
                        proposal=f"Remove '{bf}' from batch_files; only files from task_files may be used.",
                    )
                )

    return feedbacks


EVALUATE_MAP_PROMPT = """<OBJECTIVE>
Judge the quality of the batch planning step and return structured feedback with a verdict.
</OBJECTIVE>

<ROLE>
You are the `evaluate_map_batches_agent` (see WORKFLOW_CONTEXT). You are an expert judge evaluating \
whether a batch plan is coherent, well-scoped, and ready for parallel execution.
</ROLE>

<INPUT>
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- original_files_to_process ({file_count} total — all are existing files in the workspace):
{task_files}
- batch_size_constraint: {batch_size_constraint}
- proposed_batch_plan:
  - task_overview: {task_overview}
  - consolidation_instructions: {consolidation_instructions}
  - batches ({batch_count}):
{batches_summary}

Note: file existence, task_files coverage, and batch disjointness are verified algorithmically
before this runs — do NOT re-check those.

Quality language convention: when evaluating criteria derived from task_instructions or batch
instructions, map the severity based on the language used:
- MUST / REQUIRED / MANDATORY → non-compliance is an ERROR
- SHOULD / RECOMMENDED → non-compliance is a WARNING
- COULD / MAY / SUGGESTED → non-compliance is an INFO
</INPUT>

<STEPS>
Evaluate the batch plan against these criteria:
1. **Batch size**: Each batch respects the batch size constraint.
2. **Logical grouping**: Files are grouped in a way that makes sense for the task.
3. **Instructions quality**: task_overview, consolidation_instructions, and batch_instructions are
   clear and actionable.
4. **Batch coherence**: Each batch's instructions are coherent with the task_instructions and with
   the specific group of files assigned to that batch.
5. **Parallel convergence**: The task_overview provides enough shared strategy, target outputs, or
   guidelines for all batches to run independently and still converge to a consistent,
   non-conflicting result.
6. **Output files declared**: If the task involves creating or modifying output files, the expected
   output file list is declared either in task_overview (for shared outputs) or in the relevant
   batch_instructions (for batch-specific outputs).
7. **Write concurrency**: The batch grouping minimises the risk of multiple batches writing to the
   same files concurrently.
</STEPS>

<OUTPUT_FORMAT>
{{
  "judge_feedbacks": [
    {{
      "file": "general",
      "type": "OK|INFO|WARNING|ERROR",
      "description": "explanation",
      "proposal": "how to fix the issue (required for WARNING or ERROR; empty for OK or INFO)"
    }}
  ],
  "judge_verdict": "OK|INFO|WARNING|ERROR"
}}

The verdict MUST be the worst (lowest) type across all feedbacks.
</OUTPUT_FORMAT>

<WORKFLOW_CONTEXT>
{workflow_context}
</WORKFLOW_CONTEXT>"""


def evaluate_map_batches_agent(state: file_batch_workflow_state) -> dict:
    """Judge the map_batches_agent output.

    Runs deterministic algorithmic checks first (file existence, task_files coverage,
    batch disjointness), then calls the LLM judge for qualitative evaluation.
    Final verdict is the worst across both.
    """
    config = state["config"]
    task_files = state["task_files"]
    batches = state["task_file_batches"]
    task_overview = state.get("task_overview", "")
    consolidation_instructions = state.get("consolidation_instructions", "")

    # ── Deterministic checks ──────────────────────────────────────────────────
    algorithmic_feedbacks = _algorithmic_map_checks(
        config.workspace_dir, task_files, batches, config.task_files_exclude
    )

    # ── LLM qualitative judge ─────────────────────────────────────────────────
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
        workflow_context=workflow_role("evaluate_map_batches_agent", "Judge the quality of the batch planning step"),
        task_instructions=config.task_instructions,
        task_files="\n".join(task_files),
        file_count=len(task_files),
        batch_size_constraint=batch_size_constraint,
        task_overview=task_overview,
        consolidation_instructions=consolidation_instructions,
        batch_count=len(batches),
        batches_summary=batches_summary,
    )

    agent = create_agent(
        model=config.model("evaluate_map_batches_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Evaluate the batch plan."})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    llm_verdict, llm_feedbacks = parse_judge_output(content)
    del llm_verdict  # verdict is recalculated from merged feedbacks below

    # ── Merge results ─────────────────────────────────────────────────────────
    all_feedbacks = algorithmic_feedbacks + llm_feedbacks
    all_verdicts = [f.type for f in all_feedbacks]
    final_verdict = min(all_verdicts) if all_verdicts else JudgeVerdict.OK

    return {"map_judge_verdict": final_verdict, "map_judge_feedbacks": all_feedbacks}
