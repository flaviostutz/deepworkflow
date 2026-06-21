from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_evaluate_verdict
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, build_agent_prompt
from deepworkflow.shared.types import JudgeFinding, JudgeLevel, JudgeVerdict

_FIRST_PASS_VERDICT = JudgeVerdict(
    verdict=JudgeLevel.WARNING,
    findings=[
        JudgeFinding(
            level=JudgeLevel.WARNING,
            title="First pass — baseline needed for convergence check",
            reason="No previous output exists to compare against; another pass is required.",
        )
    ],
)

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Decide whether the latest execution pass made meaningful NEW progress compared to prior passes,
and whether running another pass is likely to produce further improvements."""

_ROLE = """\
You are the `batch_evaluate_convergence_agent`: a lightweight convergence checker.
You compare the current pass outputs (`batch_execute_output`, `batch_files_written`, `batch_files_read`) against the
cumulative baseline (`batch_cumulative_output`, `batch_cumulative_files_written`, `batch_cumulative_files_read`)
to decide whether meaningful new work was done that suggests another pass could still improve results.
You do NOT evaluate overall quality — that is the `batch_evaluate_quality_agent`'s responsibility."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope:
{batch_files}

Current pass (the latest execution — assess whether this added genuinely NEW work):
- batch_execute_output:
{batch_execute_output}
- batch_files_written: {batch_files_written}
- batch_files_read: {batch_files_read}

Prior passes baseline (all previous passes accumulated — use this to detect what is truly new):
- batch_cumulative_output:
{batch_cumulative_output}
- batch_cumulative_files_written: {batch_cumulative_files_written}
- batch_cumulative_files_read: {batch_cumulative_files_read}

The full execution conversation is also available in the preceding messages for additional context."""

_GUARDRAILS = """\
- Do NOT evaluate overall quality — that is `batch_evaluate_quality_agent`'s responsibility.
- Use `batch_execute_output`, `batch_files_written`, and `batch_files_read` from the current pass as the primary
  signal of what was done. Compare against the cumulative baseline to assess what is truly NEW.
- Only mark WARNING (needs another pass) when the current pass produced NEW content not already
  captured in `batch_cumulative_output` / `batch_cumulative_files_written` AND another pass is
  plausibly going to produce further non-trivial improvements.
- Mark OK (converged) when any of the following apply:
  * `batch_execute_output` is empty and no files were written in the current pass.
  * Files written contain only trivial changes (whitespace, formatting, no semantic difference).
  * The new content is low-signal — e.g. a review task with only INFO observations and no new
    WARNINGs or ERRORs.
  * All changes described are already covered by `batch_cumulative_output`.
  * Another pass is unlikely to add anything new given what has already been done.
- Produce one finding per file that was written (or one general finding if nothing was written).
- `findings[].title` MUST be < 10 words and action-oriented.
- `findings[].reason` is required when level is WARNING; describe why convergence has not been reached.
- `findings[].fix` is optional; include only what a next pass could concretely improve.
- The aggregate `verdict` MUST be the worst (lowest) level across all findings.
- No tools are available for this agent. All reasoning is done in-context."""

_OUTPUT_FORMAT = """\
Respond with a JSON object in the following format:
{{
  "verdict": "OK|WARNING",
  "findings": [
    {{
      "level": "OK|WARNING",
      "title": "<10 words — what happened in this pass for this file>",
      "reason": "<30 words max — why convergence was or was not reached; required when level is WARNING>",
      "fix": "<what a next pass could concretely improve; omit when level is OK>"
    }}
  ]
}}

Verdict meaning:
- WARNING: New meaningful work was done — another pass may improve results further (not yet converged)
- OK:      No new progress detected — the workflow has converged"""


def batch_evaluate_convergence_agent(state: file_batch_workflow_state) -> dict:
    """Evaluate whether the workflow has converged after the latest execution pass.

    On the first pass (no batch_cumulative_output), skip the LLM call entirely and return a
    synthetic WARNING verdict so the workflow always runs at least two passes before deciding
    whether to converge — a single pass has no baseline to compare against.
    """
    config = state["config"]
    cumulative_execute_output = state.get("batch_cumulative_output", "")

    if not cumulative_execute_output:
        return {"batch_evaluate_convergence_output": "", "batch_evaluate_convergence_verdict": _FIRST_PASS_VERDICT}

    batch_index = state["batch_current_index"]
    current_batch = state["map_batches"][batch_index]
    execute_messages = state.get("batch_execute_messages", [])

    files_written = state.get("batch_files_written", []) or []
    files_read = state.get("batch_files_read", []) or []
    cumulative_files_written = state.get("batch_cumulative_files_written", []) or []
    cumulative_files_read = state.get("batch_cumulative_files_read", []) or []

    system_prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
            batch_execute_output=state.get("batch_execute_output", "") or "(none)",
            batch_files_written=", ".join(files_written) if files_written else "(none)",
            batch_files_read=", ".join(files_read) if files_read else "(none)",
            batch_cumulative_output=cumulative_execute_output or "(none)",
            batch_cumulative_files_written=", ".join(cumulative_files_written)
            if cumulative_files_written
            else "(none)",
            batch_cumulative_files_read=", ".join(cumulative_files_read) if cumulative_files_read else "(none)",
        ),
        guardrails=_GUARDRAILS,
        output_format=_OUTPUT_FORMAT,
    )

    model = config.model("batch_evaluate_convergence_agent")
    messages = [SystemMessage(content=system_prompt), *execute_messages, HumanMessage(content=STANDARD_USER_MESSAGE)]
    response = model.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    content = raw if isinstance(raw, str) else str(raw)

    judge_verdict = parse_evaluate_verdict(content)
    return {"batch_evaluate_convergence_output": content, "batch_evaluate_convergence_verdict": judge_verdict}
