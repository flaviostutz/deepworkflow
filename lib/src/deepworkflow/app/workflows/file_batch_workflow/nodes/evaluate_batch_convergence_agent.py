from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from deepworkflow.adapters.connectors.deepagents_connector import bind_json_mode
from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_judge_output
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
You are the `evaluate_batch_convergence_agent`: a lightweight convergence checker.
You inspect the tool call history in the preceding messages (looking for file-write tool calls)
and compare the current pass output against `previous_execute_output` to decide whether
meaningful new work was done that suggests another pass could still improve results.
You do NOT evaluate overall quality — that is the `evaluate_batch_quality_agent`'s responsibility."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope:
{batch_files}
- previous_execute_output (summary of work done in all prior passes — use this as the baseline):
{previous_execute_output}

The full execution conversation and all tool call history are available in the messages preceding
this request. Inspect the tool calls to identify actual file writes, then compare what was written
against `previous_execute_output` to assess whether the current pass added genuinely new content."""

_GUARDRAILS = """\
- Do NOT evaluate overall quality — that is `evaluate_batch_quality_agent`'s responsibility.
- Inspect the tool call history in the preceding messages to find file-write calls. No file writes
  means no meaningful convergence signal.
- Compare what was produced in this pass against `previous_execute_output`:
  only mark WARNING (needs another pass) when the current pass wrote NEW content not already
  captured there AND another pass is plausibly going to produce further non-trivial improvements.
- Mark OK (converged) when any of the following apply:
  * No file-write tool calls were found in the history.
  * Files written contain only trivial changes (whitespace, formatting, no semantic difference).
  * The new content is low-signal — e.g. a review task with only INFO observations and no new
    WARNINGs or ERRORs.
  * All changes described are already covered by `previous_execute_output`.
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


def evaluate_batch_convergence_agent(state: file_batch_workflow_state) -> dict:
    """Evaluate whether the workflow has converged after the latest execution pass.

    On the first pass (no previous_execute_output), skip the LLM call entirely and return a
    synthetic WARNING verdict so the workflow always runs at least two passes before deciding
    whether to converge — a single pass has no baseline to compare against.
    """
    config = state["config"]
    previous_execute_output = state.get("previous_execute_output", "")

    if not previous_execute_output:
        return {"batch_convergence_output": "", "batch_convergence_verdict": _FIRST_PASS_VERDICT}

    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]
    execute_messages = state.get("execute_messages", [])

    system_prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
            previous_execute_output=previous_execute_output,
        ),
        guardrails=_GUARDRAILS,
        output_format=_OUTPUT_FORMAT,
    )

    model = bind_json_mode(config.model("evaluate_batch_convergence_agent"))
    messages = [SystemMessage(content=system_prompt), *execute_messages, HumanMessage(content=STANDARD_USER_MESSAGE)]
    response = model.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    judge_verdict = parse_judge_output(content)
    return {"batch_convergence_output": content, "batch_convergence_verdict": judge_verdict}
