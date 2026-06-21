from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_judge_output
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Perform the final quality check on the overall batch execution results and return structured
findings with a verdict."""

_ROLE = """\
You are the `evaluate_batch_quality_agent`. You run once after all execution passes for a batch
complete and decide whether the result meets the required quality bar — not whether to repeat
(that is the evaluate_batch_convergence_agent's responsibility)."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope:
{batch_files}
- execute_output: {execute_output}
- files_read: {files_read}
- files_written: {files_written}
- evaluate_quality_instructions: {evaluate_quality_instructions}"""

_STEPS = """\
1. Read the modified files from the workspace to inspect their actual content.
2. Evaluate the work against evaluate_quality_instructions and task_instructions for each file.
3. Apply the quality language severity mapping:
   - MUST / REQUIRED / MANDATORY → non-compliance is an ERROR
   - SHOULD / RECOMMENDED → non-compliance is a WARNING
   - COULD / MAY / SUGGESTED → non-compliance is an INFO
4. Produce one finding per file, plus an aggregate verdict."""

_GUARDRAILS = """\
- Do NOT decide whether to repeat passes — that is the evaluate_batch_convergence_agent's responsibility.
- The verdict MUST be the worst (lowest) level across all findings."""

_OUTPUT_FORMAT = """\
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

Verdict meaning:
- OK:    Work is correct and complete
- INFO:  Minor suggestions, but acceptable
- WARNING: Issues that should be fixed
- ERROR: Critical problems that must be fixed"""


def evaluate_batch_quality_agent(state: file_batch_workflow_state) -> dict:
    """Evaluate the overall quality of batch execution results (final quality check)."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    evaluate_quality_instructions = state["evaluate_quality_batch_instructions"]

    # Merge cumulative + current-pass values so quality is evaluated against ALL work done
    cumulative_execute_output = state.get("cumulative_execute_output", "")
    current_execute_output = state.get("execute_output", "")
    if cumulative_execute_output and current_execute_output:
        full_execute_output = cumulative_execute_output + "\n---\n" + current_execute_output
    else:
        full_execute_output = current_execute_output or cumulative_execute_output

    files_read = list(state.get("cumulative_files_read", [])) + list(state.get("files_read", []))
    files_written = list(state.get("cumulative_files_written", [])) + list(state.get("files_written", []))

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
            execute_output=full_execute_output,
            files_read=", ".join(files_read),
            files_written=", ".join(files_written),
            evaluate_quality_instructions=evaluate_quality_instructions,
        ),
        steps=_STEPS,
        guardrails=_GUARDRAILS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("evaluate_batch_quality_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
        json_mode=True,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    judge_verdict = parse_judge_output(content)
    return {
        "evaluate_quality_verdict": judge_verdict.verdict,
        "evaluate_quality_judge_verdict": judge_verdict,
    }
