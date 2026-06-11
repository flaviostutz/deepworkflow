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
feedback with a verdict."""

_ROLE = """\
You are the `evaluate_batch_quality_agent`. You are the quality judge: you run once after all
execution passes for a batch complete and decide whether the result meets the required quality
bar — not whether to repeat (that is the progress judge's responsibility)."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope:
{batch_files}
- execute_output: {execute_output}
- files_read: {files_read}
- files_written: {files_written}
- judge_instructions: {judge_instructions}"""

_STEPS = """\
1. Read the modified files from the workspace to inspect their actual content.
2. Evaluate the work against judge_instructions and task_instructions for each file.
3. Apply the quality language severity mapping:
   - MUST / REQUIRED / MANDATORY → non-compliance is an ERROR
   - SHOULD / RECOMMENDED → non-compliance is a WARNING
   - COULD / MAY / SUGGESTED → non-compliance is an INFO
4. Produce one feedback entry per file, plus an aggregate verdict."""

_GUARDRAILS = """\
- Do NOT decide whether to repeat passes — that is the progress judge's responsibility.
- The verdict MUST be the worst (lowest) type across all feedbacks."""

_OUTPUT_FORMAT = """\
{{
  "judge_feedbacks": [
    {{
      "file": "path/to/file",
      "type": "OK|INFO|WARNING|ERROR",
      "description": "explanation",
      "proposal": "how to fix the issue (required for WARNING or ERROR; empty for OK or INFO)"
    }}
  ],
  "judge_verdict": "OK|INFO|WARNING|ERROR"
}}

Verdict meaning:
- OK: Work is correct and complete
- INFO: Minor suggestions, but acceptable
- WARNING: Issues that should be fixed
- ERROR: Critical problems that must be fixed"""


def evaluate_batch_quality_agent(state: file_batch_workflow_state) -> dict:
    """Judge the overall quality of batch execution results (final quality check)."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    judge_instructions = state["judge_batch_instructions"]

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            batch_files="\n".join(current_batch.batch_files),
            execute_output=state["execute_output"],
            files_read=", ".join(state.get("files_read", [])),
            files_written=", ".join(state.get("files_written", [])),
            judge_instructions=judge_instructions,
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
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    verdict, feedbacks = parse_judge_output(content)

    return {"judge_verdict": verdict, "judge_feedbacks": feedbacks}
