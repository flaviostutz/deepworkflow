from __future__ import annotations

import json
from typing import TYPE_CHECKING

from deepworkflow.shared.agent import create_workflow_agent
from deepworkflow.shared.prompts import workflow_role
from deepworkflow.shared.types import JudgeFeedback, JudgeVerdict, WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.deepworkflow.states import WorkflowState

EVALUATE_PROMPT = """{workflow_context}

You are evaluating the quality of task execution for a single batch.

Task instructions: {task_instructions}

Files in scope: {batch_files}

Execution output:
{execute_output}

Files read: {files_read}
Files written: {files_written}

{judge_instructions}

Evaluate the work quality. For each file, provide feedback.

Respond in this JSON format:
{{
  "judge_feedbacks": [
    {{
      "file": "path/to/file",
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

The verdict should be the WORST (lowest) type across all feedbacks.
- OK: Work is correct and complete
- INFO: Minor suggestions, but acceptable
- WARNING: Issues that should be fixed
- ERROR: Critical problems that must be fixed"""


def evaluate_task_agent(state: WorkflowState) -> dict:
    """Evaluate the execution output using a separate judge agent."""
    config = state["config"]
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    judge_instructions = config.judge_instructions or (
        "Check if the task_instructions was implemented with good quality "
        "(correctness, completeness, accuracy, consistency). Check if the content "
        "of the output and the written files contains the features requested in "
        "task_description, if the output follows the format/schema specified strictly. "
        "The written files and output text should have no inconsistencies or missing "
        "aspects that should have been taken into consideration in the execution to "
        "make the initial task_instruction fulfilled with quality, correctness, "
        "completeness and accuracy."
    )

    prompt = EVALUATE_PROMPT.format(
        workflow_context=workflow_role("evaluate_task_agent", "Judge the quality of batch execution results"),
        task_instructions=config.task_instructions,
        batch_files="\n".join(current_batch.batch_files),
        execute_output=state["execute_output"],
        files_read=", ".join(state.get("files_read", [])),
        files_written=", ".join(state.get("files_written", [])),
        judge_instructions=judge_instructions,
    )

    agent = create_workflow_agent(
        model=config.model,
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": "Read the modified files from the workspace and evaluate the work."})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    verdict, feedbacks = _parse_judge_output(content)

    return {"judge_verdict": verdict, "judge_feedbacks": feedbacks}


def _parse_judge_output(content: str) -> tuple[JudgeVerdict, list[JudgeFeedback]]:
    """Parse the judge agent's JSON output into verdict and feedbacks."""
    import re

    text = content.strip()

    # Extract JSON from fenced code block if present
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict.ERROR, [
            JudgeFeedback(file="general", type=JudgeVerdict.ERROR, description="Judge produced invalid JSON output.")
        ]

    verdict_str = data.get("judge_verdict", "ERROR").upper()
    try:
        verdict = JudgeVerdict[verdict_str]
    except KeyError:
        verdict = JudgeVerdict.ERROR

    feedbacks = []
    for fb in data.get("judge_feedbacks", []):
        fb_type_str = fb.get("type", "ERROR").upper()
        try:
            fb_type = JudgeVerdict[fb_type_str]
        except KeyError:
            fb_type = JudgeVerdict.ERROR
        feedbacks.append(
            JudgeFeedback(
                file=fb.get("file", "unknown"),
                type=fb_type,
                description=fb.get("description", ""),
                proposal=fb.get("proposal", ""),
            )
        )

    return verdict, feedbacks
