from __future__ import annotations

import re
from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.config import resolveEffortConfig
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt
from deepworkflow.shared.types import WriteOption

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Analyse the task instructions and a representative sample of files to determine how much
computational effort this workflow should spend — then return a single integer effort level
between 1 and 10."""

_ROLE = """\
You are the `map_effort_analyze_agent`.  You are an expert at estimating the complexity and
risk of a task so that the right amount of LLM evaluation and iteration is applied."""

_INPUT_TEMPLATE = """\
Workflow-level inputs:
- task_instructions: {task_instructions}

Agent-specific inputs:
- files_in_scope ({file_count} total):
{files_section}"""

_STEPS = """\
1. Read a representative sample of the files listed above (especially those most central to the
   task) to understand the semantics of the work — code structure, data formats,
   interdependencies, etc.
2. Focus on the **big picture**.  Do NOT produce a full execution plan, enumerate every file,
   or begin any part of the actual task.  You are here only to judge complexity.
3. Consider the following factors when choosing a level:
   - Number of files and their interdependency (tightly coupled vs. isolated)
   - Task ambiguity: are the instructions clear and unambiguous?
   - Risk of silent errors: could incorrect output be hard to detect?
   - Need for validation: does correctness require cross-checking or domain knowledge?
   - Scope of changes: localised edits vs. wide-ranging refactoring
4. Map your assessment to an effort level using these guidelines:
   - **1**: trivial, fully mechanical task; a single LLM pass is sufficient; no validation needed
   - **2-3**: straightforward task with minor complexity; static batching acceptable
   - **4-5**: moderate complexity; LLM batching and basic quality evaluation recommended
   - **6-7**: non-trivial task requiring planning before execution and quality checks
   - **8-9**: complex task with high error risk; multiple evaluation passes advisable
   - **10**: highly complex, ambiguous, or risk-critical task; maximum agentic safeguards needed

GUARDRAIL: You MUST NOT perform any part of the actual task.  Analysis and level selection only."""

_OUTPUT_FORMAT = """\
Respond with exactly one JSON object:
{{
  "level": <integer 1-10>,
  "reasoning": "<2-4 sentences explaining the key factors that drove this level>"
}}"""

_LEVEL_RE = re.compile(r'"level"\s*:\s*([1-9]|10)')


def map_effort_analyze_agent(state: file_batch_workflow_state) -> dict:
    """Analyse the task and files to automatically select an effort level 1-10."""
    config = state["config"]
    task_files = state.get("map_files") or []

    files_section = "\n".join(task_files) if task_files else "(no files pre-selected; explore the workspace)"

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            task_instructions=config.task_instructions,
            file_count=len(task_files),
            files_section=files_section,
        ),
        steps=_STEPS,
        tool_guidance=TOOL_GUIDANCE_BASE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("map_effort_analyze_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=WriteOption.READ_ONLY,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    level = _parse_level(content)
    effort_config = resolveEffortConfig(level)
    return {"effort_config": effort_config}


def _parse_level(content: str) -> int:
    """Extract the effort level integer from the agent response."""
    match = _LEVEL_RE.search(content)
    if match:
        return int(match.group(1))
    # Fallback: look for a bare integer in the response
    numbers = re.findall(r"\b([1-9]|10)\b", content)
    if numbers:
        return int(numbers[0])
    # Default to moderate effort if parsing fails
    return 5
