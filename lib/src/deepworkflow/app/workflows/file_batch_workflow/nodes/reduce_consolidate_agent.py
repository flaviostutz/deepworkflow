from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import STANDARD_USER_MESSAGE, TOOL_GUIDANCE_BASE, build_agent_prompt

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_OBJECTIVE = """\
Review all batch execution results and the workspace state to produce a final, consolidated output."""

_ROLE = """\
You are the `reduce_consolidate_agent`. You are an expert synthesiser who combines the individual
batch results into a coherent, holistic final output following the consolidation instructions."""

_INPUT_TEMPLATE = """\
Agent-specific inputs:
- reduce_instructions: {reduce_instructions}
- batch_results_summary:
{batch_results_summary}"""

_TOOL_GUIDANCE = f"""{TOOL_GUIDANCE_BASE}

Review the workspace to inspect the combined result of all batches before producing the final
output. Use file reading tools to verify the actual state of the workspace."""

_OUTPUT_FORMAT = """\
A holistic evaluation and final consolidated output following the reduce_instructions.
The output should synthesise all batch results into a coherent whole."""


def reduce_consolidate_agent(state: file_batch_workflow_state) -> dict:
    """Consolidate all batch outputs into a final workflow output."""
    config = state["config"]
    batch_results = state.get("batch_results", [])
    reduce_instructions = state.get("reduce_instructions", "Summarize the overall results.")

    outputs_summary = []
    for i, output in enumerate(batch_results):
        outputs_summary.append(
            f"--- Batch {i + 1} ---\n"
            f"Files: {', '.join(output.batch_files)}\n"
            f"Verdict: {output.evaluate_level.name}\n"
            f"Execute output: {output.batch_execute_output[:5000]}\n"
        )

    prompt = build_agent_prompt(
        objective=_OBJECTIVE,
        role=_ROLE,
        input_section=_INPUT_TEMPLATE.format(
            reduce_instructions=reduce_instructions,
            batch_results_summary="\n".join(outputs_summary),
        ),
        tool_guidance=_TOOL_GUIDANCE,
        output_format=_OUTPUT_FORMAT,
    )

    agent = create_agent(
        model=config.model("reduce_consolidate_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=config.workspace_write_option,
    )

    result = agent.invoke({"messages": STANDARD_USER_MESSAGE})
    last_message = result["messages"][-1]
    reduce_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {"reduce_output": reduce_output}
