from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.adapters.connectors.deepagents_connector import create_agent
from deepworkflow.shared.prompts import workflow_role

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

CONSOLIDATE_PROMPT = """<OBJECTIVE>
Review all batch execution results and the workspace state to produce a final, consolidated output.
</OBJECTIVE>

<ROLE>
You are the `reduce_consolidate_agent` (see WORKFLOW_CONTEXT). You are an expert synthesiser who \
combines the individual batch results into a coherent, holistic final output following the \
consolidation instructions.
</ROLE>

<INPUT>
Agent-specific inputs:
- consolidation_instructions: {consolidation_instructions}
- batch_outputs_summary:
{batch_outputs_summary}
</INPUT>

<TOOL_GUIDANCE>
Review the workspace to inspect the combined result of all batches before producing the final
output. Use file reading tools to verify the actual state of the workspace.
</TOOL_GUIDANCE>

<OUTPUT_FORMAT>
A holistic evaluation and final consolidated output following the consolidation_instructions.
The output should synthesise all batch results into a coherent whole.
</OUTPUT_FORMAT>

<WORKFLOW_CONTEXT>
{workflow_context}
</WORKFLOW_CONTEXT>"""


def reduce_consolidate_agent(state: file_batch_workflow_state) -> dict:
    """Consolidate all batch outputs into a final workflow output."""
    config = state["config"]
    batch_outputs = state.get("batch_outputs", [])
    consolidation_instructions = state.get("consolidation_instructions", "Summarize the overall results.")

    outputs_summary = []
    for i, output in enumerate(batch_outputs):
        outputs_summary.append(
            f"--- Batch {i + 1} ---\n"
            f"Files: {', '.join(output.task_files)}\n"
            f"Verdict: {output.judge_verdict.name}\n"
            f"Execute output: {output.execute_output[:500]}\n"
        )

    prompt = CONSOLIDATE_PROMPT.format(
        workflow_context=workflow_role(
            "reduce_consolidate_agent", "Consolidate all batch results into the final workflow output"
        ),
        consolidation_instructions=consolidation_instructions,
        batch_outputs_summary="\n".join(outputs_summary),
    )

    agent = create_agent(
        model=config.model("reduce_consolidate_agent"),
        system_prompt=prompt,
        workspace_dir=config.workspace_dir,
        write_option=config.workspace_write_option,
    )

    result = agent.invoke({"messages": "Review the workspace and produce the final consolidated output."})
    last_message = result["messages"][-1]
    workflow_output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {"workflow_output": workflow_output}
