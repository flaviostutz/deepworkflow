"""Shared prompt constants for workflow context across all agents."""

WORKFLOW_CONTEXT = """\
== Workflow Context ==
This workflow processes files in batches:
resolve_globs_step → map_batches_agent → evaluate_map_batches_agent
→ [per-batch: plan_batch_agent → execute_batch_agent → reflect_batch_agent → evaluate_batch_agent]
→ reduce_consolidate_agent"""


def workflow_role(step_name: str, role_description: str) -> str:
    """Build the full workflow context header for an agent prompt."""
    return f"""{WORKFLOW_CONTEXT}
You are acting as: {step_name}
Your role: {role_description}"""
