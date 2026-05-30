"""Shared prompt constants for workflow context across all agents."""

WORKFLOW_CONTEXT = """\
== Workflow Context ==
This workflow processes files in batches:
resolve_globs → map_batches → [per-batch: plan → execute → reflect → evaluate] → consolidate"""


def workflow_role(step_name: str, role_description: str) -> str:
    """Build the full workflow context header for an agent prompt."""
    return f"""{WORKFLOW_CONTEXT}
You are acting as: {step_name}
Your role: {role_description}"""
