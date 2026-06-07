"""Shared prompt constants for workflow context across all agents."""

WORKFLOW_CONTEXT = """\
== Workflow Context ==
This workflow processes files in batches:
resolve_globs_step → map_batches_agent → evaluate_map_batches_agent
→ [per-batch: plan_batch_agent → execute_batch_agent → reflect_batch_agent
→ (evaluate_batch_progress_agent [progress judge] →)* evaluate_batch_quality_agent [quality judge]]
→ reduce_consolidate_agent

Two judges operate in the per-batch loop:
- evaluate_batch_progress_agent (progress judge): lightweight check after each pass — decides
  whether meaningful progress was made and whether to loop back for another pass (when
  batch_repeat_max > 0); does NOT evaluate final quality.
- evaluate_batch_quality_agent (quality judge): final check after all passes complete —
  evaluates the overall quality of the batch result and decides whether to accept or retry.

== Tool Usage ==
To accomplish your role effectively, actively use all available tools — do not rely on memory alone:
- Shell execution: run commands to inspect the environment, execute scripts, and verify outputs
- File grep/search: search file contents for patterns, symbols, or keywords
- File read: read files to understand current state before making decisions
- File write: create or modify files (only when your role/constraints permit writing)
- Todo list: maintain a checklist to track multi-step work in progress
- Other tools: use any additional available tools to gather context and complete the task"""


def workflow_role(step_name: str, role_description: str) -> str:
    """Build the full workflow context header for an agent prompt."""
    return f"""{WORKFLOW_CONTEXT}
You are acting as: {step_name}
Your role: {role_description}"""
