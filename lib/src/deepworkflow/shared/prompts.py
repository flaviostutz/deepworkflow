"""Shared prompt constants and builder for workflow agents."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path

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
  evaluates the overall quality of the batch result and decides whether to accept or retry."""

TOOL_GUIDANCE_BASE = """\
To accomplish your role effectively, actively use all available tools — do not rely on memory alone:
- Shell execution: run commands to inspect the environment, execute scripts, and verify outputs
- File grep/search: search file contents for patterns, symbols, or keywords
- File read: read files to understand current state before making decisions
- File write: create or modify files (only when your role/constraints permit writing)
- Todo list: maintain a checklist to track multi-step work in progress
- Other tools: use any additional available tools to gather context and complete the task
- Temporary files: you may create temporary files under /tmp in the workspace to register plans, \
checklists, specs, or any other working files that support your work

File guidance:
- Read files before modifying them; understand existing code before suggesting changes
- Do not create files unless absolutely necessary; prefer editing existing files
- Call independent tools in parallel, but do not call semantic_search in parallel; \
call dependent tools sequentially
- When reading files, prefer reading a large section at once over many small reads; \
read multiple files in parallel when possible
- When invoking a tool that takes a file path, always use the absolute file path"""

# Standard opening user message used by every agent invocation.
STANDARD_USER_MESSAGE = "reach your objective taking into consideration the inputs given"


def build_context(workspace_dir: str | None = None, max_files: int = 500) -> str:
    """Build a context string with current date, OS, and optionally workspace structure."""
    lines = [
        f"The current date is {datetime.date.today().isoformat()}.",
        f"The current OS is: {platform.system()}",
    ]
    if workspace_dir is not None:
        structure = _build_workspace_structure(workspace_dir, max_files=max_files)
        lines.append(f"I am working in a workspace that has the following structure:\n```\n{structure}\n```")
    return "\n".join(lines)


_WORKSPACE_IGNORE: frozenset[str] = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", ".mypy_cache", ".ruff_cache"}
)


def _build_workspace_structure(workspace_dir: str, max_files: int = 500) -> str:
    """Generate an indented tree of the workspace, skipping hidden entries and common ignores."""
    root = Path(workspace_dir)
    lines: list[str] = []
    counter = [0]

    def _walk(path: Path, depth: int) -> None:
        if counter[0] >= max_files:
            return
        indent = "\t" * depth
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name in _WORKSPACE_IGNORE:
                continue
            if counter[0] >= max_files:
                lines.append(f"[TRUNCATED. {counter[0]}+ total files in workspace]")
                return
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                counter[0] += 1
                _walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{entry.name}")
                counter[0] += 1

    _walk(root, 0)
    return "\n".join(lines)


def build_agent_prompt(
    *,
    objective: str,
    role: str,
    input_section: str,
    output_format: str,
    steps: str | None = None,
    guardrails: str | None = None,
    tool_guidance: str | None = None,
    context: str | None = None,
) -> str:
    """Assemble a structured XML-tagged agent system prompt.

    Sections are separated by two newlines. Optional sections (steps, guardrails,
    tool_guidance, context) are omitted when *None*. The WORKFLOW_CONTEXT section is always
    appended last.

    Section order: CONTEXT? → OBJECTIVE → ROLE → INPUT → STEPS? → GUARDRAILS? → TOOL_GUIDANCE?
                   → OUTPUT_FORMAT → WORKFLOW_CONTEXT
    """
    parts = []
    if context is not None:
        parts.append(f"<CONTEXT>\n{context}\n</CONTEXT>")
    parts += [
        f"<OBJECTIVE>\n{objective}\n</OBJECTIVE>",
        f"<ROLE>\n{role}\n</ROLE>",
        f"<INPUT>\n{input_section}\n</INPUT>",
    ]
    if steps is not None:
        parts.append(f"<STEPS>\n{steps}\n</STEPS>")
    if guardrails is not None:
        parts.append(f"<GUARDRAILS>\n{guardrails}\n</GUARDRAILS>")
    if tool_guidance is not None:
        parts.append(f"<TOOL_GUIDANCE>\n{tool_guidance}\n</TOOL_GUIDANCE>")
    parts.append(f"<OUTPUT_FORMAT>\n{output_format}\n</OUTPUT_FORMAT>")
    parts.append(f"<WORKFLOW_CONTEXT>\n{WORKFLOW_CONTEXT}\n</WORKFLOW_CONTEXT>")
    return "\n\n".join(parts)
