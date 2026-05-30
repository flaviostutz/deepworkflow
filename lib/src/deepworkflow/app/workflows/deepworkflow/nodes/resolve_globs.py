from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.deepworkflow.states import WorkflowState

# Pattern to detect glob characters (not inside line range notation like :30-40)
_GLOB_CHARS = re.compile(r"[*?\[\]]")

# Pattern to detect line range suffix (e.g. file.py:30-40)
_LINE_RANGE_SUFFIX = re.compile(r":(\d+)-(\d+)$")


def _is_glob_pattern(entry: str) -> bool:
    """Check if an entry contains glob characters (ignoring line range suffixes)."""
    # Strip line range before checking for glob chars
    base = _LINE_RANGE_SUFFIX.sub("", entry)
    return bool(_GLOB_CHARS.search(base))


def resolve_globs(state: WorkflowState) -> dict:
    """Expand glob patterns in task_files to concrete file paths.

    Explicit paths (including those with line ranges like file.py:30-40) are kept as-is.
    Glob patterns are expanded relative to workspace_dir.
    Results are deduplicated while preserving order.
    Fails the workflow if the resolved list is empty.
    """
    config = state["config"]
    workspace_dir = Path(config.workspace_dir)
    raw_files = config.task_files

    resolved: list[str] = []
    seen: set[str] = set()

    for entry in raw_files:
        if _is_glob_pattern(entry):
            # Expand glob relative to workspace_dir
            matches = sorted(str(p) for p in workspace_dir.glob(entry))
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    resolved.append(match)
        # Explicit path (possibly with line range) — keep as-is
        elif entry not in seen:
            seen.add(entry)
            resolved.append(entry)

    if not resolved:
        return {"error": "No files found after resolving glob patterns in task_files."}

    return {"task_files": resolved}
