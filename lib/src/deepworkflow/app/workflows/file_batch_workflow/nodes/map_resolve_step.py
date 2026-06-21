from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

# Pattern to detect glob characters (not inside line range notation like :30-40)
_GLOB_CHARS = re.compile(r"[*?\[\]]")

# Pattern to detect line range suffix (e.g. file.py:30-40)
_LINE_RANGE_SUFFIX = re.compile(r":(\d+)-(\d+)$")


def _is_glob_pattern(entry: str) -> bool:
    """Check if an entry contains glob characters (ignoring line range suffixes)."""
    # Strip line range before checking for glob chars
    base = _LINE_RANGE_SUFFIX.sub("", entry)
    return bool(_GLOB_CHARS.search(base))


def _to_absolute_str(path_str: str, workspace_dir: Path) -> str:
    """Return the absolute path string for a file reference, stripping any line-range suffix."""
    base = _LINE_RANGE_SUFFIX.sub("", path_str)
    p = Path(base)
    if not p.is_absolute():
        p = workspace_dir / base
    return str(p)


def _expand_excludes(workspace_dir: Path, patterns: list[str] | None) -> set[str]:
    """Expand exclude patterns to a set of absolute path strings."""
    if not patterns:
        return set()
    excluded: set[str] = set()
    for pattern in patterns:
        if _is_glob_pattern(pattern):
            excluded.update(str(p) for p in workspace_dir.glob(pattern))
        else:
            excluded.add(_to_absolute_str(pattern, workspace_dir))
    return excluded


def map_resolve_step(state: file_batch_workflow_state) -> dict:
    """Expand glob patterns in task_files to concrete file paths.

    If ``config.task_files`` is ``None``, returns an empty list so that
    ``map_plan_agent`` can discover the relevant files from the workspace.
    Explicit paths (including those with line ranges like file.py:30-40) are
    kept as-is.  Glob patterns are expanded relative to workspace_dir.
    Results are deduplicated while preserving order.
    Fails the workflow if task_files was provided but resolves to nothing.
    """
    config = state["config"]

    # When task_files is None the map agent will discover files itself
    if config.task_files is None:
        return {"map_files": []}

    workspace_dir = Path(config.workspace_dir)
    raw_files = config.task_files

    resolved: list[str] = []
    seen: set[str] = set()

    for entry in raw_files:
        if _is_glob_pattern(entry):
            # Expand glob relative to workspace_dir; keep paths relative to workspace_dir
            # so downstream nodes can safely prepend workspace_dir without double-prefixing.
            matches = sorted(str(p.relative_to(workspace_dir)) for p in workspace_dir.glob(entry))
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    resolved.append(match)
        # Explicit path (possibly with line range) — keep as-is
        elif entry not in seen:
            seen.add(entry)
            resolved.append(entry)

    # Filter out excluded files
    exclude_set = _expand_excludes(workspace_dir, config.task_files_exclude)
    if exclude_set:
        resolved = [f for f in resolved if _to_absolute_str(f, workspace_dir) not in exclude_set]

    if not resolved:
        return {"error": "No files found after resolving glob patterns in task_files."}

    return {"map_files": resolved}
