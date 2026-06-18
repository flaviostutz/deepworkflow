from __future__ import annotations

import fnmatch
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from deepworkflow.shared.types import BatchDefinition

if TYPE_CHECKING:
    from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state

_LINE_RANGE_RE = re.compile(r":\d+-\d+$")


def _base_path(file_ref: str) -> str:
    """Strip optional :start-end line-range suffix."""
    return _LINE_RANGE_RE.sub("", file_ref)


def validate_map_batches_step(state: file_batch_workflow_state) -> dict:  # noqa: C901, PLR0912
    """Deterministic validation of the batch plan produced by map_batches_agent or map_batches_step.

    Checks:
    1. Batch count does not exceed effort_config.max_batches (if set).
    2. No batch exceeds effort_config.max_files_per_batch (if set).
    3. All batch_files exist on disk.
    4. No batch file matches task_files_exclude patterns.
    5. When task_files was provided: every file is assigned to exactly one batch (coverage +
       disjointness) and no invented files are present.

    On violation:
    - static mode  → hard-fail (error returned; graph routes to fail_step).
    - agent mode   → error returned; graph routes to LLM retry loop.
    """
    config = state["config"]
    effort_config = state["effort_config"]
    task_files: list[str] = state.get("task_files") or []
    batches: list[BatchDefinition] = state.get("task_file_batches") or []

    errors: list[str] = []
    workspace = Path(config.workspace_dir)
    all_batch_files: list[str] = [f for b in batches for f in b.batch_files]

    # ── Check 1: max_batches limit ────────────────────────────────────────────
    if effort_config.max_batches is not None and len(batches) > effort_config.max_batches:
        errors.append(f"Batch count {len(batches)} exceeds max_batches={effort_config.max_batches}.")

    # ── Check 2: max_files_per_batch limit ────────────────────────────────────
    if effort_config.max_files_per_batch is not None:
        for i, batch in enumerate(batches):
            if len(batch.batch_files) > effort_config.max_files_per_batch:
                errors.append(
                    f"Batch {i + 1} has {len(batch.batch_files)} files, "
                    f"exceeding max_files_per_batch={effort_config.max_files_per_batch}."
                )

    # ── Check 3: file existence ───────────────────────────────────────────────
    for file_ref in all_batch_files:
        base = _base_path(file_ref)
        path = Path(base)
        if not path.is_absolute():
            path = workspace / base
        if not path.exists():
            errors.append(f"File '{file_ref}' does not exist in the workspace.")

    # ── Check 4: task_files_exclude ───────────────────────────────────────────
    if config.task_files_exclude:
        for file_ref in all_batch_files:
            base = _base_path(file_ref)
            abs_path = base if Path(base).is_absolute() else str(workspace / base)
            for pattern in config.task_files_exclude:
                if fnmatch.fnmatch(abs_path, pattern) or fnmatch.fnmatch(base, pattern):
                    errors.append(
                        f"File '{file_ref}' matches exclude pattern '{pattern}' and must not be included in any batch."
                    )
                    break

    # ── Check 5: task_files coverage & disjointness ───────────────────────────
    if task_files:
        task_bases = [_base_path(f) for f in task_files]
        batch_bases = [_base_path(f) for f in all_batch_files]

        task_set = set(task_bases)
        batch_counter: Counter[str] = Counter(batch_bases)

        errors.extend(
            f"File '{tf}' from task_files is not assigned to any batch." for tf in task_bases if batch_counter[tf] == 0
        )

        for bf, count in batch_counter.items():
            if count > 1 and bf in task_set:
                errors.append(f"File '{bf}' appears in {count} batches (must be in exactly one).")

        errors.extend(
            f"File '{bf}' is assigned to a batch but was not in the original task_files list."
            for bf in batch_counter
            if bf not in task_set
        )

    if errors:
        return {"error": "validate_map_batches_step: " + " | ".join(errors)}

    # Clear any previous error from a retry round
    return {"error": None}
