"""Log helper functions for the file_batch_workflow graph."""

from __future__ import annotations

from deepworkflow.shared.types import JudgeLevel, JudgeVerdict, WorkflowLogLevel


def _truncate(text: str, n_words: int) -> str:
    """Return the first *n_words* words of *text*, appending '…' if truncated."""
    words = text.split()
    if len(words) <= n_words:
        return text
    return " ".join(words[:n_words]) + "…"


def _finding_summary(verdict: JudgeVerdict) -> str:
    """Return non-zero ERROR/WARNING/INFO counts joined by '; ', or 'OK'."""
    counts = {
        JudgeLevel.ERROR: 0,
        JudgeLevel.WARNING: 0,
        JudgeLevel.INFO: 0,
    }
    for f in verdict.findings:
        if f.level in counts:
            counts[f.level] += 1
    parts = []
    if counts[JudgeLevel.ERROR]:
        parts.append(f"{counts[JudgeLevel.ERROR]} error")
    if counts[JudgeLevel.WARNING]:
        parts.append(f"{counts[JudgeLevel.WARNING]} warning")
    if counts[JudgeLevel.INFO]:
        parts.append(f"{counts[JudgeLevel.INFO]} info")
    return "; ".join(parts) if parts else "OK"


def _finding_lines(verdict: JudgeVerdict, log_level: WorkflowLogLevel) -> list[str]:
    """Return one descriptive line per non-OK finding (debug: all findings)."""
    lines = []
    for f in verdict.findings:
        if log_level != WorkflowLogLevel.DEBUG and f.level == JudgeLevel.OK:
            continue
        line = f"[{f.level.name}] {f.title}"
        if f.reason:
            line += f" — {f.reason}"
        lines.append(line)
    return lines


def _format_verdict_lines(verdict: JudgeVerdict, log_level: WorkflowLogLevel) -> list[str]:
    """Return log line(s) for a JudgeVerdict.

    OK verdicts produce a single ``"OK"`` summary line.
    Non-OK verdicts produce per-finding detail lines (each carries its own level prefix).
    """
    if verdict.verdict == JudgeLevel.OK:
        return [f"OK - {_finding_summary(verdict)}"]
    return _finding_lines(verdict, log_level)


# resolve_globs_step
def _log_resolve_globs_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    files = result.get("task_files") or []
    return [f"{len(files)} files"]


# map_batches_agent
def _log_map_batches_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:
    config = state.get("config")
    task = config.task_instructions if config is not None else ""
    text = task if log_level == WorkflowLogLevel.DEBUG else _truncate(task, 30)
    return [f"task: {text}"]


def _log_map_batches_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    if result.get("error"):
        return []
    batches = result.get("task_file_batches") or []
    overview = result.get("task_overview", "")
    avg = sum(len(b.batch_files) for b in batches) / len(batches) if batches else 0
    overview_text = overview if log_level == WorkflowLogLevel.DEBUG else _truncate(overview, 30)
    return [
        f"overview: {overview_text}",
        f"{len(batches)} batches; {avg:.0f} files/batch",
    ]


# evaluate_map_batches_agent
def _log_evaluate_map_batches_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("map_evaluate_judge_verdict")
    if verdict is not None:
        return _format_verdict_lines(verdict, log_level)
    raw_verdict = result.get("map_evaluate_quality_verdict")
    return [raw_verdict.name if raw_verdict is not None else "unknown"]


# plan_batch_agent
def _log_plan_batch_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:
    idx = state.get("current_batch_index", 0)
    batches = state.get("task_file_batches") or []
    instructions = batches[idx].batch_instructions if idx < len(batches) else ""
    text = (instructions or "") if log_level == WorkflowLogLevel.DEBUG else _truncate(instructions or "", 30)
    return [f"batch instructions: {text}"]


def _log_plan_batch_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    plan = result.get("plan_output", "")
    text = plan if log_level == WorkflowLogLevel.DEBUG else _truncate(plan, 30)
    return [f"plan: {text}"]


# analyze_task_effort_agent
def _log_analyze_effort_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    ec = result.get("effort_config")
    if ec is None:
        return ["effort_config not set"]
    return [
        f"map={ec.map_batches_mode} consolidate={ec.consolidate_mode} "
        f"eval_map_retries={ec.evaluate_map_max_retries} "
        f"eval_quality_retries={ec.evaluate_batch_quality_max_retries}"
    ]


# execute_batch_agent
def _log_execute_batch_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    output = result.get("execute_output", "")
    text = output if log_level == WorkflowLogLevel.DEBUG else _truncate(output, 30)
    return [f"output: {text}"]


# reflect_batch_agent
def _log_reflect_batch_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    written = result.get("files_written") or []
    read = result.get("files_read") or []
    return [f"{len(written)} files written; {len(read)} files read"]


# evaluate_batch_convergence_agent
def _log_evaluate_convergence_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("batch_convergence_verdict")
    if verdict is not None:
        return _format_verdict_lines(verdict, log_level)
    raw = result.get("batch_convergence_output", "")
    text = raw if log_level == WorkflowLogLevel.DEBUG else _truncate(raw, 50)
    return [text]


# evaluate_batch_quality_agent
def _log_evaluate_quality_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("evaluate_quality_judge_verdict")
    if verdict is not None:
        return _format_verdict_lines(verdict, log_level)
    raw_verdict = result.get("evaluate_quality_verdict")
    return [raw_verdict.name if raw_verdict is not None else "unknown"]


# reduce_consolidate_agent
def _log_reduce_consolidate_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    output = result.get("workflow_output", "")
    batch_outputs = state.get("batch_outputs") or []
    total_read = sum(len(b.files_read) for b in batch_outputs)
    total_written = sum(len(b.files_written) for b in batch_outputs)
    lines = ["output:\n" + (output or "")]
    lines.append(f"{total_read} files read; {total_written} files written")
    return lines
