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


# map_resolve_step
def _log_map_resolve_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    files = result.get("map_files") or []
    return [f"{len(files)} files"]


# map_plan_agent / map_plan_step
def _log_map_plan_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:
    config = state.get("config")
    task = config.task_instructions if config is not None else ""
    text = task if log_level == WorkflowLogLevel.DEBUG else _truncate(task, 30)
    return [f"task: {text}"]


def _log_map_plan_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    if result.get("error"):
        return []
    batches = result.get("map_batches") or []
    overview = result.get("map_plan_overview", "")
    overview_text = overview if log_level == WorkflowLogLevel.DEBUG else _truncate(overview, 30)
    counts = "/".join(str(len(b.batch_files)) for b in batches)
    return [
        f"overview: {overview_text}",
        f"{len(batches)} batches; {counts} files/batch",
    ]


# map_evaluate_agent
def _log_map_evaluate_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("map_evaluate_verdict")
    if verdict is not None:
        return _format_verdict_lines(verdict, log_level)
    raw_verdict = result.get("map_evaluate_level")
    return [raw_verdict.name if raw_verdict is not None else "unknown"]


# batch_plan_agent
def _log_batch_plan_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:
    idx = state.get("batch_current_index", 0)
    batches = state.get("map_batches") or []
    retry_count = state.get("batch_quality_retry_count", 0)
    batch_convergence_repeat_count = state.get("batch_convergence_repeat_count", 0)
    config = state.get("config")

    total = len(batches)
    loop_count = retry_count + batch_convergence_repeat_count
    loop_suffix = f" - loop {loop_count}" if loop_count > 0 else ""
    lines = [f">> PLAN BATCH {idx + 1}/{total}{loop_suffix}"]

    task_instructions = config.task_instructions if config is not None else ""
    lines.append(f"task instructions: {_truncate(task_instructions, 20)}")

    map_plan_overview = state.get("map_plan_overview", "")
    lines.append(f"map plan overview: {_truncate(map_plan_overview or '', 20)}")

    instructions = batches[idx].batch_instructions if idx < len(batches) else ""
    inst_text = (instructions or "") if log_level == WorkflowLogLevel.DEBUG else _truncate(instructions or "", 20)
    lines.append(f"batch instructions: {inst_text}")

    judge_verdict = state.get("batch_evaluate_verdict")
    if retry_count > 0 and judge_verdict is not None and judge_verdict.findings:
        feedback = "; ".join(f"{f.level.name}: {f.title}" for f in judge_verdict.findings if f.level.name != "OK")
        fb_text = feedback if log_level == WorkflowLogLevel.DEBUG else _truncate(feedback, 20)
        lines.append(f"evaluation feedback: {fb_text}")

    batch_cumulative_output = state.get("batch_cumulative_output", "")
    if batch_cumulative_output:
        prev_text = (
            batch_cumulative_output if log_level == WorkflowLogLevel.DEBUG else _truncate(batch_cumulative_output, 20)
        )
        lines.append(f"previous output: {prev_text}")

    return lines


def _log_batch_plan_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    plan = result.get("batch_plan", "")
    text = plan if log_level == WorkflowLogLevel.DEBUG else _truncate(plan, 20)
    return [f"plan: {text}"]


# map_effort_analyze_agent
def _log_map_effort_analyze_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    ec = result.get("effort_config")
    if ec is None:
        return ["effort_config not set"]
    return [
        f"map={ec.map_plan_mode} reduce={ec.reduce_mode} "
        f"map_eval_retries={ec.map_evaluate_max_retries} "
        f"batch_eval_quality_retries={ec.batch_evaluate_quality_max_retries}"
    ]


# batch_execute_agent
def _log_batch_execute_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:
    idx = state.get("batch_current_index", 0)
    batches = state.get("map_batches") or []
    retry_count = state.get("batch_quality_retry_count", 0)
    batch_convergence_repeat_count = state.get("batch_convergence_repeat_count", 0)
    config = state.get("config")

    total = len(batches)
    loop_count = retry_count + batch_convergence_repeat_count
    loop_suffix = f" - loop {loop_count}" if loop_count > 0 else ""
    lines = [f">> EXECUTE BATCH {idx + 1}/{total}{loop_suffix}"]

    task_instructions = config.task_instructions if config is not None else ""
    lines.append(f"task instructions: {_truncate(task_instructions, 20)}")

    map_plan_overview = state.get("map_plan_overview", "")
    lines.append(f"map plan overview: {_truncate(map_plan_overview or '', 20)}")

    instructions = batches[idx].batch_instructions if idx < len(batches) else ""
    lines.append(f"batch instructions: {_truncate(instructions or '', 20)}")

    batch_plan = state.get("batch_plan", "")
    plan_text = (batch_plan or "") if log_level == WorkflowLogLevel.DEBUG else _truncate(batch_plan or "", 20)
    lines.append(f"batch plan: {plan_text}")

    judge_verdict = state.get("batch_evaluate_verdict")
    if retry_count > 0 and judge_verdict is not None and judge_verdict.findings:
        feedback = "; ".join(f"{f.level.name}: {f.title}" for f in judge_verdict.findings if f.level.name != "OK")
        fb_text = feedback if log_level == WorkflowLogLevel.DEBUG else _truncate(feedback, 20)
        lines.append(f"evaluation feedback: {fb_text}")

    batch_cumulative_output = state.get("batch_cumulative_output", "")
    if batch_cumulative_output:
        prev_text = (
            batch_cumulative_output if log_level == WorkflowLogLevel.DEBUG else _truncate(batch_cumulative_output, 20)
        )
        lines.append(f"previous output: {prev_text}")

    return lines


def _log_batch_execute_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    output = result.get("batch_execute_output", "")
    text = output if log_level == WorkflowLogLevel.DEBUG else _truncate(output, 20)
    return [f"output: {text}"]


# batch_reflect_agent
def _log_batch_reflect_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    written = result.get("batch_files_written") or []
    read = result.get("batch_files_read") or []
    return [f"{len(written)} files written; {len(read)} files read"]


# batch_evaluate_convergence_agent
def _log_batch_evaluate_convergence_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("batch_evaluate_convergence_verdict")
    if verdict is not None:
        return _format_verdict_lines(verdict, log_level)
    raw = result.get("batch_evaluate_convergence_output", "")
    text = raw if log_level == WorkflowLogLevel.DEBUG else _truncate(raw, 50)
    return [text]


# batch_evaluate_quality_agent
def _log_batch_evaluate_quality_pre(state: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    effort_config = state.get("effort_config")
    instructions = ""
    if effort_config is not None and effort_config.batch_evaluate_quality_instructions:
        instructions = effort_config.batch_evaluate_quality_instructions
    return [f"evaluation instructions: {_truncate(instructions, 20)}"]


def _log_batch_evaluate_quality_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    verdict = result.get("batch_evaluate_verdict")
    if verdict is None:
        raw_verdict = result.get("batch_evaluate_level")
        return [f"result: {raw_verdict.name if raw_verdict is not None else 'unknown'}"]

    summary = _finding_summary(verdict)
    lines = [f"result: {verdict.verdict.name} - {summary}"]
    for f in verdict.findings:
        if log_level != WorkflowLogLevel.DEBUG and f.level.name == "OK":
            continue
        line = f"[{f.level.name}] {f.title}"
        if f.reason:
            line += f" — {_truncate(f.reason, 20)}"
        lines.append(line)
    return lines


# reduce_consolidate_agent / reduce_consolidate_step
def _log_reduce_consolidate_post(state: dict, result: dict, log_level: WorkflowLogLevel) -> list[str]:  # noqa: ARG001
    output = result.get("reduce_output", "")
    batch_results = state.get("batch_results") or []
    total_read = sum(len(b.batch_files_read) for b in batch_results)
    total_written = sum(len(b.batch_files_written) for b in batch_results)
    lines = ["output:\n" + (output or "")]
    lines.append(f"{total_read} files read; {total_written} files written")
    return lines
