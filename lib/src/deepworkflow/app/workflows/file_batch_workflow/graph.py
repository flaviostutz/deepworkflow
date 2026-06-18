from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.file_batch_workflow.nodes.analyze_task_effort_agent import analyze_task_effort_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.check_max_retries_policy_step import (
    check_max_retries_policy_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_convergence_agent import (
    evaluate_batch_convergence_agent,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent import (
    evaluate_batch_quality_agent,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent import evaluate_map_batches_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.execute_batch_agent import execute_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_batch_repeat_step import (
    increment_batch_repeat_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_retry_step import increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent import map_batches_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_step import map_batches_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_increment_retry_step import map_increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent import plan_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.record_output_step import record_output_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent import reduce_consolidate_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_step import reduce_consolidate_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent import reflect_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.resolve_globs_step import resolve_globs_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.set_effort_config_step import set_effort_config_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_batch_plan_step import skip_batch_plan_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_evaluate_quality_step import skip_evaluate_quality_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.validate_map_batches_step import validate_map_batches_step
from deepworkflow.app.workflows.file_batch_workflow.routes import (
    check_batch_convergence,
    check_map_retries,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
    route_after_map_verdict,
    route_after_reflect,
    route_effort_config,
    route_map_batches_mode,
    route_plan_batch,
    route_validate_map_limits,
)
from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import JudgeLevel, JudgeVerdict, WorkflowLogLevel
from deepworkflow.shared.workflow_log import wrap_node, wrap_route

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


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


def build_file_batch_workflow(checkpointer: Any = None) -> Any:
    """Build and compile the file_batch_workflow LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for crash recovery.

    Returns:
        Compiled LangGraph graph.

    """
    builder = StateGraph(file_batch_workflow_state)

    # === Phase 0: Resolve globs + effort config ===
    builder.add_node(
        "resolve_globs_step",
        wrap_node("resolve_globs_step", resolve_globs_step, log_post_fn=_log_resolve_globs_post),
    )
    builder.add_node(
        "set_effort_config_step",
        wrap_node("set_effort_config_step", set_effort_config_step),
    )
    builder.add_node(
        "analyze_task_effort_agent",
        wrap_node(
            "analyze_task_effort_agent",
            analyze_task_effort_agent,
            log_post_fn=_log_analyze_effort_post,
        ),
    )

    # === Phase 1: Map ===
    builder.add_node(
        "map_batches_agent",
        wrap_node(
            "map_batches_agent",
            map_batches_agent,
            log_pre_fn=_log_map_batches_pre,
            log_post_fn=_log_map_batches_post,
        ),
    )
    builder.add_node(
        "map_batches_step",
        wrap_node("map_batches_step", map_batches_step, log_post_fn=_log_map_batches_post),
    )
    builder.add_node(
        "validate_map_batches_step",
        wrap_node("validate_map_batches_step", validate_map_batches_step),
    )
    builder.add_node(
        "evaluate_map_batches_agent",
        wrap_node(
            "evaluate_map_batches_agent",
            evaluate_map_batches_agent,
            log_post_fn=_log_evaluate_map_batches_post,
        ),
    )
    builder.add_node(
        "map_increment_retry_step",
        wrap_node("map_increment_retry_step", map_increment_retry_step, stat="quality_retry"),
    )

    # === Phase 2: Execute (per batch) ===
    builder.add_node(
        "plan_batch_agent",
        wrap_node(
            "plan_batch_agent",
            plan_batch_agent,
            log_pre_fn=_log_plan_batch_pre,
            log_post_fn=_log_plan_batch_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "skip_batch_plan_step",
        wrap_node("skip_batch_plan_step", skip_batch_plan_step, show_batch_index=True),
    )
    builder.add_node(
        "execute_batch_agent",
        wrap_node(
            "execute_batch_agent",
            execute_batch_agent,
            log_post_fn=_log_execute_batch_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "reflect_batch_agent",
        wrap_node(
            "reflect_batch_agent",
            reflect_batch_agent,
            log_post_fn=_log_reflect_batch_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "evaluate_batch_convergence_agent",
        wrap_node(
            "evaluate_batch_convergence_agent",
            evaluate_batch_convergence_agent,
            log_post_fn=_log_evaluate_convergence_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "increment_batch_repeat_step",
        wrap_node("increment_batch_repeat_step", increment_batch_repeat_step, stat="convergence_retry"),
    )
    builder.add_node(
        "evaluate_batch_quality_agent",
        wrap_node(
            "evaluate_batch_quality_agent",
            evaluate_batch_quality_agent,
            log_post_fn=_log_evaluate_quality_post,
            show_batch_index=True,
        ),
    )
    builder.add_node("skip_evaluate_quality_step", wrap_node("skip_evaluate_quality_step", skip_evaluate_quality_step))
    builder.add_node("record_output_step", wrap_node("record_output_step", record_output_step))
    builder.add_node(
        "increment_retry_step",
        wrap_node("increment_retry_step", increment_retry_step, stat="quality_retry"),
    )
    builder.add_node(
        "check_max_retries_policy_step",
        wrap_node("check_max_retries_policy_step", check_max_retries_policy_step),
    )

    # === Phase 3: Reduce ===
    builder.add_node(
        "reduce_consolidate_agent",
        wrap_node(
            "reduce_consolidate_agent",
            reduce_consolidate_agent,
            log_post_fn=_log_reduce_consolidate_post,
        ),
    )
    builder.add_node(
        "reduce_consolidate_step",
        wrap_node("reduce_consolidate_step", reduce_consolidate_step, log_post_fn=_log_reduce_consolidate_post),
    )
    builder.add_node("fail_step", wrap_node("fail_step", fail_step))

    # ── Entry ──────────────────────────────────────────────────────────────────
    builder.set_entry_point("resolve_globs_step")

    # ── Phase 0 edges: resolve → effort config → map mode ────────────────────
    builder.add_conditional_edges(
        "resolve_globs_step",
        wrap_route("route_effort_config", route_effort_config),
        {"set_effort_config_step": "set_effort_config_step", "analyze_task_effort_agent": "analyze_task_effort_agent"},
    )
    for effort_node in ("set_effort_config_step", "analyze_task_effort_agent"):
        builder.add_conditional_edges(
            effort_node,
            wrap_route("route_map_batches_mode", route_map_batches_mode),
            {"map_batches_agent": "map_batches_agent", "map_batches_step": "map_batches_step"},
        )

    # ── Phase 1 edges: map → validate → evaluate or plan ─────────────────────
    for map_node in ("map_batches_agent", "map_batches_step"):
        builder.add_edge(map_node, "validate_map_batches_step")

    builder.add_conditional_edges(
        "validate_map_batches_step",
        wrap_route("route_validate_map_limits", route_validate_map_limits),
        {
            "fail_step": "fail_step",
            "map_batches_agent": "map_batches_agent",
            "evaluate_map_batches_agent": "evaluate_map_batches_agent",
            "plan_batch_agent": "plan_batch_agent",
            "skip_batch_plan_step": "skip_batch_plan_step",
        },
    )

    # After map evaluation: check verdict → plan or retry
    builder.add_conditional_edges(
        "evaluate_map_batches_agent",
        wrap_route("route_after_map_verdict", route_after_map_verdict),
        {
            "plan_batch_agent": "plan_batch_agent",
            "skip_batch_plan_step": "skip_batch_plan_step",
            "retry_or_fail": "map_increment_retry_step",
        },
    )

    # Map retry: check retries remaining
    builder.add_conditional_edges(
        "map_increment_retry_step",
        wrap_route("check_map_retries", check_map_retries),
        {"map_batches_agent": "map_batches_agent", "fail_step": "fail_step"},
    )

    # ── Phase 2 edges ─────────────────────────────────────────────────────────
    for plan_node in ("plan_batch_agent", "skip_batch_plan_step"):
        builder.add_edge(plan_node, "execute_batch_agent")
    builder.add_edge("execute_batch_agent", "reflect_batch_agent")

    # After reflect: convergence loop, evaluate quality, or skip
    builder.add_conditional_edges(
        "reflect_batch_agent",
        wrap_route("route_after_reflect", route_after_reflect),
        {
            "evaluate_convergence": "evaluate_batch_convergence_agent",
            "evaluate": "evaluate_batch_quality_agent",
            "skip": "skip_evaluate_quality_step",
        },
    )

    # After convergence check: repeat loop, evaluate quality, or skip
    builder.add_conditional_edges(
        "evaluate_batch_convergence_agent",
        wrap_route("check_batch_convergence", check_batch_convergence),
        {
            "repeat": "increment_batch_repeat_step",
            "evaluate": "evaluate_batch_quality_agent",
            "skip": "skip_evaluate_quality_step",
        },
    )
    builder.add_conditional_edges(
        "increment_batch_repeat_step",
        wrap_route("route_plan_batch", route_plan_batch),
        {"plan_batch_agent": "plan_batch_agent", "skip_batch_plan_step": "skip_batch_plan_step"},
    )

    builder.add_edge("skip_evaluate_quality_step", "record_output_step")

    # After evaluate quality: pass or retry
    builder.add_conditional_edges(
        "evaluate_batch_quality_agent",
        wrap_route("check_verdict", check_verdict),
        {"pass": "record_output_step", "retry_or_fail": "increment_retry_step"},
    )

    # Retry path
    builder.add_conditional_edges(
        "increment_retry_step",
        wrap_route("check_retries", check_retries),
        {
            "plan_batch_agent": "plan_batch_agent",
            "skip_batch_plan_step": "skip_batch_plan_step",
            "max_retries_exceeded": "check_max_retries_policy_step",
        },
    )

    # Max retries exceeded policy
    builder.add_conditional_edges(
        "check_max_retries_policy_step",
        wrap_route("check_max_retries_policy", check_max_retries_policy),
        {"fail_step": "fail_step", "record_output_step": "record_output_step"},
    )

    # After recording output: next batch or consolidate
    builder.add_conditional_edges(
        "record_output_step",
        wrap_route("next_batch", next_batch),
        {
            "plan_batch_agent": "plan_batch_agent",
            "skip_batch_plan_step": "skip_batch_plan_step",
            "reduce_consolidate_agent": "reduce_consolidate_agent",
            "reduce_consolidate_step": "reduce_consolidate_step",
        },
    )

    # ── Terminal nodes ─────────────────────────────────────────────────────────
    for terminal in ("reduce_consolidate_agent", "reduce_consolidate_step", "fail_step"):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer)


# Default workflow instance (without checkpointer)
file_batch_workflow = build_file_batch_workflow()
