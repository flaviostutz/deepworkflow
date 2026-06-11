from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.file_batch_workflow.nodes.check_max_retries_policy_step import (
    check_max_retries_policy_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent import (
    evaluate_batch_progress_agent,
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
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_increment_retry_step import map_increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent import plan_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.record_output_step import record_output_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent import reduce_consolidate_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent import reflect_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.resolve_globs_step import resolve_globs_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_judge_step import skip_judge_step
from deepworkflow.app.workflows.file_batch_workflow.routes import (
    check_batch_progress,
    check_map_retries,
    check_map_verdict,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
    route_after_reflect,
    route_map_judge,
)
from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.types import JudgeFeedback, JudgeVerdict
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


def _feedback_summary(feedbacks: list[JudgeFeedback]) -> str:
    """Return non-zero ERROR/WARNING/INFO counts joined by '; ', or 'OK'."""
    counts = {
        JudgeVerdict.ERROR: 0,
        JudgeVerdict.WARNING: 0,
        JudgeVerdict.INFO: 0,
    }
    for fb in feedbacks:
        if fb.type in counts:
            counts[fb.type] += 1
    parts = []
    if counts[JudgeVerdict.ERROR]:
        parts.append(f"{counts[JudgeVerdict.ERROR]} error")
    if counts[JudgeVerdict.WARNING]:
        parts.append(f"{counts[JudgeVerdict.WARNING]} warning")
    if counts[JudgeVerdict.INFO]:
        parts.append(f"{counts[JudgeVerdict.INFO]} info")
    return "; ".join(parts) if parts else "OK"


# resolve_globs_step
def _log_resolve_globs_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    files = result.get("task_files") or []
    return [f"{len(files)} files"]


# map_batches_agent
def _log_map_batches_pre(state: dict) -> list[str]:
    config = state.get("config")
    task = config.task_instructions if config is not None else ""
    return [f"task: {_truncate(task, 30)}"]


def _log_map_batches_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    if result.get("error"):
        return []
    batches = result.get("task_file_batches") or []
    overview = result.get("task_overview", "")
    avg = sum(len(b.batch_files) for b in batches) / len(batches) if batches else 0
    return [
        f"overview: {_truncate(overview, 30)}",
        f"{len(batches)} batches; {avg:.0f} files/batch",
    ]


# evaluate_map_batches_agent
def _log_evaluate_map_batches_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    feedbacks = result.get("map_judge_feedbacks") or []
    return [_feedback_summary(feedbacks)]


# plan_batch_agent
def _log_plan_batch_pre(state: dict) -> list[str]:
    idx = state.get("current_batch_index", 0)
    batches = state.get("task_file_batches") or []
    instructions = batches[idx].batch_instructions if idx < len(batches) else ""
    return [f"batch instructions: {_truncate(instructions or '', 30)}"]


def _log_plan_batch_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    plan = result.get("plan_output", "")
    return [f"plan: {_truncate(plan, 30)}"]


# execute_batch_agent
def _log_execute_batch_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    output = result.get("execute_output", "")
    return [f"output: {_truncate(output, 30)}"]


# reflect_batch_agent
def _log_reflect_batch_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    written = result.get("files_written") or []
    read = result.get("files_read") or []
    return [f"{len(written)} files written; {len(read)} files read"]


# evaluate_batch_progress_agent
def _log_evaluate_progress_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    return [_truncate(result.get("batch_progress_output", ""), 50)]


# evaluate_batch_quality_agent
def _log_evaluate_quality_post(state: dict, result: dict) -> list[str]:  # noqa: ARG001
    feedbacks = result.get("judge_feedbacks") or []
    return [_feedback_summary(feedbacks)]


# reduce_consolidate_agent
def _log_reduce_consolidate_post(state: dict, result: dict) -> list[str]:
    output = result.get("workflow_output", "")
    batch_outputs = state.get("batch_outputs") or []
    total_read = sum(len(b.files_read) for b in batch_outputs)
    total_written = sum(len(b.files_written) for b in batch_outputs)
    lines = ["output:"] + (output.splitlines() if output else [""])
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

    # === Phase 1: Map ===
    builder.add_node(
        "resolve_globs_step",
        wrap_node("resolve_globs_step", resolve_globs_step, log_post_fn=_log_resolve_globs_post),
    )
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
        "evaluate_batch_progress_agent",
        wrap_node(
            "evaluate_batch_progress_agent",
            evaluate_batch_progress_agent,
            log_post_fn=_log_evaluate_progress_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "increment_batch_repeat_step",
        wrap_node("increment_batch_repeat_step", increment_batch_repeat_step, stat="progress_retry"),
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
    builder.add_node("skip_judge_step", wrap_node("skip_judge_step", skip_judge_step))
    builder.add_node("record_output_step", wrap_node("record_output_step", record_output_step))
    builder.add_node(
        "increment_retry_step",
        wrap_node("increment_retry_step", increment_retry_step, stat="quality_retry"),
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
    builder.add_node("fail_step", wrap_node("fail_step", fail_step))

    # --- Entry ---
    builder.set_entry_point("resolve_globs_step")

    # --- Map phase edges ---
    builder.add_edge("resolve_globs_step", "map_batches_agent")

    # After map: route to judge or skip based on judge_skip
    builder.add_conditional_edges(
        "map_batches_agent",
        wrap_route("route_map_judge", route_map_judge),
        {"evaluate": "evaluate_map_batches_agent", "skip": "plan_batch_agent"},
    )

    # After map evaluation: pass → plan_batch_agent (first batch), retry → map_increment_retry_step
    builder.add_conditional_edges(
        "evaluate_map_batches_agent",
        wrap_route("check_map_verdict", check_map_verdict),
        {"pass": "plan_batch_agent", "retry_or_fail": "map_increment_retry_step"},
    )

    # Map retry: check retries remaining
    builder.add_conditional_edges(
        "map_increment_retry_step",
        wrap_route("check_map_retries", check_map_retries),
        {"map_batches_agent": "map_batches_agent", "fail_step": "fail_step"},
    )

    # --- Execute phase edges (per batch) ---
    builder.add_edge("plan_batch_agent", "execute_batch_agent")
    builder.add_edge("execute_batch_agent", "reflect_batch_agent")

    # After reflect: route to progress judge (if batch_repeat_max > 0), quality judge, or skip
    builder.add_conditional_edges(
        "reflect_batch_agent",
        wrap_route("route_after_reflect", route_after_reflect),
        {
            "evaluate_progress": "evaluate_batch_progress_agent",
            "evaluate": "evaluate_batch_quality_agent",
            "skip": "skip_judge_step",
        },
    )

    # After progress judge: repeat loop, quality judge, or skip
    builder.add_conditional_edges(
        "evaluate_batch_progress_agent",
        wrap_route("check_batch_progress", check_batch_progress),
        {
            "repeat": "increment_batch_repeat_step",
            "evaluate": "evaluate_batch_quality_agent",
            "skip": "skip_judge_step",
        },
    )
    builder.add_edge("increment_batch_repeat_step", "plan_batch_agent")
    builder.add_edge("skip_judge_step", "record_output_step")

    # After quality judge: check verdict
    builder.add_conditional_edges(
        "evaluate_batch_quality_agent",
        wrap_route("check_verdict", check_verdict),
        {"pass": "record_output_step", "retry_or_fail": "increment_retry_step"},
    )

    # Retry path
    builder.add_conditional_edges(
        "increment_retry_step",
        wrap_route("check_retries", check_retries),
        {"plan_batch_agent": "plan_batch_agent", "max_retries_exceeded": "check_max_retries_policy_step"},
    )

    # Max retries exceeded policy
    builder.add_node(
        "check_max_retries_policy_step",
        wrap_node("check_max_retries_policy_step", check_max_retries_policy_step),
    )
    builder.add_conditional_edges(
        "check_max_retries_policy_step",
        wrap_route("check_max_retries_policy", check_max_retries_policy),
        {"fail_step": "fail_step", "record_output_step": "record_output_step"},
    )

    # After recording output: next batch or consolidate
    builder.add_conditional_edges(
        "record_output_step",
        wrap_route("next_batch", next_batch),
        {"plan_batch_agent": "plan_batch_agent", "reduce_consolidate_agent": "reduce_consolidate_agent"},
    )

    # --- Terminal nodes ---
    builder.add_edge("reduce_consolidate_agent", END)
    builder.add_edge("fail_step", END)

    # Compile
    return builder.compile(checkpointer=checkpointer)


# Default workflow instance (without checkpointer)
file_batch_workflow = build_file_batch_workflow()
