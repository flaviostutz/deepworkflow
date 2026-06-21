from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.file_batch_workflow.graph_log import (
    _log_analyze_effort_post,
    _log_evaluate_convergence_post,
    _log_evaluate_map_batches_post,
    _log_evaluate_quality_post,
    _log_evaluate_quality_pre,
    _log_execute_batch_post,
    _log_execute_batch_pre,
    _log_map_batches_post,
    _log_map_batches_pre,
    _log_plan_batch_post,
    _log_plan_batch_pre,
    _log_reduce_consolidate_post,
    _log_reflect_batch_post,
    _log_resolve_globs_post,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.check_max_retries_policy_step import (
    check_max_retries_policy_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.effort_analyze_auto_agent import effort_analyze_auto_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.effort_static_step import effort_static_step
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
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_batch_plan_step import skip_batch_plan_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_evaluate_quality_step import skip_evaluate_quality_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_reflect_batch_step import skip_reflect_batch_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.validate_map_batches_step import validate_map_batches_step
from deepworkflow.app.workflows.file_batch_workflow.routes import (
    check_batch_convergence,
    check_map_retries,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
    route_after_execute,
    route_after_map_verdict,
    route_after_reflect,
    route_effort_config,
    route_map_batches_mode,
    route_plan_batch,
    route_validate_map_limits,
)
from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.workflow_log import wrap_node, wrap_route

# resolve_globs_step


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
        "effort_static_step",
        wrap_node("effort_static_step", effort_static_step),
    )
    builder.add_node(
        "effort_analyze_auto_agent",
        wrap_node(
            "effort_analyze_auto_agent",
            effort_analyze_auto_agent,
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
        "skip_reflect_batch_step",
        wrap_node("skip_reflect_batch_step", skip_reflect_batch_step, show_batch_index=True),
    )
    builder.add_node(
        "execute_batch_agent",
        wrap_node(
            "execute_batch_agent",
            execute_batch_agent,
            log_pre_fn=_log_execute_batch_pre,
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
            log_pre_fn=_log_evaluate_quality_pre,
            log_post_fn=_log_evaluate_quality_post,
            show_batch_index=True,
        ),
    )
    builder.add_node("skip_evaluate_quality_step", wrap_node("skip_evaluate_quality_step", skip_evaluate_quality_step))
    builder.add_node("record_output_step", wrap_node("record_output_step", record_output_step))
    builder.add_node(
        "increment_retry_step",
        wrap_node("increment_retry_step", increment_retry_step),
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
        {"effort_static_step": "effort_static_step", "effort_analyze_auto_agent": "effort_analyze_auto_agent"},
    )
    for effort_node in ("effort_static_step", "effort_analyze_auto_agent"):
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
    builder.add_conditional_edges(
        "execute_batch_agent",
        wrap_route("route_after_execute", route_after_execute),
        {"reflect_batch_agent": "reflect_batch_agent", "skip_reflect_batch_step": "skip_reflect_batch_step"},
    )
    builder.add_edge("skip_reflect_batch_step", "skip_evaluate_quality_step")

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
graph = build_file_batch_workflow()
