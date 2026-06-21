from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.file_batch_workflow.graph_log import (
    _log_batch_evaluate_convergence_post,
    _log_batch_evaluate_quality_post,
    _log_batch_evaluate_quality_pre,
    _log_batch_execute_post,
    _log_batch_execute_pre,
    _log_batch_plan_post,
    _log_batch_plan_pre,
    _log_batch_reflect_post,
    _log_map_effort_analyze_post,
    _log_map_evaluate_post,
    _log_map_plan_post,
    _log_map_plan_pre,
    _log_map_resolve_post,
    _log_reduce_consolidate_post,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_convergence_repeat_step import (
    batch_convergence_repeat_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_evaluate_convergence_agent import (
    batch_evaluate_convergence_agent,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_evaluate_quality_agent import (
    batch_evaluate_quality_agent,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_evaluate_quality_skip_step import (
    batch_evaluate_quality_skip_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_execute_agent import batch_execute_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_output_record_step import batch_output_record_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent import batch_plan_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_skip_step import batch_plan_skip_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_quality_max_retries_step import (
    batch_quality_max_retries_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_quality_retry_step import batch_quality_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_reflect_agent import batch_reflect_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_reflect_skip_step import batch_reflect_skip_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_effort_analyze_agent import map_effort_analyze_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_effort_step import map_effort_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_evaluate_agent import map_evaluate_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_evaluate_retry_step import map_evaluate_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent import map_plan_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_step import map_plan_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_validate_step import map_plan_validate_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_resolve_step import map_resolve_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent import reduce_consolidate_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_step import reduce_consolidate_step
from deepworkflow.app.workflows.file_batch_workflow.routes import (
    batch_check_convergence,
    batch_check_verdict,
    batch_quality_check_retries,
    batch_quality_max_retries,
    batch_route_after_execute,
    batch_route_after_reflect,
    batch_route_next,
    batch_route_plan,
    map_evaluate_check_retries,
    map_plan_route_validate,
    map_route_after_evaluate,
    map_route_effort,
    map_route_plan_mode,
)
from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state
from deepworkflow.shared.workflow_log import wrap_node, wrap_route


def build_file_batch_workflow(checkpointer: Any = None) -> Any:
    """Build and compile the file_batch_workflow LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for crash recovery.

    Returns:
        Compiled LangGraph graph.

    """
    builder = StateGraph(file_batch_workflow_state)

    # === Map: setup ===
    builder.add_node(
        "map_resolve_step",
        wrap_node("map_resolve_step", map_resolve_step, log_post_fn=_log_map_resolve_post),
    )
    builder.add_node(
        "map_effort_step",
        wrap_node("map_effort_step", map_effort_step),
    )
    builder.add_node(
        "map_effort_analyze_agent",
        wrap_node(
            "map_effort_analyze_agent",
            map_effort_analyze_agent,
            log_post_fn=_log_map_effort_analyze_post,
        ),
    )

    # === Map: plan ===
    builder.add_node(
        "map_plan_agent",
        wrap_node(
            "map_plan_agent",
            map_plan_agent,
            log_pre_fn=_log_map_plan_pre,
            log_post_fn=_log_map_plan_post,
        ),
    )
    builder.add_node(
        "map_plan_step",
        wrap_node("map_plan_step", map_plan_step, log_post_fn=_log_map_plan_post),
    )
    builder.add_node(
        "map_plan_validate_step",
        wrap_node("map_plan_validate_step", map_plan_validate_step),
    )

    # === Map: evaluate ===
    builder.add_node(
        "map_evaluate_agent",
        wrap_node(
            "map_evaluate_agent",
            map_evaluate_agent,
            log_post_fn=_log_map_evaluate_post,
        ),
    )
    builder.add_node(
        "map_evaluate_retry_step",
        wrap_node("map_evaluate_retry_step", map_evaluate_retry_step),
    )

    # === Batch: plan+execute (per batch) ===
    builder.add_node(
        "batch_plan_agent",
        wrap_node(
            "batch_plan_agent",
            batch_plan_agent,
            log_pre_fn=_log_batch_plan_pre,
            log_post_fn=_log_batch_plan_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "batch_plan_skip_step",
        wrap_node("batch_plan_skip_step", batch_plan_skip_step, show_batch_index=True),
    )
    builder.add_node(
        "batch_reflect_skip_step",
        wrap_node("batch_reflect_skip_step", batch_reflect_skip_step, show_batch_index=True),
    )
    builder.add_node(
        "batch_execute_agent",
        wrap_node(
            "batch_execute_agent",
            batch_execute_agent,
            log_pre_fn=_log_batch_execute_pre,
            log_post_fn=_log_batch_execute_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "batch_reflect_agent",
        wrap_node(
            "batch_reflect_agent",
            batch_reflect_agent,
            log_post_fn=_log_batch_reflect_post,
            show_batch_index=True,
        ),
    )

    # === Batch: evaluate convergence ===
    builder.add_node(
        "batch_evaluate_convergence_agent",
        wrap_node(
            "batch_evaluate_convergence_agent",
            batch_evaluate_convergence_agent,
            log_post_fn=_log_batch_evaluate_convergence_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "batch_convergence_repeat_step",
        wrap_node("batch_convergence_repeat_step", batch_convergence_repeat_step, stat="batch_convergence_retry"),
    )

    # === Batch: evaluate quality ===
    builder.add_node(
        "batch_evaluate_quality_agent",
        wrap_node(
            "batch_evaluate_quality_agent",
            batch_evaluate_quality_agent,
            log_pre_fn=_log_batch_evaluate_quality_pre,
            log_post_fn=_log_batch_evaluate_quality_post,
            show_batch_index=True,
        ),
    )
    builder.add_node(
        "batch_evaluate_quality_skip_step",
        wrap_node("batch_evaluate_quality_skip_step", batch_evaluate_quality_skip_step),
    )
    builder.add_node("batch_output_record_step", wrap_node("batch_output_record_step", batch_output_record_step))
    builder.add_node(
        "batch_quality_retry_step",
        wrap_node("batch_quality_retry_step", batch_quality_retry_step),
    )
    builder.add_node(
        "batch_quality_max_retries_step",
        wrap_node("batch_quality_max_retries_step", batch_quality_max_retries_step),
    )

    # === Reduce ===
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
    builder.set_entry_point("map_resolve_step")

    # ── Map: setup edges ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "map_resolve_step",
        wrap_route("map_route_effort", map_route_effort),
        {"map_effort_step": "map_effort_step", "map_effort_analyze_agent": "map_effort_analyze_agent"},
    )
    for effort_node in ("map_effort_step", "map_effort_analyze_agent"):
        builder.add_conditional_edges(
            effort_node,
            wrap_route("map_route_plan_mode", map_route_plan_mode),
            {"map_plan_agent": "map_plan_agent", "map_plan_step": "map_plan_step"},
        )

    # ── Map: plan → validate → evaluate ─────────────────────────────────────
    for map_node in ("map_plan_agent", "map_plan_step"):
        builder.add_edge(map_node, "map_plan_validate_step")

    builder.add_conditional_edges(
        "map_plan_validate_step",
        wrap_route("map_plan_route_validate", map_plan_route_validate),
        {
            "fail_step": "fail_step",
            "map_plan_agent": "map_plan_agent",
            "map_evaluate_agent": "map_evaluate_agent",
            "batch_plan_agent": "batch_plan_agent",
            "batch_plan_skip_step": "batch_plan_skip_step",
        },
    )

    # Map: evaluate → check verdict → plan or retry
    builder.add_conditional_edges(
        "map_evaluate_agent",
        wrap_route("map_route_after_evaluate", map_route_after_evaluate),
        {
            "batch_plan_agent": "batch_plan_agent",
            "batch_plan_skip_step": "batch_plan_skip_step",
            "retry_or_fail": "map_evaluate_retry_step",
        },
    )

    # Map: retry → check retries remaining
    builder.add_conditional_edges(
        "map_evaluate_retry_step",
        wrap_route("map_evaluate_check_retries", map_evaluate_check_retries),
        {"map_plan_agent": "map_plan_agent", "fail_step": "fail_step"},
    )

    # ── Batch: plan → execute → reflect edges ─────────────────────────────────
    for plan_node in ("batch_plan_agent", "batch_plan_skip_step"):
        builder.add_edge(plan_node, "batch_execute_agent")
    builder.add_conditional_edges(
        "batch_execute_agent",
        wrap_route("batch_route_after_execute", batch_route_after_execute),
        {"batch_reflect_agent": "batch_reflect_agent", "batch_reflect_skip_step": "batch_reflect_skip_step"},
    )
    builder.add_edge("batch_reflect_skip_step", "batch_evaluate_quality_skip_step")

    # After reflect: convergence loop, evaluate quality, or skip
    builder.add_conditional_edges(
        "batch_reflect_agent",
        wrap_route("batch_route_after_reflect", batch_route_after_reflect),
        {
            "evaluate_convergence": "batch_evaluate_convergence_agent",
            "evaluate": "batch_evaluate_quality_agent",
            "skip": "batch_evaluate_quality_skip_step",
        },
    )

    # After convergence check: repeat loop, evaluate quality, or skip
    builder.add_conditional_edges(
        "batch_evaluate_convergence_agent",
        wrap_route("batch_check_convergence", batch_check_convergence),
        {
            "repeat": "batch_convergence_repeat_step",
            "evaluate": "batch_evaluate_quality_agent",
            "skip": "batch_evaluate_quality_skip_step",
        },
    )
    builder.add_conditional_edges(
        "batch_convergence_repeat_step",
        wrap_route("batch_route_plan", batch_route_plan),
        {"batch_plan_agent": "batch_plan_agent", "batch_plan_skip_step": "batch_plan_skip_step"},
    )

    builder.add_edge("batch_evaluate_quality_skip_step", "batch_output_record_step")

    # After evaluate quality: pass or retry
    builder.add_conditional_edges(
        "batch_evaluate_quality_agent",
        wrap_route("batch_check_verdict", batch_check_verdict),
        {"pass": "batch_output_record_step", "retry_or_fail": "batch_quality_retry_step"},
    )

    # Retry path
    builder.add_conditional_edges(
        "batch_quality_retry_step",
        wrap_route("batch_quality_check_retries", batch_quality_check_retries),
        {
            "batch_plan_agent": "batch_plan_agent",
            "batch_plan_skip_step": "batch_plan_skip_step",
            "max_retries_exceeded": "batch_quality_max_retries_step",
        },
    )

    # Max retries exceeded policy
    builder.add_conditional_edges(
        "batch_quality_max_retries_step",
        wrap_route("batch_quality_max_retries", batch_quality_max_retries),
        {"fail_step": "fail_step", "batch_output_record_step": "batch_output_record_step"},
    )

    # After recording output: next batch or consolidate
    builder.add_conditional_edges(
        "batch_output_record_step",
        wrap_route("batch_route_next", batch_route_next),
        {
            "batch_plan_agent": "batch_plan_agent",
            "batch_plan_skip_step": "batch_plan_skip_step",
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
