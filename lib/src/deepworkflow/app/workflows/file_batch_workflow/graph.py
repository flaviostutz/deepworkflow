from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.file_batch_workflow.nodes.check_max_retries_policy_step import (
    check_max_retries_policy_step,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_agent import evaluate_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent import evaluate_map_batches_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.execute_batch_agent import execute_batch_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
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
    check_map_retries,
    check_map_verdict,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
    route_batch_judge,
    route_map_judge,
)
from deepworkflow.app.workflows.file_batch_workflow.states import file_batch_workflow_state


def build_file_batch_workflow(checkpointer: Any = None) -> Any:
    """Build and compile the file_batch_workflow LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for crash recovery.

    Returns:
        Compiled LangGraph graph.

    """
    builder = StateGraph(file_batch_workflow_state)

    # === Phase 1: Map ===
    builder.add_node("resolve_globs_step", resolve_globs_step)
    builder.add_node("map_batches_agent", map_batches_agent)
    builder.add_node("evaluate_map_batches_agent", evaluate_map_batches_agent)
    builder.add_node("map_increment_retry_step", map_increment_retry_step)

    # === Phase 2: Execute (per batch) ===
    builder.add_node("plan_batch_agent", plan_batch_agent)
    builder.add_node("execute_batch_agent", execute_batch_agent)
    builder.add_node("reflect_batch_agent", reflect_batch_agent)
    builder.add_node("evaluate_batch_agent", evaluate_batch_agent)
    builder.add_node("skip_judge_step", skip_judge_step)
    builder.add_node("record_output_step", record_output_step)
    builder.add_node("increment_retry_step", increment_retry_step)

    # === Phase 3: Reduce ===
    builder.add_node("reduce_consolidate_agent", reduce_consolidate_agent)
    builder.add_node("fail_step", fail_step)

    # --- Entry ---
    builder.set_entry_point("resolve_globs_step")

    # --- Map phase edges ---
    builder.add_edge("resolve_globs_step", "map_batches_agent")

    # After map: route to judge or skip based on judge_skip
    builder.add_conditional_edges(
        "map_batches_agent",
        route_map_judge,
        {"evaluate": "evaluate_map_batches_agent", "skip": "plan_batch_agent"},
    )

    # After map evaluation: pass → plan_batch_agent (first batch), retry → map_increment_retry_step
    builder.add_conditional_edges(
        "evaluate_map_batches_agent",
        check_map_verdict,
        {"pass": "plan_batch_agent", "retry_or_fail": "map_increment_retry_step"},
    )

    # Map retry: check retries remaining
    builder.add_conditional_edges(
        "map_increment_retry_step",
        check_map_retries,
        {"map_batches_agent": "map_batches_agent", "fail_step": "fail_step"},
    )

    # --- Execute phase edges (per batch) ---
    builder.add_edge("plan_batch_agent", "execute_batch_agent")
    builder.add_edge("execute_batch_agent", "reflect_batch_agent")

    # After reflect: route to judge or skip based on judge_skip
    builder.add_conditional_edges(
        "reflect_batch_agent",
        route_batch_judge,
        {"evaluate": "evaluate_batch_agent", "skip": "skip_judge_step"},
    )
    builder.add_edge("skip_judge_step", "record_output_step")

    # After task evaluation: check verdict
    builder.add_conditional_edges(
        "evaluate_batch_agent",
        check_verdict,
        {"pass": "record_output_step", "retry_or_fail": "increment_retry_step"},
    )

    # Retry path
    builder.add_conditional_edges(
        "increment_retry_step",
        check_retries,
        {"plan_batch_agent": "plan_batch_agent", "max_retries_exceeded": "check_max_retries_policy_step"},
    )

    # Max retries exceeded policy
    builder.add_node("check_max_retries_policy_step", check_max_retries_policy_step)
    builder.add_conditional_edges(
        "check_max_retries_policy_step",
        check_max_retries_policy,
        {"fail_step": "fail_step", "record_output_step": "record_output_step"},
    )

    # After recording output: next batch or consolidate
    builder.add_conditional_edges(
        "record_output_step",
        next_batch,
        {"plan_batch_agent": "plan_batch_agent", "reduce_consolidate_agent": "reduce_consolidate_agent"},
    )

    # --- Terminal nodes ---
    builder.add_edge("reduce_consolidate_agent", END)
    builder.add_edge("fail_step", END)

    # Compile
    return builder.compile(checkpointer=checkpointer)


# Default workflow instance (without checkpointer)
file_batch_workflow = build_file_batch_workflow()
