from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deepworkflow.app.workflows.deepworkflow.nodes.consolidate_agent import consolidate_agent
from deepworkflow.app.workflows.deepworkflow.nodes.evaluate_map import evaluate_map
from deepworkflow.app.workflows.deepworkflow.nodes.evaluate_task_agent import evaluate_task_agent
from deepworkflow.app.workflows.deepworkflow.nodes.execute_task_step import execute_task_step
from deepworkflow.app.workflows.deepworkflow.nodes.map_batches import map_batches
from deepworkflow.app.workflows.deepworkflow.nodes.plan_step import plan_step
from deepworkflow.app.workflows.deepworkflow.nodes.reflect_task_step import reflect_task_step
from deepworkflow.app.workflows.deepworkflow.nodes.resolve_globs import resolve_globs
from deepworkflow.app.workflows.deepworkflow.routes import (
    check_map_retries,
    check_map_verdict,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
)
from deepworkflow.app.workflows.deepworkflow.states import WorkflowState
from deepworkflow.shared.types import BatchOutput


def _record_batch_output(state: WorkflowState) -> dict:
    """Record the current batch output and advance to the next batch."""
    batch_index = state["current_batch_index"]
    current_batch = state["task_file_batches"][batch_index]

    output = BatchOutput(
        task_files=current_batch.batch_files,
        judge_verdict=state["judge_verdict"],
        judge_feedbacks=state.get("judge_feedbacks", []),
        files_read=state.get("files_read", []),
        files_written=state.get("files_written", []),
        execute_output=state.get("execute_output", ""),
    )

    batch_outputs = list(state.get("batch_outputs", []))
    batch_outputs.append(output)

    return {
        "batch_outputs": batch_outputs,
        "current_batch_index": batch_index + 1,
        "retry_count": 0,
        "judge_feedbacks": [],
    }


def _increment_retry(state: WorkflowState) -> dict:
    """Increment retry count before looping back to plan."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def _map_increment_retry(state: WorkflowState) -> dict:
    """Increment map retry count before looping back to map_batches."""
    return {"map_retry_count": state.get("map_retry_count", 0) + 1}


def _fail(state: WorkflowState) -> dict:  # noqa: ARG001
    """Mark workflow as failed."""
    return {"error": "Workflow failed"}


def build_graph(checkpointer: Any = None) -> Any:
    """Build and compile the deepworkflow LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for crash recovery.

    Returns:
        Compiled LangGraph graph.

    """
    builder = StateGraph(WorkflowState)

    # === Phase 1: Map ===
    builder.add_node("resolve_globs", resolve_globs)
    builder.add_node("map_batches", map_batches)
    builder.add_node("evaluate_map", evaluate_map)
    builder.add_node("map_increment_retry", _map_increment_retry)

    # === Phase 2: Execute (per batch) ===
    builder.add_node("plan_step", plan_step)
    builder.add_node("execute_task_step", execute_task_step)
    builder.add_node("reflect_task_step", reflect_task_step)
    builder.add_node("evaluate_task_agent", evaluate_task_agent)
    builder.add_node("record_output", _record_batch_output)
    builder.add_node("increment_retry", _increment_retry)

    # === Phase 3: Reduce ===
    builder.add_node("consolidate", consolidate_agent)
    builder.add_node("fail", _fail)

    # --- Entry ---
    builder.set_entry_point("resolve_globs")

    # --- Map phase edges ---
    builder.add_edge("resolve_globs", "map_batches")
    builder.add_edge("map_batches", "evaluate_map")

    # After map evaluation: pass → plan_step (first batch), retry → map_increment_retry
    builder.add_conditional_edges(
        "evaluate_map",
        check_map_verdict,
        {"pass": "plan_step", "retry_or_fail": "map_increment_retry"},
    )

    # Map retry: check retries remaining
    builder.add_conditional_edges(
        "map_increment_retry",
        check_map_retries,
        {"map_batches": "map_batches", "fail": "fail"},
    )

    # --- Execute phase edges (per batch) ---
    builder.add_edge("plan_step", "execute_task_step")
    builder.add_edge("execute_task_step", "reflect_task_step")
    builder.add_edge("reflect_task_step", "evaluate_task_agent")

    # After task evaluation: check verdict
    builder.add_conditional_edges(
        "evaluate_task_agent",
        check_verdict,
        {"pass": "record_output", "retry_or_fail": "increment_retry"},
    )

    # Retry path
    builder.add_conditional_edges(
        "increment_retry",
        check_retries,
        {"plan_step": "plan_step", "max_retries_exceeded": "check_max_retries_policy"},
    )

    # Max retries exceeded policy
    builder.add_node("check_max_retries_policy", lambda state: {})  # noqa: ARG005
    builder.add_conditional_edges(
        "check_max_retries_policy",
        check_max_retries_policy,
        {"fail": "fail", "record_output": "record_output"},
    )

    # After recording output: next batch or consolidate
    builder.add_conditional_edges(
        "record_output",
        next_batch,
        {"plan_step": "plan_step", "consolidate": "consolidate"},
    )

    # --- Terminal nodes ---
    builder.add_edge("consolidate", END)
    builder.add_edge("fail", END)

    # Compile
    return builder.compile(checkpointer=checkpointer)


# Default graph instance (without checkpointer)
graph = build_graph()
