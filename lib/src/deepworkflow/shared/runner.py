from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver

from deepworkflow.app.workflows.deepworkflow.graph import build_graph
from deepworkflow.shared.config import WorkflowConfig
from deepworkflow.shared.types import WorkflowResult


def run_workflow(
    config: WorkflowConfig | None = None,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | None = None,
) -> WorkflowResult:
    """Run the deepworkflow graph with optional checkpointing for crash recovery.

    Args:
        config: Workflow configuration. Required for new runs; optional when resuming.
        thread_id: Thread ID for checkpointing. If provided with checkpoint_dir,
                   the workflow can be resumed from the last checkpoint on crash.
        checkpoint_dir: Path to SQLite checkpoint database directory.
                       If None, uses in-memory checkpointer (no persistence).

    Returns:
        WorkflowResult with thread_id, output, and status.

    """
    checkpointer = None
    if checkpoint_dir:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = f"{checkpoint_dir}/checkpoints.db"
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except ImportError:
            # Fall back to memory saver if sqlite not available
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    resolved_thread_id = thread_id or str(uuid.uuid4())
    graph = build_graph(checkpointer=checkpointer)

    invoke_config = {"configurable": {"thread_id": resolved_thread_id}}

    # If config provided, start/restart with initial state
    if config is not None:
        initial_state = {"config": config}
        result = graph.invoke(initial_state, config=invoke_config)
    else:
        # Resume from checkpoint (no initial state)
        result = graph.invoke(None, config=invoke_config)

    if result.get("error"):
        return WorkflowResult(
            thread_id=resolved_thread_id,
            output=result.get("error", ""),
            status="failed",
        )

    return WorkflowResult(
        thread_id=resolved_thread_id,
        output=result.get("workflow_output", ""),
        status="completed",
    )
