from __future__ import annotations

import uuid

import mlflow
from langgraph.checkpoint.memory import MemorySaver

from deepworkflow.app.workflows.file_batch_workflow.graph import build_file_batch_workflow
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
    mlflow.langchain.autolog()
    if checkpoint_dir:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver  # ty: ignore[unresolved-import]

            db_path = f"{checkpoint_dir}/checkpoints.db"
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except ImportError:
            # Fall back to memory saver if sqlite not available
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    resolved_thread_id = thread_id or str(uuid.uuid4())
    graph = build_file_batch_workflow(checkpointer=checkpointer)

    invoke_config = {"configurable": {"thread_id": resolved_thread_id}}

    with mlflow.start_run(run_name=f"deepworkflow-{resolved_thread_id[:8]}"):
        if config is not None:
            mlflow.log_param("model", config.model)
            mlflow.log_param("judge_minimum", config.judge_minimum.name)
            mlflow.log_param("judge_max_retries", config.judge_max_retries)
            mlflow.log_param("write_option", config.task_files_write_option.value)

        # If config provided, start/restart with initial state
        if config is not None:
            initial_state = {"config": config}
            result = graph.invoke(initial_state, config=invoke_config)
        else:
            # Resume from checkpoint (no initial state)
            result = graph.invoke(None, config=invoke_config)

        if result.get("error"):
            mlflow.log_metric("success", 0)
            return WorkflowResult(
                thread_id=resolved_thread_id,
                output=result.get("error", ""),
                status="failed",
            )

        mlflow.log_metric("success", 1)
        mlflow.log_metric("output_length", len(result.get("workflow_output", "")))
        return WorkflowResult(
            thread_id=resolved_thread_id,
            output=result.get("workflow_output", ""),
            status="completed",
        )
