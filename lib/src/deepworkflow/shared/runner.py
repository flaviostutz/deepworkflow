from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import mlflow
from langgraph.checkpoint.memory import MemorySaver

from deepworkflow.app.workflows.file_batch_workflow.graph import build_file_batch_workflow
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import WorkflowLogLevel, WorkflowResult
from deepworkflow.shared.workflow_log import WorkflowStatsCallback, new_run_stats, print_summary


def _console_span_printer(span: Any) -> None:
    """Print a completed MLflow span to stdout as JSON."""
    print(json.dumps(span.to_dict(), indent=2))  # noqa: T201


def run_workflow(  # noqa: C901, PLR0912, PLR0915
    config: DeepWorkflowConfig | None = None,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | None = None,
    clone_workspace_dir: str | None = None,
) -> WorkflowResult:
    """Run the deepworkflow graph with optional checkpointing for crash recovery.

    Args:
        config: Workflow configuration. Required for new runs; optional when resuming.
        thread_id: Thread ID for checkpointing. If provided with checkpoint_dir,
                   the workflow can be resumed from the last checkpoint on crash.
        checkpoint_dir: Path to SQLite checkpoint database directory.
                       If None, uses in-memory checkpointer (no persistence).
        clone_workspace_dir: If set, copy the workspace to this directory before running.
                             Agents will use the clone; the source workspace stays untouched.
                             Fails with ValueError if the directory already exists.

    Returns:
        WorkflowResult with thread_id, output, and status.

    """
    if clone_workspace_dir is not None and config is not None:
        clone_path = Path(clone_workspace_dir)
        if clone_path.exists():
            msg = f"clone_workspace_dir already exists: {clone_workspace_dir}"
            raise ValueError(msg)
        shutil.copytree(config.workspace_dir, clone_workspace_dir)
        config = dataclasses.replace(config, workspace_dir=str(clone_path.resolve()))
    elif config is not None:
        config = dataclasses.replace(config, workspace_dir=str(Path(config.workspace_dir).resolve()))
    log_level = config.log_level if config is not None else WorkflowLogLevel.NONE

    checkpointer = None
    tracking_uri = config.mlflow_tracking_uri if config is not None else "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.langchain.autolog()

    if log_level == WorkflowLogLevel.TRACE:
        mlflow.tracing.configure(span_processors=[_console_span_printer])

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
    graph = build_file_batch_workflow(checkpointer=checkpointer)

    invoke_config: dict = {"configurable": {"thread_id": resolved_thread_id}}

    # Set up stats + callback for INFO / DEBUG / TRACE levels
    stats = None
    if log_level in (WorkflowLogLevel.INFO, WorkflowLogLevel.DEBUG, WorkflowLogLevel.TRACE):
        stats = new_run_stats()
        invoke_config["callbacks"] = [WorkflowStatsCallback(log_level)]

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"deepworkflow-{resolved_thread_id[:8]}", nested=nested):
        if config is not None:
            mlflow.log_param(
                "batch_evaluate_min",
                config.effort.batch_evaluate_min.name if config.effort.batch_evaluate_min else "N/A",
            )
            mlflow.log_param("effort_type", config.effort.type)
            mlflow.log_param("effort_level", config.effort.level)
            mlflow.log_param("write_option", config.workspace_write_option.value)

        if log_level != WorkflowLogLevel.NONE:
            print("START")  # noqa: T201

        # If config provided, start/restart with initial state
        if config is not None:
            initial_state = {"config": config}
            result = graph.invoke(initial_state, config=invoke_config)
        else:
            # Resume from checkpoint (no initial state)
            result = graph.invoke(None, config=invoke_config)

        if result.get("error"):
            mlflow.log_metric("success", 0)
            workflow_result = WorkflowResult(
                thread_id=resolved_thread_id,
                output=result.get("error", ""),
                status="failed",
            )
        else:
            mlflow.log_metric("success", 1)
            mlflow.log_metric("output_length", len(result.get("reduce_output", "")))

            map_evaluate_level = result.get("map_evaluate_level")
            if map_evaluate_level is not None:
                mlflow.log_metric("map_evaluate_level", map_evaluate_level.value)

            mlflow.log_metric("batch_quality_retries", result.get("batch_quality_retry_count", 0))
            mlflow.log_metric("batch_convergence_retries", result.get("batch_convergence_repeat_count", 0))

            batch_results = result.get("batch_results") or []
            if batch_results:
                worst = min(b.evaluate_level.value for b in batch_results)
                mlflow.log_metric("worst_batch_quality_level", worst)
                for i, b in enumerate(batch_results):
                    mlflow.log_metric(f"batch_{i}_quality_level", b.evaluate_level.value)

            workflow_result = WorkflowResult(
                thread_id=resolved_thread_id,
                output=result.get("reduce_output", ""),
                status="completed",
            )

        if log_level != WorkflowLogLevel.NONE:
            print("END")  # noqa: T201

        if stats is not None and log_level in (WorkflowLogLevel.INFO, WorkflowLogLevel.DEBUG, WorkflowLogLevel.TRACE):
            print_summary(stats, result, workflow_result)

        return workflow_result
