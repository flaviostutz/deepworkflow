from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from deepworkflow.shared.config import WorkflowConfig
from deepworkflow.shared.runner import run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def main() -> None:
    """CLI entry point for deepworkflow."""
    parser = argparse.ArgumentParser(
        prog="deepworkflow",
        description="Run a map-plan-execute-judge-reduce workflow on files",
    )
    parser.add_argument("--config", "-c", default="deepworkflow.yml", help="Path to YAML configuration file (default: deepworkflow.yml)")
    parser.add_argument("--model", "-m", help="Override model (e.g. openai:gpt-4o)")
    parser.add_argument("--thread-id", help="Thread ID for checkpoint resume")
    parser.add_argument("--checkpoint-dir", help="Directory for SQLite checkpoint database")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    config = _build_config(raw, model_override=args.model)

    try:
        result = run_workflow(config, thread_id=args.thread_id, checkpoint_dir=args.checkpoint_dir)
        if result.status == "failed":
            print(f"Workflow failed: {result.output}", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        print(result.output)  # noqa: T201
        if args.thread_id or args.checkpoint_dir:
            print(f"Thread ID: {result.thread_id}", file=sys.stderr)  # noqa: T201
    except RuntimeError as e:
        print(f"Workflow failed: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


def _build_config(raw: dict, *, model_override: str | None = None) -> WorkflowConfig:
    """Build WorkflowConfig from raw YAML dict."""
    return WorkflowConfig(
        workspace_dir=raw["workspace_dir"],
        task_instructions=raw["task_instructions"],
        task_files=raw["task_files"],
        task_files_write_option=WriteOption(raw["task_files_write_option"]),
        judge_minimum=JudgeVerdict[raw["judge_minimum"].upper()],
        judge_max_retries=raw["judge_max_retries"],
        on_max_retries_exceeded=OnMaxRetriesExceeded(raw["on_max_retries_exceeded"]),
        task_files_batch_size=raw.get("task_files_batch_size"),
        judge_instructions=raw.get("judge_instructions"),
        max_failure_retries=raw.get("max_failure_retries", 0),
        model=model_override or raw.get("model", "openai:gpt-4o"),
    )


if __name__ == "__main__":
    main()
