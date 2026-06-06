from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.runner import run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel


def main() -> None:
    """CLI entry point for deepworkflow."""
    parser = argparse.ArgumentParser(
        prog="deepworkflow",
        description="Run a map-plan-execute-judge-reduce workflow on files",
    )
    parser.add_argument(
        "--config", "-c", default="deepworkflow.yml", help="Path to YAML configuration file (default: deepworkflow.yml)"
    )
    parser.add_argument(
        "--model",
        "-m",
        help=(
            "Override model for all agents.  Value is a model string accepted by "
            "langchain's init_chat_model (e.g. openai:gpt-4o).  Overrides the 'model' "
            "dict from the config file."
        ),
    )
    parser.add_argument("--thread-id", help="Thread ID for checkpoint resume")
    parser.add_argument("--checkpoint-dir", help="Directory for SQLite checkpoint database")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    with config_path.open() as f:
        content = f.read()

    import os

    content = os.path.expandvars(content)
    raw = yaml.safe_load(content)

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


def _build_config(raw: dict, *, model_override: str | None = None) -> DeepWorkflowConfig:
    """Build DeepWorkflowConfig from raw YAML dict."""
    model_factory = _make_model_factory(raw, model_override=model_override)

    judge_min_raw = raw.get("judge_min", "WARNING")
    task_files_raw = raw.get("task_files")

    return DeepWorkflowConfig(
        workspace_dir=raw["workspace_dir"],
        task_instructions=raw["task_instructions"],
        model=model_factory,
        workspace_write_option=WriteOption(raw["workspace_write_option"]),
        judge_max_retries=raw["judge_max_retries"],
        judge_on_max_retries=OnMaxRetriesExceeded(raw["judge_on_max_retries"]),
        task_files=task_files_raw,
        judge_min=JudgeVerdict[judge_min_raw.upper()],
        task_files_batch_size=raw.get("task_files_batch_size"),
        judge_batch_instructions=raw.get("judge_batch_instructions"),
        max_failure_retries=raw.get("max_failure_retries", 0),
        judge_skip=raw.get("judge_skip", False),
        mlflow_tracking_uri=raw.get("mlflow_tracking_uri", "mlruns"),
    )


def _make_model_factory(raw: dict, *, model_override: str | None = None) -> Callable[[str], BaseChatModel]:
    """Build a model factory callable from YAML config or a CLI override string.

    API keys are read from environment variables here (the CLI adapter layer) and
    injected explicitly into init_chat_model, satisfying agentme-edr-018 rule 02
    (no implicit env-var config inside the application layer).
    """
    import os

    from langchain.chat_models import init_chat_model

    # Collect API keys from the environment at the adapter entry-point.
    env_api_keys: dict[str, str] = {}
    for env_var in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_var)
        if value:
            env_api_keys[env_var.lower()] = value

    if model_override:
        # --model flag: plain "provider:model" string, same model for all agents
        def factory_override(_agent_name: str) -> BaseChatModel:
            return init_chat_model(model_override, **env_api_keys)

        return factory_override

    # Config file: model key must be a dict of init_chat_model kwargs
    model_dict: dict = raw.get("model", {})
    if not isinstance(model_dict, dict):
        _model_type_error = (
            "'model' in config file must be a dict of init_chat_model kwargs, "
            "e.g.:\n  model:\n    model: gpt-4o\n    model_provider: openai"
        )
        raise TypeError(_model_type_error)

    # Env-var keys serve as fallback; explicit api_key in the config dict takes precedence.
    merged = {**env_api_keys, **model_dict}

    def factory_config(_agent_name: str) -> BaseChatModel:
        return init_chat_model(**merged)

    return factory_config


if __name__ == "__main__":
    main()
