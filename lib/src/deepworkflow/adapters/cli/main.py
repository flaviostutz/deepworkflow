from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.runner import run_workflow
from deepworkflow.shared.types import EffortConfig, WorkflowLogLevel, WriteOption

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for deepworkflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="deepworkflow",
        description="Run a map-plan-execute-evaluate-reduce workflow on files",
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
    parser.add_argument(
        "--loglevel",
        choices=["none", "trace", "info"],
        default=None,
        help="Console log verbosity (none/trace/info). Overrides log_level in config.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to write the consolidated workflow output to (also printed to stdout).",
    )
    parser.add_argument(
        "--clone-workspace-dir",
        default=None,
        help=(
            "Copy the workspace to this directory before running. "
            "Agents will use the clone; the source workspace stays untouched. "
            "Fails if the directory already exists."
        ),
    )

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

    config = _build_config(raw, model_override=args.model, log_level_override=args.loglevel)

    try:
        result = run_workflow(
            config,
            thread_id=args.thread_id,
            checkpoint_dir=args.checkpoint_dir,
            clone_workspace_dir=args.clone_workspace_dir,
        )
        if result.status == "failed":
            print(f"Workflow failed: {result.output}", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        print(result.output)  # noqa: T201
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.output)
        if args.thread_id or args.checkpoint_dir:
            print(f"Thread ID: {result.thread_id}", file=sys.stderr)  # noqa: T201
    except (RuntimeError, ValueError) as e:
        print(f"Workflow failed: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


def _build_config(
    raw: dict,
    *,
    model_override: str | None = None,
    log_level_override: str | None = None,
) -> DeepWorkflowConfig:
    """Build DeepWorkflowConfig from raw YAML dict."""
    model_factory = _make_model_factory(raw, model_override=model_override)

    task_files_raw = raw.get("task_files")

    # CLI flag overrides YAML value; YAML defaults to "info"
    raw_log_level = log_level_override or raw.get("log_level", "info")
    log_level = WorkflowLogLevel(raw_log_level.lower())

    return DeepWorkflowConfig(
        workspace_dir=raw["workspace_dir"],
        task_instructions=raw["task_instructions"],
        model=model_factory,
        workspace_write_option=WriteOption(raw["workspace_write_option"]),
        effort=_build_effort_config(raw),
        task_files=task_files_raw,
        max_failure_retries=raw.get("max_failure_retries", 0),
        mlflow_tracking_uri=raw.get("mlflow_tracking_uri", "sqlite:///mlflow.db"),
        log_level=log_level,
    )


def _build_effort_config(raw: dict) -> EffortConfig:
    """Build an EffortConfig from the raw YAML dict.

    YAML format::

        effort:           # optional block; defaults to EffortConfig() = level=3, type=custom
          level: 3        # 1-10 preset (used when type=custom)
          type: custom    # "custom" (default) or "auto"
          # Any EffortConfig detail field can be added to override the level default:
          map_batches_mode: agent
          max_batches: null
          max_files_per_batch: null
          evaluate_map_max_retries: 2
          skip_batch_plan: false
          evaluate_batch_convergence_max_retries: 1
          evaluate_batch_quality_max_retries: 2
          consolidate_mode: agent
          evaluate_quality_min: WARNING
          evaluate_quality_on_max_retries: continue
          evaluate_quality_batch_instructions: null

    When ``type: auto``, a specialized agent analyses ``task_instructions`` and the
    workspace files to derive the optimal effort level and quality gate settings.
    Quality-gate instructions in the prompt (MUST / SHOULD / COULD language) are
    automatically used to configure evaluation criteria.  No other fields may be
    set alongside ``type: auto``.
    """
    from deepworkflow.shared.types import JudgeLevel, OnMaxRetriesExceeded

    ec_raw = raw.get("effort")

    if ec_raw is None:
        # Default: EffortConfig() = level=3, type=custom, all fields resolved from level
        return EffortConfig()

    if not isinstance(ec_raw, dict):
        msg = "'effort' in config file must be a dict, e.g.:\n  effort:\n    level: 3"
        raise TypeError(msg)

    effort_type: str = ec_raw.get("type", "custom").lower()
    level = int(ec_raw.get("level", 3))

    # Collect only the keys explicitly present in the YAML (excluding type/level)
    kwargs: dict = {"level": level, "type": effort_type}

    _UNSET_STR = "__MISSING__"

    # Nullable int fields (None means "no limit" not "unset", so explicit null is valid)
    for key in ("max_batches", "max_files_per_batch"):
        if key in ec_raw:
            kwargs[key] = ec_raw[key]  # None or int

    if "map_batches_mode" in ec_raw:
        kwargs["map_batches_mode"] = ec_raw["map_batches_mode"]
    if "consolidate_mode" in ec_raw:
        kwargs["consolidate_mode"] = ec_raw["consolidate_mode"]
    if "evaluate_map_max_retries" in ec_raw:
        kwargs["evaluate_map_max_retries"] = int(ec_raw["evaluate_map_max_retries"])
    if "skip_batch_plan" in ec_raw:
        kwargs["skip_batch_plan"] = bool(ec_raw["skip_batch_plan"])
    if "evaluate_batch_convergence_max_retries" in ec_raw:
        kwargs["evaluate_batch_convergence_max_retries"] = int(ec_raw["evaluate_batch_convergence_max_retries"])
    if "evaluate_batch_quality_max_retries" in ec_raw:
        kwargs["evaluate_batch_quality_max_retries"] = int(ec_raw["evaluate_batch_quality_max_retries"])
    if "evaluate_quality_min" in ec_raw:
        kwargs["evaluate_quality_min"] = JudgeLevel[ec_raw["evaluate_quality_min"].upper()]
    if "evaluate_quality_on_max_retries" in ec_raw:
        kwargs["evaluate_quality_on_max_retries"] = OnMaxRetriesExceeded(
            ec_raw["evaluate_quality_on_max_retries"].lower()
        )
    if "evaluate_quality_batch_instructions" in ec_raw:
        kwargs["evaluate_quality_batch_instructions"] = ec_raw["evaluate_quality_batch_instructions"]

    return EffortConfig(**kwargs)


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
            "e.g.:\n  model:\n    model_name: gpt-4o\n    model_provider: openai"
        )
        raise TypeError(_model_type_error)

    # Remap model_name -> model for init_chat_model compatibility
    if "model_name" in model_dict:
        model_dict = {"model": model_dict["model_name"], **{k: v for k, v in model_dict.items() if k != "model_name"}}

    # Env-var keys serve as fallback; explicit api_key in the config dict takes precedence.
    merged = {**env_api_keys, **model_dict}

    api_key = merged.get("api_key") or next(
        (
            merged[k]
            for k in ("openai_api_key", "azure_openai_api_key", "anthropic_api_key", "google_api_key")
            if k in merged
        ),
        None,
    )
    if not api_key:
        msg = (
            "No API key found in model configuration. Set the appropriate environment variable"
            " (e.g. OPENAI_API_KEY, AZURE_OPENAI_API_KEY) or add 'api_key' to the model config."
        )
        raise ValueError(msg)
    logger.info("Using API key starting with: %s...", api_key[:8])

    def factory_config(_agent_name: str) -> BaseChatModel:
        return init_chat_model(**merged)

    return factory_config


if __name__ == "__main__":
    main()
