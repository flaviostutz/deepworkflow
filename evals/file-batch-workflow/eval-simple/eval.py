"""Simple evaluation: single instruction, few files, read-only mode."""

from __future__ import annotations

import getpass
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import keyring
import mlflow

from deepworkflow import DeepWorkflowConfig, EffortConfig, run_workflow
from deepworkflow.shared.types import JudgeLevel, OnMaxRetriesExceeded, WorkflowLogLevel, WriteOption

mlflow.langchain.autolog()

EVAL_WORKSPACE_DIR = "dataset/data"
EVAL_CLONE_DIR = ".workspace/dataset"
EVAL_EXPECTED_OUTPUT_PATH = Path("dataset/expected_output.jsonl")
EVAL_MIN_SIMILARITY = 0.5

EVAL_KEYCHAIN_SERVICE = "azure-openai/dev-api-key"


def _get_api_key() -> str:
    """Fetch API key from keychain (local) or env var (CI/cloud)."""
    key = keyring.get_password(EVAL_KEYCHAIN_SERVICE, getpass.getuser())
    if key:
        return key
    key = os.environ.get("EVAL_API_KEY")
    if key:
        return key
    raise RuntimeError(
        f"Secret 'dev-api-key' not found in keychain service '{EVAL_KEYCHAIN_SERVICE}'"
        " or environment variable 'EVAL_API_KEY'"
    )


def _model_factory(agent_name: str):  # noqa: ARG001
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=os.environ["DEEPWORKFLOW_MODEL"],
        model_provider=os.environ["DEEPWORKFLOW_MODEL_PROVIDER"],
        api_key=_get_api_key(),
        azure_endpoint=os.environ["DEEPWORKFLOW_AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["DEEPWORKFLOW_API_VERSION"],
    )


def _load_dataset() -> tuple[str, str]:
    """Load task_instructions and expected output from the JSONL dataset."""
    record = json.loads(EVAL_EXPECTED_OUTPUT_PATH.read_text().strip().splitlines()[0])
    return record["input"]["task_instructions"], record["expected_output"]


def _build_config(task_instructions: str) -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir=EVAL_WORKSPACE_DIR,
        task_instructions=task_instructions,
        model=_model_factory,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(
            map_batches_mode="static",
            max_batches=1,
            max_files_per_batch=None,
            evaluate_map_max_retries=0,
            skip_batch_plan=False,
            evaluate_batch_convergence_max_retries=5,
            evaluate_batch_quality_max_retries=5,
            consolidate_mode="static",
            evaluate_quality_min=JudgeLevel.INFO,
            evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
            evaluate_quality_batch_instructions=None,
        ),
        task_files=["**/*.py"],
        log_level=WorkflowLogLevel.DEBUG,
    )


def _keyword_similarity(actual: str, expected: str) -> float:
    """Compute similarity as fraction of expected keywords found in actual output."""
    expected_keywords = [kw.strip().lower() for kw in expected.replace(";", ",").split(",") if kw.strip()]
    if not expected_keywords:
        return 0.0
    actual_lower = actual.lower()
    matched = sum(1 for kw in expected_keywords if kw in actual_lower)
    return matched / len(expected_keywords)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _write_eval_report(run: Any, similarity: float, passed: bool) -> None:  # noqa: FBT001
    """Write eval-report.md programmatically from measured metrics."""
    n = 1
    accuracy = 1.0 if passed else 0.0
    f1 = accuracy
    ci_low, ci_high = _wilson_ci(accuracy, n)
    acc_status = "✓ PASS" if passed else "✗ FAIL"
    overall = "PASS" if passed else "FAIL"
    run_id = run.info.run_id if run is not None else "N/A"

    lines = [
        "# Eval Report: eval-simple",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Dataset:** dataset/",
        "**Script:** eval-simple.py",
        f"**Thresholds:** similarity ≥ {EVAL_MIN_SIMILARITY}",
        "",
        "## Overall Results",
        "",
        "| Metric    | Value | 95% CI               | Threshold | Status      |",
        "|-----------|-------|----------------------|-----------|-------------|",
        f"| Accuracy  | {accuracy:.2f}  | [{ci_low:.2f}, {ci_high:.2f}] | ≥ {EVAL_MIN_SIMILARITY}  | {acc_status} |",
        f"| F1 Score  | {f1:.2f}  | —                    | ≥ {EVAL_MIN_SIMILARITY}  | {acc_status} |",
        f"| Precision | {accuracy:.2f}  | —                    | —         | —           |",
        f"| Recall    | {accuracy:.2f}  | —                    | —         | —           |",
        f"| Samples   | {n}   | —                    | —         | —           |",
        "",
        f"**Overall: {overall}**",
        "",
        "## Per-item Results",
        "",
        "| ID  | Input Summary          | Expected           | Actual | Correct |",
        "|-----|------------------------|--------------------|--------|---------|",
        f"| 001 | dataset/data (*.py)    | similarity ≥ {EVAL_MIN_SIMILARITY} | {similarity:.2f}   | {'✓' if passed else '✗'}       |",
        "",
        "## Notes",
        "",
        f"- Similarity score: {similarity:.4f} (threshold: {EVAL_MIN_SIMILARITY})",
        f"- MLflow run ID: {run_id} — view with `mlflow ui`",
        "",
    ]
    Path("report.md").write_text("\n".join(lines), encoding="utf-8")


def run_eval() -> None:
    """Run the simple evaluation slice."""
    mlflow.set_experiment("deepworkflow-simple")
    task_instructions, expected_output = _load_dataset()
    config = _build_config(task_instructions)

    similarity = 0.0
    passed = False

    with mlflow.start_run(run_name="simple-eval") as run:
        mlflow.log_param("task_instructions", config.task_instructions)
        mlflow.log_param("evaluate_quality_min", config.effort.evaluate_quality_min.name)
        mlflow.log_param("write_option", config.workspace_write_option.value)

        try:
            result = run_workflow(config, clone_workspace_dir=EVAL_CLONE_DIR)
            similarity = _keyword_similarity(result.output, expected_output)
            passed = similarity >= EVAL_MIN_SIMILARITY
            mlflow.log_metric("success", 1)
            mlflow.log_metric("output_length", len(result.output))
            mlflow.log_metric("similarity_score", similarity)
            mlflow.log_metric("output_matches_expected", 1 if passed else 0)
        except RuntimeError as e:
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))
            _write_eval_report(run, similarity, passed)
            raise SystemExit(1) from e

        _write_eval_report(run, similarity, passed)
        if not passed:
            raise SystemExit(f"Eval failed: similarity {similarity:.2f} < {EVAL_MIN_SIMILARITY}")


if __name__ == "__main__":
    run_eval()
