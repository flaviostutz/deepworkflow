"""Simple evaluation: single instruction, few files, read-only mode."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import keyring
import mlflow

from deepworkflow import DeepWorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

mlflow.langchain.autolog()

EVAL_WORKSPACE_DIR = "dataset_simple/workspace"
EVAL_EXPECTED_OUTPUT_PATH = Path("dataset_simple/expected_output.jsonl")
EVAL_MIN_SIMILARITY = 0.5

EVAL_MODEL = "gpt-4o"
EVAL_MODEL_PROVIDER = "azure_openai"
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
        model=EVAL_MODEL,
        model_provider=EVAL_MODEL_PROVIDER,
        api_key=_get_api_key(),
        azure_endpoint=os.environ["EVAL_AZURE_ENDPOINT"],
        api_version=os.environ["EVAL_API_VERSION"],
    )


CONFIG = DeepWorkflowConfig(
    workspace_dir=EVAL_WORKSPACE_DIR,
    task_instructions="Analyze each file and report any potential bugs or issues.",
    model=_model_factory,
    workspace_write_option=WriteOption.READ_ONLY,
    task_files=["**/*.py"],
    judge_min=JudgeVerdict.WARNING,
    judge_max_retries=1,
    judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
)


def _load_expected_output() -> str:
    """Load the expected output from the JSONL dataset."""
    record = json.loads(EVAL_EXPECTED_OUTPUT_PATH.read_text().strip().splitlines()[0])
    return record["expected_output"]


def _keyword_similarity(actual: str, expected: str) -> float:
    """Compute similarity as fraction of expected keywords found in actual output."""
    expected_keywords = [kw.strip().lower() for kw in expected.replace(";", ",").split(",") if kw.strip()]
    if not expected_keywords:
        return 0.0
    actual_lower = actual.lower()
    matched = sum(1 for kw in expected_keywords if kw in actual_lower)
    return matched / len(expected_keywords)


def run_eval() -> None:
    """Run the simple evaluation slice."""
    mlflow.set_experiment("deepworkflow-simple")
    expected_output = _load_expected_output()

    with mlflow.start_run(run_name="simple-eval"):
        mlflow.log_param("task_instructions", CONFIG.task_instructions)
        mlflow.log_param("judge_min", CONFIG.judge_min.name)
        mlflow.log_param("write_option", CONFIG.workspace_write_option.value)

        try:
            result = run_workflow(CONFIG)
            similarity = _keyword_similarity(result.output, expected_output)
            mlflow.log_metric("success", 1)
            mlflow.log_metric("output_length", len(result.output))
            mlflow.log_metric("similarity_score", similarity)
            mlflow.log_metric("output_matches_expected", 1 if similarity >= EVAL_MIN_SIMILARITY else 0)
            if similarity < EVAL_MIN_SIMILARITY:
                raise SystemExit(f"Eval failed: similarity {similarity:.2f} < {EVAL_MIN_SIMILARITY}")
        except RuntimeError as e:
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))
            # Workflow crashed; exit non-zero so CI pipelines detect the failure.
            raise SystemExit(1) from e


if __name__ == "__main__":
    run_eval()
