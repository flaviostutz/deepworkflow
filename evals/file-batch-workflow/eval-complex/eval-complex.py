"""Complex evaluation: 20-file data-processing library in 1 batch, write-mode, progress + quality loops."""

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
EVAL_MIN_SIMILARITY = 0.7

EVAL_KEYCHAIN_SERVICE = "azure-openai/dev-api-key"

TASK_INSTRUCTIONS = """\
Analyze each Python file in this data-processing library and write a code-analysis report for it.

For each source file create a report at 'reports/<source_filename_without_extension>.md'.

Each report MUST include these sections exactly:
  # <filename> Analysis
  ## Summary        — purpose and overall quality of the file
  ## Findings       — list of bugs/issues, each tagged CRITICAL / MAJOR / MINOR with the line number
  ## Recommendations — actionable improvement suggestions

Quality language:
- MUST / REQUIRED / MANDATORY -> critical, non-negotiable
- SHOULD / RECOMMENDED -> important but non-blocking
- COULD / SUGGESTED -> nice-to-have\
"""

JUDGE_BATCH_INSTRUCTIONS = """\
- A report file MUST exist at reports/<source_filename_without_extension>.md for every source file in the batch.
- Every finding MUST include a severity label (CRITICAL, MAJOR, or MINOR).
- Every finding MUST reference the specific line number where the issue occurs.
- The Summary section MUST describe the file's purpose in at least 1 sentence.
- Each file's Findings section SHOULD list at least 2 distinct issues.
- Recommendations SHOULD be actionable and reference the finding they address.\
"""


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


CONFIG = DeepWorkflowConfig(
    workspace_dir=EVAL_WORKSPACE_DIR,
    task_instructions=TASK_INSTRUCTIONS,
    model=_model_factory,
    workspace_write_option=WriteOption.WRITE_ANY,
    task_files=[
        "parsers/*.py",
        "transformers/*.py",
    ],
    effort=EffortConfig(
        map_batches_mode="static",
        max_files_per_batch=20,
        evaluate_batch_convergence_max_retries=2,
        evaluate_batch_quality_max_retries=2,
        evaluate_quality_min=JudgeLevel.WARNING,
        evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        evaluate_quality_batch_instructions=JUDGE_BATCH_INSTRUCTIONS,
    ),
    log_level=WorkflowLogLevel.DEBUG,
)


def _load_expected_output() -> dict[str, str]:
    """Load the per-file expected keyword map from the JSONL dataset."""
    record = json.loads(EVAL_EXPECTED_OUTPUT_PATH.read_text().strip().splitlines()[0])
    return record["expected_output"]


def _check_report(report_path: Path, keywords_str: str) -> float:
    """Return fraction of semicolon-separated keywords found (case-insensitive) in the report."""
    keywords = [kw.strip().lower() for kw in keywords_str.split(";") if kw.strip()]
    if not keywords:
        return 0.0
    content = report_path.read_text(encoding="utf-8").lower()
    matched = sum(1 for kw in keywords if kw in content)
    return matched / len(keywords)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _write_eval_report(
    run: Any,
    expected_output: dict[str, str],
    per_file_scores: list[float],
    avg_score: float,
    passed: bool,  # noqa: FBT001
) -> None:
    """Write eval-report.md programmatically from measured metrics."""
    n = len(per_file_scores)
    num_passing = sum(1 for s in per_file_scores if s >= EVAL_MIN_SIMILARITY)
    accuracy = num_passing / n if n > 0 else 0.0
    # For this binary evaluation all expected outputs are positive; precision=1, recall=accuracy
    f1 = (2 * accuracy) / (1 + accuracy) if accuracy > 0 else 0.0
    ci_low, ci_high = _wilson_ci(accuracy, n)
    acc_status = "✓ PASS" if passed else "✗ FAIL"
    overall = "PASS" if passed else "FAIL"
    run_id = run.info.run_id if run is not None else "N/A"

    per_item_rows = []
    stems = list(expected_output.keys())
    for idx, (stem, score) in enumerate(zip(stems, per_file_scores), start=1):
        file_passed = score >= EVAL_MIN_SIMILARITY
        per_item_rows.append(
            f"| {idx:03d} | {stem}.py              | keywords in report | {score:.2f}   | {'✓' if file_passed else '✗'}       |"
        )

    lines = [
        "# Eval Report: eval-complex",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Dataset:** dataset/",
        "**Script:** eval-complex.py",
        f"**Thresholds:** avg_keyword_score ≥ {EVAL_MIN_SIMILARITY}",
        "",
        "## Overall Results",
        "",
        "| Metric    | Value | 95% CI               | Threshold | Status      |",
        "|-----------|-------|----------------------|-----------|-------------|",
        f"| Accuracy  | {accuracy:.2f}  | [{ci_low:.2f}, {ci_high:.2f}] | ≥ {EVAL_MIN_SIMILARITY}  | {acc_status} |",
        f"| F1 Score  | {f1:.2f}  | —                    | ≥ {EVAL_MIN_SIMILARITY}  | {acc_status} |",
        f"| Precision | 1.00  | —                    | —         | —           |",
        f"| Recall    | {accuracy:.2f}  | —                    | —         | —           |",
        f"| Samples   | {n}   | —                    | —         | —           |",
        "",
        f"**Overall: {overall}**",
        "",
        "## Per-item Results",
        "",
        "| ID  | Input Summary          | Expected           | Actual | Correct |",
        "|-----|------------------------|--------------------|--------|---------|",
        *per_item_rows,
        "",
        "## Notes",
        "",
        f"- Average keyword score: {avg_score:.4f} (threshold: {EVAL_MIN_SIMILARITY})",
        f"- Files passing threshold: {num_passing}/{n}",
        f"- MLflow run ID: {run_id} — view with `mlflow ui`",
        "",
    ]
    Path("eval-complex-report.md").write_text("\n".join(lines), encoding="utf-8")


def run_eval() -> None:
    """Run the complex evaluation slice."""
    mlflow.set_experiment("deepworkflow-complex")
    expected_output = _load_expected_output()

    per_file_scores: list[float] = []
    avg_score = 0.0
    passed = False

    with mlflow.start_run(run_name="complex-eval") as run:
        mlflow.log_param("task_instructions", CONFIG.task_instructions)
        mlflow.log_param("evaluate_quality_min", CONFIG.effort.evaluate_quality_min.name)
        mlflow.log_param("write_option", CONFIG.workspace_write_option.value)
        mlflow.log_param("evaluate_batch_convergence_max_retries", CONFIG.effort.evaluate_batch_convergence_max_retries)
        mlflow.log_param("evaluate_batch_quality_max_retries", CONFIG.effort.evaluate_batch_quality_max_retries)
        mlflow.log_param("max_files_per_batch", CONFIG.effort.max_files_per_batch)

        try:
            run_workflow(CONFIG, clone_workspace_dir=EVAL_CLONE_DIR)
        except RuntimeError as e:
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))
            _write_eval_report(run, expected_output, per_file_scores, avg_score, passed)
            raise SystemExit(1) from e

        reports_dir = Path(EVAL_CLONE_DIR) / "reports"
        for stem, keywords_str in expected_output.items():
            report_path = reports_dir / f"{stem}.md"
            exists = report_path.exists()
            mlflow.log_metric(f"file_{stem}_exists", 1 if exists else 0)
            score = _check_report(report_path, keywords_str) if exists else 0.0
            mlflow.log_metric(f"file_{stem}_score", score)
            per_file_scores.append(score)

        total = len(expected_output)
        files_found = sum(1 for stem in expected_output if (Path(EVAL_CLONE_DIR) / "reports" / f"{stem}.md").exists())
        coverage = files_found / total if total > 0 else 0.0
        avg_score = sum(per_file_scores) / len(per_file_scores) if per_file_scores else 0.0
        passed = avg_score >= EVAL_MIN_SIMILARITY

        mlflow.log_metric("success", 1)
        mlflow.log_metric("report_coverage", coverage)
        mlflow.log_metric("avg_keyword_score", avg_score)
        mlflow.log_metric("output_matches_expected", 1 if passed else 0)

        _write_eval_report(run, expected_output, per_file_scores, avg_score, passed)

        if not passed:
            raise SystemExit(f"Eval failed: avg_keyword_score {avg_score:.2f} < {EVAL_MIN_SIMILARITY}")


if __name__ == "__main__":
    run_eval()
