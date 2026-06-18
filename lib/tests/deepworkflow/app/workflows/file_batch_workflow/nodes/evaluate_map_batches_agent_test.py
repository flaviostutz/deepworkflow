from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

if TYPE_CHECKING:
    from pathlib import Path

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent import (
    evaluate_map_batches_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    JudgeLevel,
    JudgeVerdict,
    OnMaxRetriesExceeded,
    WriteOption,
)


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(workspace_dir: str) -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir=workspace_dir,
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort="custom",
        effort_config=resolveEffortConfig(5),
        evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with a few real files."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.py").write_text("")
    return tmp_path


class TestEvaluateMapBatchesAgent:
    def test_returns_combined_verdict_with_llm_evaluator(self, workspace: Path, mocker) -> None:
        """Test the full evaluate_map_batches_agent including the LLM evaluator call."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {
                "verdict": "OK",
                "findings": [{"level": "OK", "title": "good plan", "reason": "", "fix": ""}],
            },
        )
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")], batch_instructions="do it")]
        state = {
            "config": _make_config(str(workspace)),
            "effort_config": resolveEffortConfig(5),
            "task_files": [str(workspace / "a.py")],
            "task_file_batches": batches,
            "task_overview": "overview",
            "consolidation_instructions": "summarize",
        }
        result = evaluate_map_batches_agent(state)
        assert result["map_evaluate_quality_verdict"] == JudgeLevel.OK
        judge = result["map_evaluate_judge_verdict"]
        assert isinstance(judge, JudgeVerdict)
        assert judge.verdict == JudgeLevel.OK
        assert len(judge.findings) >= 1
        assert judge.findings[0].title

    def test_worst_verdict_wins_when_algorithmic_check_fails(self, workspace: Path, mocker) -> None:
        """When the LLM returns ERROR the final verdict is ERROR."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {"verdict": "ERROR", "findings": [{"level": "ERROR", "title": "bad plan", "reason": "x", "fix": ""}]},
        )
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")], batch_instructions="do it")]
        state = {
            "config": _make_config(str(workspace)),
            "effort_config": resolveEffortConfig(5),
            "task_files": [],
            "task_file_batches": batches,
        }
        result = evaluate_map_batches_agent(state)
        assert result["map_evaluate_quality_verdict"] == JudgeLevel.ERROR
        judge = result["map_evaluate_judge_verdict"]
        assert isinstance(judge, JudgeVerdict)
        assert judge.verdict == JudgeLevel.ERROR
        error_findings = [f for f in judge.findings if f.level == JudgeLevel.ERROR]
        assert len(error_findings) >= 1

    def test_relative_path_resolved_against_workspace(self, workspace: Path, mocker) -> None:
        """Test that relative paths in batches work when workspace is prepended."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {"verdict": "OK", "findings": [{"level": "OK", "title": "ok", "reason": "", "fix": ""}]},
        )
        batches = [BatchDefinition(batch_files=["a.py"])]
        state = {
            "config": _make_config(str(workspace)),
            "effort_config": resolveEffortConfig(5),
            "task_files": ["a.py"],
            "task_file_batches": batches,
        }
        result = evaluate_map_batches_agent(state)
        assert result["map_evaluate_quality_verdict"] == JudgeLevel.OK

    def test_relative_nonexistent_path(self, workspace: Path, mocker) -> None:
        """LLM returning OK is fine; existence check is validate_map_batches_step's job."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {"verdict": "OK", "findings": [{"level": "OK", "title": "ok", "reason": "", "fix": ""}]},
        )
        batches = [BatchDefinition(batch_files=["no_such_file.py"])]
        state = {
            "config": _make_config(str(workspace)),
            "effort_config": resolveEffortConfig(5),
            "task_files": [],
            "task_file_batches": batches,
        }
        result = evaluate_map_batches_agent(state)
        # evaluate_map_batches_agent delegates existence checks to validate_map_batches_step
        assert result["map_evaluate_quality_verdict"] == JudgeLevel.OK

