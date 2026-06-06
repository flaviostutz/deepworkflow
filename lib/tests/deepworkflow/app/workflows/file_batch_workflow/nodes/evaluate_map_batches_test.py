from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

if TYPE_CHECKING:
    from pathlib import Path

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent import (
    _algorithmic_map_checks,
    evaluate_map_batches_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(workspace_dir: str) -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir=workspace_dir,
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with a few real files."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.py").write_text("")
    return tmp_path


class TestAlgorithmicMapChecks:
    def test_all_good(self, workspace: Path) -> None:
        batches = [
            BatchDefinition(batch_files=[str(workspace / "a.py"), str(workspace / "b.py")]),
            BatchDefinition(batch_files=[str(workspace / "c.py")]),
        ]
        task_files = [str(workspace / "a.py"), str(workspace / "b.py"), str(workspace / "c.py")]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        assert feedbacks == []

    def test_non_existent_batch_file(self, workspace: Path) -> None:
        batches = [BatchDefinition(batch_files=[str(workspace / "ghost.py")])]
        feedbacks = _algorithmic_map_checks(str(workspace), [], batches)
        assert len(feedbacks) == 1
        assert feedbacks[0].type == JudgeVerdict.ERROR
        assert "ghost.py" in feedbacks[0].description
        assert "does not exist" in feedbacks[0].description

    def test_task_file_missing_from_batches(self, workspace: Path) -> None:
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        task_files = [str(workspace / "a.py"), str(workspace / "b.py")]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        errors = [f for f in feedbacks if f.type == JudgeVerdict.ERROR]
        assert any("b.py" in f.description and "not assigned" in f.description for f in errors)

    def test_task_file_in_multiple_batches(self, workspace: Path) -> None:
        dup = str(workspace / "a.py")
        batches = [
            BatchDefinition(batch_files=[dup]),
            BatchDefinition(batch_files=[dup, str(workspace / "b.py")]),
        ]
        task_files = [str(workspace / "a.py"), str(workspace / "b.py")]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        errors = [f for f in feedbacks if f.type == JudgeVerdict.ERROR]
        assert any("a.py" in f.description and "2 batches" in f.description for f in errors)

    def test_invented_file_in_batch(self, workspace: Path) -> None:
        batches = [
            BatchDefinition(batch_files=[str(workspace / "a.py"), str(workspace / "invented.py")]),
        ]
        # invented.py doesn't exist AND isn't in task_files
        task_files = [str(workspace / "a.py")]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        errors = [f for f in feedbacks if f.type == JudgeVerdict.ERROR]
        # should be flagged both for non-existence and for not being in task_files
        descriptions = " ".join(f.description for f in errors)
        assert "invented.py" in descriptions

    def test_line_range_suffix_stripped(self, workspace: Path) -> None:
        """Files with :start-end suffixes should resolve to the base file."""
        file_with_range = str(workspace / "a.py") + ":1-10"
        batches = [BatchDefinition(batch_files=[file_with_range])]
        task_files = [file_with_range]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        assert feedbacks == []

    def test_no_task_files_skips_coverage_check(self, workspace: Path) -> None:
        """When task_files is empty (agent-discovered), only existence is checked."""
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        feedbacks = _algorithmic_map_checks(str(workspace), [], batches)
        assert feedbacks == []


class TestEvaluateMapBatchesAgent:
    def test_returns_combined_verdict_with_llm_judge(self, workspace: Path, mocker) -> None:
        """Test the full evaluate_map_batches_agent including the LLM judge call."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {
                "judge_verdict": "OK",
                "judge_feedbacks": [{"file": "a.py", "type": "OK", "description": "good", "proposal": ""}],
            },
        )
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")], batch_instructions="do it")]
        state = {
            "config": _make_config(str(workspace)),
            "task_files": [str(workspace / "a.py")],
            "task_file_batches": batches,
            "task_overview": "overview",
            "consolidation_instructions": "summarize",
        }
        result = evaluate_map_batches_agent(state)
        assert result["map_judge_verdict"] == JudgeVerdict.OK
        assert len(result["map_judge_feedbacks"]) >= 1

    def test_worst_verdict_wins_when_algorithmic_check_fails(self, workspace: Path, mocker) -> None:
        """Algorithmic ERROR + LLM OK → final verdict is ERROR."""
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_map_batches_agent.create_agent",
            {"judge_verdict": "OK", "judge_feedbacks": []},
        )
        # Batch references a non-existent file → algorithmic check returns ERROR
        batches = [BatchDefinition(batch_files=[str(workspace / "ghost.py")], batch_instructions="do it")]
        state = {
            "config": _make_config(str(workspace)),
            "task_files": [],
            "task_file_batches": batches,
        }
        result = evaluate_map_batches_agent(state)
        assert result["map_judge_verdict"] == JudgeVerdict.ERROR

    def test_relative_path_resolved_against_workspace(self, workspace: Path) -> None:
        batches = [BatchDefinition(batch_files=["a.py"])]
        task_files = ["a.py"]
        feedbacks = _algorithmic_map_checks(str(workspace), task_files, batches)
        assert feedbacks == []

    def test_relative_nonexistent_path(self, workspace: Path) -> None:
        batches = [BatchDefinition(batch_files=["no_such_file.py"])]
        feedbacks = _algorithmic_map_checks(str(workspace), [], batches)
        assert len(feedbacks) == 1
        assert feedbacks[0].type == JudgeVerdict.ERROR
