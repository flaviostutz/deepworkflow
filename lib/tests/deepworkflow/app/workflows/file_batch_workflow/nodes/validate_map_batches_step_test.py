from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

if TYPE_CHECKING:
    from pathlib import Path

from deepworkflow.app.workflows.file_batch_workflow.nodes.validate_map_batches_step import validate_map_batches_step
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, EffortConfig, EffortConfig, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_effort(**kwargs) -> EffortConfig:
    defaults: dict = {
        "map_batches_mode": "agent",
        "max_batches": None,
        "max_files_per_batch": None,
    }
    defaults.update(kwargs)
    return EffortConfig(**defaults)


def _make_config(workspace_dir: str, **kwargs) -> DeepWorkflowConfig:
    defaults: dict = {
        "workspace_dir": workspace_dir,
        "task_instructions": "do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": EffortConfig(level=1),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


def _make_state(
    workspace: Path,
    batches: list[BatchDefinition],
    task_files: list[str] | None = None,
    effort: EffortConfig | None = None,
    task_files_exclude: list[str] | None = None,
) -> dict:
    effort = effort or _make_effort()
    config = _make_config(str(workspace), task_files_exclude=task_files_exclude)
    return {
        "config": config,
        "effort_config": effort,
        "task_files": task_files or [],
        "task_file_batches": batches,
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "c.py").write_text("c")
    return tmp_path


class TestValidateMapBatchesStepMaxBatches:
    def test_passes_when_batches_within_limit(self, workspace: Path):
        effort = _make_effort(map_batches_mode="agent", max_batches=2)
        batches = [
            BatchDefinition(batch_files=[str(workspace / "a.py")]),
            BatchDefinition(batch_files=[str(workspace / "b.py")]),
        ]
        state = _make_state(workspace, batches, effort=effort)
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_fails_when_batches_exceed_limit(self, workspace: Path):
        effort = _make_effort(map_batches_mode="agent", max_batches=1)
        batches = [
            BatchDefinition(batch_files=[str(workspace / "a.py")]),
            BatchDefinition(batch_files=[str(workspace / "b.py")]),
        ]
        state = _make_state(workspace, batches, effort=effort)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "max_batches" in result["error"]


class TestValidateMapBatchesStepMaxFilesPerBatch:
    def test_passes_when_within_limit(self, workspace: Path):
        effort = _make_effort(map_batches_mode="agent", max_files_per_batch=2)
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py"), str(workspace / "b.py")])]
        state = _make_state(workspace, batches, effort=effort)
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_fails_when_batch_exceeds_per_batch_limit(self, workspace: Path):
        effort = _make_effort(map_batches_mode="agent", max_files_per_batch=1)
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py"), str(workspace / "b.py")])]
        state = _make_state(workspace, batches, effort=effort)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "max_files_per_batch" in result["error"]


class TestValidateMapBatchesStepFileExistence:
    def test_passes_when_all_files_exist(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches)
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_fails_when_file_does_not_exist(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "ghost.py")])]
        state = _make_state(workspace, batches)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "ghost.py" in result["error"]

    def test_relative_paths_resolved_against_workspace(self, workspace: Path):
        batches = [BatchDefinition(batch_files=["a.py"])]
        state = _make_state(workspace, batches)
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_line_range_suffix_stripped_before_check(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py") + ":1-10"])]
        state = _make_state(workspace, batches)
        result = validate_map_batches_step(state)
        assert result["error"] is None


class TestValidateMapBatchesStepExcludePatterns:
    def test_fails_when_excluded_file_in_batch(self, workspace: Path):
        excl = str(workspace / "a.py")
        batches = [BatchDefinition(batch_files=[excl])]
        state = _make_state(workspace, batches, task_files_exclude=[excl])
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "exclude" in result["error"].lower() or "pattern" in result["error"].lower()

    def test_fails_when_glob_pattern_matches(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches, task_files_exclude=["*.py"])
        result = validate_map_batches_step(state)
        assert result["error"] is not None

    def test_passes_when_exclude_pattern_does_not_match(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches, task_files_exclude=["*.txt"])
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_no_exclude_patterns_has_no_effect(self, workspace: Path):
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches, task_files_exclude=None)
        result = validate_map_batches_step(state)
        assert result["error"] is None


class TestValidateMapBatchesStepCoverage:
    def test_passes_when_all_task_files_covered(self, workspace: Path):
        files = [str(workspace / "a.py"), str(workspace / "b.py")]
        batches = [
            BatchDefinition(batch_files=[str(workspace / "a.py")]),
            BatchDefinition(batch_files=[str(workspace / "b.py")]),
        ]
        state = _make_state(workspace, batches, task_files=files)
        result = validate_map_batches_step(state)
        assert result["error"] is None

    def test_fails_when_task_file_missing_from_batches(self, workspace: Path):
        files = [str(workspace / "a.py"), str(workspace / "b.py")]
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches, task_files=files)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "b.py" in result["error"]
        assert "not assigned" in result["error"]

    def test_fails_when_file_in_multiple_batches(self, workspace: Path):
        dup = str(workspace / "a.py")
        batches = [
            BatchDefinition(batch_files=[dup]),
            BatchDefinition(batch_files=[dup, str(workspace / "b.py")]),
        ]
        files = [str(workspace / "a.py"), str(workspace / "b.py")]
        state = _make_state(workspace, batches, task_files=files)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "a.py" in result["error"]

    def test_fails_when_invented_file_not_in_task_files(self, workspace: Path):
        files = [str(workspace / "a.py")]
        # b.py is not in task_files but is in batch
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py"), str(workspace / "b.py")])]
        state = _make_state(workspace, batches, task_files=files)
        result = validate_map_batches_step(state)
        assert result["error"] is not None
        assert "b.py" in result["error"]

    def test_no_task_files_skips_coverage_check(self, workspace: Path):
        """When task_files is empty (agent-discovered mode), only existence is checked."""
        batches = [BatchDefinition(batch_files=[str(workspace / "a.py")])]
        state = _make_state(workspace, batches, task_files=[])
        result = validate_map_batches_step(state)
        assert result["error"] is None
