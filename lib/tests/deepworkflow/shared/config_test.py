from __future__ import annotations

from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


class TestDeepWorkflowConfig:
    def test_minimal_config(self):
        config = DeepWorkflowConfig(
            workspace_dir="/tmp/workspace",
            task_instructions="Do something",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            judge_max_retries=2,
            judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        assert config.workspace_dir == "/tmp/workspace"
        assert config.task_files is None
        assert config.task_files_batch_size is None
        assert config.max_failure_retries == 0
        assert config.judge_min == JudgeVerdict.WARNING
        assert config.judge_skip is False
        assert config.model is _mock_model

    def test_full_config(self):
        config = DeepWorkflowConfig(
            workspace_dir="/workspace",
            task_instructions="Do multiple things",
            model=_mock_model,
            workspace_write_option=WriteOption.WRITE_ANY,
            judge_max_retries=3,
            judge_on_max_retries=OnMaxRetriesExceeded.FAIL,
            task_files=["a.py", "b.py"],
            judge_min=JudgeVerdict.OK,
            task_files_batch_size=1,
            judge_batch_instructions="Be strict",
            max_failure_retries=2,
            judge_skip=True,
        )
        assert config.task_files == ["a.py", "b.py"]
        assert config.task_files_batch_size == 1
        assert config.judge_batch_instructions == "Be strict"
        assert config.judge_skip is True

    def test_immutable(self):
        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="x",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            judge_max_retries=1,
            judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        import pytest

        with pytest.raises(AttributeError):
            config.workspace_dir = "/other"  # type: ignore[misc]
