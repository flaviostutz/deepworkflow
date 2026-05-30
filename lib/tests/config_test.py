from __future__ import annotations

from deepworkflow.shared.config import WorkflowConfig
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


class TestWorkflowConfig:
    def test_minimal_config(self):
        config = WorkflowConfig(
            workspace_dir="/tmp/workspace",
            task_instructions="Do something",
            task_files=["src/**/*.py"],
            task_files_write_option=WriteOption.READ_ONLY,
            judge_minimum=JudgeVerdict.WARNING,
            judge_max_retries=2,
            on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE,
        )
        assert config.workspace_dir == "/tmp/workspace"
        assert config.task_files == ["src/**/*.py"]
        assert config.task_files_batch_size is None
        assert config.max_failure_retries == 0
        assert config.model == "openai:gpt-4o"

    def test_full_config(self):
        config = WorkflowConfig(
            workspace_dir="/workspace",
            task_instructions="Do multiple things",
            task_files=["a.py", "b.py"],
            task_files_write_option=WriteOption.WRITE_ANY,
            judge_minimum=JudgeVerdict.OK,
            judge_max_retries=3,
            on_max_retries_exceeded=OnMaxRetriesExceeded.FAIL,
            task_files_batch_size=1,
            judge_instructions="Be strict",
            max_failure_retries=2,
            model="openai:gpt-4o-mini",
        )
        assert config.task_files == ["a.py", "b.py"]
        assert config.task_files_batch_size == 1
        assert config.judge_instructions == "Be strict"
        assert config.model == "openai:gpt-4o-mini"

    def test_immutable(self):
        config = WorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="x",
            task_files=["a.py"],
            task_files_write_option=WriteOption.READ_ONLY,
            judge_minimum=JudgeVerdict.WARNING,
            judge_max_retries=1,
            on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE,
        )
        import pytest

        with pytest.raises(AttributeError):
            config.workspace_dir = "/other"  # type: ignore[misc]
