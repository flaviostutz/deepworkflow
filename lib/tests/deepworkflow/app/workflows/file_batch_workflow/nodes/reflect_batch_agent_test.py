from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent import (
    _parse_reflect_output,
    reflect_batch_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config() -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


class TestReflectBatchAgent:
    def test_no_execute_messages_returns_empty(self, mocker):
        m = mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent.create_agent",
            "",
        )
        result = reflect_batch_agent({"config": _make_config(), "execute_messages": []})
        assert result == {"files_read": [], "files_written": []}
        m.assert_not_called()

    def test_returns_files_from_agent_response(self, mocker):
        response_content = "FILES_READ:\na.py\nb.py\n\nFILES_WRITTEN:\nc.py\n"
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent.create_agent",
            response_content,
        )
        existing_msg = AIMessage(content="previous execution step")
        result = reflect_batch_agent({"config": _make_config(), "execute_messages": [existing_msg]})
        assert result["files_read"] == ["a.py", "b.py"]
        assert result["files_written"] == ["c.py"]


class TestParseReflectOutput:
    def test_standard_format(self):
        content = """FILES_READ:
src/main.py
src/utils.py

FILES_WRITTEN:
src/output.py"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py", "src/utils.py"]
        assert files_written == ["src/output.py"]

    def test_no_files_written(self):
        content = """FILES_READ:
src/main.py

FILES_WRITTEN:
"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py"]
        assert files_written == []

    def test_no_files_at_all(self):
        content = """FILES_READ:

FILES_WRITTEN:
"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == []
        assert files_written == []

    def test_extra_whitespace(self):
        content = """FILES_READ:
  src/main.py  
src/utils.py

FILES_WRITTEN:
  dist/out.py  """
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py", "src/utils.py"]
        assert files_written == ["dist/out.py"]
