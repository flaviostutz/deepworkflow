from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_reflect_agent import (
    _parse_reflect_output,
    batch_reflect_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import EffortConfig, WriteOption


def _mock_model(response: str):
    model = FakeListChatModel(responses=[response])

    def _factory(_agent_name: str):
        return model

    return model, _factory


def _make_config(response: str = "") -> tuple[DeepWorkflowConfig, FakeListChatModel]:
    model, factory = _mock_model(response)
    cfg = DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=factory,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
    )
    return cfg, model


class TestReflectBatchAgent:
    def test_no_execute_messages_returns_empty(self):
        config, _ = _make_config("")
        result = batch_reflect_agent({"config": config, "batch_execute_messages": []})
        assert result == {"batch_files_read": [], "batch_files_written": []}

    def test_returns_files_from_agent_response(self):
        response_content = "FILES_READ:\na.py\nb.py\n\nFILES_WRITTEN:\nc.py\n"
        config, _model = _make_config(response_content)
        existing_msg = AIMessage(content="previous execution step")
        result = batch_reflect_agent({"config": config, "batch_execute_messages": [existing_msg]})
        assert result["batch_files_read"] == ["a.py", "b.py"]
        assert result["batch_files_written"] == ["c.py"]


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
