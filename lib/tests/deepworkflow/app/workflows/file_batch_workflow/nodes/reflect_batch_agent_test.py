from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent import (
    _parse_reflect_output,
    reflect_batch_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption


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
        effort="custom",
        effort_config=resolveEffortConfig(5),
        evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )
    return cfg, model


class TestReflectBatchAgent:
    def test_no_execute_messages_returns_empty(self):
        config, _ = _make_config("")
        result = reflect_batch_agent({"config": config, "execute_messages": []})
        assert result == {"files_read": [], "files_written": []}

    def test_returns_files_from_agent_response(self):
        response_content = "FILES_READ:\na.py\nb.py\n\nFILES_WRITTEN:\nc.py\n"
        config, _model = _make_config(response_content)
        existing_msg = AIMessage(content="previous execution step")
        result = reflect_batch_agent({"config": config, "execute_messages": [existing_msg]})
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
