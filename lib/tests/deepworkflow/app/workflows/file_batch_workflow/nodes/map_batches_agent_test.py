from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent import (
    DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
    _parse_map_output,
    map_batches_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_state(judge_batch_instructions: str | None = None) -> dict:
    config = DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="Do something MUST be done",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        judge_batch_instructions=judge_batch_instructions,
    )
    return {"config": config, "task_files": ["a.py"]}


class TestMapBatchesAgentJudgeInstructionsValidation:
    def test_none_judge_instructions_skips_validation(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent.create_agent",
            {"task_overview": "x", "consolidation_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent._derive_judge_instructions",
            return_value=DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
        )
        result = map_batches_agent(_make_state(judge_batch_instructions=None))
        assert "error" not in result
        assert result["judge_batch_instructions"] == DEFAULT_JUDGE_BATCH_INSTRUCTIONS

    def test_judge_instructions_with_must_passes(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent.create_agent",
            {"task_overview": "x", "consolidation_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent._derive_judge_instructions",
            return_value="Output MUST be valid JSON.",
        )
        result = map_batches_agent(_make_state(judge_batch_instructions="Output MUST be valid JSON."))
        assert "error" not in result
        assert result["judge_batch_instructions"] == "Output MUST be valid JSON."

    def test_judge_instructions_without_keywords_fails(self):
        result = map_batches_agent(_make_state(judge_batch_instructions="Check the output is correct."))
        assert "error" in result
        assert "MANDATORY" in result["error"] or "MUST" in result["error"]

    def test_judge_instructions_with_should_passes(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent.create_agent",
            {"task_overview": "x", "consolidation_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent._derive_judge_instructions",
            return_value="Output should be well-formatted.",
        )
        result = map_batches_agent(_make_state(judge_batch_instructions="Output should be well-formatted."))
        assert "error" not in result


class TestParseMapOutput:
    def test_valid_json(self):
        content = json.dumps(
            {
                "task_overview": "overview",
                "consolidation_instructions": "consolidate",
                "batches": [
                    {"batch_files": ["a.py", "b.py"], "batch_instructions": "group 1"},
                    {"batch_files": ["c.py"], "batch_instructions": "group 2"},
                ],
            }
        )
        result = _parse_map_output(content)
        assert result["task_overview"] == "overview"
        assert result["consolidation_instructions"] == "consolidate"
        assert len(result["task_file_batches"]) == 2
        assert result["task_file_batches"][0].batch_files == ["a.py", "b.py"]
        assert result["current_batch_index"] == 0
        assert result["retry_count"] == 0

    def test_json_in_code_block(self):
        content = """```json
{"task_overview": "x", "consolidation_instructions": "y", "batches": [{"batch_files": ["f.py"]}]}
```"""
        result = _parse_map_output(content)
        assert result["task_overview"] == "x"
        assert len(result["task_file_batches"]) == 1

    def test_invalid_json(self):
        result = _parse_map_output("not json at all")
        assert "error" in result

    def test_error_response(self):
        content = json.dumps({"error": True, "message": "unclear instructions"})
        result = _parse_map_output(content)
        assert result == {"error": "unclear instructions"}

    def test_zero_batches(self):
        content = json.dumps(
            {
                "task_overview": "x",
                "consolidation_instructions": "y",
                "batches": [],
            }
        )
        result = _parse_map_output(content)
        assert "error" in result
