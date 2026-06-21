from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent import (
    DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
    _parse_map_output,
    map_plan_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import EffortConfig, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_state(batch_evaluate_quality_instructions: str | None = None) -> dict:
    config = DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="Do something MUST be done",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
    )
    effort = resolveEffortConfig(5)
    if batch_evaluate_quality_instructions is not None:
        from dataclasses import replace

        effort = replace(effort, batch_evaluate_quality_instructions=batch_evaluate_quality_instructions)
    return {"config": config, "map_files": ["a.py"], "effort_config": effort}


class TestMapBatchesAgentJudgeInstructionsValidation:
    def test_none_evaluate_quality_instructions_skips_validation(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent.create_agent",
            {"map_plan_overview": "x", "reduce_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent._derive_evaluate_quality_instructions",
            return_value=DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
        )
        result = map_plan_agent(_make_state(batch_evaluate_quality_instructions=None))
        assert "error" not in result
        assert result["batch_evaluate_quality_instructions"] == DEFAULT_JUDGE_BATCH_INSTRUCTIONS

    def test_evaluate_quality_instructions_with_must_passes(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent.create_agent",
            {"map_plan_overview": "x", "reduce_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent._derive_evaluate_quality_instructions",
            return_value="Output MUST be valid JSON.",
        )
        result = map_plan_agent(_make_state(batch_evaluate_quality_instructions="Output MUST be valid JSON."))
        assert "error" not in result
        assert result["batch_evaluate_quality_instructions"] == "Output MUST be valid JSON."

    def test_evaluate_quality_instructions_without_keywords_fails(self):
        result = map_plan_agent(_make_state(batch_evaluate_quality_instructions="Check the output is correct."))
        assert "error" in result
        assert "MANDATORY" in result["error"] or "MUST" in result["error"]

    def test_evaluate_quality_instructions_with_should_passes(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent.create_agent",
            {"map_plan_overview": "x", "reduce_instructions": "y", "batches": [{"batch_files": ["a.py"]}]},
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent._derive_evaluate_quality_instructions",
            return_value="Output should be well-formatted.",
        )
        result = map_plan_agent(_make_state(batch_evaluate_quality_instructions="Output should be well-formatted."))
        assert "error" not in result


class TestParseMapOutput:
    def test_valid_json(self):
        content = json.dumps(
            {
                "map_plan_overview": "overview",
                "reduce_instructions": "consolidate",
                "batches": [
                    {"batch_files": ["a.py", "b.py"], "batch_instructions": "group 1"},
                    {"batch_files": ["c.py"], "batch_instructions": "group 2"},
                ],
            }
        )
        result = _parse_map_output(content)
        assert result["map_plan_overview"] == "overview"
        assert result["reduce_instructions"] == "consolidate"
        assert len(result["map_batches"]) == 2
        assert result["map_batches"][0].batch_files == ["a.py", "b.py"]
        assert result["batch_current_index"] == 0
        assert result["batch_quality_retry_count"] == 0

    def test_json_in_code_block(self):
        content = """```json
{"map_plan_overview": "x", "reduce_instructions": "y", "batches": [{"batch_files": ["f.py"]}]}
```"""
        result = _parse_map_output(content)
        assert result["map_plan_overview"] == "x"
        assert len(result["map_batches"]) == 1

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
                "map_plan_overview": "x",
                "reduce_instructions": "y",
                "batches": [],
            }
        )
        result = _parse_map_output(content)
        assert "error" in result


class TestMapBatchesAgentExcludeSection:
    def test_exclude_patterns_appear_in_prompt_when_discovering_files(self, mocker):
        """When task_files is empty and task_files_exclude is set, the prompt must mention the exclude patterns."""
        captured_prompts: list[str] = []

        def capture_create_agent(**kwargs):
            captured_prompts.append(kwargs["system_prompt"])
            agent = mocker.MagicMock()
            agent.invoke.return_value = {
                "messages": [
                    mocker.MagicMock(
                        content='{"map_plan_overview":"x","reduce_instructions":"y","batches":[{"batch_files":["a.py"]}]}'
                    )
                ]
            }
            return agent

        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent.create_agent",
            side_effect=capture_create_agent,
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent._derive_evaluate_quality_instructions",
            return_value=DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
        )

        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="Do something MUST be done",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            effort=EffortConfig(level=5),
            task_files_exclude=["*.lock", "**/__pycache__/**"],
        )
        state = {"config": config, "map_files": [], "effort_config": resolveEffortConfig(5)}
        map_plan_agent(state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "*.lock" in prompt
        assert "**/__pycache__/**" in prompt
        assert "EXCLUDE" in prompt

    def test_exclude_patterns_not_in_prompt_when_task_files_provided(self, mocker):
        """When task_files is pre-populated (exclude was already applied), prompt must NOT add exclude section."""
        captured_prompts: list[str] = []

        def capture_create_agent(**kwargs):
            captured_prompts.append(kwargs["system_prompt"])
            agent = mocker.MagicMock()
            agent.invoke.return_value = {
                "messages": [
                    mocker.MagicMock(
                        content='{"map_plan_overview":"x","reduce_instructions":"y","batches":[{"batch_files":["a.py"]}]}'
                    )
                ]
            }
            return agent

        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent.create_agent",
            side_effect=capture_create_agent,
        )
        mocker.patch(
            "deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_agent._derive_evaluate_quality_instructions",
            return_value=DEFAULT_JUDGE_BATCH_INSTRUCTIONS,
        )

        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="Do something MUST be done",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            effort=EffortConfig(level=5),
            task_files_exclude=["*.lock"],
        )
        state = {"config": config, "map_files": ["a.py"], "effort_config": resolveEffortConfig(5)}
        map_plan_agent(state)

        assert len(captured_prompts) == 1
        assert "*.lock" not in captured_prompts[0]
