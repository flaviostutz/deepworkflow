from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_agent import evaluate_batch_agent
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, JudgeVerdict, OnMaxRetriesExceeded, WriteOption


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


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "config": _make_config(),
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "execute_output": "I did the thing",
        "judge_batch_instructions": "Output MUST be valid",
    }
    defaults.update(overrides)
    return defaults


class TestEvaluateBatchAgent:
    def test_returns_verdict_and_feedbacks(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_agent.create_agent",
            {
                "judge_verdict": "OK",
                "judge_feedbacks": [{"file": "a.py", "type": "OK", "description": "looks good", "proposal": ""}],
            },
        )
        result = evaluate_batch_agent(_make_state())
        assert result["judge_verdict"] == JudgeVerdict.OK
        assert len(result["judge_feedbacks"]) == 1
        assert result["judge_feedbacks"][0].file == "a.py"

    def test_invalid_json_returns_error_verdict(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_agent.create_agent",
            "not valid json {{{",
        )
        result = evaluate_batch_agent(_make_state())
        assert result["judge_verdict"] == JudgeVerdict.ERROR
