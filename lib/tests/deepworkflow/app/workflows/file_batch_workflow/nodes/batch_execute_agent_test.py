from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_execute_agent import batch_execute_agent
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    EffortConfig,
    EvaluateFeedback,
    JudgeLevel,
    WriteOption,
)


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config() -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
    )


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "config": _make_config(),
        "batch_current_index": 0,
        "map_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "map_plan_overview": "overall plan",
        "batch_plan": "step 1: read file; step 2: summarize",
    }
    defaults.update(overrides)
    return defaults


class TestExecuteBatchAgent:
    def test_returns_execute_output_and_messages(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_execute_agent.create_agent",
            "execution complete",
        )
        result = batch_execute_agent(_make_state())
        assert result["batch_execute_output"] == "execution complete"
        assert result["batch_execute_messages"][0].content == "execution complete"

    def test_with_evaluate_quality_feedback(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_execute_agent.create_agent",
            "revised execution",
        )
        feedback = EvaluateFeedback(file="a.py", type=JudgeLevel.WARNING, description="incomplete", proposal="add more")
        result = batch_execute_agent(_make_state(batch_evaluate_feedbacks=[feedback]))
        assert result["batch_execute_output"] == "revised execution"
        assert "batch_execute_messages" in result

    def test_with_cumulative_execute_output(self, mocker):
        mock = mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_execute_agent.create_agent",
            "additional work done",
        )
        result = batch_execute_agent(_make_state(batch_cumulative_output="prior pass completed file a.py"))
        assert result["batch_execute_output"] == "additional work done"
        system_prompt = mock.call_args.kwargs["system_prompt"]
        assert "prior pass completed file a.py" in system_prompt
        assert "ADDITIONAL" in system_prompt
