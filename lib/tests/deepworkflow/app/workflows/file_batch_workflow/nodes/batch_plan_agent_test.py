from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent import batch_plan_agent
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
        task_instructions="do something MUST be done",
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
    }
    defaults.update(overrides)
    return defaults


class TestPlanBatchAgent:
    def test_returns_batch_plan(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent.create_agent",
            "step 1: do this; step 2: do that",
        )
        result = batch_plan_agent(_make_state())
        assert result["batch_plan"] == "step 1: do this; step 2: do that"

    def test_with_evaluate_quality_feedback(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent.create_agent",
            "revised plan",
        )
        feedback = EvaluateFeedback(file="a.py", type=JudgeLevel.ERROR, description="bad plan", proposal="redo it")
        result = batch_plan_agent(_make_state(batch_evaluate_feedbacks=[feedback]))
        assert result["batch_plan"] == "revised plan"

    def test_without_task_overview(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent.create_agent",
            "plan without overview",
        )
        state = _make_state()
        del state["map_plan_overview"]
        result = batch_plan_agent(state)
        assert result["batch_plan"] == "plan without overview"

    def test_cumulative_execute_output_in_prompt(self, mocker):
        mock = mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_agent.create_agent",
            "plan",
        )
        batch_plan_agent(_make_state(batch_cumulative_output="some prior work was done"))
        system_prompt = mock.call_args.kwargs["system_prompt"]
        assert "some prior work was done" in system_prompt
