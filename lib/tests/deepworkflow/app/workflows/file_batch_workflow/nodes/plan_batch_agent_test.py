from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent import plan_batch_agent
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    EvaluateFeedback,
    JudgeLevel,
    OnMaxRetriesExceeded,
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
        effort="custom",
        effort_config=resolveEffortConfig(5),
        evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "config": _make_config(),
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "task_overview": "overall plan",
    }
    defaults.update(overrides)
    return defaults


class TestPlanBatchAgent:
    def test_returns_plan_output(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent.create_agent",
            "step 1: do this; step 2: do that",
        )
        result = plan_batch_agent(_make_state())
        assert result["plan_output"] == "step 1: do this; step 2: do that"

    def test_with_evaluate_quality_feedback(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent.create_agent",
            "revised plan",
        )
        feedback = EvaluateFeedback(file="a.py", type=JudgeLevel.ERROR, description="bad plan", proposal="redo it")
        result = plan_batch_agent(_make_state(evaluate_quality_feedbacks=[feedback]))
        assert result["plan_output"] == "revised plan"

    def test_without_task_overview(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent.create_agent",
            "plan without overview",
        )
        state = _make_state()
        del state["task_overview"]
        result = plan_batch_agent(state)
        assert result["plan_output"] == "plan without overview"

    def test_previous_execute_output_not_in_prompt(self, mocker):
        mock = mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.plan_batch_agent.create_agent",
            "plan",
        )
        plan_batch_agent(_make_state(previous_execute_output="some prior work was done"))
        system_prompt = mock.call_args.kwargs["system_prompt"]
        assert "some prior work was done" not in system_prompt
