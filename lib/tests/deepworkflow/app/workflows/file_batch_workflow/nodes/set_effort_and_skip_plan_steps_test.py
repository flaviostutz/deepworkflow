from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.nodes.set_effort_config_step import set_effort_config_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_batch_plan_step import skip_batch_plan_step
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(effort_config=None):
    ec = effort_config or resolveEffortConfig(5)
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort="custom",
        effort_config=ec,
        evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


class TestSetEffortConfigStep:
    def test_copies_effort_config_from_config_into_state(self):
        ec = resolveEffortConfig(3)
        config = _make_config(effort_config=ec)
        result = set_effort_config_step({"config": config})
        assert result["effort_config"] is ec

    def test_level_1_effort_config(self):
        ec = resolveEffortConfig(1)
        config = _make_config(effort_config=ec)
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].map_batches_mode == "static"
        assert result["effort_config"].skip_batch_plan is True

    def test_level_10_effort_config(self):
        ec = resolveEffortConfig(10)
        config = _make_config(effort_config=ec)
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].map_batches_mode == "agent"
        assert result["effort_config"].skip_batch_plan is False


class TestSkipBatchPlanStep:
    def test_returns_plan_output(self):
        result = skip_batch_plan_step({})
        assert "plan_output" in result
        assert result["plan_output"]  # non-empty

    def test_plan_output_contains_planning_instruction(self):
        result = skip_batch_plan_step({})
        assert "plan" in result["plan_output"].lower()

    def test_plan_output_mentions_todo(self):
        result = skip_batch_plan_step({})
        assert "todo" in result["plan_output"].lower()
