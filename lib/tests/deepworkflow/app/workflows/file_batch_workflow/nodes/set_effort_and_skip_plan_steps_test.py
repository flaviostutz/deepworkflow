from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.nodes.set_effort_config_step import set_effort_config_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_batch_plan_step import skip_batch_plan_step
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(level: int = 5, **kwargs) -> DeepWorkflowConfig:
    effort = kwargs.pop("effort", EffortConfig(level=level))
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=effort,
        **kwargs,
    )


class TestSetEffortConfigStep:
    def test_resolves_effort_config_from_level(self):
        config = _make_config(level=3)
        result = set_effort_config_step({"config": config})
        from deepworkflow.shared.config import resolveEffortConfig
        expected = resolveEffortConfig(3)
        assert result["effort_config"] == expected

    def test_level_1_effort_config(self):
        config = _make_config(level=1)
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].map_batches_mode == "static"
        assert result["effort_config"].skip_batch_plan is True

    def test_level_10_effort_config(self):
        config = _make_config(level=10)
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].map_batches_mode == "agent"
        assert result["effort_config"].skip_batch_plan is False

    def test_quality_override_applied(self):
        config = _make_config(effort=EffortConfig(level=5, evaluate_quality_min=JudgeLevel.OK))
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].evaluate_quality_min == JudgeLevel.OK

    def test_on_max_retries_override_applied(self):
        config = _make_config(effort=EffortConfig(level=5, evaluate_quality_on_max_retries=OnMaxRetriesExceeded.FAIL))
        result = set_effort_config_step({"config": config})
        assert result["effort_config"].evaluate_quality_on_max_retries == OnMaxRetriesExceeded.FAIL


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
