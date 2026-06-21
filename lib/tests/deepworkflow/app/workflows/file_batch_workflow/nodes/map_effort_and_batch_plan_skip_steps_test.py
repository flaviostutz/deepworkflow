from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_plan_skip_step import batch_plan_skip_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_effort_step import map_effort_step
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
        result = map_effort_step({"config": config})
        from deepworkflow.shared.config import resolveEffortConfig

        expected = resolveEffortConfig(3)
        assert result["effort_config"] == expected

    def test_level_1_effort_config(self):
        config = _make_config(level=1)
        result = map_effort_step({"config": config})
        assert result["effort_config"].map_plan_mode == "static"
        assert result["effort_config"].batch_skip_plan is True

    def test_level_10_effort_config(self):
        config = _make_config(level=10)
        result = map_effort_step({"config": config})
        assert result["effort_config"].map_plan_mode == "agent"
        assert result["effort_config"].batch_skip_plan is False

    def test_quality_override_applied(self):
        config = _make_config(effort=EffortConfig(level=5, batch_evaluate_min=JudgeLevel.OK))
        result = map_effort_step({"config": config})
        assert result["effort_config"].batch_evaluate_min == JudgeLevel.OK

    def test_on_max_retries_override_applied(self):
        config = _make_config(effort=EffortConfig(level=5, batch_evaluate_on_max_retries=OnMaxRetriesExceeded.FAIL))
        result = map_effort_step({"config": config})
        assert result["effort_config"].batch_evaluate_on_max_retries == OnMaxRetriesExceeded.FAIL


class TestSkipBatchPlanStep:
    def test_returns_batch_plan(self):
        result = batch_plan_skip_step({})
        assert "batch_plan" in result
        assert result["batch_plan"]  # non-empty

    def test_batch_plan_contains_planning_instruction(self):
        result = batch_plan_skip_step({})
        assert "plan" in result["batch_plan"].lower()

    def test_batch_plan_mentions_todo(self):
        result = batch_plan_skip_step({})
        assert "todo" in result["batch_plan"].lower()
