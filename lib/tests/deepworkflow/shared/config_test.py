from __future__ import annotations

import pytest

from deepworkflow.shared.config import DeepWorkflowConfig, _ModelRef, resolveEffortConfig
from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


def _make_config(**kwargs) -> DeepWorkflowConfig:
    defaults = {
        "workspace_dir": "/tmp/workspace",
        "task_instructions": "Do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": EffortConfig(level=5),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


class TestDeepWorkflowConfig:
    def test_minimal_config(self):
        config = _make_config()
        assert config.workspace_dir == "/tmp/workspace"
        assert config.task_files is None
        assert config.max_failure_retries == 0
        assert config.effort.level == 5
        assert callable(config.model)
        assert config.model("any_agent") is None  # _mock_model returns None

    def test_default_effort(self):
        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="x",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
        )
        assert config.effort.level == 3
        assert config.effort.type == "static"

    def test_full_config(self):
        config = _make_config(
            workspace_dir="/workspace",
            task_instructions="Do multiple things",
            workspace_write_option=WriteOption.WRITE_ANY,
            effort=EffortConfig(
                level=5,
                batch_evaluate_on_max_retries=OnMaxRetriesExceeded.FAIL,
                batch_evaluate_min=JudgeLevel.OK,
                batch_evaluate_quality_instructions="Output MUST be valid",
            ),
            task_files=["a.py", "b.py"],
            max_failure_retries=2,
        )
        assert config.task_files == ["a.py", "b.py"]
        assert config.effort.batch_evaluate_quality_instructions == "Output MUST be valid"

    def test_immutable(self):
        config = _make_config()
        with pytest.raises(AttributeError):
            config.workspace_dir = "/other"  # type: ignore[misc]

    def test_effort_auto_type(self):
        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="x",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            effort=EffortConfig(type="auto"),
        )
        assert config.effort.type == "auto"

    def test_effort_auto_with_options_raises(self):
        with pytest.raises(ValueError, match="no options can be set when type='auto'"):
            EffortConfig(type="auto", batch_evaluate_min=JudgeLevel.WARNING)


class TestModelRef:
    def test_raises_when_factory_not_in_registry(self):
        ref = _ModelRef("nonexistent-config-id")
        with pytest.raises(RuntimeError, match="No model factory found"):
            ref("agent_name")


class TestResolveEffortConfig:
    def test_level_1_all_static(self):
        ec = resolveEffortConfig(1)
        assert ec.map_plan_mode == "static"
        assert ec.reduce_mode == "static"
        assert ec.map_evaluate_max_retries == 0
        assert ec.batch_evaluate_quality_max_retries == 0
        assert ec.batch_evaluate_convergence_max_retries == 0
        assert ec.batch_skip_plan is True
        assert ec.batch_skip_reflect is False
        assert ec.max_batches == 1

    def test_level_0_one_shot(self):
        ec = resolveEffortConfig(0)
        assert ec.map_plan_mode == "static"
        assert ec.reduce_mode == "static"
        assert ec.map_evaluate_max_retries == 0
        assert ec.batch_evaluate_quality_max_retries == 0
        assert ec.batch_evaluate_convergence_max_retries == 0
        assert ec.batch_skip_plan is True
        assert ec.batch_skip_reflect is True
        assert ec.max_batches == 1

    def test_level_10_all_agent(self):
        ec = resolveEffortConfig(10)
        assert ec.map_plan_mode == "agent"
        assert ec.reduce_mode == "agent"
        assert ec.map_evaluate_max_retries == 10
        assert ec.batch_evaluate_quality_max_retries == 10
        assert ec.batch_evaluate_convergence_max_retries == 10
        assert ec.batch_skip_plan is False

    def test_level_4_flips_to_agent(self):
        ec = resolveEffortConfig(4)
        assert ec.map_plan_mode == "agent"
        assert ec.reduce_mode == "agent"

    def test_level_3_still_static(self):
        ec = resolveEffortConfig(3)
        assert ec.map_plan_mode == "static"

    def test_level_6_plan_enabled(self):
        ec = resolveEffortConfig(6)
        assert ec.batch_skip_plan is False

    def test_level_5_plan_skipped(self):
        ec = resolveEffortConfig(5)
        assert ec.batch_skip_plan is True

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be between 0 and 10"):
            resolveEffortConfig(-1)
        with pytest.raises(ValueError, match="level must be between 0 and 10"):
            resolveEffortConfig(11)

    def test_returns_effort_config_instance(self):
        ec = resolveEffortConfig(5)
        assert isinstance(ec, EffortConfig)
