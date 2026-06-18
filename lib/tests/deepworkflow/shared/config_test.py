from __future__ import annotations

import pytest

from deepworkflow.shared.config import DeepWorkflowConfig, _ModelRef, resolveEffortConfig
from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


def _make_config(**kwargs) -> DeepWorkflowConfig:
    from deepworkflow.shared.config import resolveEffortConfig

    defaults = {
        "workspace_dir": "/tmp/workspace",
        "task_instructions": "Do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": "custom",
        "effort_config": resolveEffortConfig(5),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


class TestDeepWorkflowConfig:
    def test_minimal_config(self):
        config = _make_config()
        assert config.workspace_dir == "/tmp/workspace"
        assert config.task_files is None
        assert config.max_failure_retries == 0
        assert config.evaluate_quality_min == JudgeLevel.WARNING
        assert callable(config.model)
        assert config.model("any_agent") is None  # _mock_model returns None

    def test_full_config(self):
        config = _make_config(
            workspace_dir="/workspace",
            task_instructions="Do multiple things",
            workspace_write_option=WriteOption.WRITE_ANY,
            effort="custom",
            effort_config=resolveEffortConfig(5),
            evaluate_quality_on_max_retries=OnMaxRetriesExceeded.FAIL,
            task_files=["a.py", "b.py"],
            evaluate_quality_min=JudgeLevel.OK,
            evaluate_quality_batch_instructions="Output MUST be valid",
            max_failure_retries=2,
        )
        assert config.task_files == ["a.py", "b.py"]
        assert config.evaluate_quality_batch_instructions == "Output MUST be valid"

    def test_immutable(self):
        config = _make_config()
        with pytest.raises(AttributeError):
            config.workspace_dir = "/other"  # type: ignore[misc]

    def test_effort_custom_requires_effort_config(self):
        with pytest.raises(ValueError, match="effort_config is required"):
            DeepWorkflowConfig(
                workspace_dir="/tmp",
                task_instructions="x",
                model=_mock_model,
                workspace_write_option=WriteOption.READ_ONLY,
                effort="custom",
                effort_config=None,
            )

    def test_effort_auto_allows_no_effort_config(self):
        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="x",
            model=_mock_model,
            workspace_write_option=WriteOption.READ_ONLY,
            effort="auto",
            effort_config=None,
        )
        assert config.effort == "auto"
        assert config.effort_config is None


class TestModelRef:
    def test_raises_when_factory_not_in_registry(self):
        ref = _ModelRef("nonexistent-config-id")
        with pytest.raises(RuntimeError, match="No model factory found"):
            ref("agent_name")


class TestResolveEffortConfig:
    def test_level_1_all_static(self):
        ec = resolveEffortConfig(1)
        assert ec.map_batches_mode == "static"
        assert ec.consolidate_mode == "static"
        assert ec.evaluate_map_max_retries == 0
        assert ec.evaluate_batch_quality_max_retries == 0
        assert ec.evaluate_batch_convergence_max_retries == 0
        assert ec.skip_batch_plan is True
        assert ec.max_batches == 1

    def test_level_10_all_agent(self):
        ec = resolveEffortConfig(10)
        assert ec.map_batches_mode == "agent"
        assert ec.consolidate_mode == "agent"
        assert ec.evaluate_map_max_retries == 10
        assert ec.evaluate_batch_quality_max_retries == 10
        assert ec.evaluate_batch_convergence_max_retries == 10
        assert ec.skip_batch_plan is False

    def test_level_4_flips_to_agent(self):
        ec = resolveEffortConfig(4)
        assert ec.map_batches_mode == "agent"
        assert ec.consolidate_mode == "agent"

    def test_level_3_still_static(self):
        ec = resolveEffortConfig(3)
        assert ec.map_batches_mode == "static"

    def test_level_6_plan_enabled(self):
        ec = resolveEffortConfig(6)
        assert ec.skip_batch_plan is False

    def test_level_5_plan_skipped(self):
        ec = resolveEffortConfig(5)
        assert ec.skip_batch_plan is True

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be between 1 and 10"):
            resolveEffortConfig(0)
        with pytest.raises(ValueError, match="level must be between 1 and 10"):
            resolveEffortConfig(11)

    def test_returns_effort_config_instance(self):
        ec = resolveEffortConfig(5)
        assert isinstance(ec, EffortConfig)
