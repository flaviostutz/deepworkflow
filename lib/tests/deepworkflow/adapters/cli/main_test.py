from __future__ import annotations

import pytest

from deepworkflow.adapters.cli.main import _build_effort_config, _collect_effort_overrides
from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded


class TestCollectEffortOverrides:
    def test_empty_dict_returns_empty(self):
        assert _collect_effort_overrides({}) == {}

    def test_nullable_int_fields_passed_through(self):
        overrides = _collect_effort_overrides({"max_batches": 5, "max_files_per_batch": None})
        assert overrides["max_batches"] == 5
        assert overrides["max_files_per_batch"] is None

    def test_string_mode_fields_passed_through(self):
        overrides = _collect_effort_overrides({"map_batches_mode": "agent", "consolidate_mode": "static"})
        assert overrides["map_batches_mode"] == "agent"
        assert overrides["consolidate_mode"] == "static"

    def test_int_retry_fields_converted(self):
        overrides = _collect_effort_overrides({
            "evaluate_map_max_retries": "2",
            "evaluate_batch_convergence_max_retries": "1",
            "evaluate_batch_quality_max_retries": "3",
        })
        assert overrides["evaluate_map_max_retries"] == 2
        assert overrides["evaluate_batch_convergence_max_retries"] == 1
        assert overrides["evaluate_batch_quality_max_retries"] == 3

    def test_skip_batch_plan_converted_to_bool(self):
        assert _collect_effort_overrides({"skip_batch_plan": True})["skip_batch_plan"] is True
        assert _collect_effort_overrides({"skip_batch_plan": False})["skip_batch_plan"] is False

    def test_evaluate_quality_min_converted_to_enum(self):
        overrides = _collect_effort_overrides({"evaluate_quality_min": "WARNING"})
        assert overrides["evaluate_quality_min"] == JudgeLevel.WARNING

    def test_evaluate_quality_min_case_insensitive(self):
        overrides = _collect_effort_overrides({"evaluate_quality_min": "ok"})
        assert overrides["evaluate_quality_min"] == JudgeLevel.OK

    def test_evaluate_quality_on_max_retries_converted_to_enum(self):
        overrides = _collect_effort_overrides({"evaluate_quality_on_max_retries": "continue"})
        assert overrides["evaluate_quality_on_max_retries"] == OnMaxRetriesExceeded.CONTINUE

    def test_evaluate_quality_batch_instructions_passed_through(self):
        overrides = _collect_effort_overrides({"evaluate_quality_batch_instructions": "check for bugs"})
        assert overrides["evaluate_quality_batch_instructions"] == "check for bugs"

    def test_unknown_keys_ignored(self):
        overrides = _collect_effort_overrides({"type": "static", "level": 3, "unknown_key": "value"})
        assert "type" not in overrides
        assert "level" not in overrides
        assert "unknown_key" not in overrides


class TestBuildEffortConfig:
    def test_no_effort_key_returns_default(self):
        result = _build_effort_config({})
        assert isinstance(result, EffortConfig)
        assert result.level == 3
        assert result.type == "static"

    def test_effort_none_returns_default(self):
        result = _build_effort_config({"effort": None})
        assert result == EffortConfig()

    def test_effort_not_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _build_effort_config({"effort": "level: 3"})

    def test_level_and_type_set(self):
        result = _build_effort_config({"effort": {"level": 5, "type": "static"}})
        assert result.level == 5
        assert result.type == "static"

    def test_default_type_is_static(self):
        result = _build_effort_config({"effort": {"level": 2}})
        assert result.type == "static"

    def test_auto_type(self):
        result = _build_effort_config({"effort": {"type": "auto"}})
        assert result.type == "auto"

    def test_overrides_applied(self):
        result = _build_effort_config({
            "effort": {
                "level": 3,
                "map_batches_mode": "agent",
                "evaluate_batch_quality_max_retries": 2,
                "evaluate_quality_min": "OK",
            }
        })
        assert result.map_batches_mode == "agent"
        assert result.evaluate_batch_quality_max_retries == 2
        assert result.evaluate_quality_min == JudgeLevel.OK
