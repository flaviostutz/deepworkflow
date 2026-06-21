from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_quality_retry_step import batch_quality_retry_step
from deepworkflow.shared.types import EffortConfig
from deepworkflow.shared.workflow_log import WorkflowStats, _stats_var


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "batch_quality_retry_count": 0,
        "batch_convergence_repeat_count": 0,
        "effort_config": EffortConfig(level=3),
    }
    defaults.update(overrides)
    return defaults


class TestIncrementRetryStep:
    def test_increments_retry_count_from_zero(self):
        result = batch_quality_retry_step(_make_state())
        assert result["batch_quality_retry_count"] == 1

    def test_increments_retry_count_from_nonzero(self):
        result = batch_quality_retry_step(_make_state(batch_quality_retry_count=2))
        assert result["batch_quality_retry_count"] == 3

    def test_resets_batch_repeat_count(self):
        result = batch_quality_retry_step(_make_state(batch_convergence_repeat_count=5))
        assert result["batch_convergence_repeat_count"] == 0

    def test_increments_quality_retries_when_within_limit(self):
        effort = EffortConfig(batch_evaluate_quality_max_retries=3)
        stats = WorkflowStats()
        token = _stats_var.set(stats)
        try:
            batch_quality_retry_step(_make_state(batch_quality_retry_count=0, effort_config=effort))
            # new_retry_count=1 <= max_retries=3 → should count
            assert stats.batch_quality_retries == 1
        finally:
            _stats_var.reset(token)

    def test_does_not_increment_quality_retries_when_exceeds_limit(self):
        effort = EffortConfig(batch_evaluate_quality_max_retries=1)
        stats = WorkflowStats()
        token = _stats_var.set(stats)
        try:
            batch_quality_retry_step(_make_state(batch_quality_retry_count=1, effort_config=effort))
            # new_retry_count=2 > max_retries=1 → must not count
            assert stats.batch_quality_retries == 0
        finally:
            _stats_var.reset(token)

    def test_no_stats_context_does_not_raise(self):
        token = _stats_var.set(None)
        try:
            result = batch_quality_retry_step(_make_state())
            assert result["batch_quality_retry_count"] == 1
        finally:
            _stats_var.reset(token)

    def test_no_effort_config_treats_max_retries_as_zero(self):
        stats = WorkflowStats()
        token = _stats_var.set(stats)
        try:
            batch_quality_retry_step(_make_state(retry_count=0, effort_config=None))
            # new_retry_count=1 > max_retries=0 → must not count
            assert stats.batch_quality_retries == 0
        finally:
            _stats_var.reset(token)
