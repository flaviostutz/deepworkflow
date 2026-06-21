from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_convergence_repeat_step import (
    batch_convergence_repeat_step,
)


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "batch_convergence_repeat_count": 0,
        "batch_cumulative_files_read": [],
        "batch_cumulative_files_written": [],
        "batch_files_read": [],
        "batch_files_written": [],
        "batch_plan": "some plan",
        "batch_execute_output": "some output",
        "batch_execute_messages": [{"role": "user", "content": "hi"}],
    }
    defaults.update(overrides)
    return defaults


class TestIncrementBatchRepeatStep:
    def test_increments_repeat_count_from_zero(self):
        result = batch_convergence_repeat_step(_make_state())
        assert result["batch_convergence_repeat_count"] == 1

    def test_increments_repeat_count_from_nonzero(self):
        result = batch_convergence_repeat_step(_make_state(batch_convergence_repeat_count=2))
        assert result["batch_convergence_repeat_count"] == 3

    def test_merges_files_read_into_cumulative(self):
        state = _make_state(
            batch_cumulative_files_read=["prior.py"],
            batch_files_read=["a.py", "b.py"],
        )
        result = batch_convergence_repeat_step(state)
        assert result["batch_cumulative_files_read"] == ["prior.py", "a.py", "b.py"]

    def test_merges_files_written_into_cumulative(self):
        state = _make_state(
            batch_cumulative_files_written=["old.py"],
            batch_files_written=["new.py"],
        )
        result = batch_convergence_repeat_step(state)
        assert result["batch_cumulative_files_written"] == ["old.py", "new.py"]

    def test_empty_files_appends_nothing(self):
        state = _make_state(
            batch_cumulative_files_read=["existing.py"],
            batch_files_read=[],
        )
        result = batch_convergence_repeat_step(state)
        assert result["batch_cumulative_files_read"] == ["existing.py"]

    def test_resets_batch_plan(self):
        result = batch_convergence_repeat_step(_make_state(batch_plan="some plan"))
        assert result["batch_plan"] == ""

    def test_resets_execute_output(self):
        result = batch_convergence_repeat_step(_make_state(execute_output="some output"))
        assert result["batch_execute_output"] == ""

    def test_resets_execute_messages(self):
        result = batch_convergence_repeat_step(_make_state(execute_messages=[{"role": "user", "content": "hi"}]))
        assert result["batch_execute_messages"] == []

    def test_resets_files_read(self):
        result = batch_convergence_repeat_step(_make_state(files_read=["a.py"]))
        assert result["batch_files_read"] == []

    def test_resets_files_written(self):
        result = batch_convergence_repeat_step(_make_state(files_written=["a.py"]))
        assert result["batch_files_written"] == []

    def test_does_not_mutate_cumulative_lists(self):
        prior = ["prior.py"]
        state = _make_state(batch_cumulative_files_read=prior, files_read=["new.py"])
        batch_convergence_repeat_step(state)
        # original list should be unchanged
        assert prior == ["prior.py"]
