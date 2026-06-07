from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_batch_repeat_step import (
    increment_batch_repeat_step,
)


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "batch_repeat_count": 0,
        "cumulative_files_read": [],
        "cumulative_files_written": [],
        "files_read": [],
        "files_written": [],
        "plan_output": "some plan",
        "execute_output": "some output",
        "execute_messages": [{"role": "user", "content": "hi"}],
    }
    defaults.update(overrides)
    return defaults


class TestIncrementBatchRepeatStep:
    def test_increments_repeat_count_from_zero(self):
        result = increment_batch_repeat_step(_make_state())
        assert result["batch_repeat_count"] == 1

    def test_increments_repeat_count_from_nonzero(self):
        result = increment_batch_repeat_step(_make_state(batch_repeat_count=2))
        assert result["batch_repeat_count"] == 3

    def test_merges_files_read_into_cumulative(self):
        state = _make_state(
            cumulative_files_read=["prior.py"],
            files_read=["a.py", "b.py"],
        )
        result = increment_batch_repeat_step(state)
        assert result["cumulative_files_read"] == ["prior.py", "a.py", "b.py"]

    def test_merges_files_written_into_cumulative(self):
        state = _make_state(
            cumulative_files_written=["old.py"],
            files_written=["new.py"],
        )
        result = increment_batch_repeat_step(state)
        assert result["cumulative_files_written"] == ["old.py", "new.py"]

    def test_empty_files_appends_nothing(self):
        state = _make_state(
            cumulative_files_read=["existing.py"],
            files_read=[],
        )
        result = increment_batch_repeat_step(state)
        assert result["cumulative_files_read"] == ["existing.py"]

    def test_resets_plan_output(self):
        result = increment_batch_repeat_step(_make_state(plan_output="some plan"))
        assert result["plan_output"] == ""

    def test_resets_execute_output(self):
        result = increment_batch_repeat_step(_make_state(execute_output="some output"))
        assert result["execute_output"] == ""

    def test_resets_execute_messages(self):
        result = increment_batch_repeat_step(_make_state(execute_messages=[{"role": "user", "content": "hi"}]))
        assert result["execute_messages"] == []

    def test_resets_files_read(self):
        result = increment_batch_repeat_step(_make_state(files_read=["a.py"]))
        assert result["files_read"] == []

    def test_resets_files_written(self):
        result = increment_batch_repeat_step(_make_state(files_written=["a.py"]))
        assert result["files_written"] == []

    def test_does_not_mutate_cumulative_lists(self):
        prior = ["prior.py"]
        state = _make_state(cumulative_files_read=prior, files_read=["new.py"])
        increment_batch_repeat_step(state)
        # original list should be unchanged
        assert prior == ["prior.py"]
