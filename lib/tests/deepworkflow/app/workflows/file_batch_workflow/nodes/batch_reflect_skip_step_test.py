from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.batch_reflect_skip_step import batch_reflect_skip_step


class TestSkipReflectBatchStep:
    def test_returns_empty_files_read(self):
        result = batch_reflect_skip_step({})
        assert result["batch_files_read"] == []

    def test_returns_empty_files_written(self):
        result = batch_reflect_skip_step({})
        assert result["batch_files_written"] == []

    def test_ignores_state_contents(self):
        state = {"batch_files_read": ["a.py"], "batch_files_written": ["b.py"], "batch_quality_retry_count": 3}
        result = batch_reflect_skip_step(state)
        assert result["batch_files_read"] == []
        assert result["batch_files_written"] == []
