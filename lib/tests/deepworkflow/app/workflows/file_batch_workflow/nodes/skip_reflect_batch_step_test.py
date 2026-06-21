from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.skip_reflect_batch_step import skip_reflect_batch_step


class TestSkipReflectBatchStep:
    def test_returns_empty_files_read(self):
        result = skip_reflect_batch_step({})
        assert result["files_read"] == []

    def test_returns_empty_files_written(self):
        result = skip_reflect_batch_step({})
        assert result["files_written"] == []

    def test_ignores_state_contents(self):
        state = {"files_read": ["a.py"], "files_written": ["b.py"], "retry_count": 3}
        result = skip_reflect_batch_step(state)
        assert result["files_read"] == []
        assert result["files_written"] == []
