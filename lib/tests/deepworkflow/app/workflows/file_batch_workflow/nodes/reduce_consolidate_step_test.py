from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_step import reduce_consolidate_step
from deepworkflow.shared.types import BatchOutput, JudgeLevel


def _make_batch_output(
    task_files: list[str] | None = None,
    files_read: list[str] | None = None,
    files_written: list[str] | None = None,
    execute_output: str = "done",
    verdict: JudgeLevel = JudgeLevel.OK,
) -> BatchOutput:
    return BatchOutput(
        task_files=task_files or ["a.py"],
        evaluate_quality_verdict=verdict,
        evaluate_quality_feedbacks=[],
        files_read=files_read or [],
        files_written=files_written or [],
        execute_output=execute_output,
    )


class TestReduceConsolidateStep:
    def test_empty_batch_outputs_returns_placeholder(self):
        result = reduce_consolidate_step({"batch_outputs": []})
        assert "workflow_output" in result
        assert result["workflow_output"]  # non-empty string

    def test_none_batch_outputs_returns_placeholder(self):
        result = reduce_consolidate_step({})
        assert "workflow_output" in result

    def test_single_batch_included_in_output(self):
        output = _make_batch_output(task_files=["a.py"], execute_output="result text")
        result = reduce_consolidate_step({"batch_outputs": [output]})
        wf = result["workflow_output"]
        assert "a.py" in wf
        assert "result text" in wf

    def test_output_contains_quality_verdict(self):
        output = _make_batch_output(verdict=JudgeLevel.WARNING)
        result = reduce_consolidate_step({"batch_outputs": [output]})
        assert "WARNING" in result["workflow_output"]

    def test_multiple_batches_all_included(self):
        outputs = [
            _make_batch_output(task_files=["a.py"], execute_output="first"),
            _make_batch_output(task_files=["b.py"], execute_output="second"),
        ]
        result = reduce_consolidate_step({"batch_outputs": outputs})
        wf = result["workflow_output"]
        assert "a.py" in wf
        assert "b.py" in wf
        assert "first" in wf
        assert "second" in wf

    def test_files_read_and_written_included(self):
        output = _make_batch_output(files_read=["r.py"], files_written=["w.py"])
        result = reduce_consolidate_step({"batch_outputs": [output]})
        wf = result["workflow_output"]
        assert "r.py" in wf
        assert "w.py" in wf

    def test_batch_sections_separated(self):
        outputs = [
            _make_batch_output(task_files=["a.py"], execute_output="first"),
            _make_batch_output(task_files=["b.py"], execute_output="second"),
        ]
        result = reduce_consolidate_step({"batch_outputs": outputs})
        wf = result["workflow_output"]
        # Both batch headers present
        assert "Batch 1" in wf
        assert "Batch 2" in wf

    def test_empty_execute_output_handled(self):
        output = _make_batch_output(execute_output="")
        result = reduce_consolidate_step({"batch_outputs": [output]})
        assert result["workflow_output"]  # Should not crash
