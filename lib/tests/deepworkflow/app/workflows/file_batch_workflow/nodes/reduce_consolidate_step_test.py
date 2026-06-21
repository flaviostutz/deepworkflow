from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_step import reduce_consolidate_step
from deepworkflow.shared.types import BatchOutput, JudgeLevel


def _make_batch_output(
    batch_files: list[str] | None = None,
    batch_files_read: list[str] | None = None,
    batch_files_written: list[str] | None = None,
    batch_execute_output: str = "done",
    verdict: JudgeLevel = JudgeLevel.OK,
) -> BatchOutput:
    return BatchOutput(
        batch_files=batch_files or ["a.py"],
        evaluate_level=verdict,
        evaluate_feedbacks=[],
        batch_files_read=batch_files_read or [],
        batch_files_written=batch_files_written or [],
        batch_execute_output=batch_execute_output,
    )


class TestReduceConsolidateStep:
    def test_empty_batch_outputs_returns_placeholder(self):
        result = reduce_consolidate_step({"batch_results": []})
        assert "reduce_output" in result
        assert result["reduce_output"]  # non-empty string

    def test_none_batch_outputs_returns_placeholder(self):
        result = reduce_consolidate_step({})
        assert "reduce_output" in result

    def test_single_batch_included_in_output(self):
        output = _make_batch_output(batch_files=["a.py"], batch_execute_output="result text")
        result = reduce_consolidate_step({"batch_results": [output]})
        wf = result["reduce_output"]
        assert "a.py" in wf
        assert "result text" in wf

    def test_output_contains_quality_verdict(self):
        output = _make_batch_output(verdict=JudgeLevel.WARNING)
        result = reduce_consolidate_step({"batch_results": [output]})
        assert "WARNING" in result["reduce_output"]

    def test_multiple_batches_all_included(self):
        outputs = [
            _make_batch_output(batch_files=["a.py"], batch_execute_output="first"),
            _make_batch_output(batch_files=["b.py"], batch_execute_output="second"),
        ]
        result = reduce_consolidate_step({"batch_results": outputs})
        wf = result["reduce_output"]
        assert "a.py" in wf
        assert "b.py" in wf
        assert "first" in wf
        assert "second" in wf

    def test_files_read_and_written_included(self):
        output = _make_batch_output(batch_files_read=["r.py"], batch_files_written=["w.py"])
        result = reduce_consolidate_step({"batch_results": [output]})
        wf = result["reduce_output"]
        assert "r.py" in wf
        assert "w.py" in wf

    def test_batch_sections_separated(self):
        outputs = [
            _make_batch_output(batch_files=["a.py"], batch_execute_output="first"),
            _make_batch_output(batch_files=["b.py"], batch_execute_output="second"),
        ]
        result = reduce_consolidate_step({"batch_results": outputs})
        wf = result["reduce_output"]
        # Both batch headers present
        assert "Batch 1" in wf
        assert "Batch 2" in wf

    def test_empty_execute_output_handled(self):
        output = _make_batch_output(batch_execute_output="")
        result = reduce_consolidate_step({"batch_results": [output]})
        assert result["reduce_output"]  # Should not crash
