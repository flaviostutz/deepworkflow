from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.graph import _feedback_summary, _truncate
from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_retry_step import increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_increment_retry_step import map_increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.record_output_step import record_output_step
from deepworkflow.shared.types import BatchDefinition, JudgeFeedback, JudgeVerdict


class TestRecordBatchOutput:
    def test_records_and_advances_index(self):
        state = {
            "current_batch_index": 0,
            "task_file_batches": [
                BatchDefinition(batch_files=["a.py"], batch_instructions="do it"),
            ],
            "judge_verdict": JudgeVerdict.OK,
            "judge_feedbacks": [],
            "files_read": ["a.py"],
            "files_written": ["a.py"],
            "execute_output": "done",
            "batch_outputs": [],
        }
        result = record_output_step(state)
        assert result["current_batch_index"] == 1
        assert result["retry_count"] == 0
        assert len(result["batch_outputs"]) == 1
        assert result["batch_outputs"][0].task_files == ["a.py"]
        assert result["batch_outputs"][0].judge_verdict == JudgeVerdict.OK

    def test_appends_to_existing_outputs(self):
        from deepworkflow.shared.types import BatchOutput

        existing = BatchOutput(
            task_files=["x.py"],
            judge_verdict=JudgeVerdict.OK,
            judge_feedbacks=[],
            files_read=[],
            files_written=[],
            execute_output="prev",
        )
        state = {
            "current_batch_index": 1,
            "task_file_batches": [
                BatchDefinition(batch_files=["x.py"], batch_instructions=None),
                BatchDefinition(batch_files=["y.py"], batch_instructions=None),
            ],
            "judge_verdict": JudgeVerdict.INFO,
            "judge_feedbacks": [],
            "batch_outputs": [existing],
            "execute_output": "new",
        }
        result = record_output_step(state)
        assert len(result["batch_outputs"]) == 2
        assert result["current_batch_index"] == 2


class TestIncrementRetry:
    def test_increments_from_zero(self):
        result = increment_retry_step({})
        assert result["retry_count"] == 1
        assert result["batch_repeat_count"] == 0
        assert result["cumulative_files_read"] == []
        assert result["cumulative_files_written"] == []

    def test_increments_existing(self):
        result = increment_retry_step({"retry_count": 2})
        assert result["retry_count"] == 3
        assert result["batch_repeat_count"] == 0
        assert result["cumulative_files_read"] == []
        assert result["cumulative_files_written"] == []


class TestMapIncrementRetry:
    def test_increments_from_zero(self):
        assert map_increment_retry_step({}) == {"map_retry_count": 1}

    def test_increments_existing(self):
        assert map_increment_retry_step({"map_retry_count": 1}) == {"map_retry_count": 2}


class TestFail:
    def test_returns_error(self):
        assert fail_step({}) == {"error": "Workflow failed"}


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello world", 30) == "hello world"

    def test_exact_word_count_unchanged(self):
        words = " ".join(["w"] * 5)
        assert _truncate(words, 5) == words

    def test_long_text_truncated(self):
        words = " ".join(["w"] * 10)
        result = _truncate(words, 3)
        assert result == "w w w\u2026"

    def test_empty_string(self):
        assert _truncate("", 5) == ""


class TestFeedbackSummary:
    def test_no_feedbacks_returns_ok(self):
        assert _feedback_summary([]) == "OK"

    def test_counts_error_only(self):
        fb = JudgeFeedback(file="f", type=JudgeVerdict.ERROR, description="d")
        assert _feedback_summary([fb]) == "1 error"

    def test_counts_warning_only(self):
        fb = JudgeFeedback(file="f", type=JudgeVerdict.WARNING, description="d")
        assert _feedback_summary([fb]) == "1 warning"

    def test_counts_info_only(self):
        fb = JudgeFeedback(file="f", type=JudgeVerdict.INFO, description="d")
        assert _feedback_summary([fb]) == "1 info"

    def test_mixed_counts(self):
        feedbacks = [
            JudgeFeedback(file="f", type=JudgeVerdict.ERROR, description="d"),
            JudgeFeedback(file="f", type=JudgeVerdict.WARNING, description="d"),
            JudgeFeedback(file="f", type=JudgeVerdict.WARNING, description="d"),
            JudgeFeedback(file="f", type=JudgeVerdict.INFO, description="d"),
        ]
        assert _feedback_summary(feedbacks) == "1 error; 2 warning; 1 info"

    def test_ok_feedbacks_not_counted(self):
        fb = JudgeFeedback(file="f", type=JudgeVerdict.OK, description="d")
        assert _feedback_summary([fb]) == "OK"

    def test_zero_counts_omitted(self):
        fb = JudgeFeedback(file="f", type=JudgeVerdict.ERROR, description="d")
        result = _feedback_summary([fb])
        assert "warning" not in result
        assert "info" not in result
