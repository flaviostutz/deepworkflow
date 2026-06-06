from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_retry_step import increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_increment_retry_step import map_increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.record_output_step import record_output_step
from deepworkflow.shared.types import BatchDefinition, JudgeVerdict


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
        assert increment_retry_step({}) == {"retry_count": 1}

    def test_increments_existing(self):
        assert increment_retry_step({"retry_count": 2}) == {"retry_count": 3}


class TestMapIncrementRetry:
    def test_increments_from_zero(self):
        assert map_increment_retry_step({}) == {"map_retry_count": 1}

    def test_increments_existing(self):
        assert map_increment_retry_step({"map_retry_count": 1}) == {"map_retry_count": 2}


class TestFail:
    def test_returns_error(self):
        assert fail_step({}) == {"error": "Workflow failed"}
