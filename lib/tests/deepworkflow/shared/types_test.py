from __future__ import annotations

from deepworkflow.shared.types import (
    BatchDefinition,
    BatchOutput,
    EvaluateFeedback,
    JudgeLevel,
    OnMaxRetriesExceeded,
    WorkflowResult,
    WriteOption,
)


class TestEvaluateVerdict:
    def test_ordering(self):
        assert JudgeLevel.OK > JudgeLevel.INFO
        assert JudgeLevel.INFO > JudgeLevel.WARNING
        assert JudgeLevel.WARNING > JudgeLevel.ERROR

    def test_comparison_with_minimum(self):
        # OK meets any minimum
        assert JudgeLevel.OK >= JudgeLevel.OK
        assert JudgeLevel.OK >= JudgeLevel.WARNING
        assert JudgeLevel.OK >= JudgeLevel.ERROR

        # WARNING does not meet OK minimum
        assert not (JudgeLevel.WARNING >= JudgeLevel.OK)

        # ERROR does not meet WARNING minimum
        assert not (JudgeLevel.ERROR >= JudgeLevel.WARNING)


class TestWriteOption:
    def test_values(self):
        assert WriteOption.READ_ONLY == "read-only"
        assert WriteOption.WRITE_ANY == "write-any"
        assert WriteOption.WRITE_ONLY_TASK_FILES == "write-only-task-files"


class TestOnMaxRetriesExceeded:
    def test_values(self):
        assert OnMaxRetriesExceeded.FAIL == "fail"
        assert OnMaxRetriesExceeded.CONTINUE == "continue"


class TestEvaluateFeedback:
    def test_creation(self):
        fb = EvaluateFeedback(file="test.py", type=JudgeLevel.WARNING, description="needs fix")
        assert fb.file == "test.py"
        assert fb.type == JudgeLevel.WARNING
        assert fb.description == "needs fix"

    def test_immutable(self):
        fb = EvaluateFeedback(file="test.py", type=JudgeLevel.OK, description="good")
        with __import__("pytest").raises(AttributeError):
            fb.file = "other.py"  # type: ignore[misc]


class TestBatchDefinition:
    def test_creation(self):
        bd = BatchDefinition(batch_files=["a.py", "b.py"], batch_instructions="Review these")
        assert bd.batch_files == ["a.py", "b.py"]
        assert bd.batch_instructions == "Review these"

    def test_immutable(self):
        bd = BatchDefinition(batch_files=["a.py"], batch_instructions="x")
        with __import__("pytest").raises(AttributeError):
            bd.batch_files = []  # type: ignore[misc]


class TestBatchOutput:
    def test_creation(self):
        output = BatchOutput(
            batch_files=["a.py", "b.py"],
            evaluate_level=JudgeLevel.OK,
            evaluate_feedbacks=[],
            batch_files_read=["a.py"],
            batch_files_written=[],
            batch_execute_output="All good",
        )
        assert output.batch_files == ["a.py", "b.py"]
        assert output.evaluate_level == JudgeLevel.OK
        assert output.batch_execute_output == "All good"


class TestWorkflowResult:
    def test_creation(self):
        result = WorkflowResult(thread_id="abc-123", output="Done", status="completed")
        assert result.thread_id == "abc-123"
        assert result.output == "Done"
        assert result.status == "completed"
