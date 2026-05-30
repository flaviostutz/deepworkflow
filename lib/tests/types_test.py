from __future__ import annotations

from deepworkflow.shared.types import (
    BatchDefinition,
    BatchOutput,
    JudgeFeedback,
    JudgeVerdict,
    OnMaxRetriesExceeded,
    WorkflowResult,
    WriteOption,
)


class TestJudgeVerdict:
    def test_ordering(self):
        assert JudgeVerdict.OK > JudgeVerdict.INFO
        assert JudgeVerdict.INFO > JudgeVerdict.WARNING
        assert JudgeVerdict.WARNING > JudgeVerdict.ERROR

    def test_comparison_with_minimum(self):
        # OK meets any minimum
        assert JudgeVerdict.OK >= JudgeVerdict.OK
        assert JudgeVerdict.OK >= JudgeVerdict.WARNING
        assert JudgeVerdict.OK >= JudgeVerdict.ERROR

        # WARNING does not meet OK minimum
        assert not (JudgeVerdict.WARNING >= JudgeVerdict.OK)

        # ERROR does not meet WARNING minimum
        assert not (JudgeVerdict.ERROR >= JudgeVerdict.WARNING)


class TestWriteOption:
    def test_values(self):
        assert WriteOption.READ_ONLY == "read-only"
        assert WriteOption.WRITE_ANY == "write-any"
        assert WriteOption.WRITE_ONLY_TASK_FILES == "write-only-task-files"


class TestOnMaxRetriesExceeded:
    def test_values(self):
        assert OnMaxRetriesExceeded.FAIL == "fail"
        assert OnMaxRetriesExceeded.CONTINUE == "continue"


class TestJudgeFeedback:
    def test_creation(self):
        fb = JudgeFeedback(file="test.py", type=JudgeVerdict.WARNING, description="needs fix")
        assert fb.file == "test.py"
        assert fb.type == JudgeVerdict.WARNING
        assert fb.description == "needs fix"

    def test_immutable(self):
        fb = JudgeFeedback(file="test.py", type=JudgeVerdict.OK, description="good")
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
            task_files=["a.py", "b.py"],
            judge_verdict=JudgeVerdict.OK,
            judge_feedbacks=[],
            files_read=["a.py"],
            files_written=[],
            execute_output="All good",
        )
        assert output.task_files == ["a.py", "b.py"]
        assert output.judge_verdict == JudgeVerdict.OK
        assert output.execute_output == "All good"


class TestWorkflowResult:
    def test_creation(self):
        result = WorkflowResult(thread_id="abc-123", output="Done", status="completed")
        assert result.thread_id == "abc-123"
        assert result.output == "Done"
        assert result.status == "completed"
