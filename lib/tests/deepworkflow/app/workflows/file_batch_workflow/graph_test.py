from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.graph_log import (
    _finding_lines,
    _finding_summary,
    _log_evaluate_convergence_post,
    _log_evaluate_map_batches_post,
    _log_evaluate_quality_post,
    _log_execute_batch_post,
    _log_map_batches_post,
    _log_map_batches_pre,
    _log_plan_batch_post,
    _log_plan_batch_pre,
    _log_reduce_consolidate_post,
    _log_reflect_batch_post,
    _log_resolve_globs_post,
    _truncate,
)
from deepworkflow.app.workflows.file_batch_workflow.nodes.fail_step import fail_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.increment_retry_step import increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.map_increment_retry_step import map_increment_retry_step
from deepworkflow.app.workflows.file_batch_workflow.nodes.record_output_step import record_output_step
from deepworkflow.shared.types import (
    BatchDefinition,
    BatchOutput,
    JudgeFinding,
    JudgeLevel,
    JudgeVerdict,
    WorkflowLogLevel,
)


class TestRecordBatchOutput:
    def test_records_and_advances_index(self):
        state = {
            "current_batch_index": 0,
            "task_file_batches": [
                BatchDefinition(batch_files=["a.py"], batch_instructions="do it"),
            ],
            "evaluate_quality_verdict": JudgeLevel.OK,
            "evaluate_quality_feedbacks": [],
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
        assert result["batch_outputs"][0].evaluate_quality_verdict == JudgeLevel.OK

    def test_appends_to_existing_outputs(self):
        from deepworkflow.shared.types import BatchOutput

        existing = BatchOutput(
            task_files=["x.py"],
            evaluate_quality_verdict=JudgeLevel.OK,
            evaluate_quality_feedbacks=[],
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
            "evaluate_quality_verdict": JudgeLevel.INFO,
            "evaluate_quality_feedbacks": [],
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


def _warn_finding(title: str = "issue") -> JudgeFinding:
    return JudgeFinding(level=JudgeLevel.WARNING, title=title, reason="because")


def _ok_finding(title: str = "fine") -> JudgeFinding:
    return JudgeFinding(level=JudgeLevel.OK, title=title)


class TestFindingSummary:
    def test_no_findings_returns_ok(self):
        v = JudgeVerdict(verdict=JudgeLevel.OK, findings=[])
        assert _finding_summary(v) == "OK"

    def test_counts_error_only(self):
        v = JudgeVerdict(verdict=JudgeLevel.ERROR, findings=[JudgeFinding(level=JudgeLevel.ERROR, title="err")])
        assert _finding_summary(v) == "1 error"

    def test_counts_warning_only(self):
        v = JudgeVerdict(verdict=JudgeLevel.WARNING, findings=[_warn_finding()])
        assert _finding_summary(v) == "1 warning"

    def test_counts_info_only(self):
        v = JudgeVerdict(verdict=JudgeLevel.INFO, findings=[JudgeFinding(level=JudgeLevel.INFO, title="note")])
        assert _finding_summary(v) == "1 info"

    def test_mixed_counts(self):
        findings = [
            JudgeFinding(level=JudgeLevel.ERROR, title="e"),
            JudgeFinding(level=JudgeLevel.WARNING, title="w1"),
            JudgeFinding(level=JudgeLevel.WARNING, title="w2"),
            JudgeFinding(level=JudgeLevel.INFO, title="i"),
        ]
        v = JudgeVerdict(verdict=JudgeLevel.ERROR, findings=findings)
        assert _finding_summary(v) == "1 error; 2 warning; 1 info"

    def test_ok_findings_not_counted(self):
        v = JudgeVerdict(verdict=JudgeLevel.OK, findings=[_ok_finding()])
        assert _finding_summary(v) == "OK"

    def test_zero_counts_omitted(self):
        v = JudgeVerdict(verdict=JudgeLevel.ERROR, findings=[JudgeFinding(level=JudgeLevel.ERROR, title="err")])
        result = _finding_summary(v)
        assert "warning" not in result
        assert "info" not in result


class TestFindingLines:
    def test_empty_returns_empty(self):
        v = JudgeVerdict(verdict=JudgeLevel.OK, findings=[])
        assert _finding_lines(v, WorkflowLogLevel.INFO) == []

    def test_single_warning_finding_no_reason(self):
        v = JudgeVerdict(
            verdict=JudgeLevel.WARNING, findings=[JudgeFinding(level=JudgeLevel.WARNING, title="bad code")]
        )
        lines = _finding_lines(v, WorkflowLogLevel.INFO)
        assert lines == ["[WARNING] bad code"]

    def test_single_warning_finding_with_reason(self):
        v = JudgeVerdict(
            verdict=JudgeLevel.WARNING,
            findings=[JudgeFinding(level=JudgeLevel.WARNING, title="issue", reason="fix it")],
        )
        lines = _finding_lines(v, WorkflowLogLevel.INFO)
        assert lines == ["[WARNING] issue \u2014 fix it"]

    def test_ok_finding_excluded_at_info_level(self):
        v = JudgeVerdict(verdict=JudgeLevel.OK, findings=[_ok_finding()])
        lines = _finding_lines(v, WorkflowLogLevel.INFO)
        assert lines == []

    def test_ok_finding_included_at_debug_level(self):
        v = JudgeVerdict(verdict=JudgeLevel.OK, findings=[_ok_finding("all good")])
        lines = _finding_lines(v, WorkflowLogLevel.DEBUG)
        assert len(lines) == 1
        assert "[OK] all good" in lines[0]

    def test_multiple_findings(self):
        findings = [
            JudgeFinding(level=JudgeLevel.ERROR, title="err"),
            JudgeFinding(level=JudgeLevel.INFO, title="note"),
        ]
        v = JudgeVerdict(verdict=JudgeLevel.ERROR, findings=findings)
        lines = _finding_lines(v, WorkflowLogLevel.INFO)
        assert len(lines) == 2
        assert "[ERROR] err" in lines[0]
        assert "[INFO] note" in lines[1]


class TestLogCallbacks:
    """Tests for _log_* callback functions used by wrap_node."""

    def _make_batch(self, files: list[str], instructions: str = "do it") -> BatchDefinition:
        return BatchDefinition(batch_files=files, batch_instructions=instructions)

    def _make_batch_output(self, files: list[str] | None = None) -> BatchOutput:
        return BatchOutput(
            task_files=files or ["a.py"],
            evaluate_quality_verdict=JudgeLevel.OK,
            evaluate_quality_feedbacks=[],
            files_read=["a.py"],
            files_written=["b.py"],
            execute_output="done",
        )

    def _warn_verdict(self, title: str = "issue") -> JudgeVerdict:
        return JudgeVerdict(
            verdict=JudgeLevel.WARNING,
            findings=[JudgeFinding(level=JudgeLevel.WARNING, title=title, reason="because")],
        )

    def _ok_verdict(self) -> JudgeVerdict:
        return JudgeVerdict(verdict=JudgeLevel.OK, findings=[_ok_finding()])

    def test_resolve_globs_post_count(self):
        result = _log_resolve_globs_post({}, {"task_files": ["a.py", "b.py"]}, WorkflowLogLevel.INFO)
        assert result == ["2 files"]

    def test_map_batches_pre_info_truncates(self):
        config_mock = type("C", (), {"task_instructions": " ".join(["word"] * 50)})()
        state = {"config": config_mock}
        lines = _log_map_batches_pre(state, WorkflowLogLevel.INFO)
        assert lines[0].startswith("task: ")
        assert "…" in lines[0]

    def test_map_batches_pre_debug_full(self):
        long_task = " ".join(["word"] * 50)
        config_mock = type("C", (), {"task_instructions": long_task})()
        state = {"config": config_mock}
        lines = _log_map_batches_pre(state, WorkflowLogLevel.DEBUG)
        assert lines[0] == f"task: {long_task}"

    def test_map_batches_post_info_truncates_overview(self):
        long_overview = " ".join(["word"] * 50)
        batches = [self._make_batch(["a.py"])]
        result = {"task_file_batches": batches, "task_overview": long_overview}
        lines = _log_map_batches_post({}, result, WorkflowLogLevel.INFO)
        assert "…" in lines[0]

    def test_map_batches_post_debug_full_overview(self):
        long_overview = " ".join(["word"] * 50)
        batches = [self._make_batch(["a.py"])]
        result = {"task_file_batches": batches, "task_overview": long_overview}
        lines = _log_map_batches_post({}, result, WorkflowLogLevel.DEBUG)
        assert lines[0] == f"overview: {long_overview}"

    def test_map_batches_post_error_returns_empty(self):
        assert _log_map_batches_post({}, {"error": True}, WorkflowLogLevel.INFO) == []

    def test_evaluate_map_batches_post_summary_when_warning(self):
        verdict = self._warn_verdict("map issue")
        lines = _log_evaluate_map_batches_post({}, {"map_evaluate_judge_verdict": verdict}, WorkflowLogLevel.INFO)
        assert len(lines) >= 1
        assert "WARNING" in lines[0]

    def test_evaluate_map_batches_post_debug_includes_findings(self):
        verdict = self._warn_verdict("map issue")
        lines = _log_evaluate_map_batches_post({}, {"map_evaluate_judge_verdict": verdict}, WorkflowLogLevel.DEBUG)
        assert any("[WARNING]" in line for line in lines)

    def test_plan_batch_pre_info_truncates(self):
        long_instructions = " ".join(["word"] * 50)
        batches = [self._make_batch(["a.py"], instructions=long_instructions)]
        state = {"current_batch_index": 0, "task_file_batches": batches}
        lines = _log_plan_batch_pre(state, WorkflowLogLevel.INFO)
        assert "…" in lines[0]

    def test_plan_batch_pre_debug_full(self):
        long_instructions = " ".join(["word"] * 50)
        batches = [self._make_batch(["a.py"], instructions=long_instructions)]
        state = {"current_batch_index": 0, "task_file_batches": batches}
        lines = _log_plan_batch_pre(state, WorkflowLogLevel.DEBUG)
        assert lines[0] == f"batch instructions: {long_instructions}"

    def test_plan_batch_post_info_truncates(self):
        long_plan = " ".join(["word"] * 50)
        lines = _log_plan_batch_post({}, {"plan_output": long_plan}, WorkflowLogLevel.INFO)
        assert "…" in lines[0]

    def test_plan_batch_post_debug_full(self):
        long_plan = " ".join(["word"] * 50)
        lines = _log_plan_batch_post({}, {"plan_output": long_plan}, WorkflowLogLevel.DEBUG)
        assert lines[0] == f"plan: {long_plan}"

    def test_execute_batch_post_info_truncates(self):
        long_output = " ".join(["word"] * 50)
        lines = _log_execute_batch_post({}, {"execute_output": long_output}, WorkflowLogLevel.INFO)
        assert "…" in lines[0]

    def test_execute_batch_post_debug_full(self):
        long_output = " ".join(["word"] * 50)
        lines = _log_execute_batch_post({}, {"execute_output": long_output}, WorkflowLogLevel.DEBUG)
        assert lines[0] == f"output: {long_output}"

    def test_reflect_batch_post(self):
        result = {"files_written": ["a.py", "b.py"], "files_read": ["c.py"]}
        lines = _log_reflect_batch_post({}, result, WorkflowLogLevel.INFO)
        assert lines == ["2 files written; 1 files read"]

    def test_evaluate_convergence_post_info_truncates(self):
        long_output = " ".join(["word"] * 100)
        lines = _log_evaluate_convergence_post({}, {"batch_convergence_output": long_output}, WorkflowLogLevel.INFO)
        assert "…" in lines[0]

    def test_evaluate_convergence_post_debug_full(self):
        long_output = " ".join(["word"] * 100)
        lines = _log_evaluate_convergence_post({}, {"batch_convergence_output": long_output}, WorkflowLogLevel.DEBUG)
        assert lines[0] == long_output

    def test_evaluate_convergence_post_verdict_summary(self):
        verdict = self._warn_verdict("convergence check")
        lines = _log_evaluate_convergence_post({}, {"batch_convergence_verdict": verdict}, WorkflowLogLevel.INFO)
        assert "WARNING" in lines[0]

    def test_evaluate_quality_post_summary_when_warning(self):
        verdict = self._warn_verdict("quality issue")
        lines = _log_evaluate_quality_post({}, {"evaluate_quality_judge_verdict": verdict}, WorkflowLogLevel.INFO)
        assert len(lines) >= 1
        assert "WARNING" in lines[0]

    def test_evaluate_quality_post_debug_includes_findings(self):
        verdict = self._warn_verdict("quality detail")
        lines = _log_evaluate_quality_post({}, {"evaluate_quality_judge_verdict": verdict}, WorkflowLogLevel.DEBUG)
        assert any("[WARNING]" in line for line in lines)

    def test_reduce_consolidate_post(self):
        bo = self._make_batch_output()
        state = {"batch_outputs": [bo]}
        result = {"workflow_output": "final output"}
        lines = _log_reduce_consolidate_post(state, result, WorkflowLogLevel.INFO)
        assert "output:" in lines[0]
        assert "final output" in lines[0]
        assert "1 files read; 1 files written" in lines[1]
