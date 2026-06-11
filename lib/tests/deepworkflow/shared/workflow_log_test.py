from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepworkflow.shared.types import BatchOutput, JudgeVerdict, WorkflowLogLevel
from deepworkflow.shared.workflow_log import (
    WorkflowStats,
    WorkflowStatsCallback,
    _find_model_price,
    _format_tokens,
    new_run_stats,
    print_summary,
    wrap_node,
    wrap_route,
)


def _make_config(log_level: WorkflowLogLevel) -> MagicMock:
    config = MagicMock()
    config.log_level = log_level
    return config


def _make_state(log_level: WorkflowLogLevel, batch_index: int = 0) -> dict:
    return {"config": _make_config(log_level), "current_batch_index": batch_index}


class TestWrapNodeLogLevelNone:
    def test_no_output(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={"x": 1})
        wrapped = wrap_node("my_node", fn)
        wrapped(_make_state(WorkflowLogLevel.NONE))
        assert capsys.readouterr().out == ""

    def test_pre_post_not_called(self) -> None:
        pre = MagicMock(return_value=[])
        post = MagicMock(return_value=[])
        fn = MagicMock(return_value={})
        wrapped = wrap_node("n", fn, log_pre_fn=pre, log_post_fn=post)
        wrapped(_make_state(WorkflowLogLevel.NONE))
        pre.assert_not_called()
        post.assert_not_called()


class TestWrapNodeLogLevelInfo:
    def test_header_printed(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        wrap_node("my_node", fn)(_make_state(WorkflowLogLevel.INFO))
        assert "> my_node" in capsys.readouterr().out

    def test_no_start_end_words(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        wrap_node("my_node", fn)(_make_state(WorkflowLogLevel.INFO))
        out = capsys.readouterr().out
        assert "start" not in out
        assert "end" not in out

    def test_elapsed_printed(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        wrap_node("my_node", fn)(_make_state(WorkflowLogLevel.INFO))
        assert "  - elapsed:" in capsys.readouterr().out

    def test_pre_lines_printed(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        pre = MagicMock(return_value=["task: do something"])
        wrap_node("n", fn, log_pre_fn=pre)(_make_state(WorkflowLogLevel.INFO))
        out = capsys.readouterr().out
        assert "  - (in) task: do something" in out

    def test_post_lines_printed(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={"x": 1})
        post = MagicMock(return_value=["3 batches; 3 files/batch"])
        state = _make_state(WorkflowLogLevel.INFO)
        wrap_node("n", fn, log_post_fn=post)(state)
        out = capsys.readouterr().out
        assert "  - (out) 3 batches; 3 files/batch" in out
        post.assert_called_once_with(state, {"x": 1})

    def test_multiple_post_lines(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        post = MagicMock(return_value=["line one", "line two"])
        wrap_node("n", fn, log_post_fn=post)(_make_state(WorkflowLogLevel.INFO))
        out = capsys.readouterr().out
        assert "  - (out) line one" in out
        assert "  - (out) line two" in out

    def test_empty_post_lines_no_output(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        post = MagicMock(return_value=[])
        wrap_node("n", fn, log_post_fn=post)(_make_state(WorkflowLogLevel.INFO))
        assert "(out)" not in capsys.readouterr().out

    def test_show_batch_index(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        wrap_node("my_node", fn, show_batch_index=True)(_make_state(WorkflowLogLevel.INFO, batch_index=2))
        assert "> my_node[:2]" in capsys.readouterr().out

    def test_show_batch_index_zero(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value={})
        wrap_node("node", fn, show_batch_index=True)(_make_state(WorkflowLogLevel.INFO, batch_index=0))
        assert "> node[:0]" in capsys.readouterr().out


class TestWrapNodeStats:
    def test_quality_retry_incremented(self) -> None:
        from deepworkflow.shared.workflow_log import WorkflowStats, _stats_var

        stats = WorkflowStats()
        _stats_var.set(stats)
        fn = MagicMock(return_value={})
        wrap_node("n", fn, stat="quality_retry")(_make_state(WorkflowLogLevel.NONE))
        assert stats.quality_retries == 1

    def test_progress_retry_incremented(self) -> None:
        from deepworkflow.shared.workflow_log import WorkflowStats, _stats_var

        stats = WorkflowStats()
        _stats_var.set(stats)
        fn = MagicMock(return_value={})
        wrap_node("n", fn, stat="progress_retry")(_make_state(WorkflowLogLevel.NONE))
        assert stats.progress_retries == 1


class TestNewRunStats:
    def test_returns_stats_instance(self) -> None:
        from deepworkflow.shared.workflow_log import WorkflowStats, _stats_var

        stats = new_run_stats()
        assert isinstance(stats, WorkflowStats)
        assert _stats_var.get() is stats

    def test_fresh_stats_zeroed(self) -> None:
        stats = new_run_stats()
        assert stats.quality_retries == 0
        assert stats.progress_retries == 0
        assert stats.model_invocations == 0


class TestWrapRoute:
    def test_none_no_output(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value="ok")
        wrap_route("my_route", fn)(_make_state(WorkflowLogLevel.NONE))
        assert capsys.readouterr().out == ""

    def test_info_prints_result(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value="pass")
        wrap_route("my_route", fn)(_make_state(WorkflowLogLevel.INFO))
        assert "> my_route [pass]" in capsys.readouterr().out

    def test_info_no_start_line(self, capsys: pytest.CaptureFixture) -> None:
        fn = MagicMock(return_value="pass")
        wrap_route("my_route", fn)(_make_state(WorkflowLogLevel.INFO))
        assert "start" not in capsys.readouterr().out

    def test_returns_fn_result(self) -> None:
        fn = MagicMock(return_value="next_node")
        result = wrap_route("r", fn)(_make_state(WorkflowLogLevel.NONE))
        assert result == "next_node"


class TestFormatTokens:
    def test_small(self) -> None:
        assert _format_tokens(500) == "500"

    def test_thousands(self) -> None:
        assert _format_tokens(3_500) == "3k"

    def test_millions(self) -> None:
        result = _format_tokens(2_500_000)
        assert result == "2.5M"

    def test_exactly_one_million(self) -> None:
        assert _format_tokens(1_000_000) == "1.0M"


class TestFindModelPrice:
    def test_exact_match(self) -> None:
        price = _find_model_price("gpt-4o")
        assert price is not None
        assert price == (2.5, 10.0)

    def test_partial_match(self) -> None:
        price = _find_model_price("my-gpt-4o-deployment")
        assert price is not None

    def test_no_match_returns_none(self) -> None:
        assert _find_model_price("unknown-model-xyz") is None


class TestPrintSummary:
    def _make_workflow_result(self, status: str = "completed", output: str = "done"):
        from deepworkflow.shared.types import WorkflowResult

        return WorkflowResult(thread_id="t1", output=output, status=status)

    def _make_batch_output(self, verdict: JudgeVerdict = JudgeVerdict.OK) -> BatchOutput:
        return BatchOutput(
            task_files=["a.py"],
            judge_verdict=verdict,
            judge_feedbacks=[],
            files_read=["a.py"],
            files_written=["a.py"],
            execute_output="done",
        )

    def _make_config(self, judge_skip: bool = False):
        config = MagicMock()
        config.judge_min = JudgeVerdict.WARNING
        config.judge_skip = judge_skip
        return config

    def test_prints_summary_header(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        final_state = {"batch_outputs": [self._make_batch_output()], "config": self._make_config()}
        print_summary(stats, final_state, self._make_workflow_result())
        assert "> summary:" in capsys.readouterr().out

    def test_failed_status_label(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        final_state: dict = {}
        print_summary(stats, final_state, self._make_workflow_result(status="failed", output="err"))
        assert "FAILED" in capsys.readouterr().out

    def test_ok_status_label(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        final_state = {"batch_outputs": [self._make_batch_output(JudgeVerdict.OK)], "config": self._make_config()}
        print_summary(stats, final_state, self._make_workflow_result())
        assert "result: OK" in capsys.readouterr().out

    def test_judge_skip_shows_na(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        final_state = {"batch_outputs": [self._make_batch_output()], "config": self._make_config(judge_skip=True)}
        print_summary(stats, final_state, self._make_workflow_result())
        assert "N/A" in capsys.readouterr().out

    def test_model_tokens_shown(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        stats.tokens_by_model = {"gpt-4o": [1000, 200]}
        final_state = {"batch_outputs": [], "config": self._make_config()}
        print_summary(stats, final_state, self._make_workflow_result())
        out = capsys.readouterr().out
        assert "models total" in out
        assert "~US$" in out

    def test_empty_state(self, capsys: pytest.CaptureFixture) -> None:
        stats = WorkflowStats()
        print_summary(stats, {}, self._make_workflow_result())
        out = capsys.readouterr().out
        assert "> summary:" in out


class TestWorkflowStatsCallback:
    def _make_llm_result(self, in_tokens: int = 10, out_tokens: int = 5) -> MagicMock:
        result = MagicMock()
        result.llm_output = {"token_usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens}}
        return result

    def _make_serialized(self, name: str = "gpt-4o") -> dict:
        return {"name": name}

    def test_on_llm_start_tracked_in_end(self) -> None:
        from deepworkflow.shared.workflow_log import _stats_var

        stats = WorkflowStats()
        _stats_var.set(stats)
        cb = WorkflowStatsCallback(WorkflowLogLevel.INFO)
        run_id = "run-1"
        cb.on_llm_start(self._make_serialized("gpt-4o"), [], run_id=run_id)
        cb.on_llm_end(self._make_llm_result(), run_id=run_id)
        assert "gpt-4o" in stats.tokens_by_model

    def test_on_llm_end_accumulates_tokens(self) -> None:
        from deepworkflow.shared.workflow_log import _stats_var

        stats = WorkflowStats()
        _stats_var.set(stats)
        cb = WorkflowStatsCallback(WorkflowLogLevel.INFO)
        run_id = "run-2"
        cb.on_llm_start(self._make_serialized("gpt-4o"), [], run_id=run_id)
        cb.on_llm_end(self._make_llm_result(in_tokens=100, out_tokens=50), run_id=run_id)
        assert stats.model_invocations == 1
        assert stats.tokens_by_model["gpt-4o"] == [100, 50]

    def test_on_llm_end_no_stats_no_error(self) -> None:
        from deepworkflow.shared.workflow_log import _stats_var

        _stats_var.set(None)
        cb = WorkflowStatsCallback(WorkflowLogLevel.NONE)
        run_id = "run-3"
        cb.on_llm_start(self._make_serialized("gpt-4o"), [], run_id=run_id)
        cb.on_llm_end(self._make_llm_result(), run_id=run_id)  # should not raise

    def test_on_llm_end_uses_serialized_id_fallback(self) -> None:
        from deepworkflow.shared.workflow_log import _stats_var

        stats = WorkflowStats()
        _stats_var.set(stats)
        cb = WorkflowStatsCallback(WorkflowLogLevel.INFO)
        run_id = "run-4"
        cb.on_llm_start({"id": ["provider", "my-model"]}, [], run_id=run_id)
        cb.on_llm_end(self._make_llm_result(), run_id=run_id)
        assert "my-model" in stats.tokens_by_model
