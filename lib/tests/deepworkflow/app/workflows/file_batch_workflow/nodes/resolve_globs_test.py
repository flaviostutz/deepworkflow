from __future__ import annotations

import tempfile
from pathlib import Path

from deepworkflow.app.workflows.file_batch_workflow.nodes.resolve_globs_step import _is_glob_pattern, resolve_globs_step
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


def _make_config(workspace_dir: str, task_files: list[str] | None) -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir=workspace_dir,
        task_instructions="test",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=0,
        judge_on_max_retries=OnMaxRetriesExceeded.FAIL,
        task_files=task_files,
    )


class TestIsGlobPattern:
    def test_plain_file(self):
        assert _is_glob_pattern("src/main.py") is False

    def test_file_with_line_range(self):
        assert _is_glob_pattern("src/main.py:30-40") is False

    def test_star_glob(self):
        assert _is_glob_pattern("src/*.py") is True

    def test_double_star_glob(self):
        assert _is_glob_pattern("**/*.py") is True

    def test_question_mark_glob(self):
        assert _is_glob_pattern("file?.py") is True

    def test_bracket_glob(self):
        assert _is_glob_pattern("file[0-9].py") is True


class TestResolveGlobs:
    def test_explicit_paths_kept_as_is(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, ["a.py", "b.py"])})
            assert result == {"task_files": ["a.py", "b.py"]}

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, ["a.py", "b.py", "a.py"])})
            assert result == {"task_files": ["a.py", "b.py"]}

    def test_glob_expansion(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "foo.py").touch()
            Path(td, "bar.py").touch()
            Path(td, "baz.txt").touch()
            result = resolve_globs_step({"config": _make_config(td, ["*.py"])})
            assert sorted(result["task_files"]) == sorted([str(Path(td, "bar.py")), str(Path(td, "foo.py"))])

    def test_empty_result_returns_error(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, ["nonexistent_*.xyz"])})
            assert "error" in result

    def test_line_range_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, ["file.py:10-20"])})
            assert result == {"task_files": ["file.py:10-20"]}

    def test_task_files_none_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, None)})
            assert result == {"task_files": []}
