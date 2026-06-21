from __future__ import annotations

import tempfile
from pathlib import Path

from deepworkflow.app.workflows.file_batch_workflow.nodes.resolve_globs_step import _is_glob_pattern, resolve_globs_step
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import EffortConfig, WriteOption


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


def _make_config(
    workspace_dir: str,
    task_files: list[str] | None,
    task_files_exclude: list[str] | None = None,
) -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir=workspace_dir,
        task_instructions="test",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
        task_files=task_files,
        task_files_exclude=task_files_exclude,
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
            # Paths are returned relative to workspace_dir so downstream nodes
            # can safely prepend workspace_dir without creating a double-prefix.
            assert sorted(result["task_files"]) == ["bar.py", "foo.py"]

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


class TestResolveGlobsExcludes:
    def test_exclude_explicit_path_removes_file(self):
        with tempfile.TemporaryDirectory() as td:
            excl = str(Path(td, "b.py"))
            result = resolve_globs_step({"config": _make_config(td, ["a.py", excl], task_files_exclude=[excl])})
            assert result == {"task_files": ["a.py"]}

    def test_exclude_glob_pattern_removes_matching_files(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "foo.py").touch()
            Path(td, "bar.py").touch()
            Path(td, "baz.txt").touch()
            result = resolve_globs_step({"config": _make_config(td, ["*.py", "*.txt"], task_files_exclude=["*.txt"])})
            py_files = sorted(result["task_files"])
            assert all(f.endswith(".py") for f in py_files)
            assert not any(f.endswith(".txt") for f in py_files)

    def test_exclude_all_files_returns_error(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step(
                {"config": _make_config(td, ["a.py", "b.py"], task_files_exclude=["a.py", "b.py"])}
            )
            assert "error" in result

    def test_exclude_none_has_no_effect(self):
        with tempfile.TemporaryDirectory() as td:
            result = resolve_globs_step({"config": _make_config(td, ["a.py", "b.py"], task_files_exclude=None)})
            assert result == {"task_files": ["a.py", "b.py"]}

    def test_exclude_with_line_range_entry_still_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            excl = str(Path(td, "a.py"))
            result = resolve_globs_step({"config": _make_config(td, [excl + ":10-20"], task_files_exclude=[excl])})
            # The only file (with line range) was excluded → resolves to nothing
            assert "error" in result
