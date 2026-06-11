from __future__ import annotations

from typing import TYPE_CHECKING

from deepworkflow.shared.prompts import _build_workspace_structure, build_agent_prompt, build_context

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildContext:
    def test_no_workspace_dir_contains_date_and_os(self):
        result = build_context()
        assert "The current date is" in result
        assert "The current OS is:" in result
        assert "workspace" not in result

    def test_with_workspace_dir_includes_structure(self, tmp_path: Path):
        (tmp_path / "file.py").write_text("x")
        result = build_context(workspace_dir=str(tmp_path))
        assert "workspace" in result
        assert "file.py" in result


class TestBuildWorkspaceStructure:
    def test_lists_files_and_dirs(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "b.py").write_text("y")
        result = _build_workspace_structure(str(tmp_path))
        assert "a.py" in result
        assert "subdir/" in result
        assert "b.py" in result

    def test_ignores_hidden_and_venv(self, tmp_path: Path):
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "visible.py").write_text("y")
        result = _build_workspace_structure(str(tmp_path))
        assert ".hidden" not in result
        assert "__pycache__" not in result
        assert "visible.py" in result

    def test_truncates_at_max_files(self, tmp_path: Path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("x")
        result = _build_workspace_structure(str(tmp_path), max_files=2)
        assert "TRUNCATED" in result


class TestBuildAgentPrompt:
    def test_minimal_prompt_contains_required_sections(self):
        result = build_agent_prompt(
            objective="Do X",
            role="You are Y",
            input_section="Input Z",
            output_format="JSON",
        )
        assert "<OBJECTIVE>" in result
        assert "<ROLE>" in result
        assert "<INPUT>" in result
        assert "<OUTPUT_FORMAT>" in result
        assert "<STEPS>" not in result
        assert "<GUARDRAILS>" not in result
        assert "<TOOL_GUIDANCE>" not in result
        assert "<CONTEXT>" not in result

    def test_full_prompt_contains_all_sections(self):
        result = build_agent_prompt(
            objective="Do X",
            role="You are Y",
            input_section="Input Z",
            output_format="JSON",
            steps="Step 1",
            guardrails="No X",
            tool_guidance="Use tool A",
            context="Some context",
        )
        assert "<STEPS>" in result
        assert "<GUARDRAILS>" in result
        assert "<TOOL_GUIDANCE>" in result
        assert "<CONTEXT>" in result
        assert "<WORKFLOW_CONTEXT>" in result
