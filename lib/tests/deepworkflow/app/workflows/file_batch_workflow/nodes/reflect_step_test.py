from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes.reflect_batch_agent import _parse_reflect_output


class TestParseReflectOutput:
    def test_standard_format(self):
        content = """FILES_READ:
src/main.py
src/utils.py

FILES_WRITTEN:
src/output.py"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py", "src/utils.py"]
        assert files_written == ["src/output.py"]

    def test_no_files_written(self):
        content = """FILES_READ:
src/main.py

FILES_WRITTEN:
"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py"]
        assert files_written == []

    def test_no_files_at_all(self):
        content = """FILES_READ:

FILES_WRITTEN:
"""
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == []
        assert files_written == []

    def test_extra_whitespace(self):
        content = """FILES_READ:
  src/main.py  
src/utils.py

FILES_WRITTEN:
  dist/out.py  """
        files_read, files_written = _parse_reflect_output(content)
        assert files_read == ["src/main.py", "src/utils.py"]
        assert files_written == ["dist/out.py"]
