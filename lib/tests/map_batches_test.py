from __future__ import annotations

import json

from deepworkflow.app.workflows.file_batch_workflow.nodes.map_batches_agent import _parse_map_output


class TestParseMapOutput:
    def test_valid_json(self):
        content = json.dumps(
            {
                "task_overview": "overview",
                "consolidation_instructions": "consolidate",
                "batches": [
                    {"batch_files": ["a.py", "b.py"], "batch_instructions": "group 1"},
                    {"batch_files": ["c.py"], "batch_instructions": "group 2"},
                ],
            }
        )
        result = _parse_map_output(content)
        assert result["task_overview"] == "overview"
        assert result["consolidation_instructions"] == "consolidate"
        assert len(result["task_file_batches"]) == 2
        assert result["task_file_batches"][0].batch_files == ["a.py", "b.py"]
        assert result["current_batch_index"] == 0
        assert result["retry_count"] == 0

    def test_json_in_code_block(self):
        content = """```json
{"task_overview": "x", "consolidation_instructions": "y", "batches": [{"batch_files": ["f.py"]}]}
```"""
        result = _parse_map_output(content)
        assert result["task_overview"] == "x"
        assert len(result["task_file_batches"]) == 1

    def test_invalid_json(self):
        result = _parse_map_output("not json at all")
        assert "error" in result

    def test_error_response(self):
        content = json.dumps({"error": True, "message": "unclear instructions"})
        result = _parse_map_output(content)
        assert result == {"error": "unclear instructions"}

    def test_zero_batches(self):
        content = json.dumps(
            {
                "task_overview": "x",
                "consolidation_instructions": "y",
                "batches": [],
            }
        )
        result = _parse_map_output(content)
        assert "error" in result
