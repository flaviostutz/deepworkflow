from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.nodes import parse_evaluate_output as _parse_evaluate_output
from deepworkflow.shared.types import JudgeLevel


class TestParseEvaluateOutput:
    def test_valid_json(self):
        content = '{"evaluate_quality_feedbacks": [{"file": "a.py", "type": "WARNING", "description": "issue"}], "evaluate_quality_verdict": "WARNING"}'
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.WARNING
        assert len(feedbacks) == 1
        assert feedbacks[0].file == "a.py"
        assert feedbacks[0].type == JudgeLevel.WARNING
        assert feedbacks[0].description == "issue"

    def test_json_in_code_block(self):
        content = """Here is my evaluation:
```json
{"evaluate_quality_feedbacks": [{"file": "b.py", "type": "OK", "description": "looks good"}], "evaluate_quality_verdict": "OK"}
```"""
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.OK
        assert len(feedbacks) == 1

    def test_invalid_json(self):
        content = "This is not JSON at all"
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.ERROR
        assert len(feedbacks) == 1
        assert feedbacks[0].file == "general"

    def test_multiple_feedbacks(self):
        content = '{"evaluate_quality_feedbacks": [{"file": "a.py", "type": "OK", "description": "fine"}, {"file": "b.py", "type": "ERROR", "description": "broken"}], "evaluate_quality_verdict": "ERROR"}'
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.ERROR
        assert len(feedbacks) == 2

    def test_empty_feedbacks(self):
        content = '{"evaluate_quality_feedbacks": [], "evaluate_quality_verdict": "OK"}'
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.OK
        assert feedbacks == []

    def test_unknown_verdict_string_falls_back_to_error(self):
        content = '{"evaluate_quality_feedbacks": [], "evaluate_quality_verdict": "UNKNOWN_LEVEL"}'
        verdict, _ = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.ERROR

    def test_unknown_feedback_type_falls_back_to_error(self):
        content = '{"evaluate_quality_feedbacks": [{"file": "a.py", "type": "INVALID_TYPE", "description": "x"}], "evaluate_quality_verdict": "OK"}'
        verdict, feedbacks = _parse_evaluate_output(content)
        assert verdict == JudgeLevel.OK
        assert feedbacks[0].type == JudgeLevel.ERROR
