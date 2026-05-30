from __future__ import annotations

from deepworkflow.app.workflows.deepworkflow.nodes import parse_judge_output as _parse_judge_output
from deepworkflow.shared.types import JudgeVerdict


class TestParseJudgeOutput:
    def test_valid_json(self):
        content = '{"judge_feedbacks": [{"file": "a.py", "type": "WARNING", "description": "issue"}], "judge_verdict": "WARNING"}'
        verdict, feedbacks = _parse_judge_output(content)
        assert verdict == JudgeVerdict.WARNING
        assert len(feedbacks) == 1
        assert feedbacks[0].file == "a.py"
        assert feedbacks[0].type == JudgeVerdict.WARNING
        assert feedbacks[0].description == "issue"

    def test_json_in_code_block(self):
        content = """Here is my evaluation:
```json
{"judge_feedbacks": [{"file": "b.py", "type": "OK", "description": "looks good"}], "judge_verdict": "OK"}
```"""
        verdict, feedbacks = _parse_judge_output(content)
        assert verdict == JudgeVerdict.OK
        assert len(feedbacks) == 1

    def test_invalid_json(self):
        content = "This is not JSON at all"
        verdict, feedbacks = _parse_judge_output(content)
        assert verdict == JudgeVerdict.ERROR
        assert len(feedbacks) == 1
        assert feedbacks[0].file == "general"

    def test_multiple_feedbacks(self):
        content = '{"judge_feedbacks": [{"file": "a.py", "type": "OK", "description": "fine"}, {"file": "b.py", "type": "ERROR", "description": "broken"}], "judge_verdict": "ERROR"}'
        verdict, feedbacks = _parse_judge_output(content)
        assert verdict == JudgeVerdict.ERROR
        assert len(feedbacks) == 2

    def test_empty_feedbacks(self):
        content = '{"judge_feedbacks": [], "judge_verdict": "OK"}'
        verdict, feedbacks = _parse_judge_output(content)
        assert verdict == JudgeVerdict.OK
        assert feedbacks == []
