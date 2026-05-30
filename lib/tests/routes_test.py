from __future__ import annotations

from deepworkflow.app.workflows.deepworkflow.routes import (
    check_map_retries,
    check_map_verdict,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
)
from deepworkflow.shared.config import WorkflowConfig
from deepworkflow.shared.types import BatchDefinition, JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def _make_config(**kwargs) -> WorkflowConfig:
    defaults = {
        "workspace_dir": "/tmp",
        "task_instructions": "do something",
        "task_files": ["a.py"],
        "task_files_write_option": WriteOption.READ_ONLY,
        "judge_minimum": JudgeVerdict.WARNING,
        "judge_max_retries": 2,
        "on_max_retries_exceeded": OnMaxRetriesExceeded.CONTINUE,
    }
    defaults.update(kwargs)
    return WorkflowConfig(**defaults)


class TestCheckMapVerdict:
    def test_pass_when_verdict_meets_minimum(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "map_judge_verdict": JudgeVerdict.OK}
        assert check_map_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "map_judge_verdict": JudgeVerdict.ERROR}
        assert check_map_verdict(state) == "retry_or_fail"

    def test_retry_when_verdict_is_none(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "map_judge_verdict": None}
        assert check_map_verdict(state) == "retry_or_fail"


class TestCheckMapRetries:
    def test_retry_when_retries_remaining(self):
        config = _make_config(judge_max_retries=3)
        state = {"config": config, "map_retry_count": 1}
        assert check_map_retries(state) == "map_batches"

    def test_fail_when_exhausted(self):
        config = _make_config(judge_max_retries=2)
        state = {"config": config, "map_retry_count": 2}
        assert check_map_retries(state) == "fail"


class TestCheckVerdict:
    def test_pass_when_verdict_meets_minimum(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "judge_verdict": JudgeVerdict.OK}
        assert check_verdict(state) == "pass"

    def test_pass_when_verdict_equals_minimum(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "judge_verdict": JudgeVerdict.WARNING}
        assert check_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        config = _make_config(judge_minimum=JudgeVerdict.WARNING)
        state = {"config": config, "judge_verdict": JudgeVerdict.ERROR}
        assert check_verdict(state) == "retry_or_fail"


class TestCheckRetries:
    def test_retry_when_retries_remaining(self):
        config = _make_config(judge_max_retries=3)
        state = {"config": config, "retry_count": 1}
        assert check_retries(state) == "plan_step"

    def test_max_retries_when_exhausted(self):
        config = _make_config(judge_max_retries=2)
        state = {"config": config, "retry_count": 2}
        assert check_retries(state) == "max_retries_exceeded"

    def test_retry_when_zero_retries_used(self):
        config = _make_config(judge_max_retries=1)
        state = {"config": config, "retry_count": 0}
        assert check_retries(state) == "plan_step"


class TestCheckMaxRetriesPolicy:
    def test_fail_policy(self):
        config = _make_config(on_max_retries_exceeded=OnMaxRetriesExceeded.FAIL)
        state = {"config": config}
        assert check_max_retries_policy(state) == "fail"

    def test_continue_policy(self):
        config = _make_config(on_max_retries_exceeded=OnMaxRetriesExceeded.CONTINUE)
        state = {"config": config}
        assert check_max_retries_policy(state) == "record_output"


class TestNextBatch:
    def test_more_batches(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
            BatchDefinition(batch_files=["c"], batch_instructions=""),
        ]
        state = {"current_batch_index": 0, "task_file_batches": batches}
        assert next_batch(state) == "plan_step"

    def test_last_batch(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
            BatchDefinition(batch_files=["c"], batch_instructions=""),
        ]
        state = {"current_batch_index": 2, "task_file_batches": batches}
        assert next_batch(state) == "consolidate"
