from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.routes import (
    check_batch_convergence,
    check_map_retries,
    check_map_verdict,
    check_max_retries_policy,
    check_retries,
    check_verdict,
    next_batch,
    route_after_map_verdict,
    route_after_reflect,
    route_batch_evaluate_quality,
    route_map_evaluate_quality,
    route_plan_batch,
)
from deepworkflow.shared.config import DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    EffortConfig,
    JudgeFinding,
    JudgeLevel,
    JudgeVerdict,
    OnMaxRetriesExceeded,
    WriteOption,
)


def _mock_model(_agent_name: str) -> None:  # type: ignore[return]
    return None


def _make_config(**kwargs) -> DeepWorkflowConfig:
    defaults = {
        "workspace_dir": "/tmp",
        "task_instructions": "do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": "custom",
        "effort_config": resolveEffortConfig(5),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


def _make_effort(**kwargs) -> EffortConfig:
    defaults = {
        "map_batches_mode": "agent",
        "evaluate_map_max_retries": 2,
        "evaluate_batch_quality_max_retries": 2,
        "evaluate_batch_convergence_max_retries": 0,
        "skip_batch_plan": False,
        "consolidate_mode": "agent",
    }
    defaults.update(kwargs)
    return EffortConfig(**defaults)


def _warning_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        verdict=JudgeLevel.WARNING,
        findings=[JudgeFinding(level=JudgeLevel.WARNING, title="not converged", reason="more work")],
    )


def _ok_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        verdict=JudgeLevel.OK,
        findings=[JudgeFinding(level=JudgeLevel.OK, title="converged")],
    )


class TestCheckMapVerdict:
    def test_pass_when_verdict_meets_minimum(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "map_evaluate_quality_verdict": JudgeLevel.OK}
        assert check_map_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "map_evaluate_quality_verdict": JudgeLevel.ERROR}
        assert check_map_verdict(state) == "retry_or_fail"

    def test_retry_when_verdict_is_none(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "map_evaluate_quality_verdict": None}
        assert check_map_verdict(state) == "retry_or_fail"


class TestCheckMapRetries:
    def test_retry_when_retries_remaining(self):
        effort = _make_effort(evaluate_map_max_retries=3)
        state = {"effort_config": effort, "map_retry_count": 1}
        assert check_map_retries(state) == "map_batches_agent"

    def test_fail_when_exhausted(self):
        effort = _make_effort(evaluate_map_max_retries=2)
        state = {"effort_config": effort, "map_retry_count": 3}
        assert check_map_retries(state) == "fail_step"


class TestCheckVerdict:
    def test_pass_when_verdict_meets_minimum(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "evaluate_quality_verdict": JudgeLevel.OK}
        assert check_verdict(state) == "pass"

    def test_pass_when_verdict_equals_minimum(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "evaluate_quality_verdict": JudgeLevel.WARNING}
        assert check_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        state = {"config": config, "evaluate_quality_verdict": JudgeLevel.ERROR}
        assert check_verdict(state) == "retry_or_fail"


class TestCheckRetries:
    def test_retry_plan_when_retries_remaining(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=3, skip_batch_plan=False)
        state = {"effort_config": effort, "retry_count": 1}
        assert check_retries(state) == "plan_batch_agent"

    def test_retry_skip_plan_when_retries_remaining(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=3, skip_batch_plan=True)
        state = {"effort_config": effort, "retry_count": 1}
        assert check_retries(state) == "skip_batch_plan_step"

    def test_max_retries_when_exhausted(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=2)
        state = {"effort_config": effort, "retry_count": 3}
        assert check_retries(state) == "max_retries_exceeded"

    def test_retry_when_zero_retries_used(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=1, skip_batch_plan=False)
        state = {"effort_config": effort, "retry_count": 0}
        assert check_retries(state) == "plan_batch_agent"


class TestCheckMaxRetriesPolicy:
    def test_fail_policy(self):
        config = _make_config(evaluate_quality_on_max_retries=OnMaxRetriesExceeded.FAIL)
        state = {"config": config}
        assert check_max_retries_policy(state) == "fail_step"

    def test_continue_policy(self):
        config = _make_config(evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE)
        state = {"config": config}
        assert check_max_retries_policy(state) == "record_output_step"


class TestNextBatch:
    def test_more_batches_plan_agent(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
        ]
        effort = _make_effort(skip_batch_plan=False, consolidate_mode="agent")
        state = {"current_batch_index": 0, "task_file_batches": batches, "effort_config": effort}
        assert next_batch(state) == "plan_batch_agent"

    def test_more_batches_skip_plan(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
        ]
        effort = _make_effort(skip_batch_plan=True, consolidate_mode="agent")
        state = {"current_batch_index": 0, "task_file_batches": batches, "effort_config": effort}
        assert next_batch(state) == "skip_batch_plan_step"

    def test_last_batch_consolidate_agent(self):
        batches = [BatchDefinition(batch_files=["a"], batch_instructions="")]
        effort = _make_effort(consolidate_mode="agent")
        state = {"current_batch_index": 0, "task_file_batches": batches, "effort_config": effort}
        assert next_batch(state) == "reduce_consolidate_agent"

    def test_last_batch_consolidate_static(self):
        batches = [BatchDefinition(batch_files=["a"], batch_instructions="")]
        effort = _make_effort(consolidate_mode="static")
        state = {"current_batch_index": 0, "task_file_batches": batches, "effort_config": effort}
        assert next_batch(state) == "reduce_consolidate_step"


class TestRouteMapJudge:
    def test_evaluate_when_max_retries_nonzero(self):
        effort = _make_effort(evaluate_map_max_retries=2)
        assert route_map_evaluate_quality({"effort_config": effort}) == "evaluate"

    def test_skip_when_max_retries_zero(self):
        effort = _make_effort(evaluate_map_max_retries=0)
        assert route_map_evaluate_quality({"effort_config": effort}) == "skip"


class TestRouteBatchJudge:
    def test_evaluate_when_max_retries_nonzero(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=2)
        assert route_batch_evaluate_quality({"effort_config": effort}) == "evaluate"

    def test_skip_when_max_retries_zero(self):
        effort = _make_effort(evaluate_batch_quality_max_retries=0)
        assert route_batch_evaluate_quality({"effort_config": effort}) == "skip"


class TestRouteAfterReflect:
    def test_evaluate_convergence_when_convergence_retries_set(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=2)
        assert route_after_reflect({"effort_config": effort}) == "evaluate_convergence"

    def test_evaluate_convergence_even_when_quality_zero(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=1, evaluate_batch_quality_max_retries=0)
        assert route_after_reflect({"effort_config": effort}) == "evaluate_convergence"

    def test_evaluate_when_no_convergence_and_quality_not_skipped(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=0, evaluate_batch_quality_max_retries=2)
        assert route_after_reflect({"effort_config": effort}) == "evaluate"

    def test_skip_when_no_convergence_and_quality_zero(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=0, evaluate_batch_quality_max_retries=0)
        assert route_after_reflect({"effort_config": effort}) == "skip"


class TestCheckBatchConvergence:
    def test_repeat_when_not_converged_and_below_ceiling(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=3, evaluate_batch_quality_max_retries=1)
        state = {"effort_config": effort, "batch_convergence_verdict": _warning_verdict(), "batch_repeat_count": 1}
        assert check_batch_convergence(state) == "repeat"

    def test_evaluate_when_not_converged_but_at_ceiling(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=2, evaluate_batch_quality_max_retries=1)
        state = {"effort_config": effort, "batch_convergence_verdict": _warning_verdict(), "batch_repeat_count": 2}
        assert check_batch_convergence(state) == "evaluate"

    def test_evaluate_when_converged_and_quality_not_skipped(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=3, evaluate_batch_quality_max_retries=1)
        state = {"effort_config": effort, "batch_convergence_verdict": _ok_verdict(), "batch_repeat_count": 0}
        assert check_batch_convergence(state) == "evaluate"

    def test_skip_when_converged_and_quality_zero(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=3, evaluate_batch_quality_max_retries=0)
        state = {"effort_config": effort, "batch_convergence_verdict": _ok_verdict(), "batch_repeat_count": 0}
        assert check_batch_convergence(state) == "skip"

    def test_skip_when_at_ceiling_and_quality_zero(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=2, evaluate_batch_quality_max_retries=0)
        state = {"effort_config": effort, "batch_convergence_verdict": _warning_verdict(), "batch_repeat_count": 2}
        assert check_batch_convergence(state) == "skip"

    def test_repeat_count_missing_defaults_to_zero(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=1, evaluate_batch_quality_max_retries=1)
        state = {"effort_config": effort, "batch_convergence_verdict": _warning_verdict()}
        assert check_batch_convergence(state) == "repeat"

    def test_no_verdict_treats_as_converged(self):
        effort = _make_effort(evaluate_batch_convergence_max_retries=3, evaluate_batch_quality_max_retries=1)
        state = {"effort_config": effort, "batch_repeat_count": 0}
        assert check_batch_convergence(state) == "evaluate"


class TestRoutePlanBatch:
    def test_plan_agent_when_skip_false(self):
        effort = _make_effort(skip_batch_plan=False)
        assert route_plan_batch({"effort_config": effort}) == "plan_batch_agent"

    def test_skip_step_when_skip_true(self):
        effort = _make_effort(skip_batch_plan=True)
        assert route_plan_batch({"effort_config": effort}) == "skip_batch_plan_step"


class TestRouteAfterMapVerdict:
    def test_plan_agent_when_pass_and_no_skip_plan(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        effort = _make_effort(skip_batch_plan=False)
        state = {"config": config, "effort_config": effort, "map_evaluate_quality_verdict": JudgeLevel.OK}
        assert route_after_map_verdict(state) == "plan_batch_agent"

    def test_skip_batch_plan_step_when_pass_and_skip_plan(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        effort = _make_effort(skip_batch_plan=True)
        state = {"config": config, "effort_config": effort, "map_evaluate_quality_verdict": JudgeLevel.OK}
        assert route_after_map_verdict(state) == "skip_batch_plan_step"

    def test_retry_or_fail_when_verdict_below_min(self):
        config = _make_config(evaluate_quality_min=JudgeLevel.WARNING)
        effort = _make_effort(skip_batch_plan=False)
        state = {"config": config, "effort_config": effort, "map_evaluate_quality_verdict": JudgeLevel.ERROR}
        assert route_after_map_verdict(state) == "retry_or_fail"
