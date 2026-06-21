from __future__ import annotations

from deepworkflow.app.workflows.file_batch_workflow.routes import (
    batch_check_convergence,
    batch_check_verdict,
    batch_quality_check_retries,
    batch_quality_max_retries,
    batch_route_after_execute,
    batch_route_after_reflect,
    batch_route_evaluate,
    batch_route_next,
    batch_route_plan,
    map_check_verdict,
    map_evaluate_check_retries,
    map_route_after_evaluate,
    map_route_evaluate,
)
from deepworkflow.shared.config import DeepWorkflowConfig
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
        "effort": EffortConfig(level=5),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


def _make_effort(**kwargs) -> EffortConfig:
    defaults = {
        "map_plan_mode": "agent",
        "map_evaluate_max_retries": 2,
        "batch_evaluate_quality_max_retries": 2,
        "batch_evaluate_convergence_max_retries": 0,
        "batch_skip_plan": False,
        "reduce_mode": "agent",
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
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": JudgeLevel.OK}
        assert map_check_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": JudgeLevel.ERROR}
        assert map_check_verdict(state) == "retry_or_fail"

    def test_retry_when_verdict_is_none(self):
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": None}
        assert map_check_verdict(state) == "retry_or_fail"


class TestCheckMapRetries:
    def test_retry_when_retries_remaining(self):
        effort = _make_effort(map_evaluate_max_retries=3)
        state = {"effort_config": effort, "map_evaluate_retry_count": 1}
        assert map_evaluate_check_retries(state) == "map_plan_agent"

    def test_fail_when_exhausted(self):
        effort = _make_effort(map_evaluate_max_retries=2)
        state = {"effort_config": effort, "map_evaluate_retry_count": 3}
        assert map_evaluate_check_retries(state) == "fail_step"


class TestCheckVerdict:
    def test_pass_when_verdict_meets_minimum(self):
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "batch_evaluate_level": JudgeLevel.OK}
        assert batch_check_verdict(state) == "pass"

    def test_pass_when_verdict_equals_minimum(self):
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "batch_evaluate_level": JudgeLevel.WARNING}
        assert batch_check_verdict(state) == "pass"

    def test_retry_when_verdict_below_minimum(self):
        effort = _make_effort(batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "batch_evaluate_level": JudgeLevel.ERROR}
        assert batch_check_verdict(state) == "retry_or_fail"


class TestCheckRetries:
    def test_retry_plan_when_retries_remaining(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=3, batch_skip_plan=False)
        state = {"effort_config": effort, "batch_quality_retry_count": 1}
        assert batch_quality_check_retries(state) == "batch_plan_agent"

    def test_retry_skip_plan_when_retries_remaining(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=3, batch_skip_plan=True)
        state = {"effort_config": effort, "batch_quality_retry_count": 1}
        assert batch_quality_check_retries(state) == "batch_plan_skip_step"

    def test_max_retries_when_exhausted(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=2)
        state = {"effort_config": effort, "batch_quality_retry_count": 3}
        assert batch_quality_check_retries(state) == "max_retries_exceeded"

    def test_retry_when_zero_retries_used(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=1, batch_skip_plan=False)
        state = {"effort_config": effort, "batch_quality_retry_count": 0}
        assert batch_quality_check_retries(state) == "batch_plan_agent"


class TestCheckMaxRetriesPolicy:
    def test_fail_policy(self):
        effort = _make_effort(batch_evaluate_on_max_retries=OnMaxRetriesExceeded.FAIL)
        state = {"effort_config": effort}
        assert batch_quality_max_retries(state) == "fail_step"

    def test_continue_policy(self):
        effort = _make_effort(batch_evaluate_on_max_retries=OnMaxRetriesExceeded.CONTINUE)
        state = {"effort_config": effort}
        assert batch_quality_max_retries(state) == "batch_output_record_step"


class TestNextBatch:
    def test_more_batches_plan_agent(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
        ]
        effort = _make_effort(batch_skip_plan=False, reduce_mode="agent")
        state = {"batch_current_index": 0, "map_batches": batches, "effort_config": effort}
        assert batch_route_next(state) == "batch_plan_agent"

    def test_more_batches_skip_plan(self):
        batches = [
            BatchDefinition(batch_files=["a"], batch_instructions=""),
            BatchDefinition(batch_files=["b"], batch_instructions=""),
        ]
        effort = _make_effort(batch_skip_plan=True, reduce_mode="agent")
        state = {"batch_current_index": 0, "map_batches": batches, "effort_config": effort}
        assert batch_route_next(state) == "batch_plan_skip_step"

    def test_last_batch_consolidate_agent(self):
        batches = [BatchDefinition(batch_files=["a"], batch_instructions="")]
        effort = _make_effort(reduce_mode="agent")
        state = {"batch_current_index": 1, "map_batches": batches, "effort_config": effort}
        assert batch_route_next(state) == "reduce_consolidate_agent"

    def test_last_batch_consolidate_static(self):
        batches = [BatchDefinition(batch_files=["a"], batch_instructions="")]
        effort = _make_effort(reduce_mode="static")
        state = {"batch_current_index": 1, "map_batches": batches, "effort_config": effort}
        assert batch_route_next(state) == "reduce_consolidate_step"


class TestRouteMapJudge:
    def test_evaluate_when_max_retries_nonzero(self):
        effort = _make_effort(map_evaluate_max_retries=2)
        assert map_route_evaluate({"effort_config": effort}) == "evaluate"

    def test_skip_when_max_retries_zero(self):
        effort = _make_effort(map_evaluate_max_retries=0)
        assert map_route_evaluate({"effort_config": effort}) == "skip"


class TestRouteBatchJudge:
    def test_evaluate_when_max_retries_nonzero(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=2)
        assert batch_route_evaluate({"effort_config": effort}) == "evaluate"

    def test_skip_when_max_retries_zero(self):
        effort = _make_effort(batch_evaluate_quality_max_retries=0)
        assert batch_route_evaluate({"effort_config": effort}) == "skip"


class TestRouteAfterReflect:
    def test_evaluate_convergence_when_convergence_retries_set(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=2)
        assert batch_route_after_reflect({"effort_config": effort}) == "evaluate_convergence"

    def test_evaluate_convergence_even_when_quality_zero(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=1, batch_evaluate_quality_max_retries=0)
        assert batch_route_after_reflect({"effort_config": effort}) == "evaluate_convergence"

    def test_evaluate_when_no_convergence_and_quality_not_skipped(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=0, batch_evaluate_quality_max_retries=2)
        assert batch_route_after_reflect({"effort_config": effort}) == "evaluate"

    def test_skip_when_no_convergence_and_quality_zero(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=0, batch_evaluate_quality_max_retries=0)
        assert batch_route_after_reflect({"effort_config": effort}) == "skip"


class TestCheckBatchConvergence:
    def test_repeat_when_not_converged_and_below_ceiling(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=3, batch_evaluate_quality_max_retries=1)
        state = {
            "effort_config": effort,
            "batch_evaluate_convergence_verdict": _warning_verdict(),
            "batch_convergence_repeat_count": 1,
        }
        assert batch_check_convergence(state) == "repeat"

    def test_evaluate_when_not_converged_but_at_ceiling(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=2, batch_evaluate_quality_max_retries=1)
        state = {
            "effort_config": effort,
            "batch_evaluate_convergence_verdict": _warning_verdict(),
            "batch_convergence_repeat_count": 2,
        }
        assert batch_check_convergence(state) == "evaluate"

    def test_evaluate_when_converged_and_quality_not_skipped(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=3, batch_evaluate_quality_max_retries=1)
        state = {
            "effort_config": effort,
            "batch_evaluate_convergence_verdict": _ok_verdict(),
            "batch_convergence_repeat_count": 0,
        }
        assert batch_check_convergence(state) == "evaluate"

    def test_skip_when_converged_and_quality_zero(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=3, batch_evaluate_quality_max_retries=0)
        state = {
            "effort_config": effort,
            "batch_evaluate_convergence_verdict": _ok_verdict(),
            "batch_convergence_repeat_count": 0,
        }
        assert batch_check_convergence(state) == "skip"

    def test_skip_when_at_ceiling_and_quality_zero(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=2, batch_evaluate_quality_max_retries=0)
        state = {
            "effort_config": effort,
            "batch_evaluate_convergence_verdict": _warning_verdict(),
            "batch_convergence_repeat_count": 2,
        }
        assert batch_check_convergence(state) == "skip"

    def test_repeat_count_missing_defaults_to_zero(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=1, batch_evaluate_quality_max_retries=1)
        state = {"effort_config": effort, "batch_evaluate_convergence_verdict": _warning_verdict()}
        assert batch_check_convergence(state) == "repeat"

    def test_no_verdict_treats_as_converged(self):
        effort = _make_effort(batch_evaluate_convergence_max_retries=3, batch_evaluate_quality_max_retries=1)
        state = {"effort_config": effort, "batch_convergence_repeat_count": 0}
        assert batch_check_convergence(state) == "evaluate"


class TestRoutePlanBatch:
    def test_plan_agent_when_skip_false(self):
        effort = _make_effort(batch_skip_plan=False)
        assert batch_route_plan({"effort_config": effort}) == "batch_plan_agent"

    def test_skip_step_when_skip_true(self):
        effort = _make_effort(batch_skip_plan=True)
        assert batch_route_plan({"effort_config": effort}) == "batch_plan_skip_step"


class TestRouteAfterMapVerdict:
    def test_plan_agent_when_pass_and_no_skip_plan(self):
        effort = _make_effort(batch_skip_plan=False, batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": JudgeLevel.OK}
        assert map_route_after_evaluate(state) == "batch_plan_agent"

    def test_skip_batch_plan_step_when_pass_and_skip_plan(self):
        effort = _make_effort(batch_skip_plan=True, batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": JudgeLevel.OK}
        assert map_route_after_evaluate(state) == "batch_plan_skip_step"

    def test_retry_or_fail_when_verdict_below_min(self):
        effort = _make_effort(batch_skip_plan=False, batch_evaluate_min=JudgeLevel.WARNING)
        state = {"effort_config": effort, "map_evaluate_level": JudgeLevel.ERROR}
        assert map_route_after_evaluate(state) == "retry_or_fail"


class TestRouteAfterExecute:
    def test_skip_reflect_when_skip_reflect_true(self):
        effort = _make_effort(batch_skip_reflect=True)
        assert batch_route_after_execute({"effort_config": effort}) == "batch_reflect_skip_step"

    def test_reflect_agent_when_skip_reflect_false(self):
        effort = _make_effort(batch_skip_reflect=False)
        assert batch_route_after_execute({"effort_config": effort}) == "batch_reflect_agent"

    def test_level_0_routes_to_skip_reflect(self):
        effort = EffortConfig(level=0)
        assert batch_route_after_execute({"effort_config": effort}) == "batch_reflect_skip_step"

    def test_level_1_routes_to_reflect_agent(self):
        effort = EffortConfig(level=1)
        assert batch_route_after_execute({"effort_config": effort}) == "batch_reflect_agent"
