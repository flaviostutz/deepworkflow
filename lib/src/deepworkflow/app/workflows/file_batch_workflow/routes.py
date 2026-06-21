from typing import Literal

from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded

# ---------------------------------------------------------------------------
# Effort routing
# ---------------------------------------------------------------------------


def map_route_effort(state: dict) -> Literal["map_effort_analyze_agent", "map_effort_step"]:
    """Route: derive effort config automatically or use the user-supplied one."""
    if state["config"].effort.type == "auto":
        return "map_effort_analyze_agent"
    return "map_effort_step"


def map_route_plan_mode(state: dict) -> Literal["map_plan_agent", "map_plan_step"]:
    """Route: use LLM agent or deterministic algorithm for batch splitting."""
    if state["effort_config"].map_plan_mode == "static":
        return "map_plan_step"
    return "map_plan_agent"


def map_plan_route_validate(
    state: dict,
) -> Literal["fail_step", "map_plan_agent", "map_evaluate_agent", "batch_plan_agent", "batch_plan_skip_step"]:
    """Route after map_plan_validate_step.

    - If an error is present and map mode is static → hard-fail (algorithm is broken).
    - If an error is present and map mode is agent and retries remain → retry the LLM.
    - If an error is present and map mode is agent but retries exhausted → fail.
    - If no error and map_evaluate_max_retries == 0 → skip map eval, go straight to plan routing.
    - If no error and map_evaluate_max_retries > 0 → run LLM map evaluator.
    """
    effort_config = state["effort_config"]
    error = state.get("error")

    if error:
        if effort_config.map_plan_mode == "static":
            return "fail_step"
        # agent mode: retry if budget allows
        map_retry_count = state.get("map_evaluate_retry_count", 0)
        if map_retry_count < effort_config.map_evaluate_max_retries:
            return "map_plan_agent"
        return "fail_step"

    if effort_config.map_evaluate_max_retries == 0:
        return _batch_route_plan(effort_config)
    return "map_evaluate_agent"


def batch_route_plan(state: dict) -> Literal["batch_plan_agent", "batch_plan_skip_step"]:
    """Route: run plan agent or inject planning instruction into execute prompt."""
    return _batch_route_plan(state["effort_config"])


def batch_route_after_execute(state: dict) -> Literal["batch_reflect_agent", "batch_reflect_skip_step"]:
    """Route: run reflect agent or skip it (level 0 one-shot mode)."""
    if state["effort_config"].batch_skip_reflect:
        return "batch_reflect_skip_step"
    return "batch_reflect_agent"


def _batch_route_plan(effort_config: EffortConfig) -> Literal["batch_plan_agent", "batch_plan_skip_step"]:  # type: ignore[misc]
    if effort_config.batch_skip_plan:
        return "batch_plan_skip_step"
    return "batch_plan_agent"


def reduce_route_mode(state: dict) -> Literal["reduce_consolidate_agent", "reduce_consolidate_step"]:
    """Route: use LLM agent or static formatter for consolidation."""
    if state["effort_config"].reduce_mode == "static":
        return "reduce_consolidate_step"
    return "reduce_consolidate_agent"


# ---------------------------------------------------------------------------
# Map evaluation
# ---------------------------------------------------------------------------


def map_route_evaluate(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip map evaluate when map_evaluate_max_retries is 0."""
    if state["effort_config"].map_evaluate_max_retries == 0:
        return "skip"
    return "evaluate"


def map_check_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if the map evaluate verdict meets minimum threshold."""
    effort_config = state["effort_config"]
    verdict = state.get("map_evaluate_level")
    if verdict is not None and verdict >= effort_config.batch_evaluate_min:
        return "pass"
    return "retry_or_fail"


def map_route_after_evaluate(
    state: dict,
) -> Literal["batch_plan_agent", "batch_plan_skip_step", "retry_or_fail"]:
    """Combined route after map_evaluate_agent.

    Returns the plan routing target when the verdict passes, or "retry_or_fail" otherwise.
    This avoids needing an extra intermediate step between map-eval pass and plan routing.
    """
    if map_check_verdict(state) == "pass":
        return _batch_route_plan(state["effort_config"])
    return "retry_or_fail"


def map_evaluate_check_retries(state: dict) -> Literal["map_plan_agent", "fail_step"]:
    """Route: retry map if retries remaining, otherwise fail.

    ``map_evaluate_max_retries=N`` allows N retries after the first attempt.
    The increment step runs before this check, so exhaustion occurs when
    ``map_evaluate_retry_count > map_evaluate_max_retries``.
    """
    effort_config = state["effort_config"]
    retry_count = state.get("map_evaluate_retry_count", 0)
    if retry_count <= effort_config.map_evaluate_max_retries:
        return "map_plan_agent"
    return "fail_step"


# ---------------------------------------------------------------------------
# Batch execution routing
# ---------------------------------------------------------------------------


def batch_route_evaluate(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip batch evaluate quality when batch_evaluate_quality_max_retries is 0."""
    if state["effort_config"].batch_evaluate_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def batch_route_after_reflect(state: dict) -> Literal["evaluate_convergence", "evaluate", "skip"]:
    """Route after reflect.

    Goes to batch_evaluate_convergence_agent when batch_evaluate_convergence_max_retries > 0,
    else evaluates quality or skips.
    """
    effort_config = state["effort_config"]
    if effort_config.batch_evaluate_convergence_max_retries > 0:
        return "evaluate_convergence"
    if effort_config.batch_evaluate_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def batch_check_convergence(state: dict) -> Literal["repeat", "evaluate", "skip"]:
    """Route after batch_evaluate_convergence_agent.

    Repeats if the judge verdict is non-OK (WARNING) and ceiling not reached,
    else evaluates quality or skips.
    """
    effort_config = state["effort_config"]
    judge = state.get("batch_evaluate_convergence_verdict")
    needs_repeat = judge is not None and judge.verdict != JudgeLevel.OK
    batch_convergence_repeat_count = state.get("batch_convergence_repeat_count", 0)
    if needs_repeat and batch_convergence_repeat_count < effort_config.batch_evaluate_convergence_max_retries:
        return "repeat"
    if effort_config.batch_evaluate_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def batch_check_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if batch evaluate quality verdict meets minimum threshold."""
    effort_config = state["effort_config"]
    verdict = state["batch_evaluate_level"]
    if verdict >= effort_config.batch_evaluate_min:
        return "pass"
    return "retry_or_fail"


def batch_quality_check_retries(
    state: dict,
) -> Literal["batch_plan_agent", "batch_plan_skip_step", "max_retries_exceeded"]:
    """Route: retry plan if retries remaining, otherwise handle max retries.

    ``batch_evaluate_quality_max_retries=N`` allows N retries after the first attempt.
    The increment step runs before this check, so exhaustion occurs when
    ``batch_quality_retry_count > batch_evaluate_quality_max_retries``.
    """
    effort_config = state["effort_config"]
    retry_count = state.get("batch_quality_retry_count", 0)
    if retry_count <= effort_config.batch_evaluate_quality_max_retries:
        return _batch_route_plan(effort_config)
    return "max_retries_exceeded"


def batch_quality_max_retries(state: dict) -> Literal["fail_step", "batch_output_record_step"]:
    """Route: apply batch_evaluate_on_max_retries policy."""
    effort_config = state["effort_config"]
    if effort_config.batch_evaluate_on_max_retries == OnMaxRetriesExceeded.FAIL:
        return "fail_step"
    return "batch_output_record_step"


def batch_route_next(
    state: dict,
) -> Literal["batch_plan_agent", "batch_plan_skip_step", "reduce_consolidate_agent", "reduce_consolidate_step"]:
    """Route: move to next batch or proceed to consolidation."""
    batch_index = state["batch_current_index"]
    batches = state["map_batches"]
    if batch_index < len(batches):
        return _batch_route_plan(state["effort_config"])
    return reduce_route_mode(state)
