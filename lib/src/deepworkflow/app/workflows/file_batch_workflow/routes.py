from typing import Literal

from deepworkflow.shared.types import EffortConfig, JudgeLevel, OnMaxRetriesExceeded

# ---------------------------------------------------------------------------
# Effort routing
# ---------------------------------------------------------------------------


def route_effort_config(state: dict) -> Literal["analyze_task_effort_agent", "set_effort_config_step"]:
    """Route: derive effort config automatically or use the user-supplied one."""
    if state["config"].effort == "auto":
        return "analyze_task_effort_agent"
    return "set_effort_config_step"


def route_map_batches_mode(state: dict) -> Literal["map_batches_agent", "map_batches_step"]:
    """Route: use LLM agent or deterministic algorithm for batch splitting."""
    if state["effort_config"].map_batches_mode == "static":
        return "map_batches_step"
    return "map_batches_agent"


def route_validate_map_limits(
    state: dict,
) -> Literal[
    "fail_step", "map_batches_agent", "evaluate_map_batches_agent", "plan_batch_agent", "skip_batch_plan_step"
]:
    """Route after validate_map_batches_step.

    - If an error is present and map mode is static → hard-fail (algorithm is broken).
    - If an error is present and map mode is agent and retries remain → retry the LLM.
    - If an error is present and map mode is agent but retries exhausted → fail.
    - If no error and evaluate_map_max_retries == 0 → skip map eval, go straight to plan routing.
    - If no error and evaluate_map_max_retries > 0 → run LLM map evaluator.
    """
    effort_config = state["effort_config"]
    error = state.get("error")

    if error:
        if effort_config.map_batches_mode == "static":
            return "fail_step"
        # agent mode: retry if budget allows
        map_retry_count = state.get("map_retry_count", 0)
        if map_retry_count < effort_config.evaluate_map_max_retries:
            return "map_batches_agent"
        return "fail_step"

    if effort_config.evaluate_map_max_retries == 0:
        return _route_plan(effort_config)
    return "evaluate_map_batches_agent"


def route_plan_batch(state: dict) -> Literal["plan_batch_agent", "skip_batch_plan_step"]:
    """Route: run plan agent or inject planning instruction into execute prompt."""
    return _route_plan(state["effort_config"])


def _route_plan(effort_config: EffortConfig) -> Literal["plan_batch_agent", "skip_batch_plan_step"]:  # type: ignore[misc]
    if effort_config.skip_batch_plan:
        return "skip_batch_plan_step"
    return "plan_batch_agent"


def route_consolidate_mode(state: dict) -> Literal["reduce_consolidate_agent", "reduce_consolidate_step"]:
    """Route: use LLM agent or static formatter for consolidation."""
    if state["effort_config"].consolidate_mode == "static":
        return "reduce_consolidate_step"
    return "reduce_consolidate_agent"


# ---------------------------------------------------------------------------
# Map evaluation
# ---------------------------------------------------------------------------


def route_map_evaluate_quality(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip map evaluate_quality when evaluate_map_max_retries is 0."""
    if state["effort_config"].evaluate_map_max_retries == 0:
        return "skip"
    return "evaluate"


def check_map_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if the map evaluate_quality verdict meets minimum threshold."""
    config = state["config"]
    verdict = state.get("map_evaluate_quality_verdict")
    if verdict is not None and verdict >= config.evaluate_quality_min:
        return "pass"
    return "retry_or_fail"


def route_after_map_verdict(
    state: dict,
) -> Literal["plan_batch_agent", "skip_batch_plan_step", "retry_or_fail"]:
    """Combined route after evaluate_map_batches_agent.

    Returns the plan routing target when the verdict passes, or "retry_or_fail" otherwise.
    This avoids needing an extra intermediate step between map-eval pass and plan routing.
    """
    if check_map_verdict(state) == "pass":
        return _route_plan(state["effort_config"])
    return "retry_or_fail"


def check_map_retries(state: dict) -> Literal["map_batches_agent", "fail_step"]:
    """Route: retry map if retries remaining, otherwise fail.

    ``evaluate_map_max_retries=N`` allows N retries after the first attempt.
    The increment step runs before this check, so exhaustion occurs when
    ``map_retry_count > evaluate_map_max_retries``.
    """
    effort_config = state["effort_config"]
    retry_count = state.get("map_retry_count", 0)
    if retry_count <= effort_config.evaluate_map_max_retries:
        return "map_batches_agent"
    return "fail_step"


# ---------------------------------------------------------------------------
# Batch execution routing
# ---------------------------------------------------------------------------


def route_batch_evaluate_quality(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip batch evaluate_quality when evaluate_batch_quality_max_retries is 0."""
    if state["effort_config"].evaluate_batch_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def route_after_reflect(state: dict) -> Literal["evaluate_convergence", "evaluate", "skip"]:
    """Route after reflect.

    Goes to evaluate_batch_convergence_agent when evaluate_batch_convergence_max_retries > 0,
    else evaluates quality or skips.
    """
    effort_config = state["effort_config"]
    if effort_config.evaluate_batch_convergence_max_retries > 0:
        return "evaluate_convergence"
    if effort_config.evaluate_batch_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def check_batch_convergence(state: dict) -> Literal["repeat", "evaluate", "skip"]:
    """Route after evaluate_batch_convergence_agent.

    Repeats if the judge verdict is non-OK (WARNING) and ceiling not reached,
    else evaluates quality or skips.
    """
    effort_config = state["effort_config"]
    judge = state.get("batch_convergence_verdict")
    needs_repeat = judge is not None and judge.verdict != JudgeLevel.OK
    batch_repeat_count = state.get("batch_repeat_count", 0)
    if needs_repeat and batch_repeat_count < effort_config.evaluate_batch_convergence_max_retries:
        return "repeat"
    if effort_config.evaluate_batch_quality_max_retries == 0:
        return "skip"
    return "evaluate"


def check_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if batch evaluate_quality verdict meets minimum threshold."""
    config = state["config"]
    verdict = state["evaluate_quality_verdict"]
    if verdict >= config.evaluate_quality_min:
        return "pass"
    return "retry_or_fail"


def check_retries(state: dict) -> Literal["plan_batch_agent", "skip_batch_plan_step", "max_retries_exceeded"]:
    """Route: retry plan if retries remaining, otherwise handle max retries.

    ``evaluate_batch_quality_max_retries=N`` allows N retries after the first attempt.
    The increment step runs before this check, so exhaustion occurs when
    ``retry_count > evaluate_batch_quality_max_retries``.
    """
    effort_config = state["effort_config"]
    retry_count = state.get("retry_count", 0)
    if retry_count <= effort_config.evaluate_batch_quality_max_retries:
        return _route_plan(effort_config)
    return "max_retries_exceeded"


def check_max_retries_policy(state: dict) -> Literal["fail_step", "record_output_step"]:
    """Route: apply evaluate_quality_on_max_retries policy."""
    config = state["config"]
    if config.evaluate_quality_on_max_retries == OnMaxRetriesExceeded.FAIL:
        return "fail_step"
    return "record_output_step"


def next_batch(
    state: dict,
) -> Literal["plan_batch_agent", "skip_batch_plan_step", "reduce_consolidate_agent", "reduce_consolidate_step"]:
    """Route: move to next batch or proceed to consolidation."""
    batch_index = state["current_batch_index"]
    batches = state["task_file_batches"]
    if batch_index < len(batches) - 1:
        return _route_plan(state["effort_config"])
    return route_consolidate_mode(state)
