from typing import Literal

from deepworkflow.shared.types import OnMaxRetriesExceeded


def route_map_judge(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip map judge when judge_skip is True."""
    if state["config"].judge_skip:
        return "skip"
    return "evaluate"


def route_batch_judge(state: dict) -> Literal["skip", "evaluate"]:
    """Route: skip batch judge when judge_skip is True."""
    if state["config"].judge_skip:
        return "skip"
    return "evaluate"


def check_map_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if the map judge verdict meets minimum threshold."""
    config = state["config"]
    verdict = state.get("map_judge_verdict")
    if verdict is not None and verdict >= config.judge_min:
        return "pass"
    return "retry_or_fail"


def check_map_retries(state: dict) -> Literal["map_batches_agent", "fail_step"]:
    """Route: retry map if retries remaining, otherwise fail."""
    config = state["config"]
    retry_count = state.get("map_retry_count", 0)
    if retry_count < config.judge_max_retries:
        return "map_batches_agent"
    return "fail_step"


def check_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if task judge verdict meets minimum threshold."""
    config = state["config"]
    verdict = state["judge_verdict"]
    if verdict >= config.judge_min:
        return "pass"
    return "retry_or_fail"


def check_retries(state: dict) -> Literal["plan_batch_agent", "max_retries_exceeded"]:
    """Route: retry plan if retries remaining, otherwise handle max retries."""
    config = state["config"]
    retry_count = state.get("retry_count", 0)
    if retry_count < config.judge_max_retries:
        return "plan_batch_agent"
    return "max_retries_exceeded"


def check_max_retries_policy(state: dict) -> Literal["fail_step", "record_output_step"]:
    """Route: apply judge_on_max_retries policy."""
    config = state["config"]
    if config.judge_on_max_retries == OnMaxRetriesExceeded.FAIL:
        return "fail_step"
    return "record_output_step"


def next_batch(state: dict) -> Literal["plan_batch_agent", "reduce_consolidate_agent"]:
    """Route: move to next batch or proceed to consolidation."""
    batch_index = state["current_batch_index"]
    batches = state["task_file_batches"]
    if batch_index < len(batches) - 1:
        return "plan_batch_agent"
    return "reduce_consolidate_agent"
