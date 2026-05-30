from typing import Literal

from deepworkflow.shared.types import OnMaxRetriesExceeded


def check_map_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if the map judge verdict meets minimum threshold."""
    config = state["config"]
    verdict = state.get("map_judge_verdict")
    if verdict is not None and verdict >= config.judge_minimum:
        return "pass"
    return "retry_or_fail"


def check_map_retries(state: dict) -> Literal["map_batches", "fail"]:
    """Route: retry map if retries remaining, otherwise fail."""
    config = state["config"]
    retry_count = state.get("map_retry_count", 0)
    if retry_count < config.judge_max_retries:
        return "map_batches"
    return "fail"


def check_verdict(state: dict) -> Literal["pass", "retry_or_fail"]:
    """Route: check if task judge verdict meets minimum threshold."""
    config = state["config"]
    verdict = state["judge_verdict"]
    if verdict >= config.judge_minimum:
        return "pass"
    return "retry_or_fail"


def check_retries(state: dict) -> Literal["plan_step", "max_retries_exceeded"]:
    """Route: retry plan if retries remaining, otherwise handle max retries."""
    config = state["config"]
    retry_count = state.get("retry_count", 0)
    if retry_count < config.judge_max_retries:
        return "plan_step"
    return "max_retries_exceeded"


def check_max_retries_policy(state: dict) -> Literal["fail", "record_output"]:
    """Route: apply on_max_retries_exceeded policy."""
    config = state["config"]
    if config.on_max_retries_exceeded == OnMaxRetriesExceeded.FAIL:
        return "fail"
    return "record_output"


def next_batch(state: dict) -> Literal["plan_step", "consolidate"]:
    """Route: move to next batch or proceed to consolidation."""
    batch_index = state["current_batch_index"]
    batches = state["task_file_batches"]
    if batch_index < len(batches) - 1:
        return "plan_step"
    return "consolidate"
