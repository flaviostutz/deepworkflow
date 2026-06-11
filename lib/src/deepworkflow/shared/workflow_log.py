"""Workflow console logging, node/route wrappers, and runtime stats."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from deepworkflow.shared.types import WorkflowLogLevel

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.outputs import LLMResult

    from deepworkflow.shared.types import WorkflowResult

# ---------------------------------------------------------------------------
# Model price table: (input_per_1M_usd, output_per_1M_usd)
# ---------------------------------------------------------------------------

_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "o1": (15.0, 60.0),
    "o3-mini": (1.1, 4.4),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.8, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}

# ---------------------------------------------------------------------------
# Shared runtime stats
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStats:
    """Runtime statistics accumulated during a workflow run."""

    start_time: float = field(default_factory=time.time)
    quality_retries: int = 0
    progress_retries: int = 0
    model_invocations: int = 0
    tokens_by_model: dict[str, list[int]] = field(default_factory=dict)


_stats_var: ContextVar[WorkflowStats | None] = ContextVar("_stats_var", default=None)


def new_run_stats() -> WorkflowStats:
    """Create a fresh :class:`WorkflowStats` and register it in the module ContextVar."""
    stats = WorkflowStats()
    _stats_var.set(stats)
    return stats


# ---------------------------------------------------------------------------
# Node / route wrappers
# ---------------------------------------------------------------------------


def _print_pre_lines(log_pre_fn: Callable[[dict], list[str]], state: dict) -> None:
    for line in log_pre_fn(state):
        print(f"  - (in) {line}")  # noqa: T201


def _print_post_lines(log_post_fn: Callable[[dict, dict], list[str]], state: dict, result: dict) -> None:
    for line in log_post_fn(state, result):
        print(f"  - (out) {line}")  # noqa: T201


def wrap_node(  # noqa: PLR0913
    name: str,
    fn: Callable,
    *,
    stat: str | None = None,
    log_pre_fn: Callable[[dict], list[str]] | None = None,
    log_post_fn: Callable[[dict, dict], list[str]] | None = None,
    show_batch_index: bool = False,
) -> Callable:
    """Return a wrapped node function with start/end logging.

    Args:
        name: Node name used in log lines.
        fn: Original node callable.
        stat: Optional stats counter to increment — ``"quality_retry"`` or
              ``"progress_retry"``.
        log_pre_fn: Optional callable ``(state) -> list[str]`` invoked before
            ``fn``.  Each returned string is printed as ``  > (in) {line}`` at
            INFO and DEBUG levels.
        log_post_fn: Optional callable ``(state, result) -> list[str]`` invoked
            after ``fn``.  Each returned string is printed as
            ``  > (out) {line}`` at INFO and DEBUG levels.
        show_batch_index: If ``True``, appends ``[:{current_batch_index}]`` to
            the displayed node name using the value in state.

    """

    def _wrapped(state: dict) -> dict:
        config = state.get("config")
        log_level = config.log_level if config is not None else WorkflowLogLevel.NONE

        stats = _stats_var.get()
        if stats is not None:
            if stat == "quality_retry":
                stats.quality_retries += 1
            elif stat == "progress_retry":
                stats.progress_retries += 1

        display_name = f"{name}[:{state.get('current_batch_index', '')}]" if show_batch_index else name

        if log_level == WorkflowLogLevel.INFO:
            print(f"> {display_name}")  # noqa: T201
            if log_pre_fn is not None:
                _print_pre_lines(log_pre_fn, state)

        t0 = time.monotonic()
        result = fn(state)
        elapsed = time.monotonic() - t0

        if log_level == WorkflowLogLevel.INFO:
            if log_post_fn is not None:
                _print_post_lines(log_post_fn, state, result)
            print(f"  - elapsed: {elapsed:.0f}s")  # noqa: T201

        return result

    return _wrapped


def wrap_route(name: str, fn: Callable) -> Callable:
    """Return a wrapped route function with outcome logging.

    - INFO: prints ``{name} end [{result}]``
    - DEBUG: also prints ``{name} start`` and adds timing to the end line.

    """

    def _wrapped(state: dict) -> str:
        config = state.get("config")
        log_level = config.log_level if config is not None else WorkflowLogLevel.NONE

        result = fn(state)

        if log_level == WorkflowLogLevel.INFO:
            print(f"> {name} [{result}]")  # noqa: T201

        return result

    return _wrapped


# ---------------------------------------------------------------------------
# LangChain callback for token / model tracking
# ---------------------------------------------------------------------------


class WorkflowStatsCallback(BaseCallbackHandler):
    """Accumulates LLM token usage and optionally prints per-invocation debug lines."""

    def __init__(self, log_level: WorkflowLogLevel) -> None:
        super().__init__()
        self._log_level = log_level
        self._pending: dict[str, str] = {}  # str(run_id) → model_name

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],  # noqa: ARG002
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        model_name = str(serialized.get("name") or (serialized.get("id") or ["?"])[-1])
        self._pending[str(run_id)] = model_name

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        model_name = self._pending.pop(str(run_id), "?")
        stats = _stats_var.get()

        # Normalise token counts across providers (OpenAI, Anthropic, Google)
        llm_output = response.llm_output or {}
        usage: dict[str, Any] = (
            llm_output.get("token_usage") or llm_output.get("usage") or llm_output.get("usage_metadata") or {}
        )
        in_tokens: int = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tokens: int = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        if stats is not None:
            stats.model_invocations += 1
            entry = stats.tokens_by_model.setdefault(model_name, [0, 0])
            entry[0] += in_tokens
            entry[1] += out_tokens


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


_TOKENS_PER_M = 1_000_000
_TOKENS_PER_K = 1_000


def _format_tokens(count: int) -> str:
    if count >= _TOKENS_PER_M:
        return f"{count / _TOKENS_PER_M:.1f}M"
    if count >= _TOKENS_PER_K:
        return f"{count // _TOKENS_PER_K}k"
    return str(count)


def _find_model_price(model_name: str) -> tuple[float, float] | None:
    """Find price entry by exact then partial name match."""
    if model_name in _MODEL_PRICES:
        return _MODEL_PRICES[model_name]
    lower = model_name.lower()
    for key, price in _MODEL_PRICES.items():
        if key in lower:
            return price
    return None


def print_summary(stats: WorkflowStats, final_state: dict, workflow_result: WorkflowResult) -> None:
    """Print the workflow summary block to stdout."""
    elapsed = time.time() - stats.start_time
    batch_outputs = final_state.get("batch_outputs") or []
    config = final_state.get("config")

    # Determine overall result label
    if workflow_result.status == "failed":
        result_label = "FAILED"
    elif batch_outputs and config is not None:
        worst = min((b.judge_verdict for b in batch_outputs), default=None)
        result_label = "WARNING" if (worst is not None and worst < config.judge_min) else "OK"
    else:
        result_label = "OK"

    # Worst quality verdict label
    judge_skip = config.judge_skip if config is not None else False
    quality_judge_str = "N/A" if judge_skip or not batch_outputs else min(b.judge_verdict for b in batch_outputs).name

    total_files_read = sum(len(b.files_read) for b in batch_outputs)
    total_files_written = sum(len(b.files_written) for b in batch_outputs)

    total_in = sum(v[0] for v in stats.tokens_by_model.values())
    total_out = sum(v[1] for v in stats.tokens_by_model.values())

    # Cost estimation
    total_cost: float | None = None
    if stats.tokens_by_model:
        cost = 0.0
        has_price = False
        for model, (in_tok, out_tok) in stats.tokens_by_model.items():
            price = _find_model_price(model)
            if price:
                cost += (in_tok / 1_000_000) * price[0] + (out_tok / 1_000_000) * price[1]
                has_price = True
        if has_price:
            total_cost = cost

    cost_str = f" (~US$ {total_cost:.2f})" if total_cost is not None else ""
    lines = [
        "> summary:",
        f"  result: {result_label}",
        f"  quality judge: {quality_judge_str}",
        f"  total time: {elapsed:.0f}s",
        f"  total quality retries: {stats.quality_retries}",
        f"  total progress retries: {stats.progress_retries}",
        f"  total files read: {total_files_read}",
        f"  total files written: {total_files_written}",
        f"  model invocations: {stats.model_invocations}",
        f"  models total: {_format_tokens(total_in)}/{_format_tokens(total_out)} tokens{cost_str}",
    ]

    print("\n".join(lines))  # noqa: T201
