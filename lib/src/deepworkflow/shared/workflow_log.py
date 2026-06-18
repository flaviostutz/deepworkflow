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

# Prices sourced from https://developers.openai.com/api/docs/pricing (standard tier, per 1M tokens)
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI GPT-5.x flagship (confirmed June 2026)
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.5-pro": (30.0, 180.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.0, 180.0),
    # OpenAI GPT-4.1 series
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # OpenAI GPT-4o series
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    # OpenAI o-series reasoning
    "o1": (15.0, 60.0),
    "o3": (10.0, 40.0),
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
    # Anthropic Claude
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.8, 4.0),
    "claude-3-opus": (15.0, 75.0),
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}

_FALLBACK_MODEL = "gpt-4.1"

# ---------------------------------------------------------------------------
# Shared runtime stats
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStats:
    """Runtime statistics accumulated during a workflow run."""

    start_time: float = field(default_factory=time.time)
    quality_retries: int = 0
    convergence_retries: int = 0
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


def _print_pre_lines(
    log_pre_fn: Callable[[dict, WorkflowLogLevel], list[str]],
    state: dict,
    log_level: WorkflowLogLevel,
) -> None:
    for line in log_pre_fn(state, log_level):
        print(f"  - (in) {line}")  # noqa: T201


def _print_post_lines(
    log_post_fn: Callable[[dict, dict, WorkflowLogLevel], list[str]],
    state: dict,
    result: dict,
    log_level: WorkflowLogLevel,
) -> None:
    for line in log_post_fn(state, result, log_level):
        if "\n" in line:
            parts = line.split("\n")
            print(f"  - (out) {parts[0]}")  # noqa: T201
            for part in parts[1:]:
                print(part)  # noqa: T201
        else:
            print(f"  - (out) {line}")  # noqa: T201


def wrap_node(  # noqa: PLR0913, C901
    name: str,
    fn: Callable,
    *,
    stat: str | None = None,
    log_pre_fn: Callable[[dict, WorkflowLogLevel], list[str]] | None = None,
    log_post_fn: Callable[[dict, dict, WorkflowLogLevel], list[str]] | None = None,
    show_batch_index: bool = False,
) -> Callable:
    """Return a wrapped node function with start/end logging.

    Args:
        name: Node name used in log lines.
        fn: Original node callable.
        stat: Optional stats counter to increment — ``"quality_retry"`` or
              ``"convergence_retry"``.
        log_pre_fn: Optional callable ``(state, log_level) -> list[str]`` invoked
            before ``fn``.  Each returned string is printed as
            ``  > (in) {line}`` at INFO and DEBUG levels.
        log_post_fn: Optional callable ``(state, result, log_level) -> list[str]``
            invoked after ``fn``.  Each returned string is printed as
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
            elif stat == "convergence_retry":
                stats.convergence_retries += 1

        display_name = f"{name}[{state.get('current_batch_index', '')}:{state.get('retry_count', 0)}]" if show_batch_index else name

        if log_level in (WorkflowLogLevel.INFO, WorkflowLogLevel.DEBUG):
            print(f"> {display_name}")  # noqa: T201
            if log_pre_fn is not None:
                _print_pre_lines(log_pre_fn, state, log_level)

        def _total_tokens() -> tuple[int, int]:
            s = _stats_var.get()
            if s is None:
                return (0, 0)
            in_t = sum(v[0] for v in s.tokens_by_model.values())
            out_t = sum(v[1] for v in s.tokens_by_model.values())
            return (in_t, out_t)

        tokens_before = _total_tokens()
        t0 = time.monotonic()
        result = fn(state)
        elapsed = time.monotonic() - t0
        tokens_after = _total_tokens()

        if log_level in (WorkflowLogLevel.INFO, WorkflowLogLevel.DEBUG):
            if log_post_fn is not None:
                _print_post_lines(log_post_fn, state, result, log_level)
            in_delta = tokens_after[0] - tokens_before[0]
            out_delta = tokens_after[1] - tokens_before[1]
            token_str = (
                f" ({_format_tokens(in_delta)}/{_format_tokens(out_delta)} tokens)" if (in_delta or out_delta) else ""
            )
            print(f"  - elapsed: {elapsed:.0f}s{token_str}")  # noqa: T201

        return result

    return _wrapped


def wrap_route(name: str, fn: Callable) -> Callable:
    """Return a wrapped route function with outcome logging.

    - INFO: prints ``{name} [{result}]``
    - DEBUG: also adds timing to the end line.

    """

    def _wrapped(state: dict) -> str:
        config = state.get("config")
        log_level = config.log_level if config is not None else WorkflowLogLevel.NONE

        t0 = time.monotonic()
        result = fn(state)
        elapsed = time.monotonic() - t0

        if log_level in (WorkflowLogLevel.INFO, WorkflowLogLevel.DEBUG):
            suffix = f" ({elapsed:.0f}s)" if log_level == WorkflowLogLevel.DEBUG else ""
            print(f"> {name} [{result}]{suffix}")  # noqa: T201

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
        **kwargs: Any,
    ) -> None:
        invocation_params: dict[str, Any] = kwargs.get("invocation_params") or {}
        serialized_kwargs: dict[str, Any] = serialized.get("kwargs") or {}
        model_name = (
            invocation_params.get("model")
            or invocation_params.get("azure_deployment")
            or serialized_kwargs.get("model_name")
            or serialized_kwargs.get("azure_deployment")
            or serialized.get("name")
            or (serialized.get("id") or ["?"])[-1]
        )
        self._pending[str(run_id)] = str(model_name)

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
        effort_config = final_state.get("effort_config")
        worst = min((b.evaluate_quality_verdict for b in batch_outputs), default=None)
        min_quality = effort_config.evaluate_quality_min if effort_config is not None else None
        result_label = "WARNING" if (worst is not None and min_quality is not None and worst < min_quality) else "OK"
    else:
        result_label = "OK"

    # Worst quality verdict label
    effort_config = final_state.get("effort_config")
    evaluate_quality_skipped = (
        (effort_config.evaluate_batch_quality_max_retries == 0) if effort_config is not None else False
    )
    evaluate_quality_str = (
        "N/A"
        if evaluate_quality_skipped or not batch_outputs
        else min(b.evaluate_quality_verdict for b in batch_outputs).name
    )

    total_files_read = sum(len(b.files_read) for b in batch_outputs)
    total_files_written = sum(len(b.files_written) for b in batch_outputs)

    total_in = sum(v[0] for v in stats.tokens_by_model.values())
    total_out = sum(v[1] for v in stats.tokens_by_model.values())

    # Calculate cost using actual model prices; fall back to gpt-4.1 ref price when unknown
    actual_cost = 0.0
    display_models: list[str] = []
    fallback_price = _MODEL_PRICES[_FALLBACK_MODEL]
    for model_name, (in_tok, out_tok) in stats.tokens_by_model.items():
        price = _find_model_price(model_name)
        if price is not None:
            actual_cost += (in_tok / 1_000_000) * price[0] + (out_tok / 1_000_000) * price[1]
            display_models.append(model_name)
        else:
            actual_cost += (in_tok / 1_000_000) * fallback_price[0] + (out_tok / 1_000_000) * fallback_price[1]
            display_models.append(f"{_FALLBACK_MODEL}*")
    if (total_in or total_out) and display_models:
        cost_str = f" (~US$ {actual_cost:.2f} on {', '.join(display_models)})"
    else:
        cost_str = ""
    lines = [
        "> summary:",
        f"  result: {result_label}",
        f"  evaluate quality: {evaluate_quality_str}",
        f"  total time: {elapsed:.0f}s",
        f"  total quality retries: {stats.quality_retries}",
        f"  total convergence retries: {stats.convergence_retries}",
        f"  total files read: {total_files_read}",
        f"  total files written: {total_files_written}",
        f"  model invocations: {stats.model_invocations}",
        f"  models total: {_format_tokens(total_in)}/{_format_tokens(total_out)} tokens{cost_str}",
    ]

    print("\n".join(lines))  # noqa: T201
