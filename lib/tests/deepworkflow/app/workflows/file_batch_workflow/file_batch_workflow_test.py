from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.graph import build_file_batch_workflow
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

_BASE = "deepworkflow.app.workflows.file_batch_workflow.graph"


def _not_converged() -> JudgeVerdict:
    """Verdict meaning another repeat pass is needed (WARNING = not converged)."""
    return JudgeVerdict(
        verdict=JudgeLevel.WARNING,
        findings=[JudgeFinding(level=JudgeLevel.WARNING, title="progress detected", reason="keep going")],
    )


def _converged() -> JudgeVerdict:
    """Verdict meaning the batch has converged (OK = stop repeating)."""
    return JudgeVerdict(
        verdict=JudgeLevel.OK,
        findings=[JudgeFinding(level=JudgeLevel.OK, title="no new changes")],
    )


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_effort(**kwargs) -> EffortConfig:
    """Build an EffortConfig overriding specific fields."""
    defaults: dict = {
        "map_plan_mode": "agent",
        "map_evaluate_max_retries": 1,
        "batch_evaluate_quality_max_retries": 1,
        "batch_evaluate_convergence_max_retries": 0,
        "batch_skip_plan": False,
        "reduce_mode": "agent",
    }
    defaults.update(kwargs)
    return EffortConfig(**defaults)


def _make_config(**kwargs) -> DeepWorkflowConfig:
    defaults: dict = {
        "workspace_dir": "/tmp",
        "task_instructions": "do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": EffortConfig(),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


def _single_batch() -> list[BatchDefinition]:
    return [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")]


def _three_batches() -> list[BatchDefinition]:
    return [
        BatchDefinition(batch_files=["a.py"], batch_instructions="do a"),
        BatchDefinition(batch_files=["b.py"], batch_instructions="do b"),
        BatchDefinition(batch_files=["c.py"], batch_instructions="do c"),
    ]


def _map_output(batches: list[BatchDefinition]) -> dict:
    return {
        "map_batches": batches,
        "map_plan_overview": "overview",
        "reduce_instructions": "summarize",
        "batch_evaluate_quality_instructions": "Output MUST be valid",
    }


def _initial_state(config: DeepWorkflowConfig) -> dict:
    return {
        "config": config,
        "map_files": ["a.py"],
        "batch_current_index": 0,
        "batch_results": [],
    }


def _patch_all(mocker, **overrides) -> None:
    """Patch all LLM-driven agent nodes with simple return values.

    Accepted keyword overrides: resolve, set_effort, map_batch, validate_map, eval_map, plan,
    execute, reflect, eval_convergence, eval_batch, reduce.
    Each value is returned by the corresponding mock. Pass a list to use side_effect instead.
    """

    def _apply(node: str, key: str, default: dict) -> None:
        value = overrides.get(key, default)
        if isinstance(value, list):
            mocker.patch(f"{_BASE}.{node}", side_effect=value)
        else:
            mocker.patch(f"{_BASE}.{node}", return_value=value)

    _apply("map_resolve_step", "resolve", {"map_files": ["a.py"]})
    _apply("map_effort_step", "set_effort", {"effort_config": _make_effort()})
    _apply("map_plan_agent", "map_batch", _map_output(_single_batch()))
    _apply("map_plan_validate_step", "validate_map", {"error": None})
    _apply(
        "map_evaluate_agent",
        "eval_map",
        {"map_evaluate_level": JudgeLevel.OK, "map_evaluate_verdict": JudgeVerdict(verdict=JudgeLevel.OK, findings=[])},
    )
    _apply("batch_plan_agent", "plan", {"batch_plan": "plan"})
    _apply("batch_execute_agent", "execute", {"batch_execute_output": "done", "batch_execute_messages": []})
    _apply("batch_reflect_agent", "reflect", {"batch_files_read": [], "batch_files_written": []})
    _apply(
        "batch_evaluate_convergence_agent",
        "eval_convergence",
        {"batch_evaluate_convergence_verdict": _converged(), "batch_evaluate_convergence_output": ""},
    )
    _apply(
        "batch_evaluate_quality_agent",
        "eval_batch",
        {"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
    )
    _apply("reduce_consolidate_agent", "reduce", {"reduce_output": "final output"})


class TestWorkflowJudgeSkip:
    def test_evaluate_quality_skip_single_batch(self, mocker):
        """Covers route_map_evaluate_quality:skip and route_batch_evaluate_quality:skip."""
        config = _make_config()
        _patch_all(mocker)
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowSingleBatchPass:
    def test_single_batch_evaluate_quality_pass(self, mocker):
        """Covers route_map_evaluate_quality:evaluate, check_map_verdict:pass,
        route_batch_evaluate_quality:evaluate, check_verdict:pass, next_batch:reduce_consolidate_agent."""
        config = _make_config()
        _patch_all(
            mocker,
            eval_map={
                "map_evaluate_level": JudgeLevel.OK,
                "map_evaluate_verdict": JudgeVerdict(verdict=JudgeLevel.OK, findings=[]),
            },
            eval_batch={"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMapJudgeRetry:
    def test_map_evaluate_quality_retry_then_pass(self, mocker):
        """Covers route_after_map_verdict:retry_or_fail and check_map_retries:map_plan_agent."""
        config = _make_config()
        _patch_all(mocker)
        # Override: eval_map fails first call, passes second call
        mocker.patch(
            f"{_BASE}.map_evaluate_agent",
            side_effect=[
                {
                    "map_evaluate_level": JudgeLevel.ERROR,
                    "map_evaluate_verdict": JudgeVerdict(verdict=JudgeLevel.ERROR, findings=[]),
                },
                {
                    "map_evaluate_level": JudgeLevel.OK,
                    "map_evaluate_verdict": JudgeVerdict(verdict=JudgeLevel.OK, findings=[]),
                },
            ],
        )
        # map_plan_agent called twice (initial + retry)
        mocker.patch(f"{_BASE}.map_plan_agent", return_value=_map_output(_single_batch()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None

    def test_map_evaluate_quality_exhausts_retries(self, mocker):
        """Covers check_map_retries:fail_step."""
        config = _make_config()
        _patch_all(
            mocker,
            eval_map={
                "map_evaluate_level": JudgeLevel.ERROR,
                "map_evaluate_verdict": JudgeVerdict(verdict=JudgeLevel.ERROR, findings=[]),
            },
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"
        assert result.get("reduce_output") is None


class TestWorkflowBatchJudgeRetry:
    def test_batch_evaluate_quality_retry_then_pass(self, mocker):
        """Covers check_verdict:retry_or_fail and check_retries:batch_plan_agent."""
        config = _make_config()
        _patch_all(mocker)
        # Override: eval_batch fails first, passes second
        mocker.patch(
            f"{_BASE}.batch_evaluate_quality_agent",
            side_effect=[
                {"batch_evaluate_level": JudgeLevel.ERROR, "batch_evaluate_feedbacks": []},
                {"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
            ],
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None

    def test_batch_max_retries_fail_policy(self, mocker):
        """Covers check_retries:max_retries_exceeded and check_max_retries_policy:fail_step."""
        effort = _make_effort(
            batch_evaluate_quality_max_retries=1,
            batch_evaluate_min=JudgeLevel.WARNING,
            batch_evaluate_on_max_retries=OnMaxRetriesExceeded.FAIL,
        )
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"batch_evaluate_level": JudgeLevel.ERROR, "batch_evaluate_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"

    def test_batch_max_retries_continue_policy(self, mocker):
        """Covers check_max_retries_policy:batch_output_record_step."""
        effort = _make_effort(
            batch_evaluate_quality_max_retries=1,
            batch_evaluate_min=JudgeLevel.WARNING,
            batch_evaluate_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"batch_evaluate_level": JudgeLevel.ERROR, "batch_evaluate_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        # CONTINUE policy: records the failed batch and produces output anyway
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMultipleBatches:
    def test_multiple_batches_all_pass(self, mocker):
        """Covers next_batch routing with multiple batches."""
        config = _make_config()
        _patch_all(mocker, map_batch=_map_output(_three_batches()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowBatchRepeat:
    def test_repeat_loop_runs_passes_until_converged(self, mocker):
        """With batch_evaluate_convergence_max_retries=2, not-converged then converged."""
        effort = _make_effort(batch_evaluate_convergence_max_retries=2, batch_evaluate_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_convergence=[
                {"batch_evaluate_convergence_verdict": _not_converged(), "batch_evaluate_convergence_output": ""},
                {"batch_evaluate_convergence_verdict": _converged(), "batch_evaluate_convergence_output": ""},
            ],
            eval_batch={"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None

    def test_safety_ceiling_stops_loop(self, mocker):
        """With batch_evaluate_convergence_max_retries=1 and always not-converged: only one extra pass."""
        effort = _make_effort(batch_evaluate_convergence_max_retries=1, batch_evaluate_quality_max_retries=1)
        config = _make_config()
        not_converged_mock = mocker.MagicMock(
            return_value={
                "batch_evaluate_convergence_verdict": _not_converged(),
                "batch_evaluate_convergence_output": "",
            }
        )
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
        )
        mocker.patch(f"{_BASE}.batch_evaluate_convergence_agent", not_converged_mock)
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None
        assert not_converged_mock.call_count == 2

    def test_files_accumulated_across_passes(self, mocker):
        """Files from all passes are merged into batch_results."""
        effort = _make_effort(batch_evaluate_convergence_max_retries=2, batch_evaluate_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            reflect=[
                {"batch_files_read": ["a.py"], "batch_files_written": ["a.py"]},
                {"batch_files_read": ["b.py"], "batch_files_written": ["b.py"]},
            ],
            eval_convergence=[
                {"batch_evaluate_convergence_verdict": _not_converged(), "batch_evaluate_convergence_output": ""},
                {"batch_evaluate_convergence_verdict": _converged(), "batch_evaluate_convergence_output": ""},
            ],
            eval_batch={"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        batch_out = result["batch_results"][0]
        assert "a.py" in batch_out.batch_files_read
        assert "b.py" in batch_out.batch_files_read

    def test_evaluate_quality_retry_resets_repeat_count(self, mocker):
        """When evaluate_quality triggers a retry, batch_convergence_repeat_count is reset to 0."""
        effort = _make_effort(batch_evaluate_convergence_max_retries=1, batch_evaluate_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_convergence=[
                {"batch_evaluate_convergence_verdict": _converged(), "batch_evaluate_convergence_output": ""},
                {"batch_evaluate_convergence_verdict": _converged(), "batch_evaluate_convergence_output": ""},
            ],
            eval_batch=[
                {"batch_evaluate_level": JudgeLevel.ERROR, "batch_evaluate_feedbacks": []},
                {"batch_evaluate_level": JudgeLevel.OK, "batch_evaluate_feedbacks": []},
            ],
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["reduce_output"] == "final output"
        assert result.get("error") is None
