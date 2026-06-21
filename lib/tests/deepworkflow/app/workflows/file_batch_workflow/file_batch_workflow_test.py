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
        "map_batches_mode": "agent",
        "evaluate_map_max_retries": 1,
        "evaluate_batch_quality_max_retries": 1,
        "evaluate_batch_convergence_max_retries": 0,
        "skip_batch_plan": False,
        "consolidate_mode": "agent",
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
        "task_file_batches": batches,
        "task_overview": "overview",
        "consolidation_instructions": "summarize",
        "evaluate_quality_batch_instructions": "Output MUST be valid",
    }


def _initial_state(config: DeepWorkflowConfig) -> dict:
    return {
        "config": config,
        "task_files": ["a.py"],
        "current_batch_index": 0,
        "batch_outputs": [],
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

    _apply("resolve_globs_step", "resolve", {"task_files": ["a.py"]})
    _apply("effort_static_step", "set_effort", {"effort_config": _make_effort()})
    _apply("map_batches_agent", "map_batch", _map_output(_single_batch()))
    _apply("validate_map_batches_step", "validate_map", {"error": None})
    _apply(
        "evaluate_map_batches_agent",
        "eval_map",
        {"map_evaluate_quality_verdict": JudgeLevel.OK, "map_evaluate_quality_feedbacks": []},
    )
    _apply("plan_batch_agent", "plan", {"batch_plan": "plan"})
    _apply("execute_batch_agent", "execute", {"execute_output": "done", "execute_messages": []})
    _apply("reflect_batch_agent", "reflect", {"files_read": [], "files_written": []})
    _apply(
        "evaluate_batch_convergence_agent",
        "eval_convergence",
        {"batch_convergence_verdict": _converged(), "batch_convergence_output": ""},
    )
    _apply(
        "evaluate_batch_quality_agent",
        "eval_batch",
        {"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
    )
    _apply("reduce_consolidate_agent", "reduce", {"workflow_output": "final output"})


class TestWorkflowJudgeSkip:
    def test_evaluate_quality_skip_single_batch(self, mocker):
        """Covers route_map_evaluate_quality:skip and route_batch_evaluate_quality:skip."""
        config = _make_config()
        _patch_all(mocker)
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowSingleBatchPass:
    def test_single_batch_evaluate_quality_pass(self, mocker):
        """Covers route_map_evaluate_quality:evaluate, check_map_verdict:pass,
        route_batch_evaluate_quality:evaluate, check_verdict:pass, next_batch:reduce_consolidate_agent."""
        config = _make_config()
        _patch_all(
            mocker,
            eval_map={"map_evaluate_quality_verdict": JudgeLevel.OK, "map_evaluate_quality_feedbacks": []},
            eval_batch={"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMapJudgeRetry:
    def test_map_evaluate_quality_retry_then_pass(self, mocker):
        """Covers route_after_map_verdict:retry_or_fail and check_map_retries:map_batches_agent."""
        config = _make_config()
        _patch_all(mocker)
        # Override: eval_map fails first call, passes second call
        mocker.patch(
            f"{_BASE}.evaluate_map_batches_agent",
            side_effect=[
                {"map_evaluate_quality_verdict": JudgeLevel.ERROR, "map_evaluate_quality_feedbacks": []},
                {"map_evaluate_quality_verdict": JudgeLevel.OK, "map_evaluate_quality_feedbacks": []},
            ],
        )
        # map_batches_agent called twice (initial + retry)
        mocker.patch(f"{_BASE}.map_batches_agent", return_value=_map_output(_single_batch()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None

    def test_map_evaluate_quality_exhausts_retries(self, mocker):
        """Covers check_map_retries:fail_step."""
        config = _make_config()
        _patch_all(
            mocker,
            eval_map={"map_evaluate_quality_verdict": JudgeLevel.ERROR, "map_evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"
        assert result.get("workflow_output") is None


class TestWorkflowBatchJudgeRetry:
    def test_batch_evaluate_quality_retry_then_pass(self, mocker):
        """Covers check_verdict:retry_or_fail and check_retries:plan_batch_agent."""
        config = _make_config()
        _patch_all(mocker)
        # Override: eval_batch fails first, passes second
        mocker.patch(
            f"{_BASE}.evaluate_batch_quality_agent",
            side_effect=[
                {"evaluate_quality_verdict": JudgeLevel.ERROR, "evaluate_quality_feedbacks": []},
                {"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
            ],
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None

    def test_batch_max_retries_fail_policy(self, mocker):
        """Covers check_retries:max_retries_exceeded and check_max_retries_policy:fail_step."""
        effort = _make_effort(
            evaluate_batch_quality_max_retries=1,
            evaluate_quality_min=JudgeLevel.WARNING,
            evaluate_quality_on_max_retries=OnMaxRetriesExceeded.FAIL,
        )
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"evaluate_quality_verdict": JudgeLevel.ERROR, "evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"

    def test_batch_max_retries_continue_policy(self, mocker):
        """Covers check_max_retries_policy:record_output_step."""
        effort = _make_effort(
            evaluate_batch_quality_max_retries=1,
            evaluate_quality_min=JudgeLevel.WARNING,
            evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"evaluate_quality_verdict": JudgeLevel.ERROR, "evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        # CONTINUE policy: records the failed batch and produces output anyway
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMultipleBatches:
    def test_multiple_batches_all_pass(self, mocker):
        """Covers next_batch routing with multiple batches."""
        config = _make_config()
        _patch_all(mocker, map_batch=_map_output(_three_batches()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowBatchRepeat:
    def test_repeat_loop_runs_passes_until_converged(self, mocker):
        """With evaluate_batch_convergence_max_retries=2, not-converged then converged."""
        effort = _make_effort(evaluate_batch_convergence_max_retries=2, evaluate_batch_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_convergence=[
                {"batch_convergence_verdict": _not_converged(), "batch_convergence_output": ""},
                {"batch_convergence_verdict": _converged(), "batch_convergence_output": ""},
            ],
            eval_batch={"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None

    def test_safety_ceiling_stops_loop(self, mocker):
        """With evaluate_batch_convergence_max_retries=1 and always not-converged: only one extra pass."""
        effort = _make_effort(evaluate_batch_convergence_max_retries=1, evaluate_batch_quality_max_retries=1)
        config = _make_config()
        not_converged_mock = mocker.MagicMock(
            return_value={"batch_convergence_verdict": _not_converged(), "batch_convergence_output": ""}
        )
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_batch={"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
        )
        mocker.patch(f"{_BASE}.evaluate_batch_convergence_agent", not_converged_mock)
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None
        assert not_converged_mock.call_count == 2

    def test_files_accumulated_across_passes(self, mocker):
        """Files from all passes are merged into batch_outputs."""
        effort = _make_effort(evaluate_batch_convergence_max_retries=2, evaluate_batch_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            reflect=[
                {"files_read": ["a.py"], "files_written": ["a.py"]},
                {"files_read": ["b.py"], "files_written": ["b.py"]},
            ],
            eval_convergence=[
                {"batch_convergence_verdict": _not_converged(), "batch_convergence_output": ""},
                {"batch_convergence_verdict": _converged(), "batch_convergence_output": ""},
            ],
            eval_batch={"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        batch_out = result["batch_outputs"][0]
        assert "a.py" in batch_out.files_read
        assert "b.py" in batch_out.files_read

    def test_evaluate_quality_retry_resets_repeat_count(self, mocker):
        """When evaluate_quality triggers a retry, batch_repeat_count is reset to 0."""
        effort = _make_effort(evaluate_batch_convergence_max_retries=1, evaluate_batch_quality_max_retries=1)
        config = _make_config()
        _patch_all(
            mocker,
            set_effort={"effort_config": effort},
            eval_convergence=[
                {"batch_convergence_verdict": _converged(), "batch_convergence_output": ""},
                {"batch_convergence_verdict": _converged(), "batch_convergence_output": ""},
            ],
            eval_batch=[
                {"evaluate_quality_verdict": JudgeLevel.ERROR, "evaluate_quality_feedbacks": []},
                {"evaluate_quality_verdict": JudgeLevel.OK, "evaluate_quality_feedbacks": []},
            ],
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None
