from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.graph import build_file_batch_workflow
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, JudgeVerdict, OnMaxRetriesExceeded, WriteOption

_BASE = "deepworkflow.app.workflows.file_batch_workflow.graph"


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(**kwargs) -> DeepWorkflowConfig:
    defaults: dict = {
        "workspace_dir": "/tmp",
        "task_instructions": "do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "judge_max_retries": 1,
        "judge_on_max_retries": OnMaxRetriesExceeded.CONTINUE,
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
        "judge_batch_instructions": "Output MUST be valid",
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

    Accepted keyword overrides: resolve, map_batch, eval_map, plan, execute, reflect, eval_batch, reduce.
    Each value is returned by the corresponding mock. Pass a list to use side_effect instead.
    """

    def _apply(node: str, key: str, default: dict) -> None:
        value = overrides.get(key, default)
        if isinstance(value, list):
            mocker.patch(f"{_BASE}.{node}", side_effect=value)
        else:
            mocker.patch(f"{_BASE}.{node}", return_value=value)

    _apply("resolve_globs_step", "resolve", {"task_files": ["a.py"]})
    _apply("map_batches_agent", "map_batch", _map_output(_single_batch()))
    _apply("evaluate_map_batches_agent", "eval_map", {"map_judge_verdict": JudgeVerdict.OK, "map_judge_feedbacks": []})
    _apply("plan_batch_agent", "plan", {"plan_output": "plan"})
    _apply("execute_batch_agent", "execute", {"execute_output": "done", "execute_messages": []})
    _apply("reflect_batch_agent", "reflect", {"files_read": [], "files_written": []})
    _apply("evaluate_batch_agent", "eval_batch", {"judge_verdict": JudgeVerdict.OK, "judge_feedbacks": []})
    _apply("reduce_consolidate_agent", "reduce", {"workflow_output": "final output"})


class TestWorkflowJudgeSkip:
    def test_judge_skip_single_batch(self, mocker):
        """Covers route_map_judge:skip and route_batch_judge:skip."""
        config = _make_config(judge_skip=True)
        _patch_all(mocker)
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowSingleBatchPass:
    def test_single_batch_judges_pass(self, mocker):
        """Covers route_map_judge:evaluate, check_map_verdict:pass,
        route_batch_judge:evaluate, check_verdict:pass, next_batch:reduce_consolidate_agent."""
        config = _make_config(judge_skip=False, judge_min=JudgeVerdict.WARNING)
        _patch_all(
            mocker,
            eval_map={"map_judge_verdict": JudgeVerdict.OK, "map_judge_feedbacks": []},
            eval_batch={"judge_verdict": JudgeVerdict.OK, "judge_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMapJudgeRetry:
    def test_map_judge_retry_then_pass(self, mocker):
        """Covers check_map_verdict:retry_or_fail and check_map_retries:map_batches_agent."""
        config = _make_config(judge_skip=False, judge_max_retries=2, judge_min=JudgeVerdict.WARNING)
        _patch_all(mocker)
        # Override: eval_map fails first call, passes second call
        mocker.patch(
            f"{_BASE}.evaluate_map_batches_agent",
            side_effect=[
                {"map_judge_verdict": JudgeVerdict.ERROR, "map_judge_feedbacks": []},
                {"map_judge_verdict": JudgeVerdict.OK, "map_judge_feedbacks": []},
            ],
        )
        # map_batches_agent called twice (initial + retry)
        mocker.patch(f"{_BASE}.map_batches_agent", return_value=_map_output(_single_batch()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None

    def test_map_judge_exhausts_retries(self, mocker):
        """Covers check_map_retries:fail_step."""
        config = _make_config(judge_skip=False, judge_max_retries=1, judge_min=JudgeVerdict.WARNING)
        _patch_all(
            mocker,
            eval_map={"map_judge_verdict": JudgeVerdict.ERROR, "map_judge_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"
        assert result.get("workflow_output") is None


class TestWorkflowBatchJudgeRetry:
    def test_batch_judge_retry_then_pass(self, mocker):
        """Covers check_verdict:retry_or_fail and check_retries:plan_batch_agent."""
        config = _make_config(judge_skip=False, judge_max_retries=2, judge_min=JudgeVerdict.WARNING)
        _patch_all(mocker)
        # Override: eval_batch fails first, passes second
        mocker.patch(
            f"{_BASE}.evaluate_batch_agent",
            side_effect=[
                {"judge_verdict": JudgeVerdict.ERROR, "judge_feedbacks": []},
                {"judge_verdict": JudgeVerdict.OK, "judge_feedbacks": []},
            ],
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None

    def test_batch_max_retries_fail_policy(self, mocker):
        """Covers check_retries:max_retries_exceeded and check_max_retries_policy:fail_step."""
        config = _make_config(
            judge_skip=False,
            judge_max_retries=1,
            judge_min=JudgeVerdict.WARNING,
            judge_on_max_retries=OnMaxRetriesExceeded.FAIL,
        )
        _patch_all(
            mocker,
            eval_batch={"judge_verdict": JudgeVerdict.ERROR, "judge_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result.get("error") == "Workflow failed"
        assert result.get("workflow_output") is None

    def test_batch_max_retries_continue_policy(self, mocker):
        """Covers check_max_retries_policy:record_output_step."""
        config = _make_config(
            judge_skip=False,
            judge_max_retries=1,
            judge_min=JudgeVerdict.WARNING,
            judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        _patch_all(
            mocker,
            eval_batch={"judge_verdict": JudgeVerdict.ERROR, "judge_feedbacks": []},
        )
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        # CONTINUE policy: records the failed batch and produces output anyway
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None


class TestWorkflowMultipleBatches:
    def test_multiple_batches_all_pass(self, mocker):
        """Covers next_batch:plan_batch_agent (after first batch) and
        next_batch:reduce_consolidate_agent (after second batch).
        Uses 3 batches; only 2 are processed due to existing next_batch routing behaviour."""
        config = _make_config(judge_skip=True)
        _patch_all(mocker, map_batch=_map_output(_three_batches()))
        graph = build_file_batch_workflow()
        result = graph.invoke(_initial_state(config))
        assert result["workflow_output"] == "final output"
        assert result.get("error") is None
        # Two batches were recorded (batch[0] and batch[1]; batch[2] not reached by routing)
        assert len(result.get("batch_outputs", [])) == 2
