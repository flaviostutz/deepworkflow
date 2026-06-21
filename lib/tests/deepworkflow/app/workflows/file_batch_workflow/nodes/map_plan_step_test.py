from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepworkflow.app.workflows.file_batch_workflow.nodes.map_plan_step import map_plan_step
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import EffortConfig, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config(**kwargs) -> DeepWorkflowConfig:
    defaults: dict = {
        "workspace_dir": "/tmp",
        "task_instructions": "do something",
        "model": _mock_model,
        "workspace_write_option": WriteOption.READ_ONLY,
        "effort": EffortConfig(level=1),
    }
    defaults.update(kwargs)
    return DeepWorkflowConfig(**defaults)


def _make_effort(**kwargs) -> EffortConfig:
    defaults: dict = {
        "map_plan_mode": "static",
        "max_batches": 1,
        "max_files_per_batch": None,
    }
    defaults.update(kwargs)
    return EffortConfig(**defaults)


def _make_state(task_files: list[str], effort: EffortConfig) -> dict:
    return {
        "config": _make_config(),
        "effort_config": effort,
        "map_files": task_files,
    }


class TestMapBatchesStep:
    def test_single_batch_when_max_batches_is_1(self):
        effort = _make_effort(max_batches=1, max_files_per_batch=None)
        state = _make_state(["a.py", "b.py", "c.py"], effort)
        result = map_plan_step(state)
        assert "error" not in result
        batches = result["map_batches"]
        assert len(batches) == 1
        assert set(batches[0].batch_files) == {"a.py", "b.py", "c.py"}

    def test_chunks_by_max_files_per_batch(self):
        effort = _make_effort(max_batches=None, max_files_per_batch=2)
        state = _make_state(["a.py", "b.py", "c.py", "d.py"], effort)
        result = map_plan_step(state)
        assert "error" not in result
        batches = result["map_batches"]
        assert len(batches) == 2
        assert batches[0].batch_files == ["a.py", "b.py"]
        assert batches[1].batch_files == ["c.py", "d.py"]

    def test_odd_number_files_last_batch_smaller(self):
        effort = _make_effort(max_batches=None, max_files_per_batch=2)
        state = _make_state(["a.py", "b.py", "c.py"], effort)
        result = map_plan_step(state)
        assert "error" not in result
        batches = result["map_batches"]
        assert len(batches) == 2
        assert batches[1].batch_files == ["c.py"]

    def test_fallback_single_batch_when_no_limit(self):
        # max_batches=1, max_files_per_batch=None is valid and creates single batch
        effort = _make_effort(max_batches=1, max_files_per_batch=None)
        state = _make_state(["a.py", "b.py"], effort)
        result = map_plan_step(state)
        assert "error" not in result
        batches = result["map_batches"]
        assert len(batches) == 1
        assert set(batches[0].batch_files) == {"a.py", "b.py"}

    def test_error_when_task_files_empty(self):
        effort = _make_effort(max_batches=1, max_files_per_batch=None)
        state = _make_state([], effort)
        result = map_plan_step(state)
        assert "error" in result
        assert result["error"] is not None

    def test_sets_task_overview_from_task_instructions(self):
        effort = _make_effort(max_batches=1, max_files_per_batch=None)
        config = _make_config(task_instructions="my instructions")
        state = {"config": config, "effort_config": effort, "map_files": ["a.py"]}
        result = map_plan_step(state)
        assert result["map_plan_overview"] == "my instructions"

    def test_each_batch_has_instructions(self):
        effort = _make_effort(max_batches=None, max_files_per_batch=1)
        state = _make_state(["a.py", "b.py"], effort)
        result = map_plan_step(state)
        for batch in result["map_batches"]:
            assert batch.batch_instructions is not None
            assert len(batch.batch_instructions) > 0

    def test_single_file_creates_single_batch(self):
        effort = _make_effort(max_batches=None, max_files_per_batch=5)
        state = _make_state(["only.py"], effort)
        result = map_plan_step(state)
        assert len(result["map_batches"]) == 1
        assert result["map_batches"][0].batch_files == ["only.py"]
