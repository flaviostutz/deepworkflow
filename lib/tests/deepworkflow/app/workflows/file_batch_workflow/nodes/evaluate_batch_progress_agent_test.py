from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent import (
    _parse_progress_output,
    evaluate_batch_progress_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, OnMaxRetriesExceeded, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config() -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "config": _make_config(),
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "execute_output": "I did the thing",
        "files_read": ["a.py"],
        "files_written": ["a.py"],
    }
    defaults.update(overrides)
    return defaults


class TestParseProgressOutput:
    def test_yes_returns_true(self):
        assert _parse_progress_output("PROGRESS: YES\nREASON: did stuff") is True

    def test_no_returns_false(self):
        assert _parse_progress_output("PROGRESS: NO\nREASON: nothing done") is False

    def test_case_insensitive(self):
        assert _parse_progress_output("progress: yes\nreason: something") is True

    def test_missing_progress_line_returns_false(self):
        assert _parse_progress_output("I made progress on the files") is False

    def test_yes_with_extra_whitespace(self):
        assert _parse_progress_output("PROGRESS:   YES  \nREASON: ...") is True


class TestEvaluateBatchProgressAgent:
    def test_returns_true_when_progress_yes(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent.create_agent",
            "PROGRESS: YES\nREASON: Files were updated meaningfully.",
        )
        result = evaluate_batch_progress_agent(_make_state())
        assert result == {"batch_progress": True}

    def test_returns_false_when_progress_no(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent.create_agent",
            "PROGRESS: NO\nREASON: Nothing changed.",
        )
        result = evaluate_batch_progress_agent(_make_state())
        assert result == {"batch_progress": False}

    def test_returns_false_on_unparseable_output(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent.create_agent",
            "Something unexpected happened.",
        )
        result = evaluate_batch_progress_agent(_make_state())
        assert result == {"batch_progress": False}

    def test_uses_correct_agent_name(self, mocker):
        captured: list[str] = []

        def capturing_model(agent_name: str) -> FakeListChatModel:
            captured.append(agent_name)
            return FakeListChatModel(responses=["PROGRESS: YES\nREASON: ok"])

        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent.create_agent",
            "PROGRESS: YES\nREASON: ok",
        )
        config = _make_config()
        config_with_capture = DeepWorkflowConfig(
            workspace_dir=config.workspace_dir,
            task_instructions=config.task_instructions,
            model=capturing_model,
            workspace_write_option=config.workspace_write_option,
            judge_max_retries=config.judge_max_retries,
            judge_on_max_retries=config.judge_on_max_retries,
        )
        # We check the model factory is called with the right name by verifying
        # the agent is invoked (mock_deep_agent already patches create_agent)
        evaluate_batch_progress_agent(_make_state(config=config_with_capture))
        assert captured == ["evaluate_batch_progress_agent"]
