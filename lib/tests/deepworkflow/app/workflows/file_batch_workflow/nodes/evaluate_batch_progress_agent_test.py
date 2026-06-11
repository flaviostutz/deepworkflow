from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_progress_agent import (
    _parse_progress_output,
    evaluate_batch_progress_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, OnMaxRetriesExceeded, WriteOption


def _mock_model(response: str):
    model = FakeListChatModel(responses=[response])

    def _factory(_agent_name: str):
        return model

    return model, _factory


def _make_config(response: str = "PROGRESS: YES\nREASON: ok") -> tuple[DeepWorkflowConfig, FakeListChatModel]:
    model, factory = _mock_model(response)
    cfg = DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=factory,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    )
    return cfg, model


def _make_state(config: DeepWorkflowConfig | None = None, **overrides) -> dict:
    cfg, _ = _make_config() if config is None else (config, None)
    defaults: dict = {
        "config": cfg,
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "execute_messages": [],
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
    def test_returns_true_when_progress_yes(self):
        config, _ = _make_config("PROGRESS: YES\nREASON: Files were updated meaningfully.")
        result = evaluate_batch_progress_agent(_make_state(config=config))
        assert result == {
            "batch_progress": True,
            "batch_progress_output": "PROGRESS: YES\nREASON: Files were updated meaningfully.",
        }

    def test_returns_false_when_progress_no(self):
        config, _ = _make_config("PROGRESS: NO\nREASON: Nothing changed.")
        result = evaluate_batch_progress_agent(_make_state(config=config))
        assert result == {
            "batch_progress": False,
            "batch_progress_output": "PROGRESS: NO\nREASON: Nothing changed.",
        }

    def test_returns_false_on_unparseable_output(self):
        config, _ = _make_config("Something unexpected happened.")
        result = evaluate_batch_progress_agent(_make_state(config=config))
        assert result == {"batch_progress": False, "batch_progress_output": "Something unexpected happened."}

    def test_uses_correct_agent_name(self):
        captured: list[str] = []

        def capturing_factory(agent_name: str) -> FakeListChatModel:
            captured.append(agent_name)
            return FakeListChatModel(responses=["PROGRESS: YES\nREASON: ok"])

        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="do something",
            model=capturing_factory,
            workspace_write_option=WriteOption.READ_ONLY,
            judge_max_retries=1,
            judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        )
        evaluate_batch_progress_agent(_make_state(config=config))
        assert captured == ["evaluate_batch_progress_agent"]

    def test_includes_execute_messages_in_conversation(self):
        config, _ = _make_config("PROGRESS: YES\nREASON: saw tool calls")
        prior_msg = AIMessage(content="I read and modified a.py extensively")
        result = evaluate_batch_progress_agent(_make_state(config=config, execute_messages=[prior_msg]))
        assert result["batch_progress"] is True
        assert result["batch_progress_output"] == "PROGRESS: YES\nREASON: saw tool calls"

    def test_empty_execute_messages_still_evaluates(self):
        config, _ = _make_config("PROGRESS: NO\nREASON: no tool calls observed")
        result = evaluate_batch_progress_agent(_make_state(config=config, execute_messages=[]))
        assert result["batch_progress"] is False

