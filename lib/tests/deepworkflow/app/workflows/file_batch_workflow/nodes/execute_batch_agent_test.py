from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.execute_batch_agent import execute_batch_agent
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchDefinition, JudgeFeedback, JudgeVerdict, OnMaxRetriesExceeded, WriteOption


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
        "task_overview": "overall plan",
        "plan_output": "step 1: read file; step 2: summarize",
    }
    defaults.update(overrides)
    return defaults


class TestExecuteBatchAgent:
    def test_returns_execute_output_and_messages(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.execute_batch_agent.create_agent",
            "execution complete",
        )
        result = execute_batch_agent(_make_state())
        assert result["execute_output"] == "execution complete"
        assert result["execute_messages"][0].content == "execution complete"

    def test_with_judge_feedback(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.execute_batch_agent.create_agent",
            "revised execution",
        )
        feedback = JudgeFeedback(file="a.py", type=JudgeVerdict.WARNING, description="incomplete", proposal="add more")
        result = execute_batch_agent(_make_state(judge_feedbacks=[feedback]))
        assert result["execute_output"] == "revised execution"
        assert "execute_messages" in result
