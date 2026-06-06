from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent import reduce_consolidate_agent
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchOutput, JudgeVerdict, OnMaxRetriesExceeded, WriteOption


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


class TestReduceConsolidateAgent:
    def test_returns_workflow_output(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent.create_agent",
            "consolidated result",
        )
        batch_outputs = [
            BatchOutput(
                task_files=["a.py"],
                judge_verdict=JudgeVerdict.OK,
                judge_feedbacks=[],
                files_read=["a.py"],
                files_written=[],
                execute_output="done",
            ),
            BatchOutput(
                task_files=["b.py"],
                judge_verdict=JudgeVerdict.WARNING,
                judge_feedbacks=[],
                files_read=["b.py"],
                files_written=[],
                execute_output="partial",
            ),
        ]
        result = reduce_consolidate_agent(
            {
                "config": _make_config(),
                "batch_outputs": batch_outputs,
                "consolidation_instructions": "summarize all results",
            }
        )
        assert result["workflow_output"] == "consolidated result"

    def test_empty_batch_outputs(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent.create_agent",
            "nothing to consolidate",
        )
        result = reduce_consolidate_agent({"config": _make_config(), "batch_outputs": []})
        assert result["workflow_output"] == "nothing to consolidate"
