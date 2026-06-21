from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent import reduce_consolidate_agent
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import BatchOutput, EffortConfig, JudgeLevel, WriteOption


def _mock_model(_agent_name: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[""])


def _make_config() -> DeepWorkflowConfig:
    return DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=_mock_model,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
    )


class TestReduceConsolidateAgent:
    def test_returns_workflow_output(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent.create_agent",
            "consolidated result",
        )
        batch_results = [
            BatchOutput(
                batch_files=["a.py"],
                evaluate_level=JudgeLevel.OK,
                evaluate_feedbacks=[],
                batch_files_read=["a.py"],
                batch_files_written=[],
                batch_execute_output="done",
            ),
            BatchOutput(
                batch_files=["b.py"],
                evaluate_level=JudgeLevel.WARNING,
                evaluate_feedbacks=[],
                batch_files_read=["b.py"],
                batch_files_written=[],
                batch_execute_output="partial",
            ),
        ]
        result = reduce_consolidate_agent(
            {
                "config": _make_config(),
                "batch_results": batch_results,
                "reduce_instructions": "summarize all results",
            }
        )
        assert result["reduce_output"] == "consolidated result"

    def test_empty_batch_outputs(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.reduce_consolidate_agent.create_agent",
            "nothing to consolidate",
        )
        result = reduce_consolidate_agent({"config": _make_config(), "batch_results": []})
        assert result["reduce_output"] == "nothing to consolidate"
