from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import mock_deep_agent
from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent import (
    evaluate_batch_quality_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    EffortConfig,
    JudgeLevel,
    JudgeVerdict,
    WriteOption,
)


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


def _make_state(**overrides) -> dict:
    defaults: dict = {
        "config": _make_config(),
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "execute_output": "I did the thing",
        "evaluate_quality_batch_instructions": "Output MUST be valid",
    }
    defaults.update(overrides)
    return defaults


def _judge_response(verdict: str, findings: list[dict] | None = None) -> dict:
    if findings is None:
        findings = [
            {
                "level": verdict,
                "title": "looks good" if verdict == "OK" else "issue found",
                "reason": "" if verdict == "OK" else "because",
                "fix": "",
            }
        ]
    return {"verdict": verdict, "findings": findings}


class TestEvaluateBatchQualityAgent:
    def test_returns_verdict_and_judge_verdict(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent.create_agent",
            _judge_response("OK"),
        )
        result = evaluate_batch_quality_agent(_make_state())
        assert result["evaluate_quality_verdict"] == JudgeLevel.OK
        judge = result["evaluate_quality_judge_verdict"]
        assert isinstance(judge, JudgeVerdict)
        assert judge.verdict == JudgeLevel.OK
        assert len(judge.findings) >= 1
        assert judge.findings[0].title

    def test_warning_verdict_propagated(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent.create_agent",
            _judge_response(
                "WARNING", [{"level": "WARNING", "title": "issue found", "reason": "bad output", "fix": "fix it"}]
            ),
        )
        result = evaluate_batch_quality_agent(_make_state())
        assert result["evaluate_quality_verdict"] == JudgeLevel.WARNING
        judge = result["evaluate_quality_judge_verdict"]
        assert judge.verdict == JudgeLevel.WARNING
        assert judge.findings[0].level == JudgeLevel.WARNING
        assert judge.findings[0].reason == "bad output"

    def test_invalid_json_returns_error_verdict(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent.create_agent",
            "not valid json {{{",
        )
        result = evaluate_batch_quality_agent(_make_state())
        assert result["evaluate_quality_verdict"] == JudgeLevel.ERROR
        judge = result["evaluate_quality_judge_verdict"]
        assert isinstance(judge, JudgeVerdict)
        assert judge.verdict == JudgeLevel.ERROR
        assert judge.findings[0].title

    def test_multiple_findings_worst_verdict_wins(self, mocker):
        mock_deep_agent(
            mocker,
            "deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_quality_agent.create_agent",
            json.dumps(
                {
                    "verdict": "ERROR",
                    "findings": [
                        {"level": "OK", "title": "section present", "reason": "", "fix": ""},
                        {
                            "level": "ERROR",
                            "title": "critical issue",
                            "reason": "must fix",
                            "details": "details",
                            "fix": "do this",
                        },
                    ],
                }
            ),
        )
        result = evaluate_batch_quality_agent(_make_state())
        assert result["evaluate_quality_verdict"] == JudgeLevel.ERROR
        assert len(result["evaluate_quality_judge_verdict"].findings) == 2
