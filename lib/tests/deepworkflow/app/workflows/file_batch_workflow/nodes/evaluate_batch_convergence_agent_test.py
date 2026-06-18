from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from deepworkflow.app.workflows.file_batch_workflow.nodes.evaluate_batch_convergence_agent import (
    evaluate_batch_convergence_agent,
)
from deepworkflow.shared.config import DeepWorkflowConfig
from deepworkflow.shared.types import EffortConfig
from deepworkflow.shared.types import (
    BatchDefinition,
    JudgeFinding,
    JudgeLevel,
    JudgeVerdict,
    WriteOption,
)


def _judge_response(verdict: str, findings: list[dict] | None = None) -> str:
    if findings is None:
        findings = [
            {
                "level": verdict,
                "title": "desc",
                "reason": "because" if verdict != "OK" else "",
                "fix": "do more" if verdict != "OK" else "",
            }
        ]
    return json.dumps({"verdict": verdict, "findings": findings})


def _mock_model(response: str):
    model = FakeListChatModel(responses=[response])

    def _factory(_agent_name: str):
        return model

    return model, _factory


def _make_config(
    response: str | None = None,
) -> tuple[DeepWorkflowConfig, FakeListChatModel]:
    if response is None:
        response = _judge_response("WARNING")
    model, factory = _mock_model(response)
    cfg = DeepWorkflowConfig(
        workspace_dir="/tmp",
        task_instructions="do something",
        model=factory,
        workspace_write_option=WriteOption.READ_ONLY,
        effort=EffortConfig(level=5),
    )
    return cfg, model


def _make_state(config: DeepWorkflowConfig | None = None, **overrides) -> dict:
    cfg, _ = _make_config() if config is None else (config, None)
    defaults: dict = {
        "config": cfg,
        "current_batch_index": 0,
        "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
        "execute_messages": [],
        "previous_execute_output": "prior pass did some work",
    }
    defaults.update(overrides)
    return defaults


class TestFirstPassBypass:
    def test_no_llm_call_on_first_pass(self):
        """When previous_execute_output is absent, no LLM is invoked."""
        called: list[str] = []

        def recording_factory(agent_name: str) -> FakeListChatModel:
            called.append(agent_name)
            return FakeListChatModel(responses=[_judge_response("OK")])

        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="do something",
            model=recording_factory,
            workspace_write_option=WriteOption.READ_ONLY,
            effort=EffortConfig(level=5),
        )
        state = _make_state(config=config, previous_execute_output="")
        result = evaluate_batch_convergence_agent(state)

        assert called == [], "LLM must not be invoked on first pass"
        assert result["batch_convergence_output"] == ""

    def test_first_pass_returns_warning_verdict(self):
        config, _ = _make_config()
        state = _make_state(config=config, previous_execute_output="")
        result = evaluate_batch_convergence_agent(state)
        verdict = result["batch_convergence_verdict"]
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.verdict == JudgeLevel.WARNING
        assert len(verdict.findings) >= 1
        assert verdict.findings[0].title

    def test_first_pass_missing_key_also_bypasses(self):
        """State with no previous_execute_output key at all is treated as first pass."""
        config, _ = _make_config()
        state = {
            "config": config,
            "current_batch_index": 0,
            "task_file_batches": [BatchDefinition(batch_files=["a.py"], batch_instructions="do it")],
            "execute_messages": [],
        }
        result = evaluate_batch_convergence_agent(state)
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.WARNING


class TestEvaluateBatchConvergenceAgent:
    def test_returns_warning_verdict_when_not_converged(self):
        config, _ = _make_config(
            _judge_response(
                "WARNING",
                [{"level": "WARNING", "title": "new content added", "reason": "do more", "fix": "another pass"}],
            )
        )
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.WARNING

    def test_returns_ok_verdict_when_converged(self):
        config, _ = _make_config(
            _judge_response("OK", [{"level": "OK", "title": "nothing new", "reason": "", "fix": ""}])
        )
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.OK

    def test_returns_error_verdict_on_invalid_json(self):
        config, _ = _make_config("Something unexpected happened.")
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        # parse_judge_output returns ERROR on invalid JSON
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.ERROR

    def test_uses_correct_agent_name(self):
        captured: list[str] = []

        def capturing_factory(agent_name: str) -> FakeListChatModel:
            captured.append(agent_name)
            return FakeListChatModel(responses=[_judge_response("OK")])

        config = DeepWorkflowConfig(
            workspace_dir="/tmp",
            task_instructions="do something",
            model=capturing_factory,
            workspace_write_option=WriteOption.READ_ONLY,
            effort=EffortConfig(level=5),
        )
        evaluate_batch_convergence_agent(_make_state(config=config))
        assert captured == ["evaluate_batch_convergence_agent"]

    def test_includes_execute_messages_in_conversation(self):
        config, _ = _make_config(_judge_response("WARNING"))
        prior_msg = AIMessage(content="I read and modified a.py extensively")
        result = evaluate_batch_convergence_agent(_make_state(config=config, execute_messages=[prior_msg]))
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.WARNING

    def test_empty_execute_messages_still_evaluates(self):
        config, _ = _make_config(_judge_response("OK"))
        result = evaluate_batch_convergence_agent(_make_state(config=config, execute_messages=[]))
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.OK

    def test_raw_output_preserved(self):
        raw = _judge_response("WARNING")
        config, _ = _make_config(raw)
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        assert result["batch_convergence_output"] == raw


class TestEvaluateBatchConvergenceVerdict:
    def test_verdict_warning_when_findings_have_warning(self):
        raw = _judge_response(
            "WARNING",
            [{"level": "WARNING", "title": "new files written", "reason": "continue", "fix": "another pass"}],
        )
        config, _ = _make_config(raw)
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        verdict = result["batch_convergence_verdict"]
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.verdict == JudgeLevel.WARNING
        assert len(verdict.findings) == 1
        assert verdict.findings[0].level == JudgeLevel.WARNING
        assert verdict.findings[0].title
        assert verdict.findings[0].reason

    def test_verdict_ok_when_all_findings_ok(self):
        raw = _judge_response("OK", [{"level": "OK", "title": "converged", "reason": "", "fix": ""}])
        config, _ = _make_config(raw)
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        verdict = result["batch_convergence_verdict"]
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.verdict == JudgeLevel.OK
        assert verdict.findings[0].level == JudgeLevel.OK

    def test_multiple_findings_worst_verdict_wins(self):
        raw = json.dumps(
            {
                "verdict": "WARNING",
                "findings": [
                    {"level": "OK", "title": "already done", "reason": "", "fix": ""},
                    {"level": "WARNING", "title": "new content", "reason": "another pass needed", "fix": "keep going"},
                ],
            }
        )
        config, _ = _make_config(raw)
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        assert result["batch_convergence_verdict"].verdict == JudgeLevel.WARNING
        assert len(result["batch_convergence_verdict"].findings) == 2

    def test_finding_fields_populated_correctly(self):
        raw = json.dumps(
            {
                "verdict": "WARNING",
                "findings": [
                    {
                        "level": "WARNING",
                        "title": "progress made",
                        "reason": "why",
                        "details": "detail text",
                        "fix": "do this",
                    }
                ],
            }
        )
        config, _ = _make_config(raw)
        result = evaluate_batch_convergence_agent(_make_state(config=config))
        f: JudgeFinding = result["batch_convergence_verdict"].findings[0]
        assert f.level == JudgeLevel.WARNING
        assert f.title == "progress made"
        assert f.reason == "why"
        assert f.details == "detail text"
        assert f.fix == "do this"
