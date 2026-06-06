"""Basic example: run deepworkflow as a library."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from deepworkflow import DeepWorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption


def main() -> None:
    # Model factory: called once per agent with the agent name.
    # Lets you choose different models per agent, or a single model for all.
    def model_factory(agent_name: str):
        # Use a faster/cheaper model for judge agents, better model for execution
        if "evaluate" in agent_name:
            return ChatOpenAI(model="gpt-4o-mini")
        return ChatOpenAI(model="gpt-4o")

    config = DeepWorkflowConfig(
        workspace_dir=".",
        task_instructions="List all Python files and summarize their purpose.",
        model=model_factory,
        workspace_write_option=WriteOption.READ_ONLY,
        judge_min=JudgeVerdict.WARNING,
        judge_max_retries=1,
        judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
        # task_files=["src/**/*.py"],  # Optional: omit to let the agent discover files
    )

    result = run_workflow(config)
    print(result)  # noqa: T201


if __name__ == "__main__":
    main()
