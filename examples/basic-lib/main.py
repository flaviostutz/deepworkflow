"""Basic example: run deepworkflow as a library."""

from __future__ import annotations

import os

import keyring
from langchain_openai import AzureChatOpenAI

from deepworkflow import DeepWorkflowConfig, run_workflow
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

_GROUP = "deepworkflow-example-basic-lib"


def _get_api_key() -> str:
    """Fetch API key from the native OS keychain, raising if not found.

    Implements step 1 and 3 of agentme-edr-022 rule 03 fallback chain:
    1. Native OS keychain via `keyring`
    3. Raise exception with a clear message when not found.
    Run 'make setup-secrets' to store the key.
    """
    api_key = keyring.get_password(_GROUP, "api-key")
    if api_key is None:
        msg = (
            f"Secret 'api-key' not found in keychain under group '{_GROUP}'. "
            "Run 'make setup-secrets' to store it."
        )
        raise RuntimeError(msg)
    return api_key


def main() -> None:
    # Non-secret Azure config: set these in your environment or a .env file
    azure_endpoint = os.environ["DEEPWORKFLOW_AZURE_ENDPOINT"]
    api_version = os.environ["DEEPWORKFLOW_API_VERSION"]
    deployment = os.environ["DEEPWORKFLOW_MODEL"]

    # Model factory: called once per agent with the agent name.
    # Lets you choose different models per agent, or a single model for all.
    # API key is fetched at point of use from the native OS keychain (agentme-edr-022 rule 08).
    def model_factory(agent_name: str):  # noqa: ANN001, ANN202
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=_get_api_key(),
        )

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
