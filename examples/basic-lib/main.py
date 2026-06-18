"""Basic example: run deepworkflow as a library."""

from __future__ import annotations

import getpass
import os

import keyring
from langchain_openai import AzureChatOpenAI

from deepworkflow import DeepWorkflowConfig, run_workflow
from deepworkflow.shared.config import resolveEffortConfig
from deepworkflow.shared.types import WorkflowLogLevel, WriteOption

_KEYCHAIN_SERVICE = "azure-openai/dev-api-key"


def _get_api_key() -> str:
    """Fetch API key from the native OS keychain, raising if not found.

    Implements step 1 and 3 of agentme-edr-022 rule 03 fallback chain:
    1. Native OS keychain via `keyring`
    3. Raise exception with a clear message when not found.
    Run 'make setup-secrets' to store the key.
    """
    api_key = keyring.get_password(_KEYCHAIN_SERVICE, getpass.getuser())
    if api_key is None:
        msg = (
            f"Secret not found in keychain under service '{_KEYCHAIN_SERVICE}'. "
            "Run 'make setup-secrets' to store it."
        )
        raise RuntimeError(msg)
    return api_key


def main() -> None:
    # Non-secret Azure config: set these in your environment or a .env file
    azure_endpoint = os.environ["DEEPWORKFLOW_AZURE_OPENAI_ENDPOINT"]
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
        effort_config=resolveEffortConfig(3),
        log_level=WorkflowLogLevel.INFO,
        # task_files=["src/**/*.py"],  # Optional: omit to let the agent discover files
    )

    result = run_workflow(config)
    print(result)  # noqa: T201


if __name__ == "__main__":
    main()
