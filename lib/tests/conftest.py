from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_create_deep_agent():
    """Mock create_deep_agent to avoid real LLM calls in unit tests."""
    with patch("deepworkflow.shared.agent.create_deep_agent") as mock:
        agent = MagicMock()
        mock.return_value = agent
        yield mock, agent


@pytest.fixture
def mock_filesystem_backend():
    """Mock FilesystemBackend."""
    with patch("deepworkflow.shared.agent.FilesystemBackend") as mock:
        yield mock


def mock_deep_agent(mocker, patch_path: str, output: str | dict[str, Any]) -> MagicMock:
    """Patch create_agent at patch_path to return an AIMessage with the given output.

    output can be a dict (auto-serialized to JSON) or a plain string.
    Returns the patched mock.
    """
    content = json.dumps(output) if isinstance(output, dict) else output
    agent = mocker.MagicMock(invoke=mocker.MagicMock(return_value={"messages": [AIMessage(content=content)]}))
    return mocker.patch(patch_path, return_value=agent)
