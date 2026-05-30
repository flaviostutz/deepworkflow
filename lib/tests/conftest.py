from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
