from __future__ import annotations

from unittest.mock import MagicMock, patch

from deepworkflow.adapters.connectors.deepagents_connector import _build_permissions, create_agent
from deepworkflow.shared.types import WriteOption


class TestBuildPermissions:
    def test_read_only(self):
        result = _build_permissions(WriteOption.READ_ONLY)
        assert result == [{"path": "**", "mode": "read"}]

    def test_write_any(self):
        result = _build_permissions(WriteOption.WRITE_ANY)
        assert result == [{"path": "**", "mode": "write"}]

    def test_write_only_task_files(self):
        result = _build_permissions(WriteOption.WRITE_ONLY_TASK_FILES)
        assert result is None


class TestCreateWorkflowAgent:
    @patch("deepagents.backends.FilesystemBackend")
    @patch("deepworkflow.adapters.connectors.deepagents_connector.create_deep_agent")
    def test_creates_agent_with_defaults(self, mock_create, mock_backend):
        mock_create.return_value = MagicMock()
        mock_backend.return_value = MagicMock()
        mock_llm = MagicMock()

        agent = create_agent(
            model=mock_llm,
            system_prompt="Test prompt",
            workspace_dir="/tmp/workspace",
            write_option=WriteOption.READ_ONLY,
        )

        mock_backend.assert_called_once_with(root_dir="/tmp/workspace")
        mock_create.assert_called_once_with(
            model=mock_llm,
            system_prompt="Test prompt",
            backend=mock_backend.return_value,
            permissions=[{"path": "**", "mode": "read"}],
        )
        assert agent == mock_create.return_value
