"""Re-export for backward compatibility. Prefer importing from adapters.connectors.deepagents_connector."""

from deepworkflow.adapters.connectors.deepagents_connector import _build_permissions, create_workflow_agent

__all__ = ["_build_permissions", "create_workflow_agent"]
