"""agent workspace package -- workspace entities, sandbox, format handlers, and tools."""

__version__ = "0.1.0"

from threetears.agent.workspace import handlers as handlers  # noqa: F401
from threetears.agent.workspace.audit import WorkspaceAuditEnvelope
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.config import AllowConfig, ValidatorEntry, WorkspaceConfig
from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion
from threetears.agent.workspace.factory import build_workspace_tools
from threetears.agent.workspace.handlers import YamlHandler
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.agent.workspace.materialize import bind, materialize, recover
from threetears.agent.workspace.migrations import create_workspace_tables, register_workspace_migrations
from threetears.agent.workspace.pin import PinnedWorkspace, clear_pin, get_pin, set_pin
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.validators import WorkspaceValidationError

__all__ = [
    "AllowConfig",
    "PinnedWorkspace",
    "ValidatorEntry",
    "Workspace",
    "WorkspaceCollection",
    "WorkspaceConfig",
    "WorkspaceFile",
    "WorkspaceFileCollection",
    "WorkspaceFileLease",
    "WorkspaceFileVersion",
    "WorkspaceFileVersionCollection",
    "WorkspaceAuditEnvelope",
    "WorkspaceSandbox",
    "WorkspaceValidationError",
    "YamlHandler",
    "bind",
    "build_workspace_tools",
    "clear_pin",
    "create_workspace_tables",
    "get_pin",
    "materialize",
    "recover",
    "register_workspace_migrations",
    "set_pin",
]
