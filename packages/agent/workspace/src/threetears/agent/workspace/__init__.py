"""agent workspace package -- workspace entities, sandbox, format handlers, and tools."""

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-workspace")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.workspace import handlers as handlers  # noqa: F401
from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.config import (
    AllowConfig,
    BindConfig,
    ValidatorEntry,
    WorkspaceConfig,
)
from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion
from threetears.agent.workspace.factory import build_workspace_tools
from threetears.agent.workspace.handlers import YamlHandler
from threetears.agent.workspace.lease import WorkspaceFileLease
from threetears.agent.workspace.materialize import bind, materialize, recover
from threetears.agent.workspace.migrations import (
    create_workspace_tables,
    register as register_workspace_migrations,
)
from threetears.agent.workspace.pin import PinnedWorkspace, clear_pin, get_pin, set_pin
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.validators import WorkspaceValidationError

__all__ = [
    "AllowConfig",
    "BindConfig",
    "BindConflictPolicy",
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
