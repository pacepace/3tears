"""3tears-mcp: shared Model Context Protocol framework.

per-product MCP servers (metallm, hub, agent-admin) compose this
framework instead of reimplementing stdio transport, JWT auth, error
mapping, and per-tool RBAC.

modules:

- :mod:`threetears.mcp.server` -- :class:`McpServer` (RBAC-gated
  wrapper over the official ``mcp.server.Server``)
- :mod:`threetears.mcp.tool` -- :class:`McpTool`, :class:`ToolRegistry`,
  :func:`register_tool` decorator
- :mod:`threetears.mcp.http_client` -- :class:`PlatformHttpClient`
  (typed httpx with JWT login + refresh-on-401; used by both MCP
  servers and CLI scripts)
- :mod:`threetears.mcp.auth` -- :class:`Identity`,
  :class:`IdentityProvider` Protocol + :class:`EnvVarIdentityProvider`,
  :class:`Authorizer` Protocol + :class:`LocalGrantAuthorizer`
- :mod:`threetears.mcp.rbac` -- :class:`McpToolGrantCollection` over
  the ``mcp_tool_grants`` table
- :mod:`threetears.mcp.migrations` -- platform-scope migration that
  creates ``mcp_tool_grants``
"""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.mcp.auth import (
    Authorizer,
    EnvVarIdentityProvider,
    Identity,
    IdentityProvider,
    LocalGrantAuthorizer,
    PrincipalType,
)
from threetears.mcp.http_client import PlatformHttpClient, PlatformHttpError
from threetears.mcp.rbac import (
    McpToolGrantCollection,
    McpToolGrantEntity,
    mcp_tool_grants_table,
)
from threetears.mcp.server import McpServer
from threetears.mcp.tool import (
    McpTool,
    ToolHandler,
    ToolRegistry,
    get_default_registry,
    register_tool,
    reset_default_registry_for_testing,
)

__all__ = [
    "Authorizer",
    "EnvVarIdentityProvider",
    "Identity",
    "IdentityProvider",
    "LocalGrantAuthorizer",
    "McpServer",
    "McpTool",
    "McpToolGrantCollection",
    "McpToolGrantEntity",
    "PlatformHttpClient",
    "PlatformHttpError",
    "PrincipalType",
    "ToolHandler",
    "ToolRegistry",
    "get_default_registry",
    "mcp_tool_grants_table",
    "register_tool",
    "reset_default_registry_for_testing",
]
