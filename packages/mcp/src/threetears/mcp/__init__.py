"""3tears-mcp: shared Model Context Protocol framework.

per-product MCP servers compose this
framework instead of reimplementing stdio transport, JWT auth, error
mapping, and per-tool RBAC.

modules:

- :mod:`threetears.mcp.server` -- :class:`McpServer` (RBAC-gated
  wrapper over the official ``mcp.server.Server``; v1 stdio +
  v2 Streamable-HTTP entry points)
- :mod:`threetears.mcp.http_server` -- v2 Streamable-HTTP transport
  (:class:`McpHttpTransport`, :func:`build_mcp_http_app`, :func:`serve_mcp_http`,
  :func:`current_bearer_token`) running the SAME wired SDK server over
  HTTP
- :mod:`threetears.mcp.tool` -- :class:`McpTool`, :class:`McpToolAnnotations`,
  :class:`ToolRegistry`, :data:`ToolProvider`, :func:`register_tool` decorator
- :mod:`threetears.mcp.http_client` -- :class:`PlatformHttpClient`
  (typed httpx with JWT login + refresh-on-401; used by both MCP
  servers and CLI scripts)
- :mod:`threetears.mcp.auth` -- :class:`Identity`,
  :class:`IdentityProvider` Protocol + :class:`EnvVarIdentityProvider`
  (v1 stdio) + :class:`BearerTokenIdentityProvider` (v2 HTTP),
  :class:`Authorizer` Protocol + :class:`LocalGrantAuthorizer`
- :mod:`threetears.mcp.rbac` -- :class:`McpToolGrantCollection` over
  the ``mcp_tool_grants`` table
- :mod:`threetears.mcp.migrations` -- platform-scope migration that
  creates ``mcp_tool_grants``

the grant store -- :class:`LocalGrantAuthorizer`, :mod:`~threetears.mcp.rbac`
and :mod:`~threetears.mcp.migrations` -- is the ``grants`` extra
(``3tears-mcp[grants]``), the one part of the package that needs core,
epoch, nats and sqlalchemy. a server that brings its own
:class:`Authorizer` installs and imports without them, so the rbac names
below resolve on first access rather than at import.
"""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-mcp")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.mcp.auth import (
    Authorizer,
    BearerTokenIdentityProvider,
    BearerTokenResolver,
    EnvVarIdentityProvider,
    Identity,
    IdentityProvider,
    LocalGrantAuthorizer,
    PrincipalType,
    TokenSource,
)
from threetears.mcp.http_client import PlatformHttpClient, PlatformHttpError
from threetears.mcp.http_server import (
    McpHttpTransport,
    build_mcp_http_app,
    current_bearer_token,
    serve_mcp_http,
)
from threetears.mcp.server import McpServer, tool_error
from threetears.mcp.tool import (
    McpTool,
    McpToolAnnotations,
    ToolHandler,
    ToolProvider,
    ToolRegistry,
    register_tool,
)

if TYPE_CHECKING:  # the grant store's names, re-imported so type checkers resolve them
    from threetears.mcp.rbac import (
        McpToolGrantCollection,
        McpToolGrantEntity,
        mcp_tool_grants_table,
    )

#: names that live in :mod:`threetears.mcp.rbac`, which only the ``grants`` extra can import
_GRANT_STORE_NAMES = frozenset({"McpToolGrantCollection", "McpToolGrantEntity", "mcp_tool_grants_table"})

__all__ = [
    "Authorizer",
    "BearerTokenIdentityProvider",
    "BearerTokenResolver",
    "EnvVarIdentityProvider",
    "Identity",
    "IdentityProvider",
    "LocalGrantAuthorizer",
    "McpHttpTransport",
    "McpServer",
    "McpTool",
    "McpToolAnnotations",
    "McpToolGrantCollection",
    "McpToolGrantEntity",
    "PlatformHttpClient",
    "PlatformHttpError",
    "PrincipalType",
    "TokenSource",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "build_mcp_http_app",
    "current_bearer_token",
    "mcp_tool_grants_table",
    "register_tool",
    "serve_mcp_http",
    "tool_error",
]


def __getattr__(name: str) -> object:
    """resolve a grant-store name on first access.

    :param name: the attribute asked for
    :ptype name: str
    :return: the named object from :mod:`threetears.mcp.rbac`
    :rtype: object
    :raises ImportError: when the ``grants`` extra is not installed
    :raises AttributeError: for any other name
    """
    if name not in _GRANT_STORE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from threetears.mcp import rbac
    except ImportError as exc:
        raise ImportError(
            f"threetears.mcp.{name} is part of the MCP grant store, which needs the grants extra: "
            "install 3tears-mcp[grants]",
        ) from exc
    value: object = getattr(rbac, name)
    return value


def __dir__() -> list[str]:
    """the module's names, the grant store's included though they are not loaded yet.

    :return: sorted attribute names
    :rtype: list[str]
    """
    return sorted(set(globals()) | _GRANT_STORE_NAMES)
