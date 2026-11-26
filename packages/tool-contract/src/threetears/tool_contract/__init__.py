"""3tears-tool-contract: the TearsTool contract, installable without the platform.

:class:`TearsTool`, its :class:`ToolResult` and :class:`MCPToolDefinition`, the closed set of result codes, and
the input coercion :meth:`TearsTool.run` applies. A tool written against this package is served on the platform
mesh by 3tears-agent-tools (``threetears.agent.tools.base_tool`` re-exports these same objects) and over MCP by
:func:`threetears.tool_contract.mcp_face.mcp_tool` (the ``mcp`` extra).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-tool-contract")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.tool_contract.coercion import coerce_value, normalize_kwargs
from threetears.tool_contract.tool import (
    CONFLICT,
    MALFORMED_REQUEST,
    PERMISSION_DENIED,
    TOOL_AUTHORIZATION_UNAVAILABLE,
    TOOL_NOT_READY,
    TOOL_RESULT_ERROR_CODES,
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)

__all__ = [
    "CONFLICT",
    "MALFORMED_REQUEST",
    "PERMISSION_DENIED",
    "TOOL_AUTHORIZATION_UNAVAILABLE",
    "TOOL_NOT_READY",
    "TOOL_RESULT_ERROR_CODES",
    "MCPToolDefinition",
    "TearsTool",
    "ToolResult",
    "coerce_value",
    "normalize_kwargs",
]
