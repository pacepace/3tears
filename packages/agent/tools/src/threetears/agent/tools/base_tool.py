"""TearsTool, its result and its MCP-compatible definition, under their agent-tools name.

the contract lives in 3tears-tool-contract (:mod:`threetears.tool_contract`), which installs without the
platform; this module re-exports the same objects, so a tool imported from either path is the same class and every
existing import keeps working.
"""

from __future__ import annotations

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
]
