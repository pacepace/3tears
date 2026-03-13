"""MCP (Model Context Protocol) client for tool discovery and invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from threetears.core.logging import get_logger
from threetears.core.tracing import traced

log = get_logger(__name__)

_MCP_TIMEOUT_SECONDS = 30


@dataclass
class McpTool:
    """Descriptor for an MCP-exposed tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpToolResult:
    """Result of an MCP tool invocation."""

    success: bool
    content: str
    error: str | None = None


class McpClient:
    """HTTP client for MCP-compatible tool servers.

    Communicates via the ``/mcp/v1/tools/list`` and ``/mcp/v1/tools/call``
    endpoints.
    """

    def __init__(self, base_url: str, timeout: int = _MCP_TIMEOUT_SECONDS) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    @traced()
    async def list_tools(self) -> list[McpTool]:
        """Discover available tools from the MCP server."""
        try:
            resp = await self._http.post(
                f"{self._base_url}/mcp/v1/tools/list", json={}
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                McpTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in data.get("tools", [])
            ]
        except Exception as exc:
            log.error(
                "Failed to list MCP tools",
                extra={"extra_data": {"error": str(exc), "url": self._base_url}},
            )
            return []

    @traced(record_args=True)
    async def invoke_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolResult:
        """Invoke a tool on the MCP server."""
        try:
            resp = await self._http.post(
                f"{self._base_url}/mcp/v1/tools/call",
                json={"name": tool_name, "arguments": arguments},
            )
            resp.raise_for_status()
            data = resp.json()
            content_parts = data.get("content", [])
            text_content = "\n".join(
                part.get("text", "")
                for part in content_parts
                if part.get("type") == "text"
            )
            return McpToolResult(
                success=not data.get("isError", False), content=text_content
            )
        except httpx.TimeoutException:
            return McpToolResult(
                success=False, content="", error="MCP tool invocation timed out"
            )
        except Exception as exc:
            return McpToolResult(success=False, content="", error=str(exc))

    async def test_connection(self) -> bool:
        """Test whether the MCP server is reachable."""
        try:
            resp = await self._http.post(
                f"{self._base_url}/mcp/v1/tools/list", json={}
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False
