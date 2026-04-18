"""Web search tool using SearXNG JSON API."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "WebSearchInput",
    "WebSearchTool",
    "create_web_search_tool",
]


class WebSearchInput(BaseModel):
    """Input for the web search tool."""

    query: str = Field(description="Search query")


def _format_results(data: dict[str, Any]) -> str:
    """Format SearXNG JSON response into readable text."""
    results = data.get("results", [])
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results[:10], 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines).strip()


def _create_search_fn(base_url: str) -> Any:
    """Create a search function bound to a SearXNG base URL."""

    def _search(query: str) -> str:
        url = f"{base_url}/search"
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params={"q": query, "format": "json"})
            resp.raise_for_status()
            return _format_results(resp.json())
        except httpx.HTTPStatusError as exc:
            return tool_error("web_search", "search", f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return tool_error("web_search", "search", str(exc))

    return _search


def create_web_search_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a web search tool.

    Config must include ``base_url`` pointing to a SearXNG instance.
    """
    base_url = config.get("base_url")
    if not base_url:
        raise ValueError("web_search requires 'base_url' in config")
    # Strip trailing slash
    base_url = base_url.rstrip("/")
    return StructuredTool.from_function(
        func=_create_search_fn(base_url),
        name="web_search",
        description=description,
        args_schema=WebSearchInput,
    )


class WebSearchTool(TearsTool):
    """TearsTool wrapper for web search via SearXNG.

    performs web searches against SearXNG instance and returns
    formatted results with titles, URLs, and snippets.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "search query",
            },
        },
        "required": ["query"],
    }

    def __init__(self, base_url: str) -> None:
        """initialize web search tool with SearXNG base URL.

        :param base_url: base URL of SearXNG instance (trailing slash stripped)
        :ptype base_url: str
        """
        self._base_url = base_url.rstrip("/")
        self._search_fn = _create_search_fn(self._base_url)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """perform web search.

        :param kwargs: must include 'query' key with search query string
        :ptype kwargs: Any
        :return: result containing formatted search results or error
        :rtype: ToolResult
        """
        query = kwargs.get("query", "")
        content = self._search_fn(query)
        success = not content.startswith("[TOOL ERROR]")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for web search.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="search web using SearXNG and return formatted results",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.web_search"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
