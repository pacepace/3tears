"""TearsTool abstract base class and supporting dataclasses.

defines interface that all tools must implement to participate
in both direct mode and MCP server mode. no platform dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """result returned from tool execution.

    :param success: whether execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class MCPToolDefinition:
    """MCP-compatible tool definition schema.

    :param name: tool name in namespace convention ({package}.{name})
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param output_schema: optional JSON Schema for tool output
    :ptype output_schema: dict[str, Any] | None
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None


class TearsTool(ABC):
    """abstract base class for all platform tools.

    tools implement this interface to work in both direct mode
    (called from agent code) and MCP server mode (served via
    ToolServer, called through Registry proxy). no platform
    dependencies required -- standalone by design.
    """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """execute tool with given parameters.

        :param kwargs: tool-specific input parameters
        :ptype kwargs: Any
        :return: execution result
        :rtype: ToolResult
        """
        ...

    @abstractmethod
    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition.

        :return: tool definition with name, version, description, schemas
        :rtype: MCPToolDefinition
        """
        ...

    @abstractmethod
    def mcp_name(self) -> str:
        """return namespaced tool name.

        format: {package}.{name} (e.g., 'threetears.calculator')

        :return: namespaced tool name
        :rtype: str
        """
        ...

    @abstractmethod
    def mcp_version(self) -> str:
        """return semver-compatible tool version.

        :return: version string (e.g., '1.0.0')
        :rtype: str
        """
        ...
