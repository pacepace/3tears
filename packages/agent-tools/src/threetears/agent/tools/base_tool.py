"""TearsTool abstract base class and supporting dataclasses.

defines interface that all tools must implement to participate
in both direct mode and MCP server mode. no platform dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from threetears.agent.tools._coercion import normalize_kwargs


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
    :param timeout_seconds: expected maximum execution time, None uses caller default
    :ptype timeout_seconds: float | None
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None


class TearsTool(ABC):
    """abstract base class for all platform tools.

    tools implement this interface to work in both direct mode
    (called from agent code) and MCP server mode (served via
    ToolServer, called through Registry proxy). no platform
    dependencies required -- standalone by design.

    ``execute`` is a template method that normalizes loose
    LLM-supplied inputs (empty strings or JSON-encoded strings
    for object/array fields) against this tool's declared
    ``mcp_schema()`` input schema, then dispatches to subclass
    ``_execute``. subclasses override ``_execute``, never
    ``execute`` itself, so coercion always runs.
    """

    async def execute(self, **kwargs: Any) -> ToolResult:
        """normalize loose LLM-supplied kwargs then dispatch to subclass.

        coerces wrong-shape object/array values (empty strings or
        JSON-encoded strings) toward their declared JSON schema
        types, then calls subclass ``_execute`` with the normalized
        kwargs. if ``mcp_schema`` itself raises, falls back to the
        original kwargs so coercion never breaks a tool.

        :param kwargs: raw tool input parameters
        :ptype kwargs: Any
        :return: execution result from subclass ``_execute``
        :rtype: ToolResult
        """
        try:
            schema = self.mcp_schema()
            normalized = normalize_kwargs(kwargs, schema.input_schema)
        except Exception:
            normalized = kwargs
        return await self._execute(**normalized)

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult:
        """execute tool with normalized parameters.

        subclasses implement this. ``execute`` runs coercion against
        ``mcp_schema().input_schema`` before calling ``_execute``, so
        object/array kwargs arrive as native containers even when the
        LLM caller supplied empty strings or JSON-encoded strings.

        :param kwargs: tool-specific input parameters, post-coercion
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
