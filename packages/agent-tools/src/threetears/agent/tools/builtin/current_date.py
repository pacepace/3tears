"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult


def _create_date_fn(user_timezone: str | None) -> Any:
    """Create a date function, optionally bound to a user timezone."""

    def _get_current_date() -> str:
        now_utc = datetime.now(timezone.utc)
        result = f"UTC: {now_utc.strftime('%A, %B %d, %Y %I:%M %p %Z')}"

        if user_timezone:
            try:
                tz = ZoneInfo(user_timezone)
                now_local = now_utc.astimezone(tz)
                result += f"\nLocal ({user_timezone}): {now_local.strftime('%A, %B %d, %Y %I:%M %p %Z')}"
            except ZoneInfoNotFoundError, KeyError:
                result += f"\n(Could not resolve timezone: {user_timezone})"

        return result

    return _get_current_date


def create_current_date_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a current date/time tool."""
    user_timezone = config.get("_user_timezone") or config.get("timezone")
    return StructuredTool.from_function(
        func=_create_date_fn(user_timezone),
        name="current_date",
        description=description,
    )


class CurrentDateTool(TearsTool):
    """TearsTool wrapper for retrieving current date and time.

    returns current UTC date/time and optionally converts to
    specified timezone. always succeeds (no external dependencies).
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, timezone: str | None = None) -> None:
        """initialize current date tool with optional timezone.

        :param timezone: IANA timezone name for local time display
        :ptype timezone: str | None
        """
        self._timezone = timezone
        self._date_fn = _create_date_fn(timezone)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """return current date and time.

        :param kwargs: ignored (no input parameters)
        :ptype kwargs: Any
        :return: result containing current date/time string
        :rtype: ToolResult
        """
        content = self._date_fn()
        result = ToolResult(success=True, content=content)
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for current date.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="return current date and time in UTC and optional local timezone",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.current_date"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
