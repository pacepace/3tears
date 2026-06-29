"""Timezone converter tool using stdlib zoneinfo."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error

__all__ = [
    "TimezoneConverterInput",
    "TimezoneConverterTool",
    "create_timezone_converter_tool",
]

_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%I:%M %p",
    "%I:%M%p",
    "%H:%M",
    "%H:%M:%S",
]

# Sentinels that mean "the current instant". When the caller passes one of
# these, the tool reads the clock itself instead of requiring an explicit
# datetime -- the current time is data the tool must carry, never a value an
# LLM caller infers (which it cannot do reliably).
_NOW_TOKENS = frozenset({"", "now", "current", "current time", "right now"})


class TimezoneConverterInput(BaseModel):
    """Input for the timezone converter tool."""

    time_str: str = Field(
        description=(
            "Time to convert. Pass 'now' for the current time and the tool resolves "
            "it -- do not compute the current time yourself. Otherwise an explicit "
            "time, e.g. '2024-01-15 14:30' or '3:00 PM'."
        )
    )
    from_timezone: str = Field(description="Source IANA timezone (e.g. 'America/New_York')")
    to_timezone: str = Field(description="Target IANA timezone (e.g. 'Europe/London')")


def _convert_timezone(time_str: str, from_timezone: str, to_timezone: str) -> str:
    try:
        from_tz = ZoneInfo(from_timezone)
    except ZoneInfoNotFoundError, KeyError:
        return tool_error("timezone_converter", "convert", f"unknown timezone: {from_timezone}")

    try:
        to_tz = ZoneInfo(to_timezone)
    except ZoneInfoNotFoundError, KeyError:
        return tool_error("timezone_converter", "convert", f"unknown timezone: {to_timezone}")

    if time_str.strip().lower() in _NOW_TOKENS:
        # "now": resolve the current instant in the source timezone directly,
        # so the caller never has to know or pass the current datetime.
        source_dt = datetime.now(from_tz)
    else:
        parsed: datetime | None = None
        for fmt in _TIME_FORMATS:
            try:
                parsed = datetime.strptime(time_str.strip(), fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            return tool_error("timezone_converter", "convert", f"could not parse time: {time_str}")

        # If parsed year == 1900 (time-only format), use today's date
        if parsed.year == 1900:
            today = date.today()
            parsed = parsed.replace(year=today.year, month=today.month, day=today.day)

        # Localize to source timezone
        source_dt = parsed.replace(tzinfo=from_tz)

    target_dt = source_dt.astimezone(to_tz)

    fmt_str = "%A, %B %d, %Y %I:%M %p %Z"
    return f"{source_dt.strftime(fmt_str)} = {target_dt.strftime(fmt_str)}"


def create_timezone_converter_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a timezone converter tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`TimezoneConverterTool.execute` as their single
    execution body. ``config`` is unused.
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        TimezoneConverterTool(),
        description=description,
        args_schema=TimezoneConverterInput,
    )


class TimezoneConverterTool(TearsTool):
    """TearsTool wrapper for timezone conversion via stdlib zoneinfo.

    converts time strings between IANA timezones. supports multiple
    input formats including ISO 8601, 12-hour, and 24-hour notation.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "time_str": {
                "type": "string",
                "description": (
                    "Pass 'now' for the current time (resolved by the tool); "
                    "otherwise an explicit time e.g. '2024-01-15 14:30' or '3:00 PM'."
                ),
            },
            "from_timezone": {
                "type": "string",
                "description": "IANA, e.g. America/New_York.",
            },
            "to_timezone": {
                "type": "string",
                "description": "IANA, e.g. Europe/London.",
            },
        },
        "required": ["time_str", "from_timezone", "to_timezone"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """convert time between timezones.

        :param kwargs: must include 'time_str', 'from_timezone', 'to_timezone' keys
        :ptype kwargs: Any
        :return: result containing converted time or error
        :rtype: ToolResult
        """
        time_str = kwargs.get("time_str", "")
        from_timezone = kwargs.get("from_timezone", "")
        to_timezone = kwargs.get("to_timezone", "")
        content = _convert_timezone(time_str, from_timezone, to_timezone)
        success = not content.startswith("[TOOL ERROR]")
        result = ToolResult(
            success=success,
            content=content,
            error=content if not success else None,
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for timezone converter.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="convert time between IANA timezones",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.timezone_converter"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
