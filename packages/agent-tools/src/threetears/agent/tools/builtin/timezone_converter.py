"""Timezone converter tool using stdlib zoneinfo."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.utils import tool_error

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


class TimezoneConverterInput(BaseModel):
    """Input for the timezone converter tool."""

    time_str: str = Field(description="Time string to convert (e.g. '2024-01-15 14:30', '3:00 PM')")
    from_timezone: str = Field(description="Source IANA timezone (e.g. 'America/New_York')")
    to_timezone: str = Field(description="Target IANA timezone (e.g. 'Europe/London')")


def _convert_timezone(time_str: str, from_timezone: str, to_timezone: str) -> str:
    try:
        from_tz = ZoneInfo(from_timezone)
    except (ZoneInfoNotFoundError, KeyError):
        return tool_error("timezone_converter", "convert", f"unknown timezone: {from_timezone}")

    try:
        to_tz = ZoneInfo(to_timezone)
    except (ZoneInfoNotFoundError, KeyError):
        return tool_error("timezone_converter", "convert", f"unknown timezone: {to_timezone}")

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

    # Localize to source timezone and convert
    source_dt = parsed.replace(tzinfo=from_tz)
    target_dt = source_dt.astimezone(to_tz)

    fmt_str = "%A, %B %d, %Y %I:%M %p %Z"
    return f"{source_dt.strftime(fmt_str)} = {target_dt.strftime(fmt_str)}"


def create_timezone_converter_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a timezone converter tool."""
    return StructuredTool.from_function(
        func=_convert_timezone,
        name="timezone_converter",
        description=description,
        args_schema=TimezoneConverterInput,
    )
