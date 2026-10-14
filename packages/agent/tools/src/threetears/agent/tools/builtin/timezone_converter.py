"""Timezone converter tool: a time in one zone, as it is in another.

A time with no date is on the source zone's day, read from the clock in that
zone: stamped with the server's day, "3:00 PM" in Chicago at 03:21 UTC was
printed as the next day, every evening in the Americas, as a success. The result
names both zones with their offsets, says when the target is on another day, and
says how anything written loosely ("PST", "chicago", "3pm") was read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.time_words import (
    Clock,
    ReadError,
    ZoneRead,
    describe_zone,
    parse_when,
    read_zone,
    system_clock,
)
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.utils import tool_error

__all__ = [
    "TimezoneConverterTool",
    "create_timezone_converter_tool",
]

_TOOL = "timezone_converter"

# Sentinels that mean "the current instant". When the caller passes one of
# these, the tool reads the clock itself instead of requiring an explicit
# datetime -- the current time is data the tool must carry, never a value an
# LLM caller infers (which it cannot do reliably).
_NOW_TOKENS = frozenset({"", "now", "current", "current time", "right now"})

_FORMAT = "%-I:%M %p %A, %B %-d, %Y"


def _person_zone(construction_zone: str | None) -> str | None:
    """The zone of the person the call is for: the call scope's, else the one built in."""
    scope = current_scope()
    scoped = scope.context.user_timezone if scope is not None else None
    return scoped or construction_zone


def _convert(
    time_str: str,
    from_timezone: str | None,
    to_timezone: str | None,
    *,
    person_zone: str | None,
    clock: Clock,
) -> str:
    """The conversion, as the line the model reads.

    :param time_str: the time as written, or ``now``
    :ptype time_str: str
    :param from_timezone: the zone the time is in; the person's zone when left out
    :ptype from_timezone: str | None
    :param to_timezone: the zone to show it in; the person's zone when left out
    :ptype to_timezone: str | None
    :param person_zone: the zone of the person the call is for, or ``None``
    :ptype person_zone: str | None
    :param clock: what returns now
    :ptype clock: Clock
    :return: the result, or a tool error saying what to write
    :rtype: str
    """
    reads: list[str] = []
    try:
        if not (from_timezone or "").strip() and not (to_timezone or "").strip():
            raise ReadError("give the zone the time is in, the zone to show it in, or both.")
        zones: list[ZoneRead] = []
        for given, role in ((from_timezone, "from"), (to_timezone, "to")):
            if (given or "").strip():
                zones.append(read_zone(given))
            elif person_zone:
                zone = read_zone(person_zone)
                zones.append(zone)
                reads.append(f"no {role} zone given, so the person's zone, {zone.name}, was used")
            else:
                raise ReadError(f"no {role} zone given, and the person's zone is not known; name the {role} zone.")
        source, target = zones
        for zone in zones:
            if zone.how:
                reads.append(zone.how)

        now_there = clock().astimezone(source.zone)
        if time_str.strip().lower() in _NOW_TOKENS:
            source_dt = now_there
        else:
            when = parse_when(time_str, now_there.date())
            if when.instant is not None:
                source_dt = when.instant.astimezone(source.zone)
                reads.append(f"the time carried its own offset, so it was read as {describe_zone(when.instant)}")
            else:
                day = when.day
                if day is None:
                    # The source zone's day, read from the clock there: never the server's.
                    day = now_there.date()
                    reads.append(f"no date given, so today in {source.name}, {day.strftime('%A, %B %-d')}")
                clock_time = when.clock if when.clock is not None else datetime.min.time()
                if when.clock is None:
                    reads.append("no time given, so midnight at the start of that day")
                naive = datetime.combine(day, clock_time)
                source_dt = naive.replace(tzinfo=source.zone)
                round_trip = source_dt.astimezone(UTC).astimezone(source.zone).replace(tzinfo=None)
                if round_trip != naive:
                    raise ReadError(
                        f"{naive.strftime('%-I:%M %p on %B %-d, %Y')} does not happen in {source.name}: the clocks "
                        "skip that hour when summer time starts. Give a time outside it."
                    )
                if source_dt.replace(fold=1).utcoffset() != source_dt.utcoffset():
                    reads.append(
                        f"{naive.strftime('%-I:%M %p')} happens twice that day in {source.name}, as the clocks go "
                        "back; the first was used"
                    )
    except ReadError as exc:
        return tool_error(_TOOL, "read", str(exc))

    target_dt = source_dt.astimezone(target.zone)
    days = (target_dt.date() - source_dt.date()).days
    day_word = {1: ", the next day", -1: ", the day before"}.get(days, f", {days:+d} days" if days else "")
    line = (
        f"{source_dt.strftime(_FORMAT)} in {source.name} ({describe_zone(source_dt)}) = "
        f"{target_dt.strftime(_FORMAT)} in {target.name} ({describe_zone(target_dt)}){day_word}"
    )
    return line + (f"\nRead: {'; '.join(reads)}." if reads else "")


def create_timezone_converter_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a timezone converter tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer
    path share :meth:`TimezoneConverterTool.execute` as their single
    execution body. ``config["_user_timezone"]`` (legacy ``"timezone"``)
    is the person's zone, used for a zone the call leaves out.
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        TimezoneConverterTool(timezone=config.get("_user_timezone") or config.get("timezone")),
        description=description,
    )


class TimezoneConverterTool(TearsTool):
    """TearsTool wrapper for timezone conversion via stdlib zoneinfo.

    converts a time between zones written as IANA names in any case, cities,
    abbreviations or offsets. a zone left out is the person's zone (the call
    scope's, else the one given at construction).

    :param timezone: the person's zone when the call scope carries none
    :ptype timezone: str | None
    :param clock: what returns now; a test passes a frozen one
    :ptype clock: Clock | None
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "time_str": {
                "type": "string",
                "description": (
                    "The time to convert, as written: '3pm', '9:30 AM', '15:00', 'noon', 'tomorrow 9:00', "
                    "'Oct 9 3pm', '2026-10-10T14:00:00Z'. Pass 'now' for the current time; do not work it out "
                    "yourself. A time with no date is on today's date in the from zone."
                ),
            },
            "from_timezone": {
                "type": "string",
                "description": (
                    "The zone the time is in: an IANA name (America/Chicago, any case), a city (Tokyo), an "
                    "abbreviation (PST, EST, CET, IST is India) or an offset (UTC+5:30). Leave out for the "
                    "person's zone."
                ),
            },
            "to_timezone": {
                "type": "string",
                "description": "The zone to show it in, written the same ways. Leave out for the person's zone.",
            },
        },
        "required": ["time_str"],
    }

    def __init__(self, timezone: str | None = None, *, clock: Clock | None = None) -> None:
        """keep the person's zone and the clock.

        :param timezone: the person's zone when the call scope carries none
        :ptype timezone: str | None
        :param clock: what returns now
        :ptype clock: Clock | None
        """
        self._timezone = timezone
        self._clock = clock or system_clock

    async def execute(self, **kwargs: Any) -> ToolResult:
        """convert time between timezones.

        :param kwargs: 'time_str', and 'from_timezone' and/or 'to_timezone'
        :ptype kwargs: Any
        :return: result containing converted time or error
        :rtype: ToolResult
        """
        content = _convert(
            str(kwargs.get("time_str") or ""),
            kwargs.get("from_timezone"),
            kwargs.get("to_timezone"),
            person_zone=_person_zone(self._timezone),
            clock=self._clock,
        )
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
            description="convert a time from one zone to another; says which zones and which day it used",
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
