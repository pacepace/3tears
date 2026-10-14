"""Date arithmetic: the weekday of a date, a date plus or minus days, weeks, months or years, and days between.

Asked "what day is Christmas" or "90 days from today", a model counts on its
fingers and is wrong about one time in a few. This tool does the counting.
"Today" is today in the person's zone, read from the clock there, and the
result says which day and zone it took.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.time_words import Clock, ReadError, parse_date, read_zone, system_clock
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.utils import tool_error

__all__ = [
    "DateMathTool",
    "create_date_math_tool",
]

_TOOL = "date_math"

_UNITS = {
    "day": "days",
    "days": "days",
    "d": "days",
    "week": "weeks",
    "weeks": "weeks",
    "w": "weeks",
    "fortnight": "fortnights",
    "fortnights": "fortnights",
    "month": "months",
    "months": "months",
    "mo": "months",
    "year": "years",
    "years": "years",
    "y": "years",
    "yr": "years",
    "yrs": "years",
}
_AMOUNT = re.compile(r"([+-]?\s*\d+)\s*([a-z]+)", re.IGNORECASE)
_ADD_FORMS = "'90 days', '-2 weeks', '3 months', '1 year 2 months', 'minus 10 days'"
_RELATIVE = re.compile(r"\b(?:today|tomorrow|yesterday|tonight|next|last|this)\b", re.IGNORECASE)


def _long(day: date) -> str:
    return day.strftime("%A, %B %-d, %Y")


def _add_months(day: date, months: int) -> tuple[date, bool]:
    """The same day of the month ``months`` later; the month's last day when it has no such day.

    :return: the date, and whether the day was moved back to fit the month
    """
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    last = calendar.monthrange(year, month + 1)[1]
    return date(year, month + 1, min(day.day, last)), day.day > last


def _read_amount(text: str) -> dict[str, int]:
    """Read '90 days', '-2 weeks', '1 year 2 months', 'minus 10 days' into amounts by unit."""
    cleaned = text.strip().lower()
    sign = 1
    for word, value in (("minus ", -1), ("plus ", 1), ("in ", 1)):
        if cleaned.startswith(word):
            cleaned, sign = cleaned[len(word) :], value
    for word in ("ago", "before"):
        if cleaned.endswith(f" {word}"):
            cleaned, sign = cleaned[: -len(word) - 1], -sign
    parts = _AMOUNT.findall(cleaned.replace(",", " ").replace(" and ", " "))
    leftover = _AMOUNT.sub("", cleaned).replace(",", "").replace("and", "").strip()
    if not parts or leftover:
        raise ReadError(f"could not read {text!r} as an amount of time. Write {_ADD_FORMS}.")
    amounts: dict[str, int] = {}
    for number, unit in parts:
        name = _UNITS.get(unit.lower())
        if name is None:
            raise ReadError(f"{unit!r} is not a unit this tool counts in: days, weeks, months or years.")
        amounts[name] = amounts.get(name, 0) + sign * int(number.replace(" ", ""))
    return amounts


def _phrase(amounts: dict[str, int]) -> str:
    """The amount as the result says it: '+ 90 days', '- 2 weeks', '+ 1 year 2 months'."""
    order = ("years", "months", "fortnights", "weeks", "days")
    items = [(amounts[u], u) for u in order if amounts.get(u)]
    if not items:
        return "+ 0 days"

    def _one(n: int, unit: str) -> str:
        return f"{abs(n)} {unit if abs(n) != 1 else unit[:-1]}"

    if all(n > 0 for n, _ in items) or all(n < 0 for n, _ in items):
        sign = "+" if items[0][0] > 0 else "-"
        return f"{sign} " + " ".join(_one(n, u) for n, u in items)
    return " ".join(f"{'+' if n > 0 else '-'} {_one(n, u)}" for n, u in items)


def _between(start: date, end: date) -> str:
    days = (end - start).days
    span = abs(days)
    weeks, rest = divmod(span, 7)
    in_weeks = f" ({weeks} week{'s' if weeks != 1 else ''}" + (
        f" and {rest} day{'s' if rest != 1 else ''})" if rest else ")"
    )
    if days == 0:
        return f"{_long(start)} and {_long(end)} are the same day."
    when = "after" if days > 0 else "before"
    return f"{_long(end)} is {span} day{'s' if span != 1 else ''}{in_weeks if weeks else ''} {when} {_long(start)}."


def _date_math(
    *,
    date_text: str | None,
    add: str | None,
    until: str | None,
    zone_text: str | None,
    person_zone: str | None,
    clock: Clock,
) -> str:
    """The answer, as the line the model reads.

    :return: the result, or a tool error saying what to write
    :rtype: str
    """
    reads: list[str] = []
    try:
        if add and until:
            raise ReadError("pass add or until, not both: add counts from the date, until counts to another.")
        zone_name = (zone_text or "").strip() or person_zone
        zone = read_zone(zone_name) if zone_name else None
        today = clock().astimezone(zone.zone).date() if zone else clock().date()
        today_said = f"today is {_long(today)} in {zone.name if zone else 'UTC'}"
        if zone and zone.how:
            reads.append(zone.how)
        if (date_text or "").strip():
            start = parse_date(date_text or "", today)
            if _RELATIVE.search(date_text or ""):
                reads.append(today_said)
            elif not re.search(r"\d{4}", date_text or ""):
                reads.append(f"no year given, so {today.year}")
        else:
            start = today
            reads.append(today_said)

        if until is not None and until.strip():
            end = parse_date(until, today)
            if _RELATIVE.search(until) and today_said not in reads:
                reads.append(today_said)
            line = _between(start, end)
        elif add is not None and add.strip():
            amounts = _read_amount(add)
            result = start
            moved = False
            months = amounts.get("months", 0) + 12 * amounts.get("years", 0)
            if months:
                result, moved = _add_months(result, months)
            result += timedelta(
                days=amounts.get("days", 0) + 7 * amounts.get("weeks", 0) + 14 * amounts.get("fortnights", 0)
            )
            line = f"{_long(start)} {_phrase(amounts)} = {_long(result)}."
            if moved:
                reads.append(
                    f"{start.strftime('%B')} {start.day} has no match in the month reached, so its last day was used"
                )
        else:
            day_of_year = start.timetuple().tm_yday
            week = start.isocalendar().week
            line = f"{_long(start)}: day {day_of_year} of {start.year}, ISO week {week}."
    except ReadError as exc:
        return tool_error(_TOOL, "read", str(exc))
    return line + (f"\nRead: {'; '.join(reads)}." if reads else "")


def create_date_math_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a date_math tool.

    ``config["_user_timezone"]`` (legacy ``"timezone"``) is the person's zone,
    whose day "today" is.

    :param config: the tool's configuration
    :ptype config: dict[str, Any]
    :param description: the description the model reads
    :ptype description: str
    :return: the LangChain tool
    :rtype: StructuredTool
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    return to_langchain_tool(
        DateMathTool(timezone=config.get("_user_timezone") or config.get("timezone")),
        description=description,
    )


class DateMathTool(TearsTool):
    """TearsTool for date arithmetic: weekday, add or subtract, days between.

    :param timezone: the person's zone when the call scope carries none
    :ptype timezone: str | None
    :param clock: what returns now; a test passes a frozen one
    :ptype clock: Clock | None
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": (
                    "The date to start from: '2026-12-25', 'Dec 25', '25 December 2026', 'tomorrow', 'next friday'. "
                    "Leave out for today in the person's zone. Alone, it gives that date's weekday."
                ),
            },
            "add": {
                "type": "string",
                "description": f"An amount to add, or subtract with a minus: {_ADD_FORMS}.",
            },
            "until": {
                "type": "string",
                "description": "Another date: the answer is how many days from date to it.",
            },
            "timezone": {
                "type": "string",
                "description": "The zone whose today counts. Leave out for the person's zone.",
            },
        },
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
        """work out the date, or the days between two.

        :param kwargs: optional 'date', 'add', 'until', 'timezone'
        :ptype kwargs: Any
        :return: the answer, or a tool error saying what to write
        :rtype: ToolResult
        """
        scope = current_scope()
        scoped = scope.context.user_timezone if scope is not None else None
        content = _date_math(
            date_text=kwargs.get("date"),
            add=kwargs.get("add"),
            until=kwargs.get("until"),
            zone_text=kwargs.get("timezone"),
            person_zone=scoped or self._timezone,
            clock=self._clock,
        )
        success = not content.startswith("[TOOL ERROR]")
        return ToolResult(success=success, content=content, error=content if not success else None)

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for date_math.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="weekday of a date, a date plus or minus days, weeks, months or years, and days between dates",
            input_schema=self._INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.date_math"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
