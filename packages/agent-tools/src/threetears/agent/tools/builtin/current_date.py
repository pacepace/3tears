"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool


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
            except (ZoneInfoNotFoundError, KeyError):
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
