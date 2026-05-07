"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope

__all__ = [
    "CurrentDateTool",
    "create_current_date_tool",
]


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
    """Factory: create a current date/time tool.

    the StructuredTool ``func`` and the :class:`CurrentDateTool`
    ``execute`` both build their date string via
    :func:`_create_date_fn`, so the formatting logic lives in one
    place. construction-time ``timezone`` is read from
    ``config["_user_timezone"]`` (legacy ``"timezone"`` also honoured)
    for the StructuredTool path; the TearsTool path additionally reads
    per-call tz off :class:`ToolCallScope` / :class:`CallContext`.
    naming uniform: both register as ``threetears.current_date``.
    """
    user_timezone = config.get("_user_timezone") or config.get("timezone")
    return StructuredTool.from_function(
        func=_create_date_fn(user_timezone),
        name="threetears.current_date",
        description=description,
    )


class CurrentDateTool(TearsTool):
    """TearsTool wrapper for retrieving current date and time.

    returns current UTC date/time and, when an IANA timezone is
    supplied (either at construction or per-call), also returns the
    same instant rendered in that local zone. always succeeds (no
    external dependencies).
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "Optional IANA timezone name (e.g. 'America/Los_Angeles', "
                    "'Europe/Berlin') to render local time alongside UTC. "
                    "When omitted, the tool falls back to the timezone "
                    "configured on the agent at construction time, and "
                    "returns UTC only when neither is set. The agent's "
                    "system prompt carries the user's resolved timezone "
                    "so callers should pass it through verbatim when the "
                    "user asks for 'local time' or 'my time'."
                ),
            },
        },
    }

    def __init__(self, timezone: str | None = None) -> None:
        """initialize current date tool with optional construction-time timezone.

        :param timezone: IANA timezone name used as the fallback when a
            call does not supply ``timezone``; the agent runtime sets
            this from per-agent config, while per-message browser /
            channel-supplied timezones flow through the call kwargs
        :ptype timezone: str | None
        """
        self._timezone = timezone

    async def execute(self, **kwargs: Any) -> ToolResult:
        """return current date and time, optionally rendered in a local zone.

        timezone resolution priority (highest to lowest):

        1. ``timezone`` kwarg passed by the LLM in the tool call
        2. :attr:`CallContext.user_timezone` on the current
           :class:`~threetears.agent.tools.call_scope.ToolCallScope`
           (channel-adapter-resolved per-message browser / slack
           ``users.info`` / discord-locale value); this is the
           authoritative source the agent runtime stamps on every
           inbound chat
        3. ``self._timezone`` (construction-time fallback when the
           agent runtime configures a default)
        4. ``None`` -- returns UTC only

        the ``current_scope()`` read returns ``None`` when the tool is
        invoked outside the NATS dispatch path (unit tests calling
        ``execute()`` directly); that path falls through to kwargs +
        construction default with no error.

        :param kwargs: optional ``timezone`` IANA name overriding the
            other sources. unknown kwargs are ignored
        :ptype kwargs: Any
        :return: result containing UTC line and (when a tz is
            available) a paired local-time line
        :rtype: ToolResult
        """
        per_call_tz: str | None = kwargs.get("timezone")
        scope = current_scope()
        scope_tz = scope.context.user_timezone if scope is not None else None
        effective_tz = per_call_tz or scope_tz or self._timezone
        date_fn = _create_date_fn(effective_tz)
        content = date_fn()
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
            description=(
                "return current date and time in UTC, plus the same "
                "instant in a local timezone when ``timezone`` is "
                "supplied. the agent system prompt carries the user's "
                "resolved tz on every turn, so prefer passing it "
                "through whenever the user references 'local time'."
            ),
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
