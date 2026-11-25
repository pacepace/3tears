"""the per-call context a tool handler can ask for: progress, a client log channel, cancellation.

a handler is called with its validated arguments as keywords. one that also wants to talk to the
client while it runs declares a parameter annotated :class:`CallContext`; :class:`~threetears.mcp.
server.McpServer` finds it by signature when the :class:`~threetears.mcp.tool.McpTool` is built and
passes the call's context in it. a handler without one is called exactly as before::

    async def build(*, target: str, ctx: CallContext) -> str:
        await ctx.report_progress(1, 3, "cloning")
        await ctx.log("info", {"phase": "clone", "target": target})
        ...

the context carries no MCP SDK type, so a handler reaches progress, logging and cancellation
without importing the SDK. what it sends rides the call's own response stream, so it reaches the
client before the result; a Streamable-HTTP transport answering in plain JSON
(``json_response=True``) has no stream to carry it (see :class:`~threetears.mcp.http_server.
McpHttpTransport`).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any, Literal

__all__ = [
    "CallContext",
    "LogLevel",
]

LogLevel = Literal["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
"""the severity of a message to the client, the protocol's (RFC 5424) scale, least severe first."""

_ProgressSender = Callable[[float, float | None, str | None], Awaitable[None]]
"""sends one progress update for the call: ``(progress, total, message)``."""

_LogSender = Callable[[LogLevel, Any, str | None], Awaitable[None]]
"""sends one log message to the client: ``(level, data, logger)``."""


class CallContext:
    """one tool call as its handler sees it: who to report progress to, a log channel, a cancel signal.

    :class:`~threetears.mcp.server.McpServer` builds one per call. a test of a handler builds one
    itself, with senders that record what they are given.

    :param tool_name: the tool being called
    :ptype tool_name: str
    :param request_id: the protocol request id of the call, when it has one
    :ptype request_id: str | int | None
    :param progress_token: the token the client sent to ask for progress; ``None`` when it did not
        ask, and :meth:`report_progress` then sends nothing
    :ptype progress_token: str | int | None
    :param progress_sender: sends one progress update; ``None`` sends nothing
    :ptype progress_sender: Callable[[float, float | None, str | None], Awaitable[None]] | None
    :param log_sender: sends one log message to the client; ``None`` sends nothing
    :ptype log_sender: Callable[[LogLevel, Any, str | None], Awaitable[None]] | None
    :param cancelled: set when the call is cancelled; the server that builds the context sets it.
        a fresh, unset event when not given
    :ptype cancelled: asyncio.Event | None
    """

    def __init__(
        self,
        *,
        tool_name: str,
        request_id: str | int | None = None,
        progress_token: str | int | None = None,
        progress_sender: _ProgressSender | None = None,
        log_sender: _LogSender | None = None,
        cancelled: asyncio.Event | None = None,
    ) -> None:
        """capture the call's identifiers and the channels back to the client.

        :param tool_name: the tool being called
        :ptype tool_name: str
        :param request_id: the call's protocol request id
        :ptype request_id: str | int | None
        :param progress_token: the client's progress token
        :ptype progress_token: str | int | None
        :param progress_sender: sends one progress update
        :ptype progress_sender: Callable[[float, float | None, str | None], Awaitable[None]] | None
        :param log_sender: sends one log message
        :ptype log_sender: Callable[[LogLevel, Any, str | None], Awaitable[None]] | None
        :param cancelled: the call's cancellation event
        :ptype cancelled: asyncio.Event | None
        :return: nothing
        :rtype: None
        """
        self._tool_name = tool_name
        self._request_id = request_id
        self._progress_token = progress_token
        self._progress_sender = progress_sender
        self._log_sender = log_sender
        self._cancelled = cancelled if cancelled is not None else asyncio.Event()
        self._last_progress: float | None = None

    @property
    def tool_name(self) -> str:
        """the tool being called.

        :return: the tool's name
        :rtype: str
        """
        return self._tool_name

    @property
    def request_id(self) -> str | int | None:
        """the call's protocol request id, when it has one.

        :return: the request id
        :rtype: str | int | None
        """
        return self._request_id

    @property
    def progress_token(self) -> str | int | None:
        """the token the client sent to ask for progress, or ``None`` when it did not ask.

        :return: the progress token
        :rtype: str | int | None
        """
        return self._progress_token

    @property
    def cancelled(self) -> bool:
        """whether the call has been cancelled.

        the handler itself learns of a cancellation as the cancellation exception at its next
        ``await``; this is for work the call started outside its own task (a worker thread, a task
        in a longer-lived group) to see it.

        :return: ``True`` once the call is cancelled
        :rtype: bool
        """
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> None:
        """wait until the call is cancelled.

        :return: nothing, once the call is cancelled
        :rtype: None
        """
        await self._cancelled.wait()

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        """send ``notifications/progress`` for this call, when the client asked for progress.

        sends nothing when the client sent no progress token: progress is an answer to the client's
        request for it, not a broadcast.

        :param progress: how far the call has got; must be finite and greater than the last value
            reported for this call (the protocol's rule, so the client can order updates)
        :ptype progress: float
        :param total: the value ``progress`` reaches when the call is done, when it is known
        :ptype total: float | None
        :param message: a human-readable description of the step
        :ptype message: str | None
        :return: nothing
        :rtype: None
        :raises ValueError: ``progress`` or ``total`` is not finite, or ``progress`` does not
            exceed the last value reported for this call
        """
        if not math.isfinite(progress) or (total is not None and not math.isfinite(total)):
            raise ValueError(f"progress and total must be finite numbers; got progress={progress}, total={total}")
        if self._last_progress is not None and progress <= self._last_progress:
            raise ValueError(
                f"progress must increase with each report: {progress} follows {self._last_progress} "
                f"for tool {self._tool_name!r}",
            )
        self._last_progress = progress
        if self._progress_token is None or self._progress_sender is None:
            return
        await self._progress_sender(progress, total, message)

    async def log(self, level: LogLevel, data: Any, *, logger: str | None = None) -> None:
        """send a ``notifications/message`` log entry to the client, on this call's stream.

        a client on protocol 2026-07-28 or later receives it only when its request opted in at
        ``level`` or below (the ``io.modelcontextprotocol/logLevel`` key of the request's
        ``_meta``); otherwise it is not sent, as the protocol requires.

        :param level: the entry's severity
        :ptype level: LogLevel
        :param data: the entry, any JSON-serializable value
        :ptype data: Any
        :param logger: the name of the logger the entry comes from, when one is useful to the client
        :ptype logger: str | None
        :return: nothing
        :rtype: None
        """
        if self._log_sender is None:
            return
        await self._log_sender(level, data, logger)
