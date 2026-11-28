"""a mesh call's updates on the pod: the tool's progress and log entries, published towards the caller.

When a call's caller asked for updates (:class:`~threetears.core.tool_wire.CallUpdates` on the request) and the
tool's ``execute`` declares the call context, :class:`~threetears.agent.tools.server.ToolServer` passes it a
``threetears.mcp.CallContext`` whose senders are a :class:`CallUpdatePublisher`'s. The registry relays what it
publishes to the caller, whose MCP face hands it to its own client. The tool reports progress exactly as it does on
the MCP face.

The tool never waits on the link. A report is queued and the call goes on; one task per call sends what is queued,
in order, at most :data:`~threetears.nats.POD_UPDATES_PER_SECOND` a second. While it waits for budget, progress
coalesces to the latest value (a client wants where the call is, not every step it passed), and past
:data:`_MAX_PENDING` entries the oldest is dropped. Before the pod answers the call it gives what is still pending
:data:`~threetears.nats.UPDATE_FLUSH_SECONDS` to leave, so the last progress precedes the result. A publish that
fails loses that update and no other; what was dropped is logged once, when the call ends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from threetears.core.frozen_state import frozen, with_entry
from threetears.core.tool_wire import UPDATE_LOG_LEVELS, CallUpdate, CallUpdates
from threetears.nats import (
    MAX_UPDATE_BYTES,
    POD_UPDATE_BURST,
    POD_UPDATES_PER_SECOND,
    UPDATE_FLUSH_SECONDS,
    PublishError,
    Subject,
    UpdateBudget,
)
from threetears.observe import cancel_and_wait, get_logger

#: the MCP face, where the ``mcp`` extra of 3tears-tool-contract is installed: it finds and builds the context
_mcp_face: ModuleType | None
try:
    from threetears.tool_contract import mcp_face as _installed_mcp_face

    _mcp_face = _installed_mcp_face
except ImportError:
    # NOSILENT: without the mcp extra no execute can declare the MCP face's context, so a mesh call is given none
    _mcp_face = None

if TYPE_CHECKING:
    from collections.abc import Mapping

    from threetears.tool_contract.tool import TearsTool

__all__ = ["CallUpdatePublisher", "mesh_call_context"]

log = get_logger(__name__)

#: the most updates one call holds unsent; past it the oldest is dropped
_MAX_PENDING = 64

#: publishes one update's bytes on a subject (``NatsClient.publish_raw``'s shape)
_Publish = Callable[[Subject, bytes], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _Pending:
    """one update waiting to be sent: a progress report or a log entry."""

    kind: str
    progress: float | None = None
    total: float | None = None
    message: str | None = None
    level: str | None = None
    data: Any = None
    logger: str | None = None


class CallUpdatePublisher:
    """sends one mesh call's updates to the pod's progress subject, in order, bounded, never blocking the tool.

    :param publish: publishes bytes on a subject
    :ptype publish: Callable[[Subject, bytes], Awaitable[None]]
    :param updates: the call's updates request, its subject already checked to be this pod's own
    :ptype updates: CallUpdates
    :param tool_key: the tool, for the log
    :ptype tool_key: str
    :param budget: the call's send budget; the pod's default when not given
    :ptype budget: UpdateBudget | None
    """

    def __init__(
        self,
        publish: _Publish,
        updates: CallUpdates,
        *,
        tool_key: str,
        budget: UpdateBudget | None = None,
    ) -> None:
        """bind the publisher to one call.

        :param publish: publishes bytes on a subject
        :ptype publish: Callable[[Subject, bytes], Awaitable[None]]
        :param updates: the call's updates request
        :ptype updates: CallUpdates
        :param tool_key: the tool, for the log
        :ptype tool_key: str
        :param budget: the call's send budget
        :ptype budget: UpdateBudget | None
        :return: nothing
        :rtype: None
        """
        self._publish = publish
        self._subject = Subject.raw(updates.subject)
        self._tool_key = tool_key
        self._budget = (
            budget if budget is not None else UpdateBudget(rate=POD_UPDATES_PER_SECOND, burst=POD_UPDATE_BURST)
        )
        self._pending: tuple[_Pending, ...] = ()
        self._seq = 0
        self._sender: asyncio.Task[None] | None = None
        self._closed = False
        self._dropped: Mapping[str, int] = frozen()

    async def progress(self, progress: float, total: float | None, message: str | None) -> None:
        """queue one progress report; a report still waiting is replaced by it.

        :param progress: how far the call has got
        :ptype progress: float
        :param total: what ``progress`` reaches when done, when known
        :ptype total: float | None
        :param message: a description of the step
        :ptype message: str | None
        :return: nothing, at once
        :rtype: None
        """
        entry = _Pending(kind="progress", progress=progress, total=total, message=message)
        if self._pending and self._pending[-1].kind == "progress":
            # coalesced: the client wants where the call is now, and the step before it was never sent
            self._pending = (*self._pending[:-1], entry)
        else:
            self._queue(entry)

    async def log(self, level: str, data: Any, logger: str | None) -> None:
        """queue one log entry.

        :param level: the entry's severity
        :ptype level: str
        :param data: the entry
        :ptype data: Any
        :param logger: the logger it comes from
        :ptype logger: str | None
        :return: nothing, at once
        :rtype: None
        """
        self._queue(_Pending(kind="log", level=level, data=data, logger=logger))

    def _queue(self, entry: _Pending) -> None:
        """add ``entry`` to what is pending, the oldest dropped when full, and make sure a sender runs.

        :param entry: the update
        :ptype entry: _Pending
        :return: nothing
        :rtype: None
        """
        if self._closed:
            self._drop("after_answer")
            return
        pending = (*self._pending, entry)
        if len(pending) > _MAX_PENDING:
            self._drop("queue_full")
            pending = pending[1:]
        self._pending = pending
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_pending(), name=f"call-updates:{self._tool_key}")

    def _drop(self, why: str) -> None:
        """count one update not sent, by why.

        :param why: the reason
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._dropped = with_entry(self._dropped, why, self._dropped.get(why, 0) + 1)

    async def _send_pending(self) -> None:
        """send what is pending, in order, within the budget.

        :return: nothing, once nothing is pending
        :rtype: None
        """
        while self._pending:
            wait = self._budget.wait()
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            if not self._budget.take():
                continue
            entry, self._pending = self._pending[0], self._pending[1:]
            await self._send(entry)

    async def _send(self, entry: _Pending) -> None:
        """publish one update with its sequence number, or count why not.

        :param entry: the update
        :ptype entry: _Pending
        :return: nothing
        :rtype: None
        """
        self._seq += 1
        payload = self._encoded(entry, self._seq)
        if payload is None:
            return
        try:
            await self._publish(self._subject, payload)
        except PublishError:
            # NOSILENT: counted, and the count is logged when the call ends; a lost update loses only itself
            self._drop("publish_failed")

    def _encoded(self, entry: _Pending, seq: int) -> bytes | None:
        """``entry`` as the bytes of a :class:`CallUpdate`, within :data:`MAX_UPDATE_BYTES`, or ``None``.

        a log entry over the size is sent with its data replaced by a note of its size; one that cannot be encoded at
        all (data that is no JSON value) is sent with a note saying so.

        :param entry: the update
        :ptype entry: _Pending
        :param seq: its sequence number
        :ptype seq: int
        :return: the bytes, or ``None`` when it was dropped
        :rtype: bytes | None
        """
        try:
            update = CallUpdate(
                seq=seq,
                kind="progress" if entry.kind == "progress" else "log",
                progress=entry.progress,
                total=entry.total,
                message=entry.message,
                level=entry.level,  # type: ignore[arg-type]  # one of the levels the context checked
                data=entry.data,
                logger=entry.logger,
            )
            payload = update.model_dump_json().encode("utf-8")
        except (ValidationError, ValueError, TypeError) as exc:
            if entry.kind != "log":
                self._drop("unencodable")
                return None
            update = CallUpdate(
                seq=seq,
                kind="log",
                level=entry.level,  # type: ignore[arg-type]
                data={"not_sent": f"the entry's data is no JSON value: {type(exc).__name__}"},
                logger=entry.logger,
            )
            payload = update.model_dump_json().encode("utf-8")
        if len(payload) <= MAX_UPDATE_BYTES:
            return payload
        if entry.kind == "log":
            self._drop("truncated")
            note = update.model_copy(update={"data": {"not_sent": f"the entry was {len(payload)} bytes"}})
            return note.model_dump_json().encode("utf-8")
        self._drop("too_large")
        return None

    async def close(self, *, timeout: float = UPDATE_FLUSH_SECONDS) -> None:
        """let what is pending leave, for at most ``timeout``, then stop; log what was not sent.

        called before the pod answers the call, so the last progress precedes the result.

        :param timeout: the most this waits
        :ptype timeout: float
        :return: nothing
        :rtype: None
        """
        self._closed = True
        sender = self._sender
        if sender is not None and not sender.done():
            done, _ = await asyncio.wait({sender}, timeout=timeout)
            if not done:
                for _entry in self._pending:
                    self._drop("unsent_at_answer")
                self._pending = ()
                await cancel_and_wait(sender)
        if self._dropped:
            log.warning(
                "some of a call's updates were not sent to its caller",
                extra={"extra_data": {"tool_key": self._tool_key, "sent": self._seq, "dropped": dict(self._dropped)}},
            )


def _log_levels_from(updates: CallUpdates) -> frozenset[str]:
    """the levels a call's caller takes: its ``log_level`` and every more severe one, or none.

    :param updates: the call's updates request
    :ptype updates: CallUpdates
    :return: the levels
    :rtype: frozenset[str]
    """
    if updates.log_level is None:
        return frozenset()
    return frozenset(UPDATE_LOG_LEVELS[UPDATE_LOG_LEVELS.index(updates.log_level) :])


def _context_parameter(tool: TearsTool) -> str | None:
    """the parameter of ``tool.execute`` that takes the call context, or ``None``.

    read only for a call that asked for updates, as the MCP face reads it, which needs the ``mcp`` extra of
    3tears-tool-contract: without it no module can declare the MCP face's context, so none is passed. A declaration
    the MCP face refuses is logged and passed nothing: the call runs as a mesh call always has.

    :param tool: the tool
    :ptype tool: TearsTool
    :return: the parameter's name, or ``None``
    :rtype: str | None
    """
    if _mcp_face is None:
        return None
    try:
        parameter: str | None = _mcp_face.mesh_context_parameter(tool)
    except ValueError as exc:
        log.warning(
            "a tool's execute declares the call context in a form the MCP face refuses; it is given none on the mesh",
            extra={"extra_data": {"tool": tool.mcp_name(), "reason": str(exc)}},
        )
        return None
    return parameter


def mesh_call_context(
    tool: TearsTool,
    updates: CallUpdates,
    publisher: CallUpdatePublisher,
    cancelled: asyncio.Event,
) -> tuple[str, Any] | None:
    """the keyword and the call context to pass ``tool.execute`` on a mesh call that asked for updates, or ``None``.

    :param tool: the tool
    :ptype tool: TearsTool
    :param updates: the call's updates request
    :ptype updates: CallUpdates
    :param publisher: sends the context's progress and log entries
    :ptype publisher: CallUpdatePublisher
    :param cancelled: set when the call is cut short on the pod
    :ptype cancelled: asyncio.Event
    :return: ``(parameter, context)``, or ``None`` when ``execute`` declares no context
    :rtype: tuple[str, Any] | None
    """
    parameter = _context_parameter(tool)
    if parameter is None or _mcp_face is None:
        return None
    context = _mcp_face.mesh_call_context(
        tool_name=tool.mcp_name(),
        progress_token=updates.progress_token,
        progress_sender=publisher.progress,
        log_sender=publisher.log,
        log_levels=_log_levels_from(updates),
        cancelled=cancelled,
    )
    return parameter, context
