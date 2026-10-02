"""the messages a subscription has taken off the connection but not yet dispatched.

nats-py dates nothing: its ``Msg`` carries no receipt time, and a message waits in the
subscription's pending queue, unobserved, until something takes it. A request/reply
server that dates a request when its callback starts therefore counts none of the time
the request spent queued behind busy callbacks, and runs work whose caller has already
stopped waiting.

So :meth:`threetears.nats.NatsClient.subscribe` takes every message off the connection
the moment nats-py delivers it, dates it there, and holds it here until a callback is
free. That moves the backlog out of nats-py's pending queue, which is where nats-py's
slow-consumer bound lives, so this backlog carries the same bound: past it the client
stops taking messages, they wait on the connection again, and nats-py's own limit
applies to them there.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nats.aio.msg import Msg as _NatsMsg

__all__ = ["ReceiptBacklog"]


class ReceiptBacklog:
    """a bounded FIFO of received messages, each with the monotonic time it arrived.

    :param msgs_limit: most messages held at once; a put past it waits for a get
    :ptype msgs_limit: int
    :param bytes_limit: most payload bytes held at once; a put while at or past it waits
    :ptype bytes_limit: int
    :raises ValueError: when either bound is below 1
    """

    def __init__(self, *, msgs_limit: int, bytes_limit: int) -> None:
        if msgs_limit < 1:
            raise ValueError(f"msgs_limit must be at least 1; got {msgs_limit}")
        if bytes_limit < 1:
            raise ValueError(f"bytes_limit must be at least 1; got {bytes_limit}")
        self._msgs_limit = msgs_limit
        self._bytes_limit = bytes_limit
        self._held: deque[tuple[_NatsMsg, float]] = deque()
        self._bytes = 0
        self._closed = False
        self._changed = asyncio.Condition()

    def _has_room(self) -> bool:
        """report whether one more message may be taken.

        :return: True while under both bounds
        :rtype: bool
        """
        return len(self._held) < self._msgs_limit and self._bytes < self._bytes_limit

    def _has_message_or_closed(self) -> bool:
        """report whether a get can return.

        :return: True when a message is held or no more will come
        :rtype: bool
        """
        return bool(self._held) or self._closed

    async def put(self, msg: _NatsMsg, *, received: float) -> None:
        """hold a message, waiting while the backlog is at its bound.

        the caller dates the message before calling, so a wait here for room still
        counts toward the message's age.

        :param msg: the message nats-py delivered
        :ptype msg: nats.aio.msg.Msg
        :param received: when it was taken off the connection, on ``time.monotonic``
        :ptype received: float
        :return: nothing
        :rtype: None
        """
        async with self._changed:
            await self._changed.wait_for(self._has_room)
            self._held.append((msg, received))
            self._bytes += len(msg.data)
            self._changed.notify_all()

    async def get(self) -> tuple[_NatsMsg, float] | None:
        """take the oldest message, waiting while none is held.

        :return: the message and when it was received, or ``None`` once the backlog is
            closed and empty
        :rtype: tuple[nats.aio.msg.Msg, float] | None
        """
        async with self._changed:
            await self._changed.wait_for(self._has_message_or_closed)
            result: tuple[_NatsMsg, float] | None = None
            if self._held:
                result = self._held.popleft()
                self._bytes -= len(result[0].data)
                self._changed.notify_all()
            return result

    async def close(self) -> None:
        """mark that no more messages will arrive.

        messages already held are still handed out; a get on an empty closed backlog
        returns ``None``.

        :return: nothing
        :rtype: None
        """
        async with self._changed:
            self._closed = True
            self._changed.notify_all()
