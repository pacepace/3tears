"""narrow Protocols for NATS transport surfaces.

these Protocols let test fakes substitute for the live
:class:`threetears.nats.NatsClient` without subclassing it. every fake
in 3tears integration tests should satisfy one of these Protocols
rather than emulate the full client surface.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from threetears.nats.subjects import Subject

__all__ = [
    "MessageCallback",
    "RawMessageCallback",
    "StreamTransport",
]


_M = TypeVar("_M", bound="BaseModel")


MessageCallback = Callable[["BaseModel"], Awaitable[None]]
"""typed subscribe callback shape for :meth:`NatsClient.subscribe_typed`.

receives the validated Pydantic message; raising propagates to the
deadletter dispatcher when the subscription has
``deadletter_on_error=True``.
"""


RawMessageCallback = Callable[[bytes], Awaitable[None]]
"""raw subscribe callback shape for :meth:`NatsClient.subscribe`.

receives the message payload as bytes; the caller is responsible for
decoding. high-throughput sites that bypass Pydantic validation use
this shape.
"""


@runtime_checkable
class StreamTransport(Protocol):
    """minimal publish surface for streaming consumers.

    streaming sites (gateway -> agent token streams, agent -> hub token
    streams) only need ``publish_raw``; they should depend on this
    Protocol rather than the full :class:`NatsClient`. this lets tests
    inject a publish-recording fake without emulating subscribe / KV /
    request paths.
    """

    async def publish_raw(
        self,
        *,
        subject: "Subject",
        payload: bytes,
        reply_to: "Subject | None" = None,
    ) -> None:
        """publish raw bytes to subject.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: message payload bytes
        :ptype payload: bytes
        :param reply_to: optional reply subject for request-style transports
        :ptype reply_to: Subject | None
        :return: nothing
        :rtype: None
        """
        ...

    async def request_raw(
        self,
        *,
        subject: "Subject",
        payload: bytes,
        timeout: timedelta,
    ) -> bytes:
        """request raw bytes round-trip.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: request payload bytes
        :ptype payload: bytes
        :param timeout: max wait for reply
        :ptype timeout: timedelta
        :return: response payload bytes
        :rtype: bytes
        """
        ...
