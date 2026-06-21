"""narrow Protocols for NATS transport surfaces.

these Protocols let test fakes substitute for the live
:class:`threetears.nats.NatsClient` without subclassing it. every fake
in 3tears integration tests should satisfy one of these Protocols
rather than emulate the full client surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from threetears.nats.subjects import Subject

__all__ = [
    "IncomingMessage",
    "MessageCallback",
    "RawMessageCallback",
    "StreamTransport",
]


_M = TypeVar("_M", bound="BaseModel")


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """opaque envelope handed to a :class:`NatsClient.subscribe` callback.

    exposes the three surfaces a request/reply or wildcard subscribe
    server needs: payload bytes, the per-request reply subject (the
    opaque inbox nats-py populates on inbound requests), and the
    matched subject string. ``reply_subject`` is ``None`` when the
    producer published without a reply-to (pub/sub fire-and-forget
    pattern), so handlers that mix request/reply with pub/sub can
    branch on its presence rather than guessing. ``subject`` carries
    the concrete subject the message arrived on, which is what
    distinguishes one delivery from another for wildcard subscribers
    (``{ns}.audit.>``, ``{ns}.agents.route.>``, etc.) — they read
    ``subject`` to extract the wildcard segment(s).

    :param data: raw payload bytes
    :ptype data: bytes
    :param reply_subject: opaque reply subject for request/reply patterns; ``None`` for pub/sub
    :ptype reply_subject: str | None
    :param subject: concrete subject the message arrived on (matters for wildcard subscribers)
    :ptype subject: str
    """

    data: bytes
    reply_subject: str | None
    subject: str

    @property
    def reply(self) -> str | None:
        """alias for :attr:`reply_subject` matching nats-py ``Msg.reply``.

        every integration test (and several legacy handlers) reads
        ``msg.reply`` after a raw subscribe — that name comes from
        nats-py's own ``Msg`` shape. exposing it as a ``@property``
        on the canonical envelope keeps the wrapper a drop-in for
        callers that came from raw nats-py without forcing a
        rename across every test fixture.

        :return: reply subject (``None`` for pub/sub messages)
        :rtype: str | None
        """
        return self.reply_subject


MessageCallback = Callable[["BaseModel"], Awaitable[None]]
"""typed subscribe callback shape for :meth:`NatsClient.subscribe_typed`.

receives the validated Pydantic message; raising propagates to the
deadletter dispatcher when the subscription has
``deadletter_on_failure=True``.
"""


RawMessageCallback = Callable[[IncomingMessage], Awaitable[None]]
"""raw subscribe callback shape for :meth:`NatsClient.subscribe`.

receives the inbound :class:`IncomingMessage` envelope so request/reply
servers can read both the payload bytes (``msg.data``) and the
per-request reply subject (``msg.reply_subject``). pure pub/sub
subscribers ignore ``reply_subject``; handlers that respond via
:meth:`NatsClient.publish_reply` consume it.
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
