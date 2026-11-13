"""what a caller of :class:`~threetears.nats.client.NatsClient` uses of it, one slice each, as a ``Protocol``.

A consumer that takes the whole client in a parameter makes every test of it hand a whole client: a test double for the
one method it calls does not type-check as what it is. Each protocol here is one capability of the client, its
methods' signatures copied from :class:`NatsClient` (which satisfies every one of them structurally, held by
``tests/unit/test_capabilities.py``). A consumer types its parameter by the protocols it uses -- one, or a protocol of
its own that inherits several -- and a double implementing those methods is a valid argument.

- :class:`RawRequester` -- a request whose reply comes back as bytes (``request_raw``)
- :class:`Subscriber` -- a core subscription and its end (``subscribe``, ``unsubscribe``)
- :class:`TypedSubscriber` -- a subscription delivering a model, and its end (``subscribe_typed``, ``unsubscribe``)
- :class:`SubscriptionHandle` -- what ``subscribe_typed`` hands back, given back to ``unsubscribe``
- :class:`Publisher` -- a core publish of a model (``publish``)
- :class:`RawReplier` -- a reply of raw bytes to a request's reply subject (``publish_raw_reply``)
- :class:`JetStreamPublisher` -- a JetStream publish, acknowledged (``jetstream_publish``)
- :class:`DurableSubscriber` -- a durable JetStream push consumer (``jetstream_subscribe_durable``)
- :class:`PullSubscriber` -- a durable JetStream pull consumer (``jetstream_pull_subscribe``)
- :class:`KvBucketOpener` -- a KV bucket by name (``kv_bucket``), answered as the bucket slice
  :class:`~threetears.nats.kv.KvBucketLike`
- :class:`KvPublisher` -- both of :class:`KvBucketOpener` and :class:`Publisher`: a collection's client
- :class:`ChangeLogBinder` -- a change log by name (``change_log``), answered as the log slice
  :class:`~threetears.nats.change_log.ChangeLogLike`: what an epoch source or reader holds its client for

Importing this module loads no NATS client: the types it names are read only by a type checker.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from threetears.nats.change_log import ChangeLogLike
    from threetears.nats.client import JetStreamPullConsumer, JetStreamPushConsumer, Subscription
    from threetears.nats.kv import KvBucketLike
    from threetears.nats.subjects import Subject
    from threetears.nats.transport import RawMessageCallback

__all__ = [
    "ChangeLogBinder",
    "DurableSubscriber",
    "JetStreamPublisher",
    "KvBucketOpener",
    "KvPublisher",
    "Publisher",
    "PullSubscriber",
    "RawReplier",
    "RawRequester",
    "Subscriber",
    "SubscriptionHandle",
    "TypedSubscriber",
]

_T = TypeVar("_T", bound="BaseModel")


@runtime_checkable
class RawRequester(Protocol):
    """sends a request and answers its reply's bytes (:meth:`NatsClient.request_raw`)."""

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta = ...) -> bytes: ...


@runtime_checkable
class Subscriber(Protocol):
    """subscribes a callback to a subject, and ends a subscription (:meth:`NatsClient.subscribe`,
    :meth:`NatsClient.unsubscribe`)."""

    async def subscribe(
        self,
        subject: Subject | str | None = None,
        *,
        cb: RawMessageCallback | None = None,
        queue: str | None = None,
        max_in_flight: int | None = None,
        deadletter_on_failure: bool = True,
    ) -> Subscription: ...

    async def unsubscribe(self, sub: Subscription) -> None: ...


@runtime_checkable
class SubscriptionHandle(Protocol):
    """what a typed subscription hands back: the subject it holds and whether it has ended. A caller keeps it only to
    give it back to ``unsubscribe``; :class:`~threetears.nats.client.Subscription` is one, and so is a test double's
    handle."""

    @property
    def subject(self) -> Subject: ...

    @property
    def is_closed(self) -> bool: ...


@runtime_checkable
class TypedSubscriber(Protocol):
    """subscribes a callback taking a model to a subject, and ends a subscription
    (:meth:`NatsClient.subscribe_typed`, :meth:`NatsClient.unsubscribe`). The handle is the slice
    :class:`SubscriptionHandle`, so a double's own handle type-checks."""

    async def subscribe_typed(
        self,
        *,
        subject: Subject,
        cb: Callable[[_T], Awaitable[None]],
        message_type: type[_T],
        queue: str | None = None,
        max_in_flight: int | None = None,
        deadletter_on_failure: bool = True,
    ) -> SubscriptionHandle: ...

    async def unsubscribe(self, sub: SubscriptionHandle) -> None: ...


@runtime_checkable
class Publisher(Protocol):
    """publishes a model to a subject (:meth:`NatsClient.publish`)."""

    async def publish(
        self,
        *args: Any,
        subject: Subject | str | None = None,
        message: BaseModel | None = None,
        reply_to: Subject | str | None = None,
    ) -> None: ...


@runtime_checkable
class RawReplier(Protocol):
    """replies raw bytes to a request's reply subject (:meth:`NatsClient.publish_raw_reply`)."""

    async def publish_raw_reply(self, *, reply_subject: str, payload: bytes) -> None: ...


@runtime_checkable
class JetStreamPublisher(Protocol):
    """publishes to a JetStream subject and waits for its acknowledgement (:meth:`NatsClient.jetstream_publish`)."""

    async def jetstream_publish(
        self,
        *,
        subject: Subject,
        payload: bytes,
        timeout: timedelta | float = ...,
        headers: Mapping[str, str] | None = None,
    ) -> None: ...


@runtime_checkable
class DurableSubscriber(Protocol):
    """binds a callback to a durable JetStream push consumer (:meth:`NatsClient.jetstream_subscribe_durable`)."""

    async def jetstream_subscribe_durable(
        self,
        *,
        subject: Subject,
        durable: str,
        cb: Callable[[Any], Awaitable[None]],
        max_deliver: int,
        dead_letter_subject: Subject | None = None,
        ack_wait_seconds: float = 60.0,
        stream: str | None = None,
    ) -> JetStreamPushConsumer: ...


@runtime_checkable
class PullSubscriber(Protocol):
    """binds a callback to a durable JetStream pull consumer (:meth:`NatsClient.jetstream_pull_subscribe`)."""

    async def jetstream_pull_subscribe(
        self,
        *,
        subject: Subject,
        durable: str,
        cb: Callable[[Any], Awaitable[None]],
        max_deliver: int,
        dead_letter_subject: Subject | None = None,
        ack_wait_seconds: float = 60.0,
        stream: str | None = None,
        batch: int = 8,
        fetch_timeout_seconds: float = 5.0,
        error_backoff_seconds: float = ...,
    ) -> JetStreamPullConsumer: ...


@runtime_checkable
class KvBucketOpener(Protocol):
    """opens (and, asked to, makes) a KV bucket by name (:meth:`NatsClient.kv_bucket`)."""

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
        direct: bool | None = None,
        replicas: int | None = None,
    ) -> KvBucketLike: ...


@runtime_checkable
class KvPublisher(KvBucketOpener, Publisher, Protocol):
    """opens KV buckets and publishes models: what a collection uses of its client (its L2 bucket, its invalidation
    messages)."""


@runtime_checkable
class ChangeLogBinder(Protocol):
    """binds a change log by name (:meth:`NatsClient.change_log`): what an epoch source or reader
    (:class:`threetears.epoch.EpochGenerationSource`, :class:`threetears.epoch.EpochGenerationReader`) holds its
    client for."""

    def change_log(self, name: str) -> ChangeLogLike: ...
