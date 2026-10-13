"""one read of a stream's messages through a freshly NAMED push consumer: the one loop KV and the Object Store share.

A pod's grant admits a consumer create only in the named form, whose filter rides in the create
subject where the server checks it (``$JS.API.CONSUMER.CREATE.{stream}.{name}.{filter}``); nats-py's
own listings and Object Store reads create an unnamed one and are refused. Both
:meth:`threetears.nats.kv.NatsKvBucket.list_keys` and :class:`threetears.nats.object_store.NatsObjectStore`
read through :func:`read_through_named_consumer`.

The inbox is subscribed first, so nothing the consumer delivers arrives before anything listens. The
consumer acknowledges nothing and is never deleted by the reader -- a pod's grant carries no
``CONSUMER.DELETE`` -- so the server reaps it after ``inactive_threshold``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy
from threetears.observe import get_logger

from threetears.nats._publish import run_bounded

if TYPE_CHECKING:
    from nats.aio.msg import Msg

    from threetears.nats.client import NatsClient

__all__ = ["NamedRead", "read_through_named_consumer"]

log = get_logger(__name__)

#: how long the server keeps a finished read's consumer once nothing listens to its deliver subject
DEFAULT_READ_INACTIVE_THRESHOLD_SECONDS: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class NamedRead:
    """what one read asks of its consumer.

    :ivar stream: the stream to read
    :ivar filter_subject: the consumer's filter, which the create subject carries
    :ivar deliver_policy: where the consumer starts
    :ivar name_prefix: the consumer name's prefix, so an operator listing consumers can tell reads apart
    :ivar count: how many messages to read; ``None`` reads every message pending when it was created
    :ivar headers_only: deliver headers without bodies
    :ivar timeout_seconds: the ceiling on the create and on each message's wait
    """

    stream: str
    filter_subject: str
    deliver_policy: DeliverPolicy
    name_prefix: str
    count: int | None = None
    headers_only: bool = False
    timeout_seconds: float = 10.0


async def read_through_named_consumer(
    client: NatsClient, read: NamedRead, *, failure: Callable[[str, BaseException | None], Exception]
) -> list[Msg]:
    """create one named push consumer and read its messages.

    :param client: the connected wrapper client
    :ptype client: NatsClient
    :param read: what to read
    :ptype read: NamedRead
    :param failure: builds the caller's error from a message and its cause, so each caller raises its
        own type
    :ptype failure: Callable[[str, BaseException | None], Exception]
    :return: the messages, in stream order
    :rtype: list[nats.aio.msg.Msg]
    :raises Exception: what ``failure`` builds, when the connection is closed, the create fails or is
        never answered, or a message does not arrive in time
    """
    raw = client.raw
    if raw.is_closed:
        raise failure(f"cannot read {read.stream}: the NATS connection is closed", None)
    inbox = raw.new_inbox()
    subscription = await raw.subscribe(inbox)
    messages: list[Any] = []
    try:
        config = ConsumerConfig(
            name=f"{read.name_prefix}{uuid.uuid7().hex}",
            deliver_subject=inbox,
            filter_subject=read.filter_subject,
            deliver_policy=read.deliver_policy,
            ack_policy=AckPolicy.NONE,
            headers_only=read.headers_only or None,
            inactive_threshold=DEFAULT_READ_INACTIVE_THRESHOLD_SECONDS,
            mem_storage=True,
        )
        js = client.jetstream_context()
        try:
            info = await run_bounded(
                lambda: js.add_consumer(read.stream, config=config),
                timeout=read.timeout_seconds,
                what=f"read consumer create on {read.stream}",
            )
        except Exception as exc:
            raise failure(
                f"a read consumer on {read.stream} could not be created: {exc}. an ungranted create is never "
                f"answered -- check this principal's grant on $JS.API.CONSUMER.CREATE.{read.stream}.*.{read.filter_subject}",
                exc,
            ) from exc
        wanted = int(info.num_pending or 0) if read.count is None else read.count
        while len(messages) < wanted:
            try:
                messages.append(await subscription.next_msg(timeout=read.timeout_seconds))
            except TimeoutError as exc:
                raise failure(
                    f"reading {read.filter_subject} from {read.stream} stalled after {len(messages)} of {wanted} messages",
                    exc,
                ) from exc
    finally:
        try:
            await subscription.unsubscribe()
        # NOSILENT: logged; the consumer behind it is reaped by the server once nothing listens
        except Exception as exc:  # noqa: BLE001 -- teardown continues regardless
            log.debug("named read unsubscribe on %s failed: %s", read.filter_subject, exc)
    return messages
