"""The shipped NATS double is each capability protocol a consumer types its client by (hub spec 12a).

A consumer typed by ``TypedSubscriber``, ``Publisher``, ``KvBucketOpener`` or ``KvPublisher`` takes a
:class:`FakeNatsClient` in its tests only when the double's signatures are the protocol's: mypy holds that
through the assignments below (``isinstance`` on a runtime-checkable protocol checks names, not signatures).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from threetears.core.testing.kv import FakeChangeLog, FakeKvBucket, FakeNatsClient
from threetears.nats.errors import PublishError
from threetears.nats import (
    ChangeLogBinder,
    KvBucketOpener,
    KvPublisher,
    Publisher,
    SubscriptionHandle,
    TypedSubscriber,
)
from threetears.nats.change_log import ChangeLogLike
from threetears.nats.kv import KvBucketLike, KvDeclaringClient
from threetears.nats.subjects import Subject


# held by mypy (typecheck.sh checks tests): the double is each capability, signatures and all
def _typed_subscriber(client: FakeNatsClient) -> TypedSubscriber:
    return client


def _publisher(client: FakeNatsClient) -> Publisher:
    return client


def _kv_bucket_opener(client: FakeNatsClient) -> KvBucketOpener:
    return client


def _kv_publisher(client: FakeNatsClient) -> KvPublisher:
    return client


def _kv_declaring_client(client: FakeNatsClient) -> KvDeclaringClient:
    return client


def _bucket(bucket: FakeKvBucket) -> KvBucketLike:
    return bucket


def _change_log_binder(client: FakeNatsClient) -> ChangeLogBinder:
    return client


def _change_log(log: FakeChangeLog) -> ChangeLogLike:
    return log


class _Message(BaseModel):
    """a typed envelope."""

    value: str


async def _keep(seen: list[str], message: _Message) -> None:
    seen.append(message.value)


@pytest.mark.asyncio
async def test_a_subscriber_typed_by_the_protocol_hears_a_publisher_typed_by_the_protocol() -> None:
    client = FakeNatsClient()
    subscriber: TypedSubscriber = client
    publisher: Publisher = client
    seen: list[str] = []
    subject = Subject.raw("a.b")
    handle: SubscriptionHandle = await subscriber.subscribe_typed(
        subject=subject, cb=lambda m: _keep(seen, m), message_type=_Message
    )
    assert handle.subject == subject
    assert not handle.is_closed
    await publisher.publish(subject=subject, message=_Message(value="one"))
    await subscriber.unsubscribe(handle)
    assert handle.is_closed
    await publisher.publish(subject=subject, message=_Message(value="two"))
    assert seen == ["one"]


@pytest.mark.asyncio
async def test_the_positional_raw_publish_reaches_a_typed_subscriber_as_the_real_client_s_does() -> None:
    client = FakeNatsClient()
    seen: list[str] = []
    await client.subscribe_typed(subject=Subject.raw("a.b"), cb=lambda m: _keep(seen, m), message_type=_Message)
    await client.publish("a.b", _Message(value="raw").model_dump_json().encode())
    assert seen == ["raw"]


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [("a.b",), ("a.b", b"{}", b"{}"), ("a.b", "not bytes")])
async def test_a_malformed_positional_publish_is_refused_as_the_real_client_refuses_it(
    args: tuple[object, ...],
) -> None:
    with pytest.raises(PublishError):
        await FakeNatsClient().publish(*args)


@pytest.mark.asyncio
async def test_a_bucket_opened_through_the_protocol_is_a_bucket_and_takes_replicas() -> None:
    opener: KvBucketOpener = FakeNatsClient()
    bucket = await opener.kv_bucket(name="b", replicas=3)
    assert await bucket.put(key="k", value=b"v") == 1
    assert await bucket.get(key="k") == b"v"


def test_the_double_binds_a_change_log_an_epoch_source_and_reader_take() -> None:
    client = FakeNatsClient()
    assert isinstance(client, ChangeLogBinder)
    assert isinstance(client.change_log("epoch-changes"), ChangeLogLike)
