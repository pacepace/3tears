"""the shipped NATS double's pub/sub surface, which consumers write their listener tests against.

A double that cannot express one listener stopping while another keeps running makes those tests
pass or fail for reasons unrelated to the code under test -- and the platform depends on exactly
that shape: two L2-live registries in one process subscribe and stop independently, while two
listeners on ONE registry is the thing that is forbidden.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from threetears.core.testing.kv import FakeNatsClient


class _Message(BaseModel):
    """a typed envelope, as every subject on this bus carries."""

    value: str


class _Subject:
    """a subject object, which the real client takes rather than a bare string."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __str__(self) -> str:
        return self.path


@pytest.mark.asyncio
async def test_a_published_message_reaches_every_subscriber_on_its_subject() -> None:
    client = FakeNatsClient()
    seen: list[str] = []
    await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(seen, m), message_type=_Message)
    await client.publish(subject=_Subject("a"), message=_Message(value="one"))
    assert seen == ["one"]
    assert [m.value for m in client.published] == ["one"]


@pytest.mark.asyncio
async def test_a_message_does_not_reach_another_subject() -> None:
    client = FakeNatsClient()
    seen: list[str] = []
    await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(seen, m), message_type=_Message)
    await client.publish(subject=_Subject("b"), message=_Message(value="one"))
    assert seen == []


@pytest.mark.asyncio
async def test_unsubscribing_one_listener_leaves_the_other_running() -> None:
    # the property the fake exists to express. Clearing every subscriber made this test pass
    # whatever the code under test did.
    client = FakeNatsClient()
    first: list[str] = []
    second: list[str] = []
    handle = await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(first, m), message_type=_Message)
    await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(second, m), message_type=_Message)

    await client.unsubscribe(handle)
    await client.publish(subject=_Subject("a"), message=_Message(value="after"))

    assert first == [], "the unsubscribed listener still received a message"
    assert second == ["after"], "unsubscribing one listener stopped the other"


@pytest.mark.asyncio
async def test_unsubscribing_twice_leaves_the_rest_alone() -> None:
    client = FakeNatsClient()
    seen: list[str] = []
    handle = await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(seen, m), message_type=_Message)
    other = await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(seen, m), message_type=_Message)
    await client.unsubscribe(handle)
    await client.unsubscribe(handle)  # a stale handle
    await client.publish(subject=_Subject("a"), message=_Message(value="after"))
    assert seen == ["after"], "a repeated unsubscribe took another listener with it"
    await client.unsubscribe(other)
    await client.publish(subject=_Subject("a"), message=_Message(value="later"))
    assert seen == ["after"]


@pytest.mark.asyncio
async def test_a_foreign_handle_unsubscribes_nothing() -> None:
    # the silent early return, asserted: a handle from somewhere else must not look successful
    # while quietly leaving -- or removing -- a listener.
    client = FakeNatsClient()
    seen: list[str] = []
    await client.subscribe_typed(subject=_Subject("a"), cb=lambda m: _record(seen, m), message_type=_Message)
    await client.unsubscribe(object())
    await client.publish(subject=_Subject("a"), message=_Message(value="still here"))
    assert seen == ["still here"], "a foreign handle removed a live listener"


async def _record(sink: list[str], message: _Message) -> None:
    """collect a delivered message's value.

    :param sink: where to record it
    :ptype sink: list[str]
    :param message: the delivered envelope
    :ptype message: _Message
    :return: nothing
    :rtype: None
    """
    sink.append(message.value)


@pytest.mark.asyncio
async def test_bucket_age_makes_an_established_bucket() -> None:
    """The affordance every verifier test needs, and the reason it exists.

    `ReplayGuard` refuses an artifact issued before its bucket was created plus the verifier's
    tolerance. A fake bucket created at the instant a test mints its artifact is inside that
    window by construction, so the obvious test of replay semantics fails with a replay refusal
    that has nothing to do with replay -- the same trap the first real call after a broker
    restart hits.
    """
    aged = FakeNatsClient(bucket_age=timedelta(hours=1))
    bucket = await aged.kv_bucket(name="nonces")
    age = datetime.now(UTC) - await bucket.date_created()
    assert age >= timedelta(minutes=59), "bucket_age did not push the creation time back"


@pytest.mark.asyncio
async def test_without_bucket_age_a_bucket_is_brand_new() -> None:
    # the default must stay the real client's behaviour, or a watermark test written against
    # the fake would silently stop exercising the refusal it exists for.
    client = FakeNatsClient()
    bucket = await client.kv_bucket(name="nonces")
    age = datetime.now(UTC) - await bucket.date_created()
    assert age < timedelta(seconds=5), "a fresh bucket should report roughly now"


def test_a_negative_bucket_age_is_refused() -> None:
    # a bucket created in the future would make the watermark refuse everything, which reads as
    # a guard defect rather than as a mis-built double.
    with pytest.raises(ValueError, match="bucket_age"):
        FakeNatsClient(bucket_age=timedelta(seconds=-1))
