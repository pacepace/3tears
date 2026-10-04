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
from threetears.nats.errors import KvBucketNotFoundError, KvError
from threetears.nats.kv import KvDeclaring


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


@pytest.mark.asyncio
async def test_a_vanished_bucket_is_recreated_by_its_next_operation() -> None:
    # the real wrapper's self-heal: a bucket a broker restart lost is recreated by whatever
    # operation next reaches it, empty, with that operation's moment as its creation time.
    client = FakeNatsClient(bucket_age=timedelta(hours=1))
    bucket = await client.kv_bucket(name="nonces")
    await bucket.put(key="k", value=b"v")
    bucket.vanish()
    assert bucket.keys() == ()

    assert await bucket.get(key="k") is None
    age = datetime.now(UTC) - await bucket.date_created()
    assert age < timedelta(seconds=5), "the next operation should have recreated the bucket now"


@pytest.mark.asyncio
async def test_a_wiped_bucket_restarts_its_revisions_at_one() -> None:
    # a recreated stream starts its sequence again, and the revision IS the sequence. A double that
    # kept counting would hide every consumer that orders writes by revision alone and so refuses
    # every write after a broker restart.
    client = FakeNatsClient()
    bucket = await client.kv_bucket(name="collections")
    for n in range(3):
        await bucket.put(key=f"k{n}", value=b"v")
    bucket.wipe()
    assert await bucket.create(key="k0", value=b"v") == 1


@pytest.mark.asyncio
async def test_a_delete_leaves_a_marker_whose_revision_fences_a_later_write() -> None:
    # a real delete publishes a marker with its own revision. A write expecting the revision the
    # writer saw before the delete must lose; one expecting the marker's must land; and a create
    # lands over a marker, which is exactly why a create cannot fence a seed.
    client = FakeNatsClient()
    bucket = await client.kv_bucket(name="collections")
    assert await bucket.get_latest(key="k") == (None, 0)
    written = await bucket.put(key="k", value=b"v")
    assert await bucket.get_latest(key="k") == (b"v", written)
    await bucket.delete(key="k")
    assert await bucket.get_entry(key="k") is None
    _, marker = await bucket.get_latest(key="k")
    assert marker > written
    assert await bucket.update(key="k", value=b"stale", revision=written) is None
    assert await bucket.update(key="k", value=b"seed", revision=0) is None, "revision 0 expects no message at all"
    assert await bucket.update(key="k", value=b"seed", revision=marker) is not None


@pytest.mark.asyncio
async def test_an_update_at_revision_zero_lands_only_on_a_key_with_no_message() -> None:
    client = FakeNatsClient()
    bucket = await client.kv_bucket(name="collections")
    assert await bucket.update(key="fresh", value=b"v", revision=0) is not None
    assert await bucket.update(key="fresh", value=b"v2", revision=0) is None


@pytest.mark.asyncio
async def test_a_vanished_bucket_restarts_its_revisions_at_one() -> None:
    client = FakeNatsClient()
    bucket = await client.kv_bucket(name="collections")
    for n in range(3):
        await bucket.put(key=f"k{n}", value=b"v")
    bucket.vanish()
    assert await bucket.put(key="k0", value=b"v") == 1


@pytest.mark.asyncio
async def test_reconnect_runs_every_hook_in_order_past_a_failing_one() -> None:
    client = FakeNatsClient()
    ran: list[str] = []

    async def _first() -> None:
        ran.append("first")

    async def _failing() -> None:
        ran.append("failing")
        raise RuntimeError("hook failed")

    async def _last() -> None:
        ran.append("last")

    for hook in (_first, _failing, _last):
        client.add_reconnect_callback(hook)
    await client.reconnect()
    assert ran == ["first", "failing", "last"]


@pytest.mark.asyncio
async def test_the_double_declares_as_well_as_opens() -> None:
    # a consumer that declares its bucket takes a KvDeclaring, and the shipped double has to
    # satisfy that by construction, or every such consumer's test needs a double of its own.
    client = FakeNatsClient()
    assert isinstance(client, KvDeclaring)
    bucket = await client.ensure_kv_bucket(name="versions", ttl=timedelta(minutes=5), direct=True)
    assert (bucket.storage, bucket.direct, bucket.ttl) == ("memory", True, timedelta(minutes=5))
    assert await client.kv_bucket(name="versions") is bucket, "a declaration and a later open share one handle"


@pytest.mark.asyncio
async def test_a_declaration_reconciles_a_bucket_somebody_opened_first() -> None:
    # the real declaration updates the live stream in place; entries stay.
    client = FakeNatsClient()
    opened = await client.kv_bucket(name="versions")
    await opened.put(key="k", value=b"v")
    declared = await client.ensure_kv_bucket(name="versions", direct=True)
    assert declared is opened
    assert declared.direct is True
    assert await declared.get(key="k") == b"v"


@pytest.mark.asyncio
async def test_a_declaration_keeps_a_live_buckets_expiry_unless_it_owns_it() -> None:
    # the real declaration reconciles max_age only for a declarer that owns the bucket's expiry.
    client = FakeNatsClient()
    await client.kv_bucket(name="ratelimits", ttl=timedelta(seconds=300))

    kept = await client.ensure_kv_bucket(name="ratelimits", ttl=None)
    assert kept.ttl == timedelta(seconds=300)

    owned = await client.ensure_kv_bucket(name="ratelimits", ttl=None, owns_expiry=True)
    assert owned is kept
    assert owned.ttl is None


@pytest.mark.asyncio
async def test_a_bind_only_declaration_cannot_own_expiry() -> None:
    client = FakeNatsClient(declared_buckets=["ratelimits"])
    with pytest.raises(ValueError, match="owns_expiry"):
        await client.ensure_kv_bucket(name="ratelimits", create_if_missing=False, owns_expiry=True)


@pytest.mark.asyncio
async def test_a_bind_only_declaration_of_an_absent_bucket_raises() -> None:
    client = FakeNatsClient()
    with pytest.raises(KvBucketNotFoundError, match="versions") as caught:
        await client.ensure_kv_bucket(name="versions", create_if_missing=False)
    assert caught.value.bucket == "versions"


@pytest.mark.asyncio
async def test_a_bind_only_open_of_an_absent_bucket_raises_what_the_real_client_raises() -> None:
    # the real client raises KvBucketNotFoundError, which is a KvError: code that degrades on
    # KvError must see the same thing over the double, not a KeyError that escapes its catch.
    client = FakeNatsClient()
    with pytest.raises(KvBucketNotFoundError, match="versions") as caught:
        await client.kv_bucket(name="versions", create_if_missing=False)
    assert isinstance(caught.value, KvError)
    assert caught.value.bucket == "versions"


@pytest.mark.asyncio
async def test_a_bind_only_declaration_of_a_vanished_bucket_raises() -> None:
    # a declaration never reads the client's cache; it asks the broker, which no longer has it.
    client = FakeNatsClient(declared_buckets=["hub-owned"])
    bucket = await client.kv_bucket(name="hub-owned", create_if_missing=False)
    bucket.vanish()
    with pytest.raises(KvBucketNotFoundError, match="hub-owned"):
        await client.ensure_kv_bucket(name="hub-owned", create_if_missing=False)


@pytest.mark.asyncio
async def test_a_bind_only_handle_on_a_vanished_bucket_raises_rather_than_recreating_it() -> None:
    # the real bind-only handle re-binds, finds nothing, and raises once its wait for the declarer
    # is spent: only the declarer may create the bucket. The double heals only a handle that may.
    client = FakeNatsClient(declared_buckets=["hub-owned"])
    bucket = await client.kv_bucket(name="hub-owned", create_if_missing=False)
    await bucket.put(key="k", value=b"v")
    bucket.vanish()

    with pytest.raises(KvBucketNotFoundError, match="hub-owned"):
        await bucket.get(key="k")
    assert not client.bucket_exists("hub-owned"), "a bind-only handle must not have created the bucket"


@pytest.mark.asyncio
async def test_a_declaration_brings_a_vanished_bucket_back_for_its_binders() -> None:
    client = FakeNatsClient(declared_buckets=["hub-owned"])
    bound = await client.kv_bucket(name="hub-owned", create_if_missing=False)
    await bound.put(key="k", value=b"old")
    await bound.put(key="k", value=b"older")
    bound.vanish()
    await client.ensure_kv_bucket(name="hub-owned")  # the declarer is back

    assert client.bucket_exists("hub-owned")
    assert await bound.put(key="k", value=b"v") == 1, "the redeclared bucket starts its revisions again"


@pytest.mark.asyncio
async def test_only_a_declaration_that_may_create_is_remembered_whatever_its_storage() -> None:
    # the real client remembers exactly these and creates them again after every reconnect: a
    # restart can lose file storage as well as memory (on Kubernetes the volume goes with the pod).
    # an ordinary open and a bind-only declaration are never remembered.
    client = FakeNatsClient(declared_buckets=["hub-owned"])
    await client.ensure_kv_bucket(name="declared")
    await client.ensure_kv_bucket(name="durable", storage="file")
    await client.ensure_kv_bucket(name="hub-owned", create_if_missing=False)
    await client.kv_bucket(name="opened")
    assert client.remembered_declarations == frozenset({"declared", "durable"})


@pytest.mark.asyncio
async def test_a_file_declaration_comes_back_empty_after_a_restart_that_lost_its_storage() -> None:
    client = FakeNatsClient()
    durable = await client.ensure_kv_bucket(name="durable", storage="file")
    await durable.put(key="k", value=b"v")
    seen_by_hook: list[bool] = []

    async def _hook() -> None:
        seen_by_hook.append(client.bucket_exists("durable"))

    client.add_reconnect_callback(_hook)
    await client.restart_broker()

    assert seen_by_hook == [True]
    assert durable.keys() == ()
    assert durable.storage == "file"


@pytest.mark.asyncio
async def test_a_broker_restart_puts_back_only_what_this_client_declared() -> None:
    # the real restart: every memory bucket loses its entries; the client re-creates each one it
    # declared before any reconnect hook runs; anything else is gone until an operation heals it.
    client = FakeNatsClient()
    declared = await client.ensure_kv_bucket(name="declared")
    opened = await client.kv_bucket(name="opened")
    await declared.put(key="k", value=b"v")
    await opened.put(key="k", value=b"v")
    seen_by_hook: list[tuple[bool, bool]] = []

    async def _hook() -> None:
        seen_by_hook.append((client.bucket_exists("declared"), client.bucket_exists("opened")))

    client.add_reconnect_callback(_hook)
    await client.restart_broker()

    assert seen_by_hook == [(True, False)]
    assert declared.keys() == ()
    assert await opened.get(key="k") is None
    assert client.bucket_exists("opened"), "an operation through a handle heals a vanished bucket"
