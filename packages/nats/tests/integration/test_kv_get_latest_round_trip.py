"""Integration test: ``get_latest`` reports a deletion's revision, and a write fenced on it holds.

A read that seeds L2 from L3 writes at the key's latest revision as it read it before its query,
so the seed lands only if nothing happened to the key since. That rests on two broker facts a
fake can only assert: a delete leaves a marker whose revision is the key's latest, and an update
expecting revision ``0`` lands only on a key with no message at all -- where a create lands over a
deletion marker too, which is the whole reason the seed cannot use one.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_a_deletion_is_reported_by_revision_and_fences_a_later_write(nats_container: str) -> None:
    namespace = f"latest{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="latest"
    ) as nc:
        bucket = await nc.kv_bucket(name="latest")
        assert await bucket.get_latest(key="k") == (None, 0)

        written = await bucket.put(key="k", value=b"v")
        assert await bucket.get_latest(key="k") == (b"v", written)

        assert await bucket.delete(key="k")
        assert await bucket.get_entry(key="k") is None
        value, marker = await bucket.get_latest(key="k")
        assert value is None
        assert marker > written, "a delete left no marker revision"

        # the seed's view was taken before the delete: it must lose.
        assert await bucket.update(key="k", value=b"stale", revision=written) is None
        # revision 0 expects no message at all, and the marker is one.
        assert await bucket.update(key="k", value=b"stale", revision=0) is None
        # a create lands over the marker -- which is why a seed cannot be a create.
        assert await bucket.create(key="k", value=b"created") is not None


async def test_an_update_at_revision_zero_lands_only_on_a_key_never_written(nats_container: str) -> None:
    namespace = f"latest{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="latest-zero"
    ) as nc:
        bucket = await nc.kv_bucket(name="latest")
        assert await bucket.update(key="fresh", value=b"v", revision=0) is not None
        assert await bucket.update(key="fresh", value=b"v2", revision=0) is None


async def test_many_keys_read_in_batched_direct_gets_answer_as_one_at_a_time(nats_container: str) -> None:
    """Flips if a batched read answers any key otherwise than get_latest does (a live value, a deletion's
    revision, a key never written), or reads a thousand keys a request each."""
    namespace = f"latest{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="latest-many"
    ) as nc:
        bucket = await nc.kv_bucket(name="latest", direct=True)
        keys = [f"k{n:04d}" for n in range(1_200)]
        for key in keys[:1_000]:
            await bucket.put(key=key, value=key.encode())
        for key in keys[:100]:
            await bucket.delete(key=key)
        one_at_a_time = {key: await bucket.get_latest(key=key) for key in keys}
        requests = 0
        # rationale: the bucket issues its requests on the client's own nats-py connection; counting them there
        connection = nc.raw
        real_publish = connection.publish

        async def counting(subject: str, *args: object, **kwargs: object) -> object:
            nonlocal requests
            requests += subject.startswith("$JS.API.DIRECT.GET.")
            return await real_publish(subject, *args, **kwargs)

        connection.publish = counting  # type: ignore[method-assign]
        try:
            batched = await bucket.get_latest_many(keys=keys, batched=True)
        finally:
            connection.publish = real_publish  # type: ignore[method-assign]

        assert batched == one_at_a_time
        assert requests == 3


async def test_a_batched_read_nobody_answers_is_probed_briefly_and_not_tried_again(nats_container: str) -> None:
    """Flips if a batched read whose request the grant refuses (never answered) waits out its deadline, or is
    tried again, rather than reading the keys one at a time."""
    namespace = f"latest{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="latest-refused"
    ) as nc:
        bucket = await nc.kv_bucket(name="refused", direct=True)
        keys = [f"k{n}" for n in range(20)]
        for key in keys:
            await bucket.put(key=key, value=key.encode())
        # rationale: the bucket issues its requests on the client's own nats-py connection; a refused one is dropped
        connection = nc.raw
        real_publish = connection.publish
        batches = 0

        async def refusing(subject: str, *args: object, **kwargs: object) -> object:
            nonlocal batches
            if subject == f"$JS.API.DIRECT.GET.KV_{bucket.name}":
                batches += 1
                return None
            return await real_publish(subject, *args, **kwargs)

        connection.publish = refusing  # type: ignore[method-assign]
        try:
            began = asyncio.get_running_loop().time()
            first = await bucket.get_latest_many(keys=keys, batched=True)
            took = asyncio.get_running_loop().time() - began
            again = await bucket.get_latest_many(keys=keys, batched=True)
        finally:
            connection.publish = real_publish  # type: ignore[method-assign]

        assert first == again == {key: (key.encode(), first[key][1]) for key in keys}
        assert took < 3, f"the unanswered batch waited {took:.1f} s"
        assert batches == 1, "a refused bucket was asked again"
