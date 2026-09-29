"""Integration test: ``get_latest`` reports a deletion's revision, and a write fenced on it holds.

A read that seeds L2 from L3 writes at the key's latest revision as it read it before its query,
so the seed lands only if nothing happened to the key since. That rests on two broker facts a
fake can only assert: a delete leaves a marker whose revision is the key's latest, and an update
expecting revision ``0`` lands only on a key with no message at all -- where a create lands over a
deletion marker too, which is the whole reason the seed cannot use one.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

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
