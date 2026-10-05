"""Integration test: a declaring ReplayGuard recreates its nonce bucket on memory when it is live on file.

Against a real broker, because JetStream cannot change a live stream's storage: the owner must
delete and recreate it, and only the server says which storage the stream ended up on. A bind-only
guard over the same kind of bucket leaves it exactly as it found it.

Each case opens the file bucket on one connection and binds the guard on a second, as a service
restarting onto a bucket an older release left behind would. Uses the session-scoped
``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from nats.js.api import StorageType

from threetears.core.coordination import ReplayGuard
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_NAMESPACE = "replayowns"
_TOLERANCE = timedelta(seconds=5)


async def _file_bucket(nats_url: str, name: str) -> str:
    """leave a nonce bucket live on file storage with one entry, as an older release would.

    :param nats_url: the broker
    :ptype nats_url: str
    :param name: the bucket suffix
    :ptype name: str
    :return: the bucket's full name
    :rtype: str
    """
    async with await NatsClient.connect(nats_url=nats_url, nats_subject_namespace=_NAMESPACE, client_name="old") as nc:
        bucket = await nc.kv_bucket(name=name, ttl=timedelta(seconds=120), storage="file")
        await bucket.put(key="left", value=b"1")
        return bucket.name


async def _storage_and_keys(nc: NatsClient, full_name: str) -> tuple[StorageType | None, int]:
    """the bucket's storage and how many messages its stream holds, as the server reports them.

    :param nc: a connected client
    :ptype nc: NatsClient
    :param full_name: the bucket's full name
    :ptype full_name: str
    :return: (storage, message count)
    :rtype: tuple[StorageType | None, int]
    """
    info = await nc.jetstream_context().stream_info(f"KV_{full_name}")
    return info.config.storage, info.state.messages


async def test_the_declaring_guard_recreates_a_file_bucket_on_memory(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    name = f"own_{uuid4().hex[:8]}"
    full_name = await _file_bucket(nats_container, name)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="restarted"
    ) as nc:
        guard = ReplayGuard(nc, bucket_name=name, ttl_seconds=120, verifier_future_tolerance=_TOLERANCE)
        await guard.bind()
        assert await _storage_and_keys(nc, full_name) == (StorageType.MEMORY, 0)


async def test_a_bind_only_guard_leaves_a_file_bucket_as_it_is(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    name = f"bind_{uuid4().hex[:8]}"
    full_name = await _file_bucket(nats_container, name)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="binder"
    ) as nc:
        guard = ReplayGuard(
            nc, bucket_name=name, ttl_seconds=120, verifier_future_tolerance=_TOLERANCE, create_if_missing=False
        )
        await guard.bind()
        assert await _storage_and_keys(nc, full_name) == (StorageType.FILE, 1)
