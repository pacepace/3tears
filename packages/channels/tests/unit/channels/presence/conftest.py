"""shared in-process NATS stand-in for presence unit tests.

mirrors :mod:`packages.registry.tests.integration.test_heartbeat_collection`'s
``InMemoryNatsBus`` but adds the CAS surface
(:meth:`_InMemoryKvBucket.get_entry` / ``create`` / ``update`` /
revision-guarded ``delete``) that :class:`RoomIndexCollection`'s member
CAS uses. unit tests exercise the L1+L2 contract + the CAS read-modify-
write without a docker container; the real-NATS proof lives in the
integration suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.channels.presence.collection import PresenceCollection
from threetears.channels.presence.l1_cache import create_presence_l1_backend
from threetears.channels.presence.room_state import RoomState
from threetears.channels.presence.sweeper import PresenceSweeper
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket


class _InMemoryKvBucket(FakeKvBucket):
    """the shared collections bucket, as the published fake models it.

    Once a hand-rolled subset of get/put/delete; now the published double, so it carries the
    revision history -- deletion markers and ``get_latest`` included -- that a collection's read
    path fences its L2 seed on. A subset double that cannot answer the real bucket's questions is
    how a KV bug ships green.
    """

    def __init__(self) -> None:
        """open an empty bucket.

        :return: None
        :rtype: None
        """
        super().__init__(bucket_name="collections")


class InMemoryNatsBus:
    """typed-wrapper NATS stand-in: one shared KV bucket + typed pub/sub.

    every collection wired against the same bus shares the bucket + the
    invalidation subject, mirroring two pods reading one NATS cluster.
    """

    def __init__(self) -> None:
        self._bucket = _InMemoryKvBucket()
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}

    async def kv_bucket(self, *, name: str, **_: Any) -> _InMemoryKvBucket:  # noqa: ARG002
        return self._bucket

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:  # noqa: ARG002
        subject_str = str(subject)
        for cb, message_type in self._subscribers.get(subject_str, []):
            payload = message.model_dump_json()
            decoded = message_type.model_validate_json(payload)
            await cb(decoded)

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any, **_: Any) -> None:
        subject_str = str(subject)
        self._subscribers.setdefault(subject_str, []).append((cb, message_type))


def make_pod(nats: Any | None) -> tuple[PresenceCollection, CollectionRegistry]:
    """construct one pod's PresenceCollection + its registry.

    :param nats: shared L2 bus, or ``None`` for L1-only operation
    :ptype nats: Any | None
    :return: (collection, registry) pair
    :rtype: tuple[PresenceCollection, CollectionRegistry]
    """
    l1 = create_presence_l1_backend()
    registry = CollectionRegistry()
    # replicas of ONE channel-adapter principal, so one scope: the cross-pod L2 assertions
    # are only meaningful when both pods land on the same key.
    registry.configure(l1_backend=l1, l2_client=nats, kv_key_scope="test-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = PresenceCollection(registry, config, nats_client=nats)
    return collection, registry


@pytest.fixture
def bus() -> InMemoryNatsBus:
    """a shared in-memory NATS bus for a test."""
    return InMemoryNatsBus()


@pytest.fixture
def pod_a(bus: InMemoryNatsBus) -> tuple[PresenceCollection, CollectionRegistry, RoomState, PresenceSweeper]:
    """pod A: collection + registry + RoomState + sweeper over the shared bus."""
    collection, registry = make_pod(bus)
    state = RoomState(collection, pod_id="pod-a")
    sweeper = PresenceSweeper(collection, check_interval=100.0, timeout=30.0)
    return collection, registry, state, sweeper


@pytest.fixture
def pod_b(bus: InMemoryNatsBus) -> tuple[PresenceCollection, CollectionRegistry, RoomState, PresenceSweeper]:
    """pod B: a second collection + registry + RoomState over the SAME bus."""
    collection, registry = make_pod(bus)
    state = RoomState(collection, pod_id="pod-b")
    sweeper = PresenceSweeper(collection, check_interval=100.0, timeout=30.0)
    return collection, registry, state, sweeper
