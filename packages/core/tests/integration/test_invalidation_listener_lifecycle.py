"""cross-pod integration test for the invalidation listener's stop path.

the unit tests in ``packages/core/tests/test_registry.py`` drive the real
wrapper over a fake nats-py client, so they prove the handle is retained,
released and re-acquired. what they cannot prove is the property a consumer
actually depends on: that ONE pod tearing its listener down leaves every
other pod still receiving invalidations.

that is the failure this shard could plausibly introduce -- a teardown that
took a shared consumer down with it would look identical in a single-process
test and would silently stop cross-pod cache coherence in a fleet. so two
registries subscribe against a real NATS, one stops, and the other must still
evict.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "invalidation_lifecycle_rows"
_ROW_ID = "row-1"


def _make_metadata() -> MetaData:
    """one-table schema for the pod-local L1 caches.

    :return: metadata carrying the single test table
    :rtype: MetaData
    """
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
    )
    return metadata


class _StubCollection:
    """the two attributes the invalidation listener reads off a collection.

    a real :class:`BaseCollection` would drag an L3 store and a config into a
    test whose subject is the subscription lifecycle. the listener looks up
    ``table_name`` (to register) and ``primary_key_columns`` (to check the
    message's pk arity) and touches nothing else on the way to the eviction.
    """

    @property
    def table_name(self) -> str:
        """the registered table.

        :return: table name
        :rtype: str
        """
        return _TABLE

    @property
    def primary_key_columns(self) -> tuple[str, ...]:
        """pk columns in declared order.

        :return: the single-column pk
        :rtype: tuple[str, ...]
        """
        return ("id",)


@pytest.fixture
async def nats_clients(nats_container: str) -> AsyncIterator[tuple[NatsClient, NatsClient]]:
    """two independently connected NATS clients: one per simulated pod.

    :param nats_container: the container fixture's connection url
    :ptype nats_container: str
    :return: async iterator yielding both connected clients
    :rtype: AsyncIterator[tuple[NatsClient, NatsClient]]
    """
    set_default_namespace("3tears")
    first = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="3tears",
        client_name="invalidation-pod-a",
    )
    second = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="3tears",
        client_name="invalidation-pod-b",
    )
    try:
        yield (first, second)
    finally:
        await first.shutdown()
        await second.shutdown()


def _make_pod() -> tuple[CollectionRegistry, SQLiteBackend]:
    """build one pod: its own L1 backend, registry and registered collection.

    :return: the pod's registry and its L1 backend
    :rtype: tuple[CollectionRegistry, SQLiteBackend]
    """
    l1 = SQLiteBackend(db_name=f"invalidation_lifecycle_{uuid.uuid4().hex[:8]}")
    l1.initialize(_make_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1)
    registry.register(_StubCollection())
    return registry, l1


def _seed(*backends: SQLiteBackend) -> None:
    """write the same row into every backend's L1.

    :param backends: the pods' L1 backends
    :ptype backends: SQLiteBackend
    :return: nothing
    :rtype: None
    """
    for backend in backends:
        backend.upsert(_TABLE, {"id": _ROW_ID, "name": "seeded"}, "id")


def _is_cached(backend: SQLiteBackend) -> bool:
    """whether the seeded row is still in this backend's L1.

    :param backend: the pod's L1 backend
    :ptype backend: SQLiteBackend
    :return: True while the row survives
    :rtype: bool
    """
    return backend.select_by_id(_TABLE, _ROW_ID, "id") is not None


async def _wait_until_evicted(backend: SQLiteBackend, timeout: float = 10.0) -> None:
    """poll until the seeded row leaves ``backend``'s L1, or fail.

    :param backend: the pod's L1 backend
    :ptype backend: SQLiteBackend
    :param timeout: seconds to wait before giving up
    :ptype timeout: float
    :return: nothing
    :rtype: None
    :raises AssertionError: if the row is still cached when the timeout lapses
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while _is_cached(backend):
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"row still cached after {timeout}s -- the invalidation never landed")
        await asyncio.sleep(0.05)


class TestStopLeavesOtherSubscribersAlone:
    """the cross-pod half of SUB-02: one teardown, one casualty."""

    async def test_stopping_one_pod_does_not_deafen_the_other(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """pod A stops listening; pod B must keep evicting on every broadcast.

        the first round is not decoration -- without it, "pod A did not evict"
        is satisfied just as well by a listener that never worked, and the test
        would pass against a broken subscribe.

        :param nats_clients: one connected client per pod
        :ptype nats_clients: tuple[NatsClient, NatsClient]
        :return: nothing
        :rtype: None
        """
        client_a, client_b = nats_clients
        registry_a, l1_a = _make_pod()
        registry_b, l1_b = _make_pod()
        # a third registry publishes: its origin differs from both listeners',
        # so neither receiver takes the self-origin early return.
        publisher = CollectionRegistry()

        await registry_a.start_invalidation_listener(client_a)
        await registry_b.start_invalidation_listener(client_b)
        # subscribe -> flush -> publish. nats-py does not flush on subscribe, so
        # without the barrier pod B's SUB can reach the server after the PUB it
        # was meant to catch, and the test fails on a race rather than on the
        # behaviour it is asserting.
        await client_a.flush()
        await client_b.flush()
        try:
            _seed(l1_a, l1_b)
            await publisher.publish_invalidation(client_a, _TABLE, _ROW_ID)
            await _wait_until_evicted(l1_a)
            await _wait_until_evicted(l1_b)

            _seed(l1_a, l1_b)
            await registry_a.stop_invalidation_listener()
            await publisher.publish_invalidation(client_a, _TABLE, _ROW_ID)

            await _wait_until_evicted(l1_b)
            assert _is_cached(l1_a), "pod A unsubscribed, so its L1 must be untouched by the broadcast"
        finally:
            await registry_a.stop_invalidation_listener()
            await registry_b.stop_invalidation_listener()

    async def test_a_stopped_pod_hears_again_after_restarting(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """SUB-05 across a real connection: the registry is reusable.

        :param nats_clients: one connected client per pod
        :ptype nats_clients: tuple[NatsClient, NatsClient]
        :return: nothing
        :rtype: None
        """
        client_a, _ = nats_clients
        registry_a, l1_a = _make_pod()
        publisher = CollectionRegistry()

        await registry_a.start_invalidation_listener(client_a)
        await registry_a.stop_invalidation_listener()
        await registry_a.start_invalidation_listener(client_a)
        await client_a.flush()
        try:
            _seed(l1_a)
            await publisher.publish_invalidation(client_a, _TABLE, _ROW_ID)
            await _wait_until_evicted(l1_a)
        finally:
            await registry_a.stop_invalidation_listener()
