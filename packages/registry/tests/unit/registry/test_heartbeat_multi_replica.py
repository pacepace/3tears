"""two registry replicas on one heartbeat store: eviction is timed and complete as with one.

Every replica hears every heartbeat (no queue group) and sweeps on its own interval; the heartbeat
rows are shared. One :class:`HeartbeatCollection` shared by two subscribers, each with its own
catalog, stands in for two registry processes on one bucket. What is pinned:

- a pod is evicted after ``max_consecutive_misses`` sweeps of a replica, not after that many sweeps
  summed over the replicas (a miss count shared through the row rose once per replica);
- a pod one replica evicts leaves the other replica's catalog too, not only its tracking;
- a replica sweeps every pod its catalog holds, so copies loaded from the bucket for a pod that died
  before it started are evicted, while a pod that registered within the timeout is left alone;
- a pod quarantined by one replica is revived there when a peer records its heartbeat.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.l1_cache import create_registry_l1_backend

from .copy_entries import uniform_entry

_TOOL = "threetears.calculator@1.0.0"
_TIMEOUT = 30.0


def _store() -> HeartbeatCollection:
    """one heartbeat store both replicas read and write.

    :return: the collection
    :rtype: HeartbeatCollection
    """
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend())
    return HeartbeatCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


def _entry(pod_id: str, *, registered: datetime | None = None) -> CatalogEntry:
    """a catalog entry served by one pod.

    :param pod_id: the pod
    :ptype pod_id: str
    :param registered: when the copy last registered; now when ``None``
    :ptype registered: datetime | None
    :return: the entry
    :rtype: CatalogEntry
    """
    endpoint = ToolEndpoint(pod_id=pod_id, status="available")
    if registered is not None:
        endpoint.date_last_heartbeat = registered
    return uniform_entry(
        tool_name="threetears.calculator",
        tool_version="1.0.0",
        full_name=_TOOL,
        description="test tool",
        input_schema={"type": "object", "properties": {}},
        endpoints=[endpoint],
    )


async def _replica(
    store: HeartbeatCollection, *pods: CatalogEntry, misses: int = 3
) -> tuple[HeartbeatSubscriber, ToolCatalog]:
    """one registry replica: its own catalog, the shared store.

    :param store: the shared heartbeat store
    :ptype store: HeartbeatCollection
    :param pods: the entries its catalog holds
    :ptype pods: CatalogEntry
    :param misses: its eviction threshold
    :ptype misses: int
    :return: the subscriber and its catalog
    :rtype: tuple[HeartbeatSubscriber, ToolCatalog]
    """
    catalog = ToolCatalog()
    for entry in pods:
        await catalog.register(entry)
    subscriber = HeartbeatSubscriber(
        catalog, store, namespace="test", check_interval=5.0, timeout=_TIMEOUT, max_consecutive_misses=misses
    )
    return subscriber, catalog


async def _stale_row(store: HeartbeatCollection, pod_id: str) -> None:
    """a heartbeat row last written a minute ago, past the timeout.

    :param store: the shared store
    :ptype store: HeartbeatCollection
    :param pod_id: the pod
    :ptype pod_id: str
    :return: nothing
    :rtype: None
    """
    entity = store.create(
        {
            "pod_id": pod_id,
            "date_last_heartbeat": datetime.now(UTC) - timedelta(seconds=60),
            "tools": [_TOOL],
            "tools_count": 1,
            "status": "healthy",
            "consecutive_misses": 0,
        }
    )
    await store.save_entity(entity)


def _status(catalog: ToolCatalog) -> str | None:
    entry = catalog.get(_TOOL)
    return None if entry is None else entry.endpoints[0].status


class TestEvictionTimingIsEachReplicas:
    async def test_two_replicas_evict_after_the_threshold_of_sweeps_not_half_of_it(self) -> None:
        store = _store()
        await _stale_row(store, "pod-dead")
        first, first_catalog = await _replica(store, _entry("pod-dead"))
        second, second_catalog = await _replica(store, _entry("pod-dead"))

        for _ in range(2):
            await first.run_health_check()
            await second.run_health_check()

        # two intervals, each swept by both: two misses each, under the threshold of three
        assert _status(first_catalog) == "unavailable"
        assert _status(second_catalog) == "unavailable"
        assert await store.get("pod-dead") is not None

        await first.run_health_check()
        assert first_catalog.get(_TOOL) is None, "the third miss evicts"


class TestAPeersEvictionLeavesThisCatalog:
    async def test_a_pod_a_peer_evicted_is_dropped_from_this_catalog(self) -> None:
        store = _store()
        await _stale_row(store, "pod-dead")
        # a live pod re-registers with every heartbeat, so a dead pod's copy is as old as its last one
        last_heard = datetime.now(UTC) - timedelta(seconds=60)
        evicting, evicting_catalog = await _replica(store, _entry("pod-dead", registered=last_heard), misses=1)
        peer, peer_catalog = await _replica(store, _entry("pod-dead", registered=last_heard), misses=1)
        peer.track_pod("pod-dead")

        await evicting.run_health_check()
        assert evicting_catalog.get(_TOOL) is None
        assert await store.get("pod-dead") is None

        await peer.run_health_check()
        assert peer_catalog.get(_TOOL) is None, "the peer no longer offers the evicted pod's copy"
        assert "pod-dead" not in peer.known_pod_ids


class TestAReplicaSweepsWhatItsCatalogHolds:
    async def test_copies_loaded_for_a_pod_that_died_before_this_replica_started_are_evicted(self) -> None:
        store = _store()
        long_ago = datetime.now(UTC) - timedelta(hours=1)
        # loaded from the bucket: never heard here, its row long since deleted
        late, catalog = await _replica(store, _entry("pod-gone", registered=long_ago))

        await late.run_health_check()

        assert catalog.get(_TOOL) is None

    async def test_a_pod_that_registered_within_the_timeout_is_left_while_its_row_is_unwritten(self) -> None:
        store = _store()
        fresh, catalog = await _replica(store, _entry("pod-new"))

        await fresh.run_health_check()

        assert _status(catalog) == "available"
        assert "pod-new" in fresh.known_pod_ids

    async def test_a_stale_row_for_a_loaded_copy_is_counted_and_evicted(self) -> None:
        store = _store()
        await _stale_row(store, "pod-dead")
        late, catalog = await _replica(store, _entry("pod-dead"), misses=1)

        await late.run_health_check()

        assert catalog.get(_TOOL) is None
        assert await store.get("pod-dead") is None


class TestAPeerHeardTheRevival:
    async def test_a_quarantined_pod_is_revived_when_a_peer_records_its_heartbeat(self) -> None:
        store = _store()
        await _stale_row(store, "pod-blip")
        quarantining, catalog = await _replica(store, _entry("pod-blip"))

        await quarantining.run_health_check()
        assert _status(catalog) == "unavailable"

        # the heartbeat reached a peer, which wrote the row; this replica did not hear it
        row = await store.get("pod-blip")
        assert row is not None
        row.date_last_heartbeat = datetime.now(UTC)
        row.consecutive_misses = 0
        row.status = "healthy"
        await store.save_entity(row)

        await quarantining.run_health_check()
        assert _status(catalog) == "available"
