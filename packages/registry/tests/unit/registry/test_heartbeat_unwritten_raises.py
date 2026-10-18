"""heartbeats live in L2 alone: an L2 failure is never read as "saved" or as "gone".

HeartbeatCollection has no L3, so a write L2 refused is a heartbeat no peer registry sees, and a
read L2 could not answer is not an absent pod. The collection raises for both; the subscriber
records the failure and keeps judging liveness from what this registry itself heard, because
quarantining a pod that stopped heartbeating must not wait for L2 to come back.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.agent.tools.server import HeartbeatMessage
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import L2WriteRefusedError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats.errors import KvError

from threetears.registry.catalog import ToolCatalog, ToolEndpoint
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend

from .copy_entries import uniform_entry

#: short enough that a test outlives it with a brief sleep
_TIMEOUT = 0.05


def _collection(nats: FakeNatsClient) -> HeartbeatCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend(), l2_client=nats, kv_key_scope="registry-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return HeartbeatCollection(registry, config, nats_client=nats)


def _row(pod_id: str, *, last: datetime | None = None) -> dict[str, object]:
    return {
        "pod_id": pod_id,
        "date_last_heartbeat": last if last is not None else datetime.now(UTC),
        "tools": [f"threetears.{pod_id}@1.0.0"],
        "tools_count": 1,
        "status": "healthy",
        "consecutive_misses": 0,
    }


async def _bucket(nats: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await nats.kv_bucket(name=BaseCollection.L2_BUCKET_SUFFIX)
    return bucket


def _outage() -> KvError:
    return KvError("broker unreachable")


def _heartbeat(pod_id: str) -> MagicMock:
    msg = MagicMock()
    msg.data = HeartbeatMessage(pod_id=pod_id, timestamp=datetime.now(UTC).isoformat(), tools_count=1).model_dump_json()
    return msg


async def _catalog_serving(*pod_ids: str) -> ToolCatalog:
    catalog = ToolCatalog()
    for pod_id in pod_ids:
        await catalog.register(
            uniform_entry(
                tool_name=f"threetears.{pod_id}",
                tool_version="1.0.0",
                description="a tool",
                input_schema={"type": "object", "properties": {}},
                endpoints=[ToolEndpoint(pod_id=pod_id, status="available")],
            )
        )
    return catalog


def _status(catalog: ToolCatalog, pod_id: str) -> str | None:
    entry = catalog.get(f"threetears.{pod_id}@1.0.0")
    return None if entry is None else entry.endpoints[0].status


def _subscriber(catalog: ToolCatalog, collection: HeartbeatCollection, *, misses: int) -> HeartbeatSubscriber:
    return HeartbeatSubscriber(
        catalog, collection, namespace="test", timeout=_TIMEOUT, check_interval=5.0, max_consecutive_misses=misses
    )


class TestTheCollectionRaisesForAnL2Failure:
    @pytest.mark.asyncio
    async def test_a_save_l2_refused_raises_and_the_handle_still_reads(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        (await _bucket(nats)).become_unreachable(_outage())

        entity = collection.create(_row("pod-1"))
        with pytest.raises(L2WriteRefusedError, match="was not saved"):
            await collection.save_entity(entity)

        assert entity.is_dirty is True
        assert entity.to_dict()["pod_id"] == "pod-1", "the handle keeps the row it was saving"
        assert collection.get_row_sync("pod-1") is None

    @pytest.mark.asyncio
    async def test_a_delete_l2_refused_raises_and_the_row_is_still_there(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        await collection.save_entity(collection.create(_row("pod-1")))
        bucket = await _bucket(nats)
        bucket.become_unreachable(_outage())

        with pytest.raises(L2WriteRefusedError, match="was not deleted"):
            await collection.delete("pod-1")

        bucket.become_reachable()
        assert await collection.get("pod-1") is not None

    @pytest.mark.asyncio
    async def test_a_read_l2_could_not_answer_raises_rather_than_reading_absent(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        await collection.save_entity(collection.create(_row("pod-1")))
        collection.evict_from_cache_sync("pod-1")
        (await _bucket(nats)).become_unreachable(_outage())

        with pytest.raises(KvError):
            await collection.get("pod-1")

    @pytest.mark.asyncio
    async def test_with_no_nats_heartbeats_stay_in_l1_and_nothing_raises(self) -> None:
        registry = CollectionRegistry()
        registry.configure(l1_backend=create_registry_l1_backend())
        collection = HeartbeatCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS"))

        await collection.save_entity(collection.create(_row("pod-1")))

        assert await collection.get("pod-1") is not None
        assert await collection.delete("pod-1") is True


class TestLivenessIsStillJudgedDuringAnL2Outage:
    @pytest.mark.asyncio
    async def test_a_pod_that_stops_heartbeating_is_still_quarantined(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        catalog = await _catalog_serving("pod-a")
        subscriber = _subscriber(catalog, collection, misses=3)
        await subscriber.handle_heartbeat(_heartbeat("pod-a"))
        collection.evict_from_cache_sync("pod-a")
        (await _bucket(nats)).become_unreachable(_outage())

        await asyncio.sleep(_TIMEOUT * 2)
        await subscriber.run_health_check()

        assert _status(catalog, "pod-a") == "unavailable"
        assert "pod-a" in subscriber.known_pod_ids

    @pytest.mark.asyncio
    async def test_a_pod_first_seen_during_the_outage_is_still_swept(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        catalog = await _catalog_serving("pod-new")
        subscriber = _subscriber(catalog, collection, misses=1)
        bucket = await _bucket(nats)
        bucket.become_unreachable(_outage())

        await subscriber.handle_heartbeat(_heartbeat("pod-new"))  # recorded, not raised
        assert "pod-new" in subscriber.known_pod_ids
        await asyncio.sleep(_TIMEOUT * 2)
        await subscriber.run_health_check()

        assert _status(catalog, "pod-new") is None, "deregistered on this registry's own record"
        assert "pod-new" in subscriber.known_pod_ids, "its row's delete is owed until L2 answers"

        bucket.become_reachable()
        await subscriber.run_health_check()
        assert "pod-new" not in subscriber.known_pod_ids

    @pytest.mark.asyncio
    async def test_an_unreadable_row_is_not_taken_for_a_peer_s_eviction(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        catalog = await _catalog_serving("pod-a")
        subscriber = _subscriber(catalog, collection, misses=3)
        await collection.save_entity(collection.create(_row("pod-a")))
        collection.evict_from_cache_sync("pod-a")
        subscriber.track_pod("pod-a")
        (await _bucket(nats)).become_unreachable(_outage())

        await subscriber.run_health_check()

        assert "pod-a" in subscriber.known_pod_ids
        assert _status(catalog, "pod-a") == "available"


class TestASweepGoesOnPastARefusedDelete:
    @pytest.mark.asyncio
    async def test_one_refused_delete_does_not_stop_the_others(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        catalog = await _catalog_serving("pod-a", "pod-b")
        subscriber = _subscriber(catalog, collection, misses=1)
        stale = datetime.now(UTC) - timedelta(minutes=1)
        for pod_id in ("pod-a", "pod-b"):
            await collection.save_entity(collection.create(_row(pod_id, last=stale)))
            subscriber.track_pod(pod_id)
        bucket = await _bucket(nats)
        working_delete = bucket.delete
        refused_key = collection.l2_key("pod-a")

        async def _delete(*, key: str, revision: int | None = None) -> bool:
            if key == refused_key:
                raise _outage()
            return await working_delete(key=key, revision=revision)

        bucket.delete = AsyncMock(side_effect=_delete)  # type: ignore[method-assign]

        await subscriber.run_health_check()

        assert subscriber.known_pod_ids == {"pod-a"}, "pod-a's delete is retried next sweep; pod-b is done"
        assert await collection.get("pod-b") is None
        assert await collection.get("pod-a") is not None
