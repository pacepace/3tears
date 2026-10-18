"""a heartbeat write or delete that L2 refused raises: L2 is where heartbeats live.

HeartbeatCollection has no L3, so a write L2 refused is a heartbeat no peer registry sees. It used
to be reported as saved, leaving this pod's L1 serving a row no store held.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats.errors import KvError

from threetears.registry.catalog import ToolCatalog
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend


def _collection(nats: FakeNatsClient) -> HeartbeatCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend(), l2_client=nats, kv_key_scope="registry-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return HeartbeatCollection(registry, config, nats_client=nats)


def _row(pod_id: str, *, last: datetime | None = None) -> dict[str, object]:
    return {
        "pod_id": pod_id,
        "date_last_heartbeat": last if last is not None else datetime.now(UTC),
        "tools": ["threetears.calculator@1.0.0"],
        "tools_count": 1,
        "status": "healthy",
        "consecutive_misses": 0,
    }


async def _bucket(nats: FakeNatsClient) -> object:
    return await nats.kv_bucket(name=BaseCollection.L2_BUCKET_SUFFIX)


class TestAHeartbeatL2RefusedRaises:
    @pytest.mark.asyncio
    async def test_a_save_l2_refused_raises_and_leaves_no_l1_row(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        (await _bucket(nats)).become_unreachable(KvError("broker unreachable"))  # type: ignore[attr-defined]

        entity = collection.create(_row("pod-1"))
        with pytest.raises(KvError, match="was not saved"):
            await collection.save_entity(entity)

        assert entity.is_dirty is True
        assert collection.get_row_sync("pod-1") is None

    @pytest.mark.asyncio
    async def test_a_delete_l2_refused_raises_and_the_row_is_still_there(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        await collection.save_entity(collection.create(_row("pod-1")))
        bucket = await _bucket(nats)
        bucket.become_unreachable(KvError("broker unreachable"))  # type: ignore[attr-defined]

        with pytest.raises(KvError, match="was not deleted"):
            await collection.delete("pod-1")

        bucket.become_reachable()  # type: ignore[attr-defined]
        assert await collection.get("pod-1") is not None

    @pytest.mark.asyncio
    async def test_with_no_nats_heartbeats_stay_in_l1_and_nothing_raises(self) -> None:
        registry = CollectionRegistry()
        registry.configure(l1_backend=create_registry_l1_backend())
        collection = HeartbeatCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS"))

        await collection.save_entity(collection.create(_row("pod-1")))

        assert await collection.get("pod-1") is not None
        assert await collection.delete("pod-1") is True


class TestASweepWhoseDeleteL2RefusedTriesAgain:
    @pytest.mark.asyncio
    async def test_the_pod_stays_known_so_the_next_sweep_deletes_it(self) -> None:
        nats = FakeNatsClient()
        collection = _collection(nats)
        subscriber = HeartbeatSubscriber(
            ToolCatalog(), collection, namespace="test", timeout=30.0, check_interval=5.0, max_consecutive_misses=1
        )
        await collection.save_entity(
            collection.create(_row("pod-stale", last=datetime.now(UTC) - timedelta(minutes=1)))
        )
        subscriber.track_pod("pod-stale")
        bucket = await _bucket(nats)
        working_delete = bucket.delete  # type: ignore[attr-defined]
        bucket.delete = AsyncMock(side_effect=KvError("broker unreachable"))  # type: ignore[attr-defined]

        with pytest.raises(KvError, match="was not deleted"):
            await subscriber.run_health_check()
        assert "pod-stale" in subscriber.known_pod_ids

        bucket.delete = working_delete  # type: ignore[attr-defined]
        await subscriber.run_health_check()
        assert "pod-stale" not in subscriber.known_pod_ids
        assert await collection.get("pod-stale") is None
