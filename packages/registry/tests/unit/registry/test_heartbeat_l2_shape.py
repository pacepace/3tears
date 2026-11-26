"""a heartbeat in L2 of another shape reads as no heartbeat, never as this release's row.

HeartbeatCollection keeps its rows in L2 alone (``l2_is_the_store``): there is no L3 to refill from. An entry an older
release wrote (no shape tag) or another shape's is ignored, counted, and the pod reads as absent until its next beat
writes its row again (``threetears.core.collections.l2_shape``). An entry of its own shape is read by every replica.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from prometheus_client import REGISTRY
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend

_SCOPE = "heartbeat-shape"
_POD = "pod-shape"


def _replica(nats: FakeNatsClient) -> HeartbeatCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend(), l2_client=nats, kv_key_scope=_SCOPE)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return HeartbeatCollection(registry, config, nats_client=nats)


def _row() -> dict[str, Any]:
    return {
        "pod_id": _POD,
        "date_last_heartbeat": datetime.now(UTC),
        "tools": ["threetears.calculator@1.0.0"],
        "tools_count": 1,
        "status": "healthy",
        "consecutive_misses": 0,
    }


def _mismatches() -> float:
    return (
        REGISTRY.get_sample_value("threetears_collection_l2_shape_mismatches_total", {"table": "pod_heartbeats"}) or 0.0
    )


class TestAHeartbeatOfAnotherShape:
    @pytest.mark.asyncio
    async def test_reads_as_absent_and_is_counted(self) -> None:
        nats = FakeNatsClient()
        writer = _replica(nats)
        bucket: FakeKvBucket = await nats.kv_bucket(name="collections")
        # as every release before 0.95.0 wrote it: the codec's bytes, no tag
        await bucket.put(key=writer.l2_key(_POD), value=writer.serialize(_row()))
        before = _mismatches()

        reader = _replica(nats)
        assert await reader.get(_POD) is None, "a heartbeat of another shape was read as this release's"
        assert _mismatches() == before + 1

    @pytest.mark.asyncio
    async def test_a_saved_heartbeat_is_read_by_another_replica(self) -> None:
        nats = FakeNatsClient()
        writer = _replica(nats)
        await writer.save_entity(writer.create(_row()))

        reader = _replica(nats)
        reader.evict_from_cache_sync(_POD)  # the registry L1 is one per process: read L2, not the writer's L1 copy
        read = await reader.get(_POD)

        assert read is not None and read.status == "healthy"
