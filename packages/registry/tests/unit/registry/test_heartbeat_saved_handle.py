"""a saved heartbeat handle keeps reading the row it saved after L1 drops the key.

HeartbeatCollection is L1+L2 only, and its L1 is a cache of L2: a peer's invalidation broadcast
evicts the key at any time. A handle that read its fields through L1 would then answer every
field as missing -- the same defect that made a saved identity root read ``version_id`` as None
once a caller transaction settled its key.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend


def _collection() -> HeartbeatCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend())
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return HeartbeatCollection(registry, config)


class TestASavedHeartbeatHandle:
    @pytest.mark.asyncio
    async def test_reads_its_row_after_l1_drops_the_key(self) -> None:
        collection = _collection()
        entity = collection.create(
            {
                "pod_id": "pod-held",
                "date_last_heartbeat": datetime.now(UTC),
                "tools": ["threetears.calculator@1.0.0"],
                "tools_count": 1,
                "status": "healthy",
                "consecutive_misses": 0,
            }
        )
        await collection.save_entity(entity)
        collection.evict_from_cache_sync("pod-held")
        assert entity.status == "healthy"
        assert entity.date_updated is not None
