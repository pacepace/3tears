"""HeartbeatCollection must never be given an L1 max-age bound.

The heartbeat collection is L1+L2 only: `heartbeat_collection.py` forces
``l3_pool = None`` and every L3 method raises. So there is nothing to pull
through from, and an expired row would not be a miss that repairs itself on the
next read -- it would read as "this pod never reported", which is the same shape
as a dead pod. Liveness decided by a cache timer rather than by an actual missed
heartbeat is exactly the failure the sweeper exists to make deliberate.

The gate lives in ``BaseCollection.l1_max_age_seconds``, which refuses a bound
when there is no L3 pool. These tests hold the facts that gate depends on, for
this collection specifically -- the presence half is covered by
``packages/channels/tests/unit/channels/presence/test_no_l1_expiry.py``.
"""

from __future__ import annotations

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend


def _collection() -> tuple[HeartbeatCollection, CollectionRegistry]:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend())
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return HeartbeatCollection(registry, config), registry


class TestHeartbeatHasNoL3ToPullThroughFrom:
    def test_the_collection_reports_no_l3_pool(self) -> None:
        collection, _ = _collection()
        assert collection.l3_pool is None


class TestABoundIsRefusedEvenWhenConfigured:
    def test_configuring_a_max_age_does_not_give_heartbeats_one(self) -> None:
        """Configuration is not the gate; having somewhere to fall back to is.

        An operator turning expiry on fleet-wide must not be able to reach this
        table by doing so.
        """
        collection, registry = _collection()
        registry.set_l1_max_age(collection.table_name, 1.0)
        assert collection.l1_max_age_seconds is None

    def test_the_registry_still_records_what_was_asked_for(self) -> None:
        """The refusal is at the point of use, not by discarding the setting.

        If the registry dropped the value instead, a collection that later
        gained an L3 would quietly stay unbounded.
        """
        collection, registry = _collection()
        registry.set_l1_max_age(collection.table_name, 1.0)
        assert registry.get_l1_max_age(collection.table_name) == 1.0
