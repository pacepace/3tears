"""Presence collections must never be given an L1 max-age bound.

The bound is a cache mechanism, and these collections are not caches: they are
L1+L2 only, with no L3 to pull through from. An expired row would therefore not
be a miss that repairs itself on the next read -- it would read as "this row
does not exist", and the member CAS that follows would write a fresh one-member
room over a live ten-member one, with nothing raising.

The gate lives in ``BaseCollection.l1_max_age_seconds``, which refuses a bound
when there is no L3 pool. These tests hold the two facts that gate depends on.
"""

from __future__ import annotations

from typing import Any


from .conftest import InMemoryNatsBus, make_pod


def _collections(collection: Any) -> list[Any]:
    """the two L1+L2-only collections a PresenceCollection composes."""
    return [collection.connections, collection.rooms]


class TestPresenceHasNoL3ToPullThroughFrom:
    def test_every_presence_collection_reports_no_l3_pool(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        for sub in _collections(collection):
            assert sub.l3_pool is None


class TestABoundIsRefusedEvenWhenConfigured:
    def test_configuring_a_max_age_does_not_give_presence_one(self, bus: InMemoryNatsBus) -> None:
        """Configuration is not the gate; having somewhere to fall back to is.

        A future operator turning expiry on fleet-wide must not be able to
        reach these tables by doing so.
        """
        collection, registry = make_pod(bus)
        for sub in _collections(collection):
            registry.set_l1_max_age(sub.table_name, 1.0)

        for sub in _collections(collection):
            assert sub.l1_max_age_seconds is None

    def test_the_registry_still_records_what_was_asked_for(self, bus: InMemoryNatsBus) -> None:
        """The refusal is at the point of use, not by discarding the setting.

        Worth pinning: if the registry silently dropped the value instead, a
        collection that later gained an L3 would quietly stay unbounded.
        """
        collection, registry = make_pod(bus)
        table = _collections(collection)[0].table_name
        registry.set_l1_max_age(table, 1.0)
        assert registry.get_l1_max_age(table) == 1.0
