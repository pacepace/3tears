"""the visibility-scan cache: what it stores, and what must evict it.

The scan this caches is RBAC-filtered, so eviction is a security property, not
a freshness nicety. A scan whose result depends on ``role_assignments`` must
drop when a grant is revoked; leaving that to the TTL would let a revoked caller
keep reading rows.
"""

from __future__ import annotations

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.scan_cache import ScanCache, ScanCacheKey


def _cache(ttl: float = 60.0) -> ScanCache:
    """build a scan cache over a real in-memory L1 backend.

    Deliberately NOT a mock: the point of the class is that it uses the pod's
    L1 rather than a process-local dict, so the test exercises the real one.

    :param ttl: backstop expiry
    :ptype ttl: float
    :return: a scan cache
    :rtype: ScanCache
    """
    return ScanCache(SQLiteBackend(), ttl_seconds=ttl)


_ROWS = [{"id": "a", "name": "concept one"}, {"id": "b", "name": "concept two"}]


class TestRoundTrip:
    def test_miss_before_anything_is_stored(self) -> None:
        cache = _cache()
        assert cache.get(ScanCacheKey("concepts", "user-1"), now_monotonic=0.0) is None

    def test_stored_rows_come_back(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.get(key, now_monotonic=1.0) == _ROWS

    def test_a_different_caller_is_a_different_entry(self) -> None:
        """two callers with different grants must never share a result."""
        cache = _cache()
        cache.put(ScanCacheKey("concepts", "user-1"), _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.get(ScanCacheKey("concepts", "user-2"), now_monotonic=0.0) is None


class TestEviction:
    def test_write_to_the_owning_table_evicts(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts", "role_assignments"), now_monotonic=0.0)
        assert cache.drop_for_table("concepts") == 1
        assert cache.get(key, now_monotonic=0.0) is None

    def test_write_to_an_rbac_table_evicts(self) -> None:
        """THE SECURITY CASE.

        The visibility predicate JOINs ``role_assignments``. If a revoked grant
        did not drop the entry, the caller would keep reading rows they can no
        longer see until the TTL lapsed.
        """
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts", "role_assignments"), now_monotonic=0.0)
        assert cache.drop_for_table("role_assignments") == 1
        assert cache.get(key, now_monotonic=0.0) is None

    def test_an_unrelated_table_does_not_evict(self) -> None:
        """or every write anywhere would flush the cache and it would buy nothing."""
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.drop_for_table("conversations") == 0
        assert cache.get(key, now_monotonic=0.0) == _ROWS


class TestTtlBackstop:
    def test_expired_entry_is_a_miss(self) -> None:
        cache = _cache(ttl=30.0)
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.get(key, now_monotonic=31.0) is None

    def test_entry_inside_the_ttl_survives(self) -> None:
        cache = _cache(ttl=30.0)
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.get(key, now_monotonic=29.0) == _ROWS


class TestDisabled:
    def test_no_l1_backend_is_a_no_op_not_an_error(self) -> None:
        """a pod with no L1 must still serve, just uncached."""
        cache = ScanCache(None)
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)
        assert cache.get(key, now_monotonic=0.0) is None
        assert cache.drop_for_table("concepts") == 0


class TestLocalWriteEvictsLocalScans:
    """the pod that WRITES is the one pod that never hears its own broadcast.

    The invalidation listener skips self-published messages on purpose: for a
    by-pk row the writer holds the freshest copy, so evicting it would force a
    needless re-read. A SCAN inverts that. The write changed which rows match,
    so the writer's cached result is stale the moment it commits -- and it is
    guaranteed never to be told.

    Shipped without this in 0.23.6: a hub that imported knowledge kept serving
    the pre-import concept set until the TTL lapsed.
    """

    @pytest.mark.asyncio
    async def test_publish_invalidation_drops_dependent_scans(self) -> None:
        from threetears.core.collections.registry import CollectionRegistry

        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend())
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)

        await registry.publish_invalidation(None, "concepts", "some-id")

        assert registry.scan_cache.get(key, now_monotonic=0.0) is None

    @pytest.mark.asyncio
    async def test_eviction_happens_without_a_nats_client(self) -> None:
        """local eviction is not a broadcast; no bus must not mean no eviction.

        The method returns early when there is no NATS client. Ordering the
        scan drop after that return would leave devx, tests, and any pod whose
        bus is down serving stale scans forever.
        """
        from threetears.core.collections.registry import CollectionRegistry

        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend())
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, depends_on=("concepts", "role_assignments"), now_monotonic=0.0)

        await registry.publish_invalidation(None, "role_assignments", "grant-id")

        assert registry.scan_cache.get(key, now_monotonic=0.0) is None

    @pytest.mark.asyncio
    async def test_an_unrelated_write_leaves_the_scan_alone(self) -> None:
        from threetears.core.collections.registry import CollectionRegistry

        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend())
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, depends_on=("concepts",), now_monotonic=0.0)

        await registry.publish_invalidation(None, "conversations", "other-id")

        assert registry.scan_cache.get(key, now_monotonic=0.0) == _ROWS
