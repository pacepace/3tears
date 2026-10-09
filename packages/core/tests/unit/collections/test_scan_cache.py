"""the visibility-scan cache: what it stores, and what must evict it.

The scan this caches is RBAC-filtered, so eviction is a security property, not
a freshness nicety. A scan whose result depends on ``role_assignments`` must
drop when a grant is revoked; leaving that to the TTL would let a revoked caller
keep reading rows.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.scan_cache import ScanCache, ScanCacheKey


def _cache() -> ScanCache:
    """build a scan cache over a real in-memory L1 backend, every table followed.

    Deliberately NOT a mock: the point of the class is that it uses the pod's
    L1 rather than a process-local dict, so the test exercises the real one.

    :return: a scan cache
    :rtype: ScanCache
    """
    return ScanCache(SQLiteBackend(db_name=f"scan-{uuid4().hex}"), trusted=lambda _tables: True)


def _followed_registry(*tables: str) -> object:
    """a registry over a real L1 whose given tables are followed with their watches running.

    :param tables: the tables
    :ptype tables: str
    :return: the registry
    :rtype: CollectionRegistry
    """
    from threetears.core.collections.registry import CollectionRegistry

    registry = CollectionRegistry()
    registry.configure(l1_backend=SQLiteBackend())
    for table in tables:
        registry.follow_generation(table)
        registry.watched_by(table, lambda: True)
    return registry


_ROWS = [{"id": "a", "name": "concept one"}, {"id": "b", "name": "concept two"}]


class TestRoundTrip:
    def test_miss_before_anything_is_stored(self) -> None:
        cache = _cache()
        assert cache.get(ScanCacheKey("concepts", "user-1")) is None

    def test_stored_rows_come_back(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts",)))
        assert cache.get(key) == _ROWS

    def test_a_different_caller_is_a_different_entry(self) -> None:
        """two callers with different grants must never share a result."""
        cache = _cache()
        cache.put(ScanCacheKey("concepts", "user-1"), _ROWS, token=cache.begin_read(("concepts",)))
        assert cache.get(ScanCacheKey("concepts", "user-2")) is None


class TestEviction:
    def test_write_to_the_owning_table_evicts(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts", "role_assignments")))
        assert cache.drop_for_table("concepts") == 1
        assert cache.get(key) is None

    def test_write_to_an_rbac_table_evicts(self) -> None:
        """THE SECURITY CASE.

        The visibility predicate JOINs ``role_assignments``. If a revoked grant
        did not drop the entry, the caller would keep reading rows they can no
        longer see.
        """
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts", "role_assignments")))
        assert cache.drop_for_table("role_assignments") == 1
        assert cache.get(key) is None

    def test_an_unrelated_table_does_not_evict(self) -> None:
        """or every write anywhere would flush the cache and it would buy nothing."""
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts",)))
        assert cache.drop_for_table("conversations") == 0
        assert cache.get(key) == _ROWS


class TestNoAgeOnlyTrust:
    """nothing expires; an entry is served and stored only while every dependency is followed."""

    def test_an_entry_is_served_however_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as _time

        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts",)))
        later = _time.monotonic() + 365 * 24 * 3600
        monkeypatch.setattr(_time, "monotonic", lambda: later)
        assert cache.get(key) == _ROWS

    def test_a_cache_nothing_follows_stores_nothing(self) -> None:
        cache = ScanCache(SQLiteBackend())
        key = ScanCacheKey("concepts", "user-1")
        assert cache.put(key, _ROWS, token=cache.begin_read(("concepts",))) is False
        assert cache.get(key) is None

    def test_an_entry_is_not_served_while_a_dependency_is_not_followed(self) -> None:
        followed = {"concepts": True, "role_assignments": True}
        cache = ScanCache(SQLiteBackend(), trusted=lambda tables: all(followed.get(t, False) for t in tables))
        key = ScanCacheKey("concepts", "user-1")
        assert cache.put(key, _ROWS, token=cache.begin_read(("concepts", "role_assignments")))
        followed["role_assignments"] = False
        assert cache.get(key) is None
        followed["role_assignments"] = True
        assert cache.get(key) == _ROWS

    def test_the_registry_trusts_a_table_only_while_it_is_followed_and_watched(self) -> None:
        from threetears.core.collections.registry import CollectionRegistry

        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend())
        assert not registry.tables_trusted(("concepts",))
        registry.follow_generation("concepts")
        assert not registry.tables_trusted(("concepts",)), "followed but nobody watches it"
        watching = [True]

        def watch() -> bool:
            return watching[0]

        registry.watched_by("concepts", watch)
        assert registry.tables_trusted(("concepts",))
        watching[0] = False
        assert not registry.tables_trusted(("concepts",))
        registry.not_watched_by("concepts", watch)
        watching[0] = True
        assert not registry.tables_trusted(("concepts",))
        key = ScanCacheKey("concepts", "user-1")
        assert registry.scan_cache.put(key, _ROWS, token=registry.scan_cache.begin_read(("concepts",))) is False

    def test_a_table_two_followers_watch_is_trusted_only_while_both_watches_run(self) -> None:
        """a second follower must not hide the first one's failed watch, nor its withdrawal the second's."""
        from threetears.core.collections.registry import CollectionRegistry

        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend())
        registry.follow_generation("concepts")
        first, second = [True], [True]

        def first_watch() -> bool:
            return first[0]

        def second_watch() -> bool:
            return second[0]

        registry.watched_by("concepts", first_watch)
        registry.watched_by("concepts", second_watch)
        assert registry.tables_trusted(("concepts",))
        first[0] = False
        assert not registry.tables_trusted(("concepts",)), "the first follower's watch is down"
        first[0] = True
        second[0] = False
        assert not registry.tables_trusted(("concepts",)), "the second follower's watch is down"
        second[0] = True
        registry.not_watched_by("concepts", second_watch)
        assert registry.tables_trusted(("concepts",)), "the first follower still watches"
        first[0] = False
        assert not registry.tables_trusted(("concepts",))
        registry.not_watched_by("concepts", first_watch)
        first[0] = True
        assert not registry.tables_trusted(("concepts",)), "nobody watches it now"


class TestDisabled:
    def test_no_l1_backend_is_a_no_op_not_an_error(self) -> None:
        """a pod with no L1 must still serve, just uncached."""
        cache = ScanCache(None)
        key = ScanCacheKey("concepts", "user-1")
        cache.put(key, _ROWS, token=cache.begin_read(("concepts",)))
        assert cache.get(key) is None
        assert cache.drop_for_table("concepts") == 0


class TestLocalWriteEvictsLocalScans:
    """the pod that WRITES is the one pod that never hears its own broadcast.

    The invalidation listener skips self-published messages on purpose: for a
    by-pk row the writer holds the freshest copy, so evicting it would force a
    needless re-read. A SCAN inverts that. The write changed which rows match,
    so the writer's cached result is stale the moment it commits -- and it is
    guaranteed never to be told.

    Shipped without this in 0.23.6: a hub that imported knowledge kept serving
    the pre-import concept set.
    """

    @pytest.mark.asyncio
    async def test_publish_invalidation_drops_dependent_scans(self) -> None:
        registry: Any = _followed_registry("concepts", "role_assignments")
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, token=registry.scan_cache.begin_read(("concepts",)))

        await registry.publish_invalidation(None, "concepts", "some-id")

        assert registry.scan_cache.get(key) is None

    @pytest.mark.asyncio
    async def test_eviction_happens_without_a_nats_client(self) -> None:
        """local eviction is not a broadcast; no bus must not mean no eviction.

        The method returns early when there is no NATS client. Ordering the
        scan drop after that return would leave devx, tests, and any pod whose
        bus is down serving stale scans forever.
        """
        registry: Any = _followed_registry("concepts", "role_assignments")
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, token=registry.scan_cache.begin_read(("concepts", "role_assignments")))

        await registry.publish_invalidation(None, "role_assignments", "grant-id")

        assert registry.scan_cache.get(key) is None

    @pytest.mark.asyncio
    async def test_an_unrelated_write_leaves_the_scan_alone(self) -> None:
        registry: Any = _followed_registry("concepts", "role_assignments")
        key = ScanCacheKey("concepts", "user-1")
        registry.scan_cache.put(key, _ROWS, token=registry.scan_cache.begin_read(("concepts",)))

        await registry.publish_invalidation(None, "conversations", "other-id")

        assert registry.scan_cache.get(key) == _ROWS


class TestAReadOvertakenByAnEvictionIsNotStored:
    """eviction drops what is stored; it cannot drop what a reader has not stored YET.

    A scan that read L3 before a write committed, and reaches ``put`` after that write's
    eviction ran, would store the pre-write rows where no eviction can reach them -- and
    serve them until the TTL. The read token closes that: ``put`` refuses a result whose
    dependencies were evicted after the token was taken.
    """

    def test_eviction_during_the_read_refuses_the_store(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        token = cache.begin_read(("concepts", "role_assignments"))
        # the write commits and evicts while the read is in flight, before anything is stored
        cache.drop_for_table("concepts")
        assert cache.put(key, _ROWS, token=token) is False
        assert cache.get(key) is None

    def test_rbac_eviction_during_the_read_refuses_the_store(self) -> None:
        """the security case: a grant revoked mid-read must not leave the old result cached."""
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        token = cache.begin_read(("concepts", "role_assignments"))
        cache.drop_for_table("role_assignments")
        assert cache.put(key, _ROWS, token=token) is False
        assert cache.get(key) is None

    def test_an_unrelated_eviction_during_the_read_still_stores(self) -> None:
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        token = cache.begin_read(("concepts",))
        cache.drop_for_table("conversations")
        assert cache.put(key, _ROWS, token=token) is True
        assert cache.get(key) == _ROWS

    def test_an_eviction_before_the_token_does_not_refuse(self) -> None:
        """the read began after the eviction, so it saw the write; its result is safe."""
        cache = _cache()
        key = ScanCacheKey("concepts", "user-1")
        cache.drop_for_table("concepts")
        token = cache.begin_read(("concepts",))
        assert cache.put(key, _ROWS, token=token) is True
        assert cache.get(key) == _ROWS

    def test_a_token_from_another_cache_is_refused_loudly(self) -> None:
        """a token's counts belong to the cache that issued it; comparing them elsewhere is meaningless."""
        issuing, other = _cache(), _cache()
        token = issuing.begin_read(("concepts",))
        with pytest.raises(ValueError, match="different ScanCache"):
            other.put(ScanCacheKey("concepts", "user-1"), _ROWS, token=token)

    @pytest.mark.asyncio
    async def test_a_local_write_committing_mid_read_refuses_the_store(self) -> None:
        """the writing pod's own post-commit eviction runs through publish_invalidation."""
        registry: Any = _followed_registry("concepts", "role_assignments")
        key = ScanCacheKey("concepts", "user-1")
        token = registry.scan_cache.begin_read(("concepts",))

        await registry.publish_invalidation(None, "concepts", "some-id")

        assert registry.scan_cache.put(key, _ROWS, token=token) is False
        assert registry.scan_cache.get(key) is None
