"""the access tables are followed and the cache derived from them is bound, in one call.

The watches themselves are ``threetears.epoch.TableFollower``'s, tested with it; what is pinned here
is the half that belongs to access control: which tables are followed, and that the cache is trusted
only while their watches run.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from threetears.agent.acl import ACCESS_TABLES
from threetears.core.collections import CollectionRegistry


async def _listening_registry() -> CollectionRegistry:
    """a registry whose invalidation listener runs, as a follower requires."""
    from threetears.core.testing.kv import FakeNatsClient

    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="pod")
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return registry


class _Pushing:
    """a watcher that pushes one generation per table and then waits."""

    async def watch(self, table_name: str) -> Any:
        yield "inc:1"
        await asyncio.Event().wait()


async def test_a_watch_that_is_pushed_a_value_is_healthy() -> None:
    from threetears.agent.acl.generation_follow import follow_access_tables
    from threetears.agent.acl import AclCache

    from threetears.core.testing.kv import FakeNatsClient

    registry = CollectionRegistry()
    bus = FakeNatsClient()
    registry.configure(l2_client=bus, kv_key_scope="pod")
    loader: Any = object()
    cache = AclCache(membership_loader=loader, grant_loader=loader)
    with pytest.raises(RuntimeError, match="invalidation listener"):
        follow_access_tables(registry, cache, _Pushing())
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    following = follow_access_tables(registry, cache, _Pushing())
    try:
        for _ in range(100):
            if following.healthy:
                break
            await asyncio.sleep(0.01)
        assert following.healthy
        assert all(h.pushes >= 1 for h in following.follower.health.values())
        assert set(following.follower.health) == set(ACCESS_TABLES)
        assert all(registry.generation_marks.follows(table) for table in ACCESS_TABLES)
        assert registry.has_derived_caches("group_members")
    finally:
        await following.stop()
    assert not registry.has_derived_caches("group_members")
    assert not following.healthy


class _FailsWhenTold:
    """a watcher whose every watch runs until told to fail, then fails every time."""

    def __init__(self) -> None:
        self.fail = asyncio.Event()

    async def watch(self, table_name: str) -> Any:
        await self.fail.wait()
        raise ConnectionError("the bucket could not be reached")
        yield "never"  # pragma: no cover


async def test_an_acl_cache_is_trusted_only_while_its_watches_run_and_never_after_they_stop() -> None:
    """with no age on any entry, a cache whose watches fail could serve a missed write forever."""
    from uuid import uuid4

    from threetears.agent.acl import AclCache
    from threetears.agent.acl.cache import ActorMembershipKey
    from threetears.agent.acl.generation_follow import follow_access_tables

    watches = _FailsWhenTold()
    loader: Any = object()
    cache = AclCache(membership_loader=loader, grant_loader=loader)
    following = follow_access_tables(
        await _listening_registry(),
        cache,
        watches,
        restart_delay=timedelta(milliseconds=1),
    )
    key = ActorMembershipKey(actor_kind="user", actor_id=uuid4())
    try:
        assert cache.trusted
        cache.put_membership(key, ())
        assert cache.get_membership(key) is not None
        # the watches fail from here: nothing held is served, and nothing is kept
        watches.fail.set()
        for _ in range(200):
            if not following.follower.watching:
                break
            await asyncio.sleep(0.005)
        assert not following.follower.watching
        assert cache.get_membership(key) is None
        assert cache.size == 0
        cache.put_membership(key, ())
        assert cache.get_membership(key) is None
    finally:
        await following.stop()
    assert not cache.trusted
    assert cache.size == 0
