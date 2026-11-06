"""a person's membership cache and the shape answers' dependencies are each bound and followed in one call.

What is pinned: which tables each follows, that each is trusted only while its watches run, that the
dependencies start from the generation the first watch judged, and that stopping unbinds.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.acl import ACCESS_TABLES, AccessShape, CallerAccessCache, CallerKey, ShapeDependencies
from threetears.agent.acl.generation_follow import follow_access_shape_answers, follow_caller_memberships
from threetears.core.collections import CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient


async def _listening_registry() -> CollectionRegistry:
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="hub")
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return registry


class _Pushing:
    """a watcher that pushes one generation per table and then waits."""

    async def watch(self, table_name: str) -> Any:
        yield "inc:3"
        await asyncio.Event().wait()


async def _until(condition: Any) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.005)


async def test_a_membership_cache_follows_group_members_alone() -> None:
    registry = await _listening_registry()
    cache: CallerAccessCache[AccessShape] = CallerAccessCache()
    following = follow_caller_memberships(registry, cache, _Pushing())
    key = CallerKey(uuid4(), uuid4())
    try:
        await _until(lambda: following.healthy)
        assert set(following.follower.health) == {"group_members"}
        assert cache.trusted
        assert cache.put(key, AccessShape.of([]), fence=cache.read_fence())
        assert registry.has_derived_caches("group_members")
        assert not registry.has_derived_caches("role_assignments")
    finally:
        await following.stop()
    assert not cache.trusted
    assert cache.get(key) is None
    assert not registry.has_derived_caches("group_members")


async def test_shape_dependencies_follow_every_table_and_start_from_the_judged_generation() -> None:
    registry = await _listening_registry()
    tables = (*ACCESS_TABLES, "rest_shares")
    dependencies = ShapeDependencies(registry, tables)
    assert not dependencies.trusted
    following = follow_access_shape_answers(registry, dependencies, _Pushing())
    try:
        await _until(lambda: dependencies.trusted)
        assert dependencies.trusted
        assert set(following.follower.health) == set(tables)
        assert dependencies.stamp() == dict.fromkeys(tables, "inc:3")
        assert all(registry.has_derived_caches(table) for table in tables)
    finally:
        await following.stop()
    assert not dependencies.trusted
    assert not any(registry.has_derived_caches(table) for table in tables)


async def test_following_needs_the_listener() -> None:
    registry = CollectionRegistry()
    registry.configure(l2_client=FakeNatsClient(), kv_key_scope="hub")
    with pytest.raises(RuntimeError, match="invalidation listener"):
        follow_access_shape_answers(registry, ShapeDependencies(registry, ACCESS_TABLES), _Pushing())
    with pytest.raises(RuntimeError, match="invalidation listener"):
        follow_caller_memberships(registry, CallerAccessCache(), _Pushing())
