"""the access-table follower keeps one watch per table running for as long as it is started.

A watch ends when its connection closes and fails when the bucket cannot be reached; a table left
followed with no watch has nothing to judge its mark, so a missed broadcast would never be caught.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from threetears.agent.acl import ACCESS_TABLES
from threetears.agent.acl import generation_follow
from threetears.agent.acl.generation_follow import AccessTableFollower
from threetears.core.collections import CollectionRegistry


async def _listening_registry() -> CollectionRegistry:
    """a registry whose invalidation listener runs, as a follower requires."""
    from threetears.core.testing.kv import FakeNatsClient

    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="pod")
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return registry


def test_a_follower_refuses_to_start_before_the_listener_runs() -> None:
    follower = AccessTableFollower(CollectionRegistry(), object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="invalidation listener"):
        follower.start()
    assert not follower.running


class _Watches:
    """stands in for ``follow_generation_key``: each table's first watch fails, the next one ends, the third runs."""

    def __init__(self) -> None:
        self.started: dict[str, int] = {}

    async def __call__(self, registry: Any, reader: Any, table: str, *, grace: timedelta) -> None:
        count = self.started[table] = self.started.get(table, 0) + 1
        if count == 1:
            raise ConnectionError("the bucket could not be reached")
        if count == 2:
            return
        await asyncio.Event().wait()


async def test_every_table_is_followed_and_a_failed_or_ended_watch_is_started_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watches = _Watches()
    monkeypatch.setattr(generation_follow, "follow_generation_key", watches)
    registry = await _listening_registry()
    follower = AccessTableFollower(registry, object(), restart_delay=timedelta(milliseconds=1))  # type: ignore[arg-type]
    follower.start()
    try:
        for table in ACCESS_TABLES:
            assert registry.generation_marks.follows(table)
        for _ in range(200):
            if all(watches.started.get(table) == 3 for table in ACCESS_TABLES):
                break
            await asyncio.sleep(0.005)
        assert {table: watches.started.get(table) for table in ACCESS_TABLES} == dict.fromkeys(ACCESS_TABLES, 3)
        follower.start()  # a no-op while running
        assert all(count == 3 for count in watches.started.values())
    finally:
        await follower.stop()
    assert not follower.running
    await follower.stop()  # idempotent


class _FailingWatches:
    """every watch fails at once; records when each attempt started."""

    def __init__(self) -> None:
        self.started: list[float] = []

    async def __call__(self, registry: Any, reader: Any, table: str, *, grace: timedelta) -> None:
        self.started.append(asyncio.get_running_loop().time())
        raise ConnectionError("no grant on the epoch bucket")


async def test_a_watch_that_keeps_failing_backs_off_to_a_cap_and_reports_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watches = _FailingWatches()
    monkeypatch.setattr(generation_follow, "follow_generation_key", watches)
    follower = AccessTableFollower(
        await _listening_registry(),
        object(),  # type: ignore[arg-type]
        tables=("groups",),
        restart_delay=timedelta(milliseconds=10),
        max_restart_delay=timedelta(milliseconds=40),
    )
    follower.start()
    try:
        while len(watches.started) < 6:
            await asyncio.sleep(0.005)
    finally:
        await follower.stop()
    gaps = [later - earlier for earlier, later in zip(watches.started, watches.started[1:], strict=False)]
    # 10ms, 20ms, 40ms, then held at the 40ms cap
    assert gaps[1] > gaps[0] * 1.5
    assert all(gap < 0.04 * 3 for gap in gaps)
    health = follower.health["groups"]
    assert health.consecutive_failures >= 5
    assert health.last_error is not None and "no grant" in health.last_error
    assert not follower.healthy


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
        assert registry.has_derived_caches("group_members")
    finally:
        await following.stop()
    assert not registry.has_derived_caches("group_members")
    assert not following.healthy
