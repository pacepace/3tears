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
    assert not follower.watching


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


class _Silent:
    """a watcher on keys never written: running, pushed nothing."""

    async def watch(self, table_name: str) -> Any:
        await asyncio.Event().wait()
        yield "never"


async def test_a_watch_on_a_key_never_written_is_watching_but_not_yet_healthy() -> None:
    follower = AccessTableFollower(await _listening_registry(), _Silent())  # type: ignore[arg-type]
    assert not follower.watching
    follower.start()
    try:
        await asyncio.sleep(0.02)
        assert follower.watching
        assert not follower.healthy
    finally:
        await follower.stop()
    assert not follower.watching


class _FailsWhenTold:
    """stands in for ``follow_generation_key``: watches until told to fail, then fails every time."""

    def __init__(self) -> None:
        self.fail = asyncio.Event()

    async def __call__(self, registry: Any, reader: Any, table: str, *, grace: timedelta) -> None:
        await self.fail.wait()
        raise ConnectionError("the bucket could not be reached")


async def test_an_acl_cache_is_trusted_only_while_its_watches_run_and_never_after_they_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with no age on any entry, a cache whose watches fail could serve a missed write forever."""
    from uuid import uuid4

    from threetears.agent.acl import AclCache
    from threetears.agent.acl.cache import ActorMembershipKey
    from threetears.agent.acl.generation_follow import follow_access_tables

    watches = _FailsWhenTold()
    monkeypatch.setattr(generation_follow, "follow_generation_key", watches)
    loader: Any = object()
    cache = AclCache(membership_loader=loader, grant_loader=loader)
    following = follow_access_tables(
        await _listening_registry(),
        cache,
        object(),
        restart_delay=timedelta(milliseconds=1),  # type: ignore[arg-type]
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


class _ClosedConnection:
    """a watcher whose connection is closed: every watch fails at once, as on a drained client."""

    closed = True

    def __init__(self) -> None:
        self.attempts = 0

    async def watch(self, table_name: str) -> Any:
        self.attempts += 1
        raise ConnectionError(f"cannot watch {table_name}: the NATS connection is closed")
        yield "never"  # pragma: no cover


async def test_a_watch_whose_connection_closed_stops_instead_of_retrying() -> None:
    """a pod's shutdown drains its client; a retry loop against it kept the process alive."""
    reader = _ClosedConnection()
    follower = AccessTableFollower(
        await _listening_registry(),
        reader,  # type: ignore[arg-type]
        tables=("groups",),
        restart_delay=timedelta(milliseconds=1),
    )
    follower.start()
    for _ in range(100):
        if follower.health["groups"].consecutive_failures:
            break
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)
    assert reader.attempts == 1
    assert not follower.watching
    await follower.stop()


class _IgnoresCancellation:
    """stands in for ``follow_generation_key`` stuck in a call that swallows cancellation, until released."""

    def __init__(self) -> None:
        self.released = asyncio.Event()

    async def __call__(self, registry: Any, reader: Any, table: str, *, grace: timedelta) -> None:
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue
        raise asyncio.CancelledError


async def test_stop_returns_within_its_bound_even_when_a_watch_will_not_end(monkeypatch: pytest.MonkeyPatch) -> None:
    stubborn = _IgnoresCancellation()
    monkeypatch.setattr(generation_follow, "follow_generation_key", stubborn)
    follower = AccessTableFollower(
        await _listening_registry(),
        object(),  # type: ignore[arg-type]
        tables=("groups",),
        stop_timeout=timedelta(milliseconds=50),
    )
    follower.start()
    await asyncio.sleep(0.01)
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(follower.stop(), timeout=2)
    assert asyncio.get_running_loop().time() - started < 1
    assert not follower.running
    stubborn.released.set()
    await asyncio.sleep(0.01)


async def test_the_epoch_reader_says_when_its_client_is_closed() -> None:
    from unittest.mock import MagicMock

    from threetears.epoch import EpochGenerationReader

    client = MagicMock()
    client.is_closed = False
    reader = EpochGenerationReader(client)
    assert reader.closed is False
    client.is_closed = True
    assert reader.closed is True


async def test_the_registry_trusts_a_followed_table_only_while_its_watch_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the scan cache has no age: it may hold a scan only while every table it read is watched."""
    from threetears.agent.acl.generation_follow import follow_tables

    scanned = ("concepts", "playbook_entries")
    watches = _FailsWhenTold()
    monkeypatch.setattr(generation_follow, "follow_generation_key", watches)
    registry = await _listening_registry()
    assert not registry.tables_trusted(scanned)
    following = follow_tables(
        registry,
        object(),  # type: ignore[arg-type]
        scanned,
        restart_delay=timedelta(milliseconds=1),
    )
    try:
        assert following.running
        assert registry.tables_trusted(scanned)
        assert not registry.tables_trusted((*scanned, "role_assignments")), "not followed here"
        watches.fail.set()
        for _ in range(200):
            if not registry.tables_trusted(scanned):
                break
            await asyncio.sleep(0.005)
        assert not registry.tables_trusted(scanned)
    finally:
        await following.stop()
    assert not registry.tables_trusted(scanned)


async def test_a_second_follower_of_a_table_neither_hides_nor_withdraws_the_first() -> None:
    from threetears.agent.acl.generation_follow import follow_tables

    watches = _Silent()
    registry = await _listening_registry()
    first = follow_tables(registry, watches, ("concepts",))  # type: ignore[arg-type]
    second = follow_tables(registry, watches, ("concepts",))  # type: ignore[arg-type]
    try:
        assert registry.tables_trusted(("concepts",))
        await second.stop()
        assert registry.tables_trusted(("concepts",)), "the first follower still watches"
    finally:
        await first.stop()
    assert not registry.tables_trusted(("concepts",))
