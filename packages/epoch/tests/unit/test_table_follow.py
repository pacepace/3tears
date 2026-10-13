"""the table follower keeps one watch per table running for as long as it is started.

A watch ends when its connection closes and fails when the bucket cannot be reached; a table left
followed with no watch has nothing to judge its mark, so a missed broadcast would never be caught.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.collections import CollectionRegistry
from threetears.epoch import TableFollower, follow_tables, table_follow

#: any tables: the follower knows nothing about what they hold
_TABLES = ("concepts", "playbook_entries", "groups")


async def _listening_registry() -> CollectionRegistry:
    """a registry whose invalidation listener runs, as a follower requires."""
    from threetears.core.testing.kv import FakeNatsClient

    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="pod")
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return registry


def test_a_follower_refuses_to_start_before_the_listener_runs() -> None:
    follower = TableFollower(CollectionRegistry(), object(), _TABLES)  # type: ignore[arg-type]
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
    monkeypatch.setattr(table_follow, "follow_generation_key", watches)
    registry = await _listening_registry()
    follower = TableFollower(registry, object(), _TABLES, restart_delay=timedelta(milliseconds=1))  # type: ignore[arg-type]
    follower.start()
    try:
        for table in _TABLES:
            assert registry.generation_marks.follows(table)
        for _ in range(200):
            if all(watches.started.get(table) == 3 for table in _TABLES):
                break
            await asyncio.sleep(0.005)
        assert {table: watches.started.get(table) for table in _TABLES} == dict.fromkeys(_TABLES, 3)
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
    monkeypatch.setattr(table_follow, "follow_generation_key", watches)
    follower = TableFollower(
        await _listening_registry(),
        object(),  # type: ignore[arg-type]
        ("groups",),
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


class _Silent:
    """a watcher on keys never written: running, pushed nothing."""

    async def watch(self, table_name: str) -> Any:
        await asyncio.Event().wait()
        yield "never"


async def test_a_watch_on_a_key_never_written_is_watching_but_not_yet_healthy() -> None:
    follower = TableFollower(await _listening_registry(), _Silent(), _TABLES)  # type: ignore[arg-type]
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
    follower = TableFollower(
        await _listening_registry(),
        reader,  # type: ignore[arg-type]
        ("groups",),
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
    monkeypatch.setattr(table_follow, "follow_generation_key", stubborn)
    follower = TableFollower(
        await _listening_registry(),
        object(),  # type: ignore[arg-type]
        ("groups",),
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
    scanned = ("concepts", "playbook_entries")
    watches = _FailsWhenTold()
    monkeypatch.setattr(table_follow, "follow_generation_key", watches)
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
