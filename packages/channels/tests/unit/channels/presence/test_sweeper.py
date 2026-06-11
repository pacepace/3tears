"""unit tests for PresenceSweeper — heartbeat-driven self-heal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from threetears.channels.presence.room_state import RoomState
from threetears.channels.presence.sweeper import PresenceSweeper

from .conftest import InMemoryNatsBus, make_pod

pytestmark = pytest.mark.asyncio


def _setup(bus: InMemoryNatsBus) -> tuple[RoomState, PresenceSweeper]:
    collection, _ = make_pod(bus)
    state = RoomState(collection, pod_id="pod-a")
    sweeper = PresenceSweeper(collection, check_interval=100.0, timeout=30.0)
    return state, sweeper


async def _make_stale(state: RoomState, connection_id: str) -> None:
    """force a connection's heartbeat into the stale window."""
    coll = state._collection.connections  # noqa: SLF001 -- test reaches into the wired collection
    entity = await coll.get(connection_id)
    assert entity is not None
    entity.date_last_heartbeat = datetime.now(UTC) - timedelta(seconds=120)
    await coll.save_entity(entity)


class TestSweep:
    async def test_fresh_connection_survives(self, bus: InMemoryNatsBus) -> None:
        state, sweeper = _setup(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        sweeper.track("conn-1")
        evicted = await sweeper.run_sweep()
        assert evicted == []
        members = await state.members("cust:s:main:f.md")
        assert [m.connection_id for m in members] == ["conn-1"]
        assert "conn-1" in sweeper.known_connection_ids

    async def test_stale_connection_evicted_and_pruned(self, bus: InMemoryNatsBus) -> None:
        state, sweeper = _setup(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        sweeper.track("conn-1")
        await _make_stale(state, "conn-1")

        evicted = await sweeper.run_sweep()
        assert evicted == ["conn-1"]
        # pruned from the room index AND the connection row gone
        assert await state.members("cust:s:main:f.md") == []
        assert await state._collection.connections.get("conn-1") is None  # noqa: SLF001
        assert "conn-1" not in sweeper.known_connection_ids

    async def test_stale_one_of_two_only_evicts_stale(self, bus: InMemoryNatsBus) -> None:
        state, sweeper = _setup(bus)
        await state.join("cust:s:main:f.md", "conn-stale", "user-1", "cust")
        await state.join("cust:s:main:f.md", "conn-fresh", "user-2", "cust")
        sweeper.track_many(["conn-stale", "conn-fresh"])
        await _make_stale(state, "conn-stale")

        evicted = await sweeper.run_sweep()
        assert evicted == ["conn-stale"]
        remaining = [m.connection_id for m in await state.members("cust:s:main:f.md")]
        assert remaining == ["conn-fresh"]

    async def test_already_gone_connection_dropped_from_tracking(self, bus: InMemoryNatsBus) -> None:
        """a tracked connection whose row a peer already evicted is untracked."""
        state, sweeper = _setup(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        sweeper.track("conn-1")
        await state._collection.connections.delete("conn-1")  # noqa: SLF001 -- peer-evicted

        evicted = await sweeper.run_sweep()
        assert evicted == []
        assert "conn-1" not in sweeper.known_connection_ids

    async def test_empty_tracking_set_is_noop(self, bus: InMemoryNatsBus) -> None:
        _state, sweeper = _setup(bus)
        assert await sweeper.run_sweep() == []

    async def test_untrack_stops_examining(self, bus: InMemoryNatsBus) -> None:
        state, sweeper = _setup(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        sweeper.track("conn-1")
        sweeper.untrack("conn-1")
        await _make_stale(state, "conn-1")
        evicted = await sweeper.run_sweep()
        # untracked: the sweep does not see it, so it is not evicted by THIS sweeper
        assert evicted == []

    async def test_start_stop_lifecycle(self, bus: InMemoryNatsBus) -> None:
        _state, sweeper = _setup(bus)
        assert not sweeper.has_active_check_task
        await sweeper.start()
        assert sweeper.has_active_check_task
        await sweeper.stop()
        assert not sweeper.has_active_check_task
