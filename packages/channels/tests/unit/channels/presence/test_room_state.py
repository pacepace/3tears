"""unit tests for RoomState — socket map sync + membership surface.

covers the synchronized ``connection_id → live socket`` map (register /
unregister / snapshot iteration under concurrency), the
join/leave/heartbeat/members/local_sockets surface, and tenancy
isolation by room key.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.channels.presence.room_state import RoomMember, RoomState

from .conftest import InMemoryNatsBus, make_pod

pytestmark = pytest.mark.asyncio


class _FakeSocket:
    """a stand-in live socket handle."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


def _state(bus: InMemoryNatsBus, pod_id: str = "pod-a") -> RoomState:
    collection, _ = make_pod(bus)
    return RoomState(collection, pod_id=pod_id)


# ---------------------------------------------------------------------------
# socket map
# ---------------------------------------------------------------------------


class TestSocketMap:
    """the synchronized pod-local live-socket map."""

    async def test_register_and_resolve(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        sock = _FakeSocket("s1")
        await state.register("conn-1", sock)
        assert await state.local_socket("conn-1") is sock
        assert await state.local_connection_count() == 1

    async def test_unregister_removes(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        sock = _FakeSocket("s1")
        await state.register("conn-1", sock)
        await state.unregister("conn-1")
        assert await state.local_socket("conn-1") is None
        assert await state.local_connection_count() == 0

    async def test_unregister_unknown_is_safe(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        await state.unregister("ghost")
        assert await state.local_connection_count() == 0

    async def test_unregister_only_matching_socket(self, bus: InMemoryNatsBus) -> None:
        """a reconnect replaces the handle; unregistering the OLD one is a no-op."""
        state = _state(bus)
        old = _FakeSocket("old")
        new = _FakeSocket("new")
        await state.register("conn-1", old)
        await state.register("conn-1", new)  # reconnect replaced the handle
        await state.unregister("conn-1", old)  # stale handle: must NOT drop the new one
        assert await state.local_socket("conn-1") is new

    async def test_local_sockets_returns_snapshot_list(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        s1, s2 = _FakeSocket("s1"), _FakeSocket("s2")
        await state.register("conn-1", s1)
        await state.register("conn-2", s2)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        await state.join("cust:s:main:f.md", "conn-2", "user-2", "cust")
        sockets = await state.local_sockets("cust:s:main:f.md")
        assert set(id(s) for s in sockets) == {id(s1), id(s2)}

    async def test_local_sockets_skips_remote_members(self, bus: InMemoryNatsBus) -> None:
        """a member whose socket lives on another pod is not in local_sockets."""
        state = _state(bus, pod_id="pod-a")
        s1 = _FakeSocket("s1")
        await state.register("conn-local", s1)
        await state.join("cust:s:main:f.md", "conn-local", "user-1", "cust")
        # a member with NO local handle (it lives on a peer pod)
        await state.join("cust:s:main:f.md", "conn-remote", "user-2", "cust", pod_id="pod-b")
        sockets = await state.local_sockets("cust:s:main:f.md")
        assert sockets == [s1]


class TestSnapshotIterationNoRace:
    """join/leave churn concurrent with a local_sockets read must not race."""

    async def test_churn_during_reads_does_not_raise(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        room = "cust:s:main:f.md"
        # seed a pool of connections + sockets
        for i in range(20):
            await state.register(f"conn-{i}", _FakeSocket(f"s{i}"))
            await state.join(room, f"conn-{i}", f"user-{i}", "cust")

        stop = False

        async def churn() -> None:
            i = 0
            while not stop:
                cid = f"churn-{i}"
                await state.register(cid, _FakeSocket(cid))
                await state.join(room, cid, "user-c", "cust")
                await state.leave(room, cid)
                await state.unregister(cid)
                i += 1
                await asyncio.sleep(0)

        async def read() -> None:
            for _ in range(200):
                # must never raise "dict changed size during iteration" or KeyError
                sockets = await state.local_sockets(room)
                for s in sockets:  # iterate the snapshot
                    _ = s.name
                await asyncio.sleep(0)

        churn_task = asyncio.create_task(churn())
        try:
            await asyncio.gather(read(), read(), read())
        finally:
            stop = True
            await churn_task
        # the 20 seeded connections survive the churn
        assert await state.local_connection_count() >= 20


# ---------------------------------------------------------------------------
# membership surface
# ---------------------------------------------------------------------------


class TestMembership:
    """join/leave/members over the collection."""

    async def test_join_then_members(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        members = await state.members("cust:s:main:f.md")
        assert members == [RoomMember("conn-1", "user-1", "pod-a", "cust")]

    async def test_join_stamps_pod_id(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus, pod_id="pod-z")
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        members = await state.members("cust:s:main:f.md")
        assert members[0].pod_id == "pod-z"

    async def test_leave_removes_member_and_connection(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        await state.leave("cust:s:main:f.md", "conn-1")
        assert await state.members("cust:s:main:f.md") == []

    async def test_members_skips_connection_with_no_presence_row(self, bus: InMemoryNatsBus) -> None:
        """a member id in the room index whose presence row is gone is skipped."""
        collection, _ = make_pod(bus)
        state = RoomState(collection, pod_id="pod-a")
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        # simulate a mid-sweep window: connection row deleted, index not yet pruned
        await collection.connections.delete("conn-1")
        assert await state.members("cust:s:main:f.md") == []

    async def test_heartbeat_refreshes_and_returns_true(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        await state.join("cust:s:main:f.md", "conn-1", "user-1", "cust")
        assert await state.heartbeat("conn-1") is True

    async def test_heartbeat_unknown_returns_false(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        assert await state.heartbeat("ghost") is False


class TestTenancyIsolation:
    """two customers produce disjoint room keys — isolation holds."""

    async def test_distinct_customers_distinct_rooms(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        # same story/branch/file, different customer prefix → different room
        await state.join("custA:s:main:f.md", "conn-a", "user-1", "custA")
        await state.join("custB:s:main:f.md", "conn-b", "user-2", "custB")

        members_a = await state.members("custA:s:main:f.md")
        members_b = await state.members("custB:s:main:f.md")
        assert [m.connection_id for m in members_a] == ["conn-a"]
        assert [m.connection_id for m in members_b] == ["conn-b"]
        assert members_a[0].customer_id == "custA"
        assert members_b[0].customer_id == "custB"

    async def test_leave_one_customer_leaves_other_intact(self, bus: InMemoryNatsBus) -> None:
        state = _state(bus)
        await state.join("custA:s:main:f.md", "conn-a", "user-1", "custA")
        await state.join("custB:s:main:f.md", "conn-b", "user-2", "custB")
        await state.leave("custA:s:main:f.md", "conn-a")
        assert await state.members("custA:s:main:f.md") == []
        assert [m.connection_id for m in await state.members("custB:s:main:f.md")] == ["conn-b"]
