"""unit tests for the L1+L2-only presence collections.

covers the pk-keyed entry logic (per-connection + room-index), the L3
raise-loudly guards, entry serialize/deserialize round-trips, the
``l2_key`` colon-sanitization, and the room-index member CAS semantics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from threetears.channels.presence.collection import (
    PresenceConnectionCollection,
    RoomIndexCollection,
)

from .conftest import InMemoryNatsBus, make_pod

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# L3 raise-loudly guards (L1+L2 only)
# ---------------------------------------------------------------------------


class TestL3Guards:
    """presence is ephemeral; every L3 path must raise loudly."""

    @pytest.mark.parametrize("table", ["connections", "rooms"])
    async def test_fetch_from_postgres_raises(self, bus: InMemoryNatsBus, table: str) -> None:
        collection, _ = make_pod(bus)
        sub = getattr(collection, table)
        with pytest.raises(RuntimeError):
            await sub.fetch_from_postgres("anything")

    @pytest.mark.parametrize("table", ["connections", "rooms"])
    async def test_save_to_postgres_raises(self, bus: InMemoryNatsBus, table: str) -> None:
        collection, _ = make_pod(bus)
        sub = getattr(collection, table)
        with pytest.raises(RuntimeError):
            await sub.save_to_postgres({"x": 1})

    @pytest.mark.parametrize("table", ["connections", "rooms"])
    async def test_delete_from_postgres_raises(self, bus: InMemoryNatsBus, table: str) -> None:
        collection, _ = make_pod(bus)
        sub = getattr(collection, table)
        with pytest.raises(RuntimeError):
            await sub.delete_from_postgres("anything")

    async def test_l3_pool_is_none(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        assert collection.connections.l3_pool is None
        assert collection.rooms.l3_pool is None


# ---------------------------------------------------------------------------
# per-connection entry (pk = connection_id)
# ---------------------------------------------------------------------------


class TestConnectionEntry:
    """per-connection pk-keyed entry logic."""

    async def test_save_then_get_round_trips_identity(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        now = datetime.now(UTC)
        entity = collection.connections.create(
            {
                "connection_id": "conn-1",
                "room_id": "cust:story:main:scene.md",
                "user_id": "user-1",
                "pod_id": "pod-a",
                "customer_id": "cust",
                "date_last_heartbeat": now,
            }
        )
        await collection.connections.save_entity(entity)

        hit = await collection.connections.get("conn-1")
        assert hit is not None
        assert hit.connection_id == "conn-1"
        assert hit.room_id == "cust:story:main:scene.md"
        assert hit.user_id == "user-1"
        assert hit.pod_id == "pod-a"
        assert hit.customer_id == "cust"

    async def test_get_unknown_returns_none(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        assert await collection.connections.get("nope") is None

    async def test_heartbeat_refresh_only_touches_connection(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        old = datetime(2020, 1, 1, tzinfo=UTC)
        entity = collection.connections.create(
            {
                "connection_id": "conn-hb",
                "room_id": "cust:story:main:scene.md",
                "user_id": "user-1",
                "pod_id": "pod-a",
                "customer_id": "cust",
                "date_last_heartbeat": old,
            }
        )
        await collection.connections.save_entity(entity)
        fresh = await collection.connections.get("conn-hb")
        assert fresh is not None
        fresh.date_last_heartbeat = datetime.now(UTC)
        await collection.connections.save_entity(fresh)
        again = await collection.connections.get("conn-hb")
        assert again is not None
        assert again.date_last_heartbeat > old


# ---------------------------------------------------------------------------
# serialize / deserialize round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    """L2 codec round-trips, including datetime rehydration."""

    async def test_connection_serialize_deserialize_round_trip(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        coll = collection.connections
        now = datetime.now(UTC)
        row = {
            "connection_id": "conn-x",
            "room_id": "cust:story:main:scene.md",
            "user_id": "user-1",
            "pod_id": "pod-a",
            "customer_id": "cust",
            "date_last_heartbeat": now,
            "date_created": now,
            "date_updated": now,
        }
        restored = coll.deserialize(coll.serialize(row))
        assert restored["connection_id"] == "conn-x"
        # datetimes come back as aware-UTC datetime objects, not strings
        assert isinstance(restored["date_last_heartbeat"], datetime)
        assert restored["date_last_heartbeat"].tzinfo is not None
        assert restored["date_last_heartbeat"] == now

    async def test_naive_datetime_coerced_to_aware_utc(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        coll = collection.connections
        naive_iso = "2024-06-01T12:00:00"
        payload = json.dumps({"connection_id": "c", "date_last_heartbeat": naive_iso}).encode()
        restored = coll.deserialize(payload)
        assert restored["date_last_heartbeat"].tzinfo is UTC

    async def test_room_members_round_trip(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        coll = collection.rooms
        now = datetime.now(UTC)
        row = {
            "room_id": "cust:story:main:scene.md",
            "customer_id": "cust",
            "members": ["conn-1", "conn-2"],
            "date_created": now,
            "date_updated": now,
        }
        restored = coll.deserialize(coll.serialize(row))
        assert restored["members"] == ["conn-1", "conn-2"]


# ---------------------------------------------------------------------------
# l2_key colon sanitization
# ---------------------------------------------------------------------------


class TestRoomKeySanitization:
    """the room id must map to a JetStream-grammar-safe, collision-safe KV key."""

    async def test_room_l2_key_is_grammar_safe_and_collision_safe(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        rooms = collection.rooms
        key = rooms.l2_key("cust:story-7:main:chapter/scene.md")

        # SHA-256 hex key: no colon, grammar-valid (hex), scoped by table.
        prefix, _, digest = key.partition(".")
        assert prefix == "presence_rooms"
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        assert ":" not in key

        # deterministic — reads/writes/CAS must resolve the same key.
        assert key == rooms.l2_key("cust:story-7:main:chapter/scene.md")

        # collision-safe: two ids that the old ':'->'=' replace would collide
        # onto one key ("x=y=z") must map to DISTINCT keys.
        assert rooms.l2_key("x=y:z") != rooms.l2_key("x:y=z")

        # out-of-grammar characters (a space) still produce a valid key (no KvError).
        space_digest = rooms.l2_key("cust:s:main:my file.md").partition(".")[2]
        assert len(space_digest) == 64 and all(c in "0123456789abcdef" for c in space_digest)

    async def test_connection_l2_key_is_plain(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        key = collection.connections.l2_key("conn-1")
        assert key == "presence_connections.conn-1"


# ---------------------------------------------------------------------------
# room-index member CAS
# ---------------------------------------------------------------------------


class TestRoomIndexMembers:
    """room-index member add/remove + the empty-room cleanup."""

    async def test_add_member_creates_room_index(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        assert await collection.rooms.members("cust:s:main:f.md") == ["conn-1"]

    async def test_add_member_is_idempotent(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        assert await collection.rooms.members("cust:s:main:f.md") == ["conn-1"]

    async def test_add_multiple_members(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-2")
        assert set(await collection.rooms.members("cust:s:main:f.md")) == {"conn-1", "conn-2"}

    async def test_remove_member(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-2")
        await collection.rooms.remove_member("cust:s:main:f.md", "conn-1")
        assert await collection.rooms.members("cust:s:main:f.md") == ["conn-2"]

    async def test_remove_last_member_deletes_room_index(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.remove_member("cust:s:main:f.md", "conn-1")
        assert await collection.rooms.members("cust:s:main:f.md") == []
        assert await collection.rooms.get("cust:s:main:f.md") is None

    async def test_remove_unknown_member_is_noop(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.remove_member("cust:s:main:f.md", "ghost")
        assert await collection.rooms.members("cust:s:main:f.md") == ["conn-1"]

    async def test_remove_from_missing_room_is_noop(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        await collection.rooms.remove_member("cust:s:main:absent.md", "conn-1")
        assert await collection.rooms.members("cust:s:main:absent.md") == []

    async def test_members_of_missing_room_is_empty(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        assert await collection.rooms.members("cust:s:main:none.md") == []


class TestRoomIndexL1Only:
    """member mutation must work with no NATS client (single-pod / unit mode)."""

    async def test_add_and_remove_l1_only(self) -> None:
        collection, _ = make_pod(None)
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-1")
        await collection.rooms.add_member("cust:s:main:f.md", "cust", "conn-2")
        assert set(await collection.rooms.members("cust:s:main:f.md")) == {"conn-1", "conn-2"}
        await collection.rooms.remove_member("cust:s:main:f.md", "conn-1")
        assert await collection.rooms.members("cust:s:main:f.md") == ["conn-2"]
        await collection.rooms.remove_member("cust:s:main:f.md", "conn-2")
        assert await collection.rooms.get("cust:s:main:f.md") is None


class TestPrimaryKeyShape:
    """the two collections stay strictly single-pk-keyed."""

    async def test_connection_pk(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        assert isinstance(collection.connections, PresenceConnectionCollection)
        assert collection.connections.primary_key_columns == ("connection_id",)

    async def test_room_pk(self, bus: InMemoryNatsBus) -> None:
        collection, _ = make_pod(bus)
        assert isinstance(collection.rooms, RoomIndexCollection)
        assert collection.rooms.primary_key_columns == ("room_id",)
