"""a presence entry in L2 of another shape reads as absent, never as this release's row.

The presence collections keep their rows in L2 alone (``l2_is_the_store``): there is no L3 to refill from. An entry an
older release wrote (no shape tag) or another shape's is ignored and counted, and the connection or room reads as absent
until it is written again (``threetears.core.collections.l2_shape``). An entry of its own shape is read by every pod.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from prometheus_client import REGISTRY

from .conftest import InMemoryNatsBus, make_pod

pytestmark = pytest.mark.asyncio

_ROOM = "cust:story:main:scene.md"


def _mismatches(table: str) -> float:
    return REGISTRY.get_sample_value("threetears_collection_l2_shape_mismatches_total", {"table": table}) or 0.0


def _connection() -> dict[str, Any]:
    return {
        "connection_id": "conn-shape",
        "room_id": _ROOM,
        "user_id": "user-1",
        "pod_id": "pod-a",
        "customer_id": "cust",
        "date_last_heartbeat": datetime.now(UTC),
    }


class TestAConnectionOfAnotherShape:
    async def test_reads_as_absent_and_is_counted(self, bus: InMemoryNatsBus) -> None:
        writer, _ = make_pod(bus)
        coll = writer.connections
        bucket = await bus.kv_bucket(name=coll.L2_BUCKET_SUFFIX)
        # as every release before 0.95.0 wrote it: the codec's bytes, no tag
        await bucket.put(key=coll.l2_key("conn-shape"), value=coll.serialize(_connection()))
        before = _mismatches(coll.table_name)

        reader, _ = make_pod(bus)
        assert await reader.connections.get("conn-shape") is None
        assert _mismatches(coll.table_name) == before + 1

    async def test_a_saved_connection_is_read_by_another_pod(self, bus: InMemoryNatsBus) -> None:
        writer, _ = make_pod(bus)
        await writer.connections.save_entity(writer.connections.create(_connection()))

        reader, _ = make_pod(bus)
        hit = await reader.connections.get("conn-shape")

        assert hit is not None and hit.user_id == "user-1"


class TestARoomOfAnotherShape:
    async def test_a_member_added_over_it_starts_the_room_afresh(self, bus: InMemoryNatsBus) -> None:
        writer, _ = make_pod(bus)
        rooms = writer.rooms
        bucket = await bus.kv_bucket(name=rooms.L2_BUCKET_SUFFIX)
        now = datetime.now(UTC)
        older = {
            "room_id": _ROOM,
            "customer_id": "cust",
            "members": ["conn-gone"],
            "date_created": now,
            "date_updated": now,
        }
        await bucket.put(key=rooms.l2_key(_ROOM), value=rooms.serialize(older))
        before = _mismatches(rooms.table_name)

        reader, _ = make_pod(bus)
        await reader.rooms.add_member(_ROOM, "cust", "conn-1")

        raw = await bucket.get(key=rooms.l2_key(_ROOM))
        assert raw is not None
        row = reader.rooms.decode_l2_entry(raw)
        assert row is not None and row["members"] == ["conn-1"], "a member list of another shape was carried on"
        assert _mismatches(rooms.table_name) == before + 1
