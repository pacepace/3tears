"""Tests for ThreeTierCheckpointSaver.

Tests the serialization helpers, protocol interactions, and sync-method guards.
Full integration tests require a Postgres instance and are in the host app.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver, _UUIDSafeSerializer


class TestUUIDSafeSerializer:
    """_UUIDSafeSerializer sanitizes uuid_utils.UUID objects."""

    def test_roundtrip_simple(self):
        serde = _UUIDSafeSerializer()
        data = {"key": "value", "num": 42, "nested": {"list": [1, 2, 3]}}
        typed = serde.dumps_typed(data)
        result = serde.loads_typed(typed)
        assert result == data

    def test_sanitizes_uuid_utils(self):
        import uuid_utils

        serde = _UUIDSafeSerializer()
        uid = uuid_utils.uuid7()
        data = {"id": uid, "nested": {"ids": [uid]}}
        typed = serde.dumps_typed(data)
        result = serde.loads_typed(typed)
        assert result["id"] == str(uid)
        assert result["nested"]["ids"][0] == str(uid)

    def test_sanitizes_tuple(self):
        import uuid_utils

        serde = _UUIDSafeSerializer()
        uid = uuid_utils.uuid7()
        sanitized = _UUIDSafeSerializer._sanitize((uid, "hello"))
        assert sanitized == (str(uid), "hello")


class TestCacheSerializationHelpers:
    """Test serialize/deserialize checkpoint tuple for cache storage."""

    def _make_saver(self) -> ThreeTierCheckpointSaver:
        pool = MagicMock()
        return ThreeTierCheckpointSaver(postgres_pool=pool)

    def test_roundtrip(self):
        saver = self._make_saver()
        checkpoint = {"id": "cp-123", "ts": "2026-01-01", "channel_values": {}}
        metadata = {"source": "loop", "step": 1}
        parent_id = "cp-122"
        pending = [("task-1", "messages", {"content": "hello"})]

        blob = saver._serialize_checkpoint_tuple(checkpoint, metadata, parent_id, pending)
        result = saver._deserialize_checkpoint_tuple(blob)

        assert result["checkpoint"]["id"] == "cp-123"
        assert result["metadata"]["source"] == "loop"
        assert result["parent_checkpoint_id"] == "cp-122"
        assert len(result["pending_writes"]) == 1

    def test_roundtrip_no_parent(self):
        saver = self._make_saver()
        checkpoint = {"id": "cp-1", "ts": "2026-01-01", "channel_values": {}}
        metadata = {}

        blob = saver._serialize_checkpoint_tuple(checkpoint, metadata, None, [])
        result = saver._deserialize_checkpoint_tuple(blob)

        assert result["parent_checkpoint_id"] is None
        assert result["pending_writes"] == []


class TestL1Degradation:
    """L1 cache failures degrade gracefully."""

    async def test_l1_get_returns_none_on_error(self):
        l1 = AsyncMock()
        l1.get.side_effect = RuntimeError("L1 down")

        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool, l1_cache=l1)

        result = await saver._l1_get("thread-1", "")
        assert result is None

    async def test_l1_put_swallows_error(self):
        l1 = AsyncMock()
        l1.put.side_effect = RuntimeError("L1 down")

        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool, l1_cache=l1)

        # Should not raise
        await saver._l1_put("thread-1", "", b"data")

    async def test_l1_delete_swallows_error(self):
        l1 = AsyncMock()
        l1.delete.side_effect = RuntimeError("L1 down")

        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool, l1_cache=l1)

        await saver._l1_delete("thread-1")


class TestL2Degradation:
    """L2 cache failures degrade gracefully."""

    async def test_l2_get_returns_none_on_error(self):
        l2 = AsyncMock()
        l2.get.side_effect = RuntimeError("L2 down")

        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool, l2_cache=l2)

        result = await saver._l2_get("thread-1", "")
        assert result is None

    async def test_l2_key_with_ns(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        assert saver._l2_key("thread-1", "") == "thread-1"
        assert saver._l2_key("thread-1", "ns1") == "thread-1.ns1"


class TestNoCacheProvided:
    """When no L1/L2 provided, all cache ops are no-ops."""

    async def test_l1_ops_are_noop(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        assert await saver._l1_get("t", "") is None
        await saver._l1_put("t", "", b"data")  # no-op
        await saver._l1_delete("t")  # no-op

    async def test_l2_ops_are_noop(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        assert await saver._l2_get("t", "") is None
        await saver._l2_put("t", "", b"data")  # no-op
        await saver._l2_delete("t")  # no-op


class TestSyncMethodsRaise:
    """Sync methods raise NotImplementedError."""

    def test_get_tuple_raises(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        with pytest.raises(NotImplementedError):
            saver.get_tuple({"configurable": {"thread_id": "t1"}})

    def test_list_raises(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        with pytest.raises(NotImplementedError):
            list(saver.list(None))

    def test_put_raises(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        with pytest.raises(NotImplementedError):
            saver.put({"configurable": {"thread_id": "t1"}}, {}, {}, {})

    def test_put_writes_raises(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        with pytest.raises(NotImplementedError):
            saver.put_writes({"configurable": {"thread_id": "t1"}}, [], "task-1")

    def test_delete_thread_raises(self):
        pool = MagicMock()
        saver = ThreeTierCheckpointSaver(postgres_pool=pool)

        with pytest.raises(NotImplementedError):
            saver.delete_thread("t1")
