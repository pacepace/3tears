"""Tests for ThreeTierCheckpointSaver.

Tests the serialization helpers, protocol interactions, and sync-method guards.
Full integration tests require a Postgres instance and are in the host app.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.protocols import AsyncpgPoolAdapter
from threetears.langgraph.serde import UUIDSafeSerializer


def _make_executor() -> Any:
    """build a MagicMock standing in for an AsyncQueryExecutor.

    every protocol method is an AsyncMock so call sites can assert
    invocation counts and arguments without a live database.

    :return: mock executor with async fetch/fetchrow/execute
    :rtype: Any
    """
    executor = MagicMock()
    executor.fetch = AsyncMock(return_value=[])
    executor.fetchrow = AsyncMock(return_value=None)
    executor.execute = AsyncMock(return_value="INSERT 0 1")
    return executor


class TestUUIDSafeSerializer:
    """UUIDSafeSerializer sanitizes uuid_utils.UUID objects."""

    def test_roundtrip_simple(self):
        serde = UUIDSafeSerializer()
        data = {"key": "value", "num": 42, "nested": {"list": [1, 2, 3]}}
        typed = serde.dumps_typed(data)
        result = serde.loads_typed(typed)
        assert result == data

    def testsanitizes_uuid_utils(self):
        import uuid_utils

        serde = UUIDSafeSerializer()
        uid = uuid_utils.uuid7()
        data = {"id": uid, "nested": {"ids": [uid]}}
        typed = serde.dumps_typed(data)
        result = serde.loads_typed(typed)
        assert result["id"] == str(uid)
        assert result["nested"]["ids"][0] == str(uid)

    def testsanitizes_tuple(self):
        import uuid_utils

        uid = uuid_utils.uuid7()
        sanitized = UUIDSafeSerializer.sanitize((uid, "hello"))
        assert sanitized == (str(uid), "hello")


class TestCacheSerializationHelpers:
    """Test serialize/deserialize checkpoint tuple for cache storage."""

    def _make_saver(self) -> ThreeTierCheckpointSaver:
        return ThreeTierCheckpointSaver(executor=_make_executor())

    def test_roundtrip(self):
        saver = self._make_saver()
        checkpoint = {"id": "cp-123", "ts": "2026-01-01", "channel_values": {}}
        metadata = {"source": "loop", "step": 1}
        parent_id = "cp-122"
        pending = [("task-1", "messages", {"content": "hello"})]

        blob = saver.serialize_checkpoint_tuple(checkpoint, metadata, parent_id, pending)
        result = saver.deserialize_checkpoint_tuple(blob)

        assert result["checkpoint"]["id"] == "cp-123"
        assert result["metadata"]["source"] == "loop"
        assert result["parent_checkpoint_id"] == "cp-122"
        assert len(result["pending_writes"]) == 1

    def test_roundtrip_no_parent(self):
        saver = self._make_saver()
        checkpoint = {"id": "cp-1", "ts": "2026-01-01", "channel_values": {}}
        metadata = {}

        blob = saver.serialize_checkpoint_tuple(checkpoint, metadata, None, [])
        result = saver.deserialize_checkpoint_tuple(blob)

        assert result["parent_checkpoint_id"] is None
        assert result["pending_writes"] == []


class TestL1Degradation:
    """L1 cache failures degrade gracefully."""

    async def testl1_get_returns_none_on_error(self):
        l1 = AsyncMock()
        l1.get.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1)

        result = await saver.l1_get("thread-1", "")
        assert result is None

    async def testl1_put_swallows_error(self):
        l1 = AsyncMock()
        l1.put.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1)

        # Should not raise
        await saver.l1_put("thread-1", "", b"data")

    async def testl1_delete_swallows_error(self):
        l1 = AsyncMock()
        l1.delete.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1)

        await saver.l1_delete("thread-1")


class TestL2Degradation:
    """L2 cache failures degrade gracefully."""

    async def testl2_get_returns_none_on_error(self):
        l2 = AsyncMock()
        l2.get.side_effect = RuntimeError("L2 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2)

        result = await saver.l2_get("thread-1", "")
        assert result is None

    async def testl2_key_with_ns(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert saver.l2_key("thread-1", "") == "thread-1"
        assert saver.l2_key("thread-1", "ns1") == "thread-1.ns1"


class TestNoCacheProvided:
    """When no L1/L2 provided, all cache ops are no-ops."""

    async def test_l1_ops_are_noop(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert await saver.l1_get("t", "") is None
        await saver.l1_put("t", "", b"data")  # no-op
        await saver.l1_delete("t")  # no-op

    async def test_l2_ops_are_noop(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        assert await saver.l2_get("t", "") is None
        await saver.l2_put("t", "", b"data")  # no-op
        await saver.l2_delete("t")  # no-op


class TestSyncMethodsRaise:
    """Sync methods raise NotImplementedError."""

    def test_get_tuple_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(NotImplementedError):
            saver.get_tuple({"configurable": {"thread_id": "t1"}})

    def test_list_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(NotImplementedError):
            list(saver.list(None))

    def test_put_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(NotImplementedError):
            saver.put({"configurable": {"thread_id": "t1"}}, {}, {}, {})

    def test_put_writes_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(NotImplementedError):
            saver.put_writes({"configurable": {"thread_id": "t1"}}, [], "task-1")

    def test_delete_thread_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor())

        with pytest.raises(NotImplementedError):
            saver.delete_thread("t1")


class TestProtocolFlow:
    """verify executor protocol methods are invoked with expected sql."""

    async def test_aput_invokes_executor_execute(self):
        """aput() writes the checkpoint INSERT via executor.execute."""
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor)

        checkpoint = {
            "id": "cp-1",
            "ts": "2026-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
        metadata = {"source": "input", "step": 0}

        result = await saver.aput(config, checkpoint, metadata, {})

        assert result["configurable"]["checkpoint_id"] == "cp-1"
        executor.execute.assert_called_once()
        sql_stmt = executor.execute.call_args.args[0]
        assert "INSERT INTO checkpoints" in sql_stmt

    async def test_aget_tuple_returns_none_when_executor_empty(self):
        """aget_tuple() returns None when executor.fetchrow returns None."""
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor)

        result = await saver.aget_tuple(
            {"configurable": {"thread_id": "thread-404", "checkpoint_ns": ""}},
        )

        assert result is None
        executor.fetchrow.assert_called_once()

    async def test_adelete_thread_issues_two_delete_statements(self):
        """adelete_thread() runs DELETE on writes and checkpoints tables."""
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor)

        await saver.adelete_thread("thread-42")

        assert executor.execute.call_count == 2
        first_sql = executor.execute.call_args_list[0].args[0]
        second_sql = executor.execute.call_args_list[1].args[0]
        assert "DELETE FROM checkpoint_writes" in first_sql
        assert "DELETE FROM checkpoints" in second_sql

    async def test_flush_callback_runs_after_aput(self):
        """flush_callback is invoked after aput() writes succeed."""
        executor = _make_executor()
        flush = AsyncMock(return_value=3)
        saver = ThreeTierCheckpointSaver(executor=executor, flush_callback=flush)

        checkpoint = {
            "id": "cp-1",
            "ts": "2026-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

        await saver.aput(config, checkpoint, {}, {})

        flush.assert_awaited_once()


class TestAsyncpgPoolAdapter:
    """AsyncpgPoolAdapter wraps asyncpg.Pool to satisfy AsyncQueryExecutor.

    the adapter acquires a connection from the pool, runs the call,
    and converts asyncpg.Record results into plain dicts so the
    checkpoint saver sees the same shape whether the executor is an
    adapter or a NatsProxyL3Backend.
    """

    async def test_fetch_returns_list_of_dicts(self):
        record_one = {"checkpoint_id": "cp-1", "type": "msgpack"}
        record_two = {"checkpoint_id": "cp-2", "type": "msgpack"}
        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=[record_one, record_two])
        pool = _build_pool_with_conn(conn)

        adapter = AsyncpgPoolAdapter(pool)
        rows = await adapter.fetch("SELECT ...")

        assert rows == [record_one, record_two]
        conn.fetch.assert_awaited_once_with("SELECT ...")

    async def test_fetchrow_returns_dict_or_none(self):
        conn = MagicMock()
        conn.fetchrow = AsyncMock(return_value={"col": "v"})
        pool = _build_pool_with_conn(conn)

        adapter = AsyncpgPoolAdapter(pool)
        row = await adapter.fetchrow("SELECT col")
        assert row == {"col": "v"}

        conn.fetchrow = AsyncMock(return_value=None)
        pool_empty = _build_pool_with_conn(conn)
        adapter_empty = AsyncpgPoolAdapter(pool_empty)
        assert await adapter_empty.fetchrow("SELECT none") is None

    async def test_execute_returns_status_tag(self):
        conn = MagicMock()
        conn.execute = AsyncMock(return_value="UPDATE 2")
        pool = _build_pool_with_conn(conn)

        adapter = AsyncpgPoolAdapter(pool)
        result = await adapter.execute("UPDATE foo SET x=1")

        assert result == "UPDATE 2"
        conn.execute.assert_awaited_once_with("UPDATE foo SET x=1")

    async def test_checkpoint_saver_accepts_adapter(self):
        """end-to-end: saver driven via an adapter performs the same
        executor calls as when driven via a protocol-native backend.
        """
        conn = MagicMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        pool = _build_pool_with_conn(conn)

        saver = ThreeTierCheckpointSaver(executor=AsyncpgPoolAdapter(pool))

        checkpoint = {
            "id": "cp-1",
            "ts": "2026-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

        result = await saver.aput(config, checkpoint, {}, {})
        assert result["configurable"]["checkpoint_id"] == "cp-1"
        conn.execute.assert_awaited_once()

        tup = await saver.aget_tuple(
            {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
        )
        assert tup is None
        conn.fetchrow.assert_awaited_once()


def _build_pool_with_conn(conn: Any) -> Any:
    """build a MagicMock pool whose acquire() yields the given conn.

    :param conn: mock asyncpg connection
    :ptype conn: Any
    :return: pool-shaped mock with async context-managed acquire()
    :rtype: Any
    """
    pool = MagicMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool
