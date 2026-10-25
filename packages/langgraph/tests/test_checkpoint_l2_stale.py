"""A checkpoint L2 entry that cannot be refreshed is deleted, never left holding an older checkpoint.

The saver's L2 key is per thread and namespace, not per checkpoint. A put that does not land
therefore leaves the PREVIOUS checkpoint under the key, and every worker whose L1 misses is served
that older state from L2 without reaching L3. Production hit it on 2026-10-23: three checkpoint
writes for one turn were over the NATS server's ``max_payload`` and each left the last one cached.

These tests use an in-memory L2 and a mocked executor, so they run in CI; the end-to-end case
against a real NATS server and Postgres is ``integration/test_checkpoint_l2_oversize_live.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import CheckpointL2Cache, CheckpointL2ValueLimit

_UNSCOPED = CheckpointScope.unscoped(reason="tests pin the un-tenanted keyspace")

_BUCKET = "checkpoints"


def _checkpoint(checkpoint_id: str) -> dict[str, Any]:
    """a minimal LangGraph checkpoint carrying ``checkpoint_id``.

    :param checkpoint_id: the checkpoint's id
    :ptype checkpoint_id: str
    :return: the checkpoint mapping
    :rtype: dict[str, Any]
    """
    return {
        "id": checkpoint_id,
        "ts": "2026-10-23T06:06:55Z",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _make_executor() -> Any:
    """an executor stand-in whose reads find nothing.

    :return: mock executor with async fetch/fetchrow/execute
    :rtype: Any
    """
    executor = MagicMock()
    executor.fetch = AsyncMock(return_value=[])
    executor.fetchrow = AsyncMock(return_value=None)
    executor.execute = AsyncMock(return_value="INSERT 0 1")
    return executor


class _DictL2Cache(CheckpointL2Cache):
    """an exact-key L2 over a dict whose put and delete can be made to fail, recording every call."""

    def __init__(self) -> None:
        """start empty, with nothing failing.

        :return: nothing
        :rtype: None
        """
        self.store: dict[str, bytes] = {}
        self.fail_put: Exception | None = None
        self.fail_delete: Exception | None = None
        self.puts: list[str] = []
        self.gets: list[str] = []

    async def get(self, bucket: str, key: str) -> bytes | None:
        """read a value, or None on a miss.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: stored bytes or None
        :rtype: bytes | None
        """
        self.gets.append(key)
        return self.store.get(key)

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """write a value, or raise the configured failure.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: nothing
        :rtype: None
        :raises Exception: the configured put failure
        """
        self.puts.append(key)
        if self.fail_put is not None:
            raise self.fail_put
        self.store[key] = value

    async def delete(self, bucket: str, key: str) -> None:
        """drop one key, or raise the configured failure.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: nothing
        :rtype: None
        :raises Exception: the configured delete failure
        """
        if self.fail_delete is not None:
            raise self.fail_delete
        self.store.pop(key, None)


class _LimitedL2Cache(_DictL2Cache, CheckpointL2ValueLimit):
    """a dict L2 that states the largest value it accepts, as a NATS bucket's ``max_payload`` does."""

    def __init__(self, limit: int | None) -> None:
        """start empty with ``limit`` as the stated maximum.

        :param limit: the largest value accepted, or None for unknown
        :ptype limit: int | None
        :return: nothing
        :rtype: None
        """
        super().__init__()
        self.limit = limit

    @property
    def max_value_bytes(self) -> int | None:
        """the stated limit.

        :return: the limit, or None when unknown
        :rtype: int | None
        """
        return self.limit


class TestAnOversizedBlobIsNotCached:
    """a blob over the cache's stated limit deletes the entry instead of attempting a doomed put."""

    async def test_the_entry_is_deleted_and_no_put_is_attempted(self, caplog: pytest.LogCaptureFixture) -> None:
        l2 = _LimitedL2Cache(limit=10)
        l2.store["t-1"] = b"older checkpoint"
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        with caplog.at_level(logging.WARNING):
            await saver.l2_put("t-1", "", b"x" * 11)

        assert "t-1" not in l2.store, "the older checkpoint is still cached under the key"
        assert l2.puts == []
        warnings = [r for r in caplog.records if "over the cache's value limit" in r.getMessage()]
        assert len(warnings) == 1
        assert (warnings[0].size_bytes, warnings[0].max_value_bytes) == (11, 10)

    async def test_a_blob_at_the_limit_is_cached(self) -> None:
        l2 = _LimitedL2Cache(limit=10)
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        await saver.l2_put("t-1", "", b"x" * 10)

        assert l2.store["t-1"] == b"x" * 10

    async def test_an_unknown_limit_attempts_the_put(self) -> None:
        l2 = _LimitedL2Cache(limit=None)
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        await saver.l2_put("t-1", "", b"x" * 100)

        assert l2.store["t-1"] == b"x" * 100

    async def test_an_oversized_checkpoint_leaves_the_next_worker_nothing_old_to_read(self) -> None:
        """through aput: the newer checkpoint is in L3, so L2 must not still answer with the older one."""
        l2 = _LimitedL2Cache(limit=None)
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        config = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}
        await saver.aput(config, _checkpoint("cp-1"), {}, {})
        assert "t-1" in l2.store

        l2.limit = 1
        await saver.aput(config, _checkpoint("cp-2"), {}, {})

        assert "t-1" not in l2.store


class TestAFailedPutDeletesTheEntry:
    """any other put failure deletes the entry too, and only a failed delete bypasses the cache."""

    async def test_the_older_checkpoint_is_deleted(self) -> None:
        l2 = _DictL2Cache()
        l2.store["t-1"] = b"older checkpoint"
        l2.fail_put = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        await saver.l2_put("t-1", "", b"newer checkpoint")

        assert "t-1" not in l2.store

    async def test_a_deleted_entry_does_not_take_the_thread_off_the_cache(self) -> None:
        """the delete was the fix, so the thread keeps reading L2 (a miss there, then L3)."""
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "", b"newer checkpoint")

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert l2.gets == ["t-1"]

    async def test_a_failed_delete_reads_the_thread_from_l3(self) -> None:
        l2 = _DictL2Cache()
        l2.store["t-1"] = b"older checkpoint"
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "", b"newer checkpoint")

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert l2.gets == [], "the thread read the older checkpoint still cached in L2"
        executor.fetchrow.assert_awaited_once()
        # only that thread: another still reads its cache
        await saver.aget_tuple({"configurable": {"thread_id": "t-2", "checkpoint_ns": ""}})
        assert l2.gets == ["t-2"]

    async def test_an_oversized_blob_whose_delete_fails_reads_the_thread_from_l3(self) -> None:
        l2 = _LimitedL2Cache(limit=1)
        l2.store["t-1"] = b"older checkpoint"
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "", b"newer checkpoint")

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert l2.gets == []


class TestAPinnedReadDoesNotWarmTheCache:
    """the cache keys are read as "latest", so only a read of the latest checkpoint may fill them."""

    @staticmethod
    def _row(saver: ThreeTierCheckpointSaver, checkpoint_id: str) -> dict[str, Any]:
        """an L3 checkpoint row for ``checkpoint_id``, serialized the way aput stores it.

        :param saver: the saver whose serde encodes the row
        :ptype saver: ThreeTierCheckpointSaver
        :param checkpoint_id: the checkpoint's id
        :ptype checkpoint_id: str
        :return: the row as the executor returns it
        :rtype: dict[str, Any]
        """
        cp_type, cp_blob = saver.serde.dumps_typed(_checkpoint(checkpoint_id))
        _md_type, md_blob = saver.serde.dumps_typed({})
        return {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": None,
            "type": cp_type,
            "checkpoint": cp_blob,
            "metadata_": md_blob,
        }

    async def test_a_read_pinned_to_an_older_checkpoint_leaves_latest_cached(self) -> None:
        l2 = _DictL2Cache()
        l2.store["t-1"] = b"latest checkpoint"
        l1 = AsyncMock()
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = self._row(saver, "cp-1")

        tup = await saver.aget_tuple(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "cp-1"}}
        )

        assert tup is not None and tup.checkpoint["id"] == "cp-1"
        assert l2.store["t-1"] == b"latest checkpoint"
        assert l2.puts == []
        l1.put.assert_not_awaited()

    async def test_a_read_of_the_latest_checkpoint_warms_both_caches(self) -> None:
        """the counterpart: the guard stops the pinned read only, not every warm."""
        l2 = _DictL2Cache()
        l1 = AsyncMock()
        l1.get.return_value = None
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = self._row(saver, "cp-2")

        tup = await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert tup is not None and tup.checkpoint["id"] == "cp-2"
        assert saver.deserialize_checkpoint_tuple(l2.store["t-1"])["checkpoint"]["id"] == "cp-2"
        l1.put.assert_awaited_once()
