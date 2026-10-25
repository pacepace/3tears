"""A checkpoint L2 entry that cannot be refreshed is deleted, never left holding an older checkpoint.

The saver's L2 key is per thread and namespace, not per checkpoint. A put that does not land
therefore leaves the PREVIOUS checkpoint under the key, and every worker whose L1 misses is served
that older state from L2 without reaching L3. Production hit it on 2026-10-23: three checkpoint
writes for one turn were over the NATS server's ``max_payload`` and each left the last one cached.

These tests use an in-memory L2 and a mocked executor, so they run in CI; the end-to-end case
against a real NATS server and Postgres is ``integration/test_checkpoint_l2_oversize_live.py``.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus_client import REGISTRY

from threetears.langgraph.checkpoint import (
    L2_WRITE_BYPASS_LIMIT,
    CheckpointPurgeIncompleteError,
    ThreeTierCheckpointSaver,
)
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import CheckpointL2Cache, CheckpointL2RevisionCache, CheckpointL2ValueLimit

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


def _fresh(exc: Exception) -> Exception:
    """a new exception like ``exc``, so each raise starts its own traceback.

    raising one stored instance again appends every new raise's frames to its traceback, and a
    logged failure then formats a longer one each time: 4,097 raises took hours.

    :param exc: the configured failure
    :ptype exc: Exception
    :return: a fresh instance of the same type and arguments
    :rtype: Exception
    """
    return type(exc)(*exc.args)


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
            raise _fresh(self.fail_put)
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
            raise _fresh(self.fail_delete)
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


class _RevisionDictL2Cache(_DictL2Cache, CheckpointL2RevisionCache):
    """a dict L2 that keeps a revision per key, a deletion included, as a NATS KV bucket does."""

    def __init__(self) -> None:
        """start empty at revision 0.

        :return: nothing
        :rtype: None
        """
        super().__init__()
        self.revisions: dict[str, int] = {}
        self.sequence = 0
        self.updates: list[tuple[str, int, bool]] = []
        self.fail_get_latest: Exception | None = None
        self.fail_update: Exception | None = None

    def _bump(self, key: str) -> int:
        """advance the bucket's sequence and make it ``key``'s revision.

        :param key: the key written or deleted
        :ptype key: str
        :return: the new revision
        :rtype: int
        """
        self.sequence += 1
        self.revisions[key] = self.sequence
        return self.sequence

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """write a value at a new revision.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: nothing
        :rtype: None
        """
        await super().put(bucket, key, value)
        self._bump(key)

    async def delete(self, bucket: str, key: str) -> None:
        """drop a key, leaving a deletion at a new revision.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        await super().delete(bucket, key)
        self._bump(key)

    async def get_latest(self, bucket: str, key: str) -> tuple[bytes | None, int]:
        """the value and the key's latest revision.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: value or None, and the revision (0 for a key never written)
        :rtype: tuple[bytes | None, int]
        :raises Exception: the configured read failure
        """
        self.gets.append(key)
        if self.fail_get_latest is not None:
            raise _fresh(self.fail_get_latest)
        return self.store.get(key), self.revisions.get(key, 0)

    async def update(self, bucket: str, key: str, value: bytes, revision: int) -> int | None:
        """write only while the key's revision is ``revision``.

        :param bucket: bucket name
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: the expected current revision
        :ptype revision: int
        :return: the new revision, or None when the key changed
        :rtype: int | None
        :raises Exception: the configured write failure
        """
        if self.fail_update is not None:
            raise _fresh(self.fail_update)
        landed = self.revisions.get(key, 0) == revision
        self.updates.append((key, revision, landed))
        if not landed:
            return None
        self.store[key] = value
        return self._bump(key)


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

    async def test_a_read_pinned_to_an_older_checkpoint_leaves_latest_cached(self) -> None:
        l2 = _RevisionDictL2Cache()
        await l2.put(_BUCKET, "t-1", b"latest checkpoint")
        l2.puts.clear()
        l1 = AsyncMock()
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-1")

        tup = await saver.aget_tuple(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "cp-1"}}
        )

        assert tup is not None and tup.checkpoint["id"] == "cp-1"
        assert l2.store["t-1"] == b"latest checkpoint"
        assert l2.puts == [] and l2.updates == []
        l1.put.assert_not_awaited()

    async def test_a_read_of_the_latest_checkpoint_warms_both_caches(self) -> None:
        """the counterpart: the guard stops the pinned read only, not every warm."""
        l2 = _RevisionDictL2Cache()
        l1 = AsyncMock()
        l1.get.return_value = None
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-2")

        tup = await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert tup is not None and tup.checkpoint["id"] == "cp-2"
        assert saver.deserialize_checkpoint_tuple(l2.store["t-1"])["checkpoint"]["id"] == "cp-2"
        l1.put.assert_awaited_once()


class TestARefillCannotReplaceANewerCheckpoint:
    """a refill from an L3 read lands only if nothing reached the key since the revision read before it."""

    async def test_an_older_refill_landing_after_a_newer_put_leaves_the_newer_one(self) -> None:
        """the race, made deterministic: the newer put lands while the reader's query is running."""
        l2 = _RevisionDictL2Cache()
        reader_l1 = AsyncMock()
        reader_l1.get.return_value = None
        executor = _make_executor()
        writer = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        reader = ThreeTierCheckpointSaver(executor=executor, l1_cache=reader_l1, l2_cache=l2, scope=_UNSCOPED)
        config = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}
        older_row = _row(reader, "cp-1")

        async def _query_then_a_newer_write_lands(*_args: object) -> dict[str, Any]:
            await writer.aput(config, _checkpoint("cp-2"), {}, {})
            return older_row

        executor.fetchrow.side_effect = _query_then_a_newer_write_lands

        tup = await reader.aget_tuple(config)

        assert tup is not None and tup.checkpoint["id"] == "cp-1", "the reader answers with what it read"
        assert reader.deserialize_checkpoint_tuple(l2.store["t-1"])["checkpoint"]["id"] == "cp-2", (
            "the older refill replaced the newer checkpoint in L2"
        )
        reader_l1.put.assert_not_awaited()
        fresh = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        fresh_tup = await fresh.aget_tuple(config)
        assert fresh_tup is not None and fresh_tup.checkpoint["id"] == "cp-2"

    async def test_a_refill_after_a_deletion_since_the_read_writes_nothing(self) -> None:
        """a delete (a failed put's, a purge's) since the read refuses the refill too: create-if-absent would not."""
        l2 = _RevisionDictL2Cache()
        executor = _make_executor()
        reader = ThreeTierCheckpointSaver(executor=executor, l2_cache=l2, scope=_UNSCOPED)
        older_row = _row(reader, "cp-1")

        async def _query_then_the_key_is_deleted(*_args: object) -> dict[str, Any]:
            await l2.put(_BUCKET, "t-1", b"newer")
            await l2.delete(_BUCKET, "t-1")
            return older_row

        executor.fetchrow.side_effect = _query_then_the_key_is_deleted

        await reader.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert "t-1" not in l2.store

    async def test_a_cache_that_cannot_fence_is_not_refilled_from_a_read(self) -> None:
        """without get_latest/update only aput writes L2; L1 is still warmed."""
        l2 = _DictL2Cache()
        l1 = AsyncMock()
        l1.get.return_value = None
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-1")

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert l2.puts == [] and l2.store == {}
        l1.put.assert_awaited_once()

    async def test_an_oversized_refill_writes_nothing_and_deletes_nothing(self) -> None:
        l2 = _RevisionLimitedL2Cache(limit=1)
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-1")

        await saver.aget_tuple({"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}})

        assert l2.updates == [] and l2.revisions == {}


class _RevisionLimitedL2Cache(_RevisionDictL2Cache, CheckpointL2ValueLimit):
    """a fencing dict L2 with a stated value limit."""

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


def _contained(site: str, exc_type: str) -> float:
    """how many errors ``contain()`` has counted at ``site``.

    :param site: the containing function's site label
    :ptype site: str
    :param exc_type: the contained exception's type name
    :ptype exc_type: str
    :return: the count so far
    :rtype: float
    """
    labels = {"site": f"threetears.langgraph.checkpoint:ThreeTierCheckpointSaver.{site}", "exc_type": exc_type}
    return REGISTRY.get_sample_value("contained_errors_total", labels) or 0.0


class TestAFencedCacheThatFails:
    """a fenced cache's failed revision read or failed refill degrades; neither fails the read."""

    _CONFIG: dict[str, Any] = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}

    async def test_a_failed_revision_read_is_answered_from_l3_and_counted(self) -> None:
        l2 = _RevisionDictL2Cache()
        assert isinstance(l2, CheckpointL2RevisionCache), "the double must reach the fenced branch"
        l2.fail_get_latest = RuntimeError("nats: timeout")
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-1")
        before = _contained("_l2_read_with_revision", "CheckpointCacheError")

        tup = await saver.aget_tuple(self._CONFIG)

        assert tup is not None and tup.checkpoint["id"] == "cp-1", "the durable store answered"
        assert _contained("_l2_read_with_revision", "CheckpointCacheError") - before == 1
        assert l2.updates == [], "no revision was read, so nothing is refilled"

    async def test_a_failed_refill_still_answers_and_still_warms_l1(self) -> None:
        l2 = _RevisionDictL2Cache()
        assert isinstance(l2, CheckpointL2RevisionCache), "the double must reach the fenced branch"
        l2.fail_update = RuntimeError("nats: timeout")
        l1 = AsyncMock()
        l1.get.return_value = None
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)
        executor.fetchrow.return_value = _row(saver, "cp-1")
        before = _contained("_refill_l2", "RuntimeError")

        tup = await saver.aget_tuple(self._CONFIG)

        assert tup is not None and tup.checkpoint["id"] == "cp-1"
        assert _contained("_refill_l2", "RuntimeError") - before == 1
        assert l2.store == {}
        l1.put.assert_awaited_once()


class TestAWriteBypassLiftsWhenTheKeyIsWrittenAgain:
    """an outage's bypass ends with the next write that lands; a purge's bypass does not."""

    _CONFIG: dict[str, Any] = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}

    async def test_an_outage_bypasses_the_key_and_the_next_landed_put_lifts_it(self) -> None:
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "", b"cp-1")
        await saver.aget_tuple(self._CONFIG)
        assert l2.gets == [], "during the outage the key is read from L3"

        l2.fail_put = l2.fail_delete = None
        await saver.l2_put("t-1", "", b"cp-2")
        await saver.aget_tuple(self._CONFIG)

        assert l2.gets == ["t-1"], "the key holds the newest checkpoint again, so it is read from cache"

    async def test_a_landed_delete_lifts_it_too(self) -> None:
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "", b"cp-1")

        l2.fail_delete = None
        await saver.l2_put("t-1", "", b"cp-2")
        await saver.aget_tuple(self._CONFIG)

        assert l2.gets == ["t-1"]

    async def test_the_bypass_is_per_namespace(self) -> None:
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        await saver.l2_put("t-1", "sub", b"cp-1")

        await saver.aget_tuple(self._CONFIG)

        assert l2.gets == ["t-1"], "the root namespace was never bypassed"

    async def test_a_purge_bypass_survives_a_landed_put(self) -> None:
        l2 = _DictL2Cache()
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)
        with pytest.raises(CheckpointPurgeIncompleteError):
            await saver.adelete_thread("t-1")

        l2.fail_delete = None
        await saver.l2_put("t-1", "", b"written after the purge")
        await saver.aget_tuple(self._CONFIG)

        assert l2.gets == [], "an erased thread is read from L3 until a purge of it succeeds"

    async def test_the_write_bypass_is_bounded_oldest_first(self, caplog: pytest.LogCaptureFixture) -> None:
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        limit = 3
        saver = ThreeTierCheckpointSaver(
            executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED, l2_write_bypass_limit=limit
        )

        with caplog.at_level(logging.WARNING):
            for n in range(limit + 1):
                await saver.l2_put(f"t-{n}", "", b"cp")

        full = [r for r in caplog.records if "write bypass is full" in r.getMessage()]
        assert len(full) == 1 and full[0].l2_key == "t-0" and full[0].limit == limit
        await saver.aget_tuple({"configurable": {"thread_id": "t-0", "checkpoint_ns": ""}})
        await saver.aget_tuple({"configurable": {"thread_id": f"t-{limit}", "checkpoint_ns": ""}})
        assert l2.gets == ["t-0"], "the oldest key was dropped from the bypass; the newest is still on it"

    def test_the_bypass_holds_the_published_limit_by_default(self) -> None:
        default = inspect.signature(ThreeTierCheckpointSaver).parameters["l2_write_bypass_limit"].default

        assert default == L2_WRITE_BYPASS_LIMIT

    async def test_a_reused_failure_does_not_grow_its_traceback(self) -> None:
        l2 = _DictL2Cache()
        l2.fail_put = RuntimeError("nats: timeout")
        l2.fail_delete = RuntimeError("nats: timeout")
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        for n in range(50):
            await saver.l2_put(f"t-{n}", "", b"cp")

        assert l2.fail_put.__traceback__ is None, "the stored failure is never raised itself"
