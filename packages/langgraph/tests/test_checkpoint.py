"""Tests for ThreeTierCheckpointSaver.

Tests the serialization helpers, protocol interactions, and sync-method guards.
Full integration tests require a Postgres instance and are in the host app.

Every saver here is built on one shared unscoped :class:`CheckpointScope`, so
this file keeps pinning the un-tenanted keyspace exactly as it did before the
scope parameter existed. The customer dimension has its own suite in
``test_checkpoint_tenancy.py``; that split is what proves the unscoped path did
not move.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph.checkpoint.base import WRITES_IDX_MAP

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import AsyncpgPoolAdapter
from threetears.langgraph.serde import UUIDSafeSerializer

#: built once at import so the warning it emits lands outside every ``caplog``
#: block in this file.
_UNSCOPED = CheckpointScope.unscoped(reason="tests pin the un-tenanted keyspace")


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
        return ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

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

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, scope=_UNSCOPED)

        result = await saver.l1_get("thread-1", "")
        assert result is None

    async def testl1_put_swallows_error(self):
        l1 = AsyncMock()
        l1.put.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, scope=_UNSCOPED)

        # Should not raise
        await saver.l1_put("thread-1", "", b"data")

    async def testl1_delete_swallows_error(self):
        l1 = AsyncMock()
        l1.delete.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, scope=_UNSCOPED)

        await saver.l1_delete("thread-1")


class TestCrashRecoveryWritesDegrade:
    """A crash-recovery write must not kill a live turn -- but a control write must.

    ``aput_writes`` receives two kinds of row and they get opposite treatment.

    Ordinary channel writes only let a CRASHED run resume. LangGraph calls the
    method from its executor teardown, so an unguarded failure there propagates
    out of ``run_graph`` and the whole turn terminates as ``AGENT_FAILED`` --
    discarding an answer that already exists in order to protect the ability to
    recover an answer nobody needs any more. Observed on cobalt-dev: two
    identical L3 timeouts on ``aibots.l3.query`` in one log window. The first hit
    memory retrieval, which guards its call, and soft-failed -- that turn
    survived. The second hit ``aput_writes``, which guarded nothing, and killed
    the eval case ``county-name-from-fips``. Same infrastructure event, opposite
    outcomes, decided purely by whether the call site had a try/except.

    The members of ``WRITES_IDX_MAP`` are NOT that. ``__interrupt__`` /
    ``__resume__`` / ``__error__`` / ``__scheduled__`` carry control flow: pregel
    builds a snapshot's interrupts from the rows this method writes, and
    ``detect_interrupt`` reads the pause from nothing else. Swallowing a failed
    ``__interrupt__`` write therefore ends the turn as an ordinary
    ``StreamEndEvent`` -- a human-approval gate silently skipped, its payload
    lost. A dead turn is recoverable; a vanished approval gate is not.
    """

    _LOGGER = "threetears.langgraph.checkpoint"
    _CONFIG = {"configurable": {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "c-1"}}

    async def testaput_writes_degrades_and_reports_the_loss(self, caplog):
        """the warning is the ENTIRE detection surface for a lost write, so pin it.

        Asserting only "did not raise" passes against ``except Exception: pass``,
        which would make the incident that motivated the guard invisible again.
        """
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("NATS request failed: nats: timeout")

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            await saver.aput_writes(self._CONFIG, [("channel-a", "value-a")], "task-1")

        records = [r for r in caplog.records if "re-execute" in r.getMessage()]
        assert len(records) == 1
        record = records[0]
        assert record.levelno == logging.WARNING
        assert record.thread_id == "t-1"
        assert record.checkpoint_ns == ""
        assert record.checkpoint_id == "c-1"
        assert record.task_id == "task-1"
        assert record.write_count == 1
        # the timeout itself, not merely the fact of one
        assert record.exc_info is not None

    @pytest.mark.parametrize("channel", sorted(WRITES_IDX_MAP))
    async def testaput_writes_reraises_for_a_control_channel(self, channel, caplog):
        """losing a control write silently changes what the run DOES."""
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("NATS request failed: nats: timeout")

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with caplog.at_level(logging.WARNING, logger=self._LOGGER), pytest.raises(RuntimeError, match="timeout"):
            await saver.aput_writes(self._CONFIG, [(channel, "value")], "task-1")

        # the channel names ARE the diagnosis, and a traceback does not carry
        # them -- so the re-raise branch logs before it re-raises
        records = [r for r in caplog.records if "control-channel" in r.getMessage()]
        assert len(records) == 1
        assert records[0].control_channels == [channel]

    async def testaput_writes_reraises_when_a_control_write_shares_the_set(self):
        """a set cannot be half-degraded, so one control write makes it must-persist."""
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("NATS request failed: nats: timeout")

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with pytest.raises(RuntimeError, match="timeout"):
            await saver.aput_writes(
                self._CONFIG,
                [("channel-a", "value-a"), ("__interrupt__", "approve?")],
                "task-1",
            )

    @pytest.mark.parametrize("config", [{}, {"configurable": None}, {"configurable": {}}])
    async def testaput_writes_does_not_raise_on_a_malformed_config(self, config):
        """the ids are read defensively, so the guard covers the likeliest bad shapes.

        Subscripting them ahead of the ``try`` made "degrades, never raises"
        false for exactly the input most likely to be malformed. An explicit
        ``configurable: None`` is included because a ``.get`` default does not
        cover it -- the next lookup still raises AttributeError.
        """
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        await saver.aput_writes(config, [("channel-a", "value-a")], "task-1")

    async def testaput_writes_degrades_when_a_malformed_config_reaches_the_write(self, caplog):
        """a malformed config must take the DEGRADE path, not just avoid raising.

        The no-raise test above runs against a succeeding executor, so it never
        enters the guard it is named for; this one does.
        """
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("null value in column violates not-null constraint")

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            await saver.aput_writes({"configurable": {}}, [("channel-a", "value-a")], "task-1")

        assert any("re-execute" in r.getMessage() for r in caplog.records)

    async def testa_malformed_config_still_raises_for_a_control_channel(self):
        """the degrade carve-out is scoped by CHANNEL, never by config shape.

        Every other re-raise test uses a well-formed config, which would let a
        guard that keyed off the config instead of the channel pass.
        """
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("NATS request failed: nats: timeout")

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        with pytest.raises(RuntimeError, match="timeout"):
            await saver.aput_writes({"configurable": {}}, [("__interrupt__", "approve?")], "task-1")

    async def testcontrol_channel_writes_are_last_wins(self):
        """a second Command(resume=...) must REPLACE the first, not be dropped.

        The reference saver skips a duplicate only when ``idx >= 0``; a control
        channel has a negative index and overwrites. Getting this backwards
        resumes a paused graph on the stale earlier approve/deny -- a wrong human
        decision applied silently.
        """
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.aput_writes(
            self._CONFIG,
            [("__resume__", "first-answer"), ("__resume__", "second-answer")],
            "task-1",
        )

        query, *params = executor.execute.await_args.args
        # deduplicated to one row, keeping the LAST value
        assert params.count("__resume__") == 1
        assert "second-answer" in str(params)
        assert "first-answer" not in str(params)
        # and the statement lets a control row overwrite one already stored,
        # while leaving an ordinary row alone
        assert "DO UPDATE" in query
        assert "WHERE checkpoint_writes.idx < 0" in query

    async def testordinary_channel_writes_are_first_wins(self):
        """an ordinary channel keeps the row already stored, per the reference saver."""
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.aput_writes(self._CONFIG, [("channel-a", "value-a")], "task-1")

        query, *_ = executor.execute.await_args.args
        # the predicate is what makes the upsert asymmetric: false for idx >= 0
        assert "WHERE checkpoint_writes.idx < 0" in query

    async def testcontrol_channels_map_to_their_reserved_index(self):
        """the reserved negative index is what makes a control row upsertable."""
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.aput_writes(self._CONFIG, [("__interrupt__", "approve?")], "task-1")

        _, *params = executor.execute.await_args.args
        assert WRITES_IDX_MAP["__interrupt__"] in params

    async def testcontrol_channel_write_invalidates_the_caches(self):
        """a cached bundle carries pending_writes=[] and would hide the interrupt.

        ``aput`` warms L1/L2 with an empty pending-writes list and
        ``aget_tuple`` serves that verbatim when no checkpoint_id is pinned --
        which is how interrupt detection reads state. Without invalidation the
        approval gate vanishes however successfully the row was written.
        """
        l1, l2 = AsyncMock(), AsyncMock()

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)

        await saver.aput_writes(self._CONFIG, [("__interrupt__", "approve?")], "task-1")

        l1.delete.assert_awaited_once_with("t-1")
        assert l2.delete.await_count == 1
        # the namespace we WROTE, not a hardcoded root: l2 is exact-key, so
        # invalidating "" would leave a namespaced bundle stale
        assert l2.delete.await_args.args[1] == "t-1"

    async def testinvalidation_clears_the_namespaced_key_not_the_root(self):
        """L2 is exact-key, so invalidating "" leaves a namespaced bundle stale.

        ``l2_delete`` used to hardcode the root namespace, which made this exact
        case a silent miss: the interrupt row lands, the cache still answers with
        the pre-interrupt bundle, and the approval gate vanishes anyway -- the
        failure the invalidation exists to prevent.
        """
        l1, l2 = AsyncMock(), AsyncMock()

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)

        await saver.aput_writes(
            {"configurable": {"thread_id": "t-1", "checkpoint_ns": "inner", "checkpoint_id": "c-1"}},
            [("__interrupt__", "approve?")],
            "task-1",
        )

        # l2_key("t-1", "inner") -- the key aput actually wrote
        assert l2.delete.await_args.args[1] == "t-1.inner"

    async def testa_failed_invalidation_after_a_control_write_raises(self):
        """a silently-failed invalidation is the original bug wearing a hat.

        The row reaches L3, the cache keeps serving the pre-interrupt bundle,
        detect_interrupt finds nothing, and the approval gate is skipped -- the
        same end state as never writing the row. So this invalidation does NOT go
        through l1_delete/l2_delete, which degrade by design.
        """
        l1, l2 = AsyncMock(), AsyncMock()
        l2.delete.side_effect = RuntimeError("NATS request failed: nats: timeout")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)

        with pytest.raises(RuntimeError, match="timeout"):
            await saver.aput_writes(self._CONFIG, [("__interrupt__", "approve?")], "task-1")

    async def testa_failed_l1_invalidation_after_a_control_write_also_raises(self):
        """both tiers can hide the interrupt, so neither may fail quietly."""
        l1, l2 = AsyncMock(), AsyncMock()
        l1.delete.side_effect = RuntimeError("L1 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)

        with pytest.raises(RuntimeError, match="L1 down"):
            await saver.aput_writes(self._CONFIG, [("__interrupt__", "approve?")], "task-1")

    async def testordinary_write_does_not_invalidate_the_caches(self):
        """invalidation is scoped to control channels, not every write."""
        l1, l2 = AsyncMock(), AsyncMock()

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l1_cache=l1, l2_cache=l2, scope=_UNSCOPED)

        await saver.aput_writes(self._CONFIG, [("channel-a", "value-a")], "task-1")

        assert l1.delete.await_count == 0
        assert l2.delete.await_count == 0

    async def testan_unserializable_value_surfaces_rather_than_degrading(self):
        """a programming error must not hide behind the transport guard.

        Serialization is deterministic: degrading it would log one warning per
        turn forever instead of surfacing a permanent bug.
        """
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        class _Unserializable:
            def __reduce__(self):
                raise TypeError("cannot serialize this")

        with pytest.raises(TypeError, match="not msgpack serializable"):
            await saver.aput_writes(self._CONFIG, [("channel-a", _Unserializable())], "task-1")

        assert executor.execute.await_count == 0

    async def testaput_writes_persists_the_set_in_one_statement(self):
        """all-or-nothing, because a truncated set is worse than a lost one.

        Pregel applies a task's pending writes to channels and skips any task
        that has writes, so resuming from half a set continues from partially
        updated channels -- divergence rather than lost resumability.
        """
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.aput_writes(
            self._CONFIG,
            [("channel-a", "value-a"), ("channel-b", "value-b")],
            "task-1",
        )

        assert executor.execute.await_count == 1
        query, *params = executor.execute.await_args.args
        assert "channel-a" in params
        assert "channel-b" in params
        # every placeholder bound exactly once, in order, with no gaps -- catches
        # a desynchronised parameter list, which SQL alone would not reveal
        assert [int(n) for n in re.findall(r"\$(\d+)", query)] == list(range(1, len(params) + 1))

    async def testaput_writes_issues_no_statement_for_an_empty_set(self):
        """no rows to persist is not a failure to persist rows."""
        executor = _make_executor()

        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        await saver.aput_writes(self._CONFIG, [], "task-1")

        assert executor.execute.await_count == 0


class TestL2Degradation:
    """L2 cache failures degrade gracefully."""

    async def testl2_get_returns_none_on_error(self):
        l2 = AsyncMock()
        l2.get.side_effect = RuntimeError("L2 down")

        saver = ThreeTierCheckpointSaver(executor=_make_executor(), l2_cache=l2, scope=_UNSCOPED)

        result = await saver.l2_get("thread-1", "")
        assert result is None

    async def testl2_key_with_ns(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        assert saver.l2_key("thread-1", "") == "thread-1"
        assert saver.l2_key("thread-1", "ns1") == "thread-1.ns1"


class TestNoCacheProvided:
    """When no L1/L2 provided, all cache ops are no-ops."""

    async def test_l1_ops_are_noop(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        assert await saver.l1_get("t", "") is None
        await saver.l1_put("t", "", b"data")  # no-op
        await saver.l1_delete("t")  # no-op

    async def test_l2_ops_are_noop(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        assert await saver.l2_get("t", "") is None
        await saver.l2_put("t", "", b"data")  # no-op
        await saver.l2_delete("t", "")  # no-op


class TestSyncMethodsRaise:
    """Sync methods raise NotImplementedError."""

    def test_get_tuple_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError):
            saver.get_tuple({"configurable": {"thread_id": "t1"}})

    def test_list_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError):
            list(saver.list(None))

    def test_put_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError):
            saver.put({"configurable": {"thread_id": "t1"}}, {}, {}, {})

    def test_put_writes_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError):
            saver.put_writes({"configurable": {"thread_id": "t1"}}, [], "task-1")

    def test_delete_thread_raises(self):
        saver = ThreeTierCheckpointSaver(executor=_make_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError):
            saver.delete_thread("t1")


class TestProtocolFlow:
    """verify executor protocol methods are invoked with expected sql."""

    async def test_aput_invokes_executor_execute(self):
        """aput() writes the checkpoint INSERT via executor.execute."""
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

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
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

        result = await saver.aget_tuple(
            {"configurable": {"thread_id": "thread-404", "checkpoint_ns": ""}},
        )

        assert result is None
        executor.fetchrow.assert_called_once()

    async def test_adelete_thread_issues_two_delete_statements(self):
        """adelete_thread() runs DELETE on writes and checkpoints tables."""
        executor = _make_executor()
        saver = ThreeTierCheckpointSaver(executor=executor, scope=_UNSCOPED)

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
        saver = ThreeTierCheckpointSaver(executor=executor, flush_callback=flush, scope=_UNSCOPED)

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

        saver = ThreeTierCheckpointSaver(executor=AsyncpgPoolAdapter(pool), scope=_UNSCOPED)

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
