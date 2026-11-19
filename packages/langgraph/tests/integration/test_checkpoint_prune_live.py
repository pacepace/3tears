"""Integration: pruning checkpoint history against a real Postgres, and the maintenance tools around it.

``ThreeTierCheckpointSaver.aprune`` keeps the newest checkpoint of every namespace and its writes and
deletes the rest; a paused run must still resume from what is kept. ``checkpoint_maintenance``
measures the tables, prunes them in batches and deletes threads an app no longer owns. Every test
here drives real LangGraph graphs or real saver writes into a schema the package's own migration
runner built.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import json
import operator
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict
from uuid import UUID

import asyncpg
import pytest
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from threetears.core.data.migrations import MigrationRunner
from threetears.langgraph.checkpoint import (
    DELTA_COUNTERS_METADATA_KEY,
    CheckpointPurgeIncompleteError,
    ThreeTierCheckpointSaver,
)
from threetears.langgraph.checkpoint_maintenance import (
    checkpoint_usage,
    delete_unowned_threads,
    prune_threads,
)
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.migrations import register as register_langgraph
from threetears.langgraph.protocols import (
    AsyncpgPoolAdapter,
    CheckpointL1Cache,
    CheckpointL2Cache,
    CheckpointL2PrefixCache,
)

pytestmark = pytest.mark.integration

_UNSCOPED = CheckpointScope.unscoped(reason="integration test of the un-tenanted keyspace")
_CUSTOMER_A = UUID("00000000-0000-4000-8000-00000000000a")
_CUSTOMER_B = UUID("00000000-0000-4000-8000-00000000000b")
_BUCKET = "checkpoints"


class _MigrationStore:
    """the two calls the migration runner makes, over one asyncpg connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """wrap ``conn``.

        :param conn: connection whose search_path selects the test schema
        :ptype conn: asyncpg.Connection
        """
        self.conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """run a statement.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: status tag
        :rtype: str
        """
        result: str = await self.conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """fetch rows as dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: rows
        :rtype: list[dict[str, Any]]
        """
        return [dict(r) for r in await self.conn.fetch(sql, *params)]


@dataclass(frozen=True)
class _Database:
    """one migrated schema: where it is and an executor over it."""

    dsn: str
    schema: str
    executor: AsyncpgPoolAdapter


@pytest.fixture
async def database(db_container: str) -> AsyncIterator[_Database]:
    schema = f"prune_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_langgraph(runner)
        await runner.apply_for_agent_schema(_MigrationStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert pool is not None
    try:
        yield _Database(dsn=db_container, schema=schema, executor=AsyncpgPoolAdapter(pool))
    finally:
        await pool.close()
        admin = await asyncpg.connect(db_container)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


@pytest.fixture
def executor(database: _Database) -> AsyncpgPoolAdapter:
    return database.executor


class _DictL1Cache(CheckpointL1Cache):
    """a pod-local L1 over a dict, keyed as the protocol keys it."""

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self.store: dict[tuple[str, str], bytes] = {}

    async def get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """
        :param thread_id: thread
        :ptype thread_id: str
        :param checkpoint_ns: namespace
        :ptype checkpoint_ns: str
        :return: the cached bundle, or None
        :rtype: bytes | None
        """
        return self.store.get((thread_id, checkpoint_ns))

    async def put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """
        :param thread_id: thread
        :ptype thread_id: str
        :param checkpoint_ns: namespace
        :ptype checkpoint_ns: str
        :param data: bundle
        :ptype data: bytes
        """
        self.store[(thread_id, checkpoint_ns)] = data

    async def delete(self, thread_id: str) -> None:
        """
        :param thread_id: thread whose every namespace is dropped
        :ptype thread_id: str
        """
        for key in [key for key in self.store if key[0] == thread_id]:
            del self.store[key]


class _SweepingL2Cache(CheckpointL2Cache, CheckpointL2PrefixCache):
    """a shared L2 over a dict with the prefix sweep, whose deletes can be made to fail."""

    def __init__(self) -> None:
        """start empty, nothing failing.

        :return: nothing
        :rtype: None
        """
        self.store: dict[str, bytes] = {}
        self.fail_delete = False

    async def get(self, bucket: str, key: str) -> bytes | None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :return: value, or None
        :rtype: bytes | None
        """
        return self.store.get(key)

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :param value: value
        :ptype value: bytes
        """
        self.store[key] = value

    async def delete(self, bucket: str, key: str) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :raises ConnectionError: when deletes are set to fail
        """
        if self.fail_delete:
            raise ConnectionError("L2 unreachable")
        self.store.pop(key, None)

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param prefix: key prefix
        :ptype prefix: str
        :raises ConnectionError: when deletes are set to fail
        """
        if self.fail_delete:
            raise ConnectionError("L2 unreachable")
        for key in [key for key in self.store if key.startswith(prefix)]:
            del self.store[key]


# ---------------------------------------------------------------------------
# a real graph: a parent with a subgraph that can pause for approval
# ---------------------------------------------------------------------------


class _State(TypedDict):
    """the run's log, and whether this turn asks for approval."""

    log: Annotated[list[str], operator.add]
    ask: bool


def _draft(state: _State) -> dict[str, Any]:
    """
    :param state: graph state
    :ptype state: _State
    :return: the update
    :rtype: dict[str, Any]
    """
    return {"log": ["draft"]}


def _approve(state: _State) -> dict[str, Any]:
    """pause for a decision when the turn asks for one.

    :param state: graph state
    :ptype state: _State
    :return: the update
    :rtype: dict[str, Any]
    """
    if not state["ask"]:
        return {"log": ["auto"]}
    decision = interrupt("send it?")
    return {"log": [f"decision:{decision}"]}


def _graph(saver: ThreeTierCheckpointSaver) -> Any:
    """compile the parent graph, whose second step is a subgraph that may interrupt.

    :param saver: the checkpoint saver
    :ptype saver: ThreeTierCheckpointSaver
    :return: the compiled graph
    :rtype: Any
    """
    child = StateGraph(_State)
    child.add_node("review", _draft)
    child.add_node("approve", _approve)
    child.add_edge(START, "review")
    child.add_edge("review", "approve")
    child.add_edge("approve", END)
    parent = StateGraph(_State)
    parent.add_node("draft", _draft)
    parent.add_node("child", child.compile())
    parent.add_edge(START, "draft")
    parent.add_edge("draft", "child")
    parent.add_edge("child", END)
    return parent.compile(checkpointer=saver)


async def _paused_thread(graph: Any, thread: str) -> dict[str, Any]:
    """two complete turns, then a third that pauses inside the subgraph.

    :param graph: the compiled graph
    :ptype graph: Any
    :param thread: the thread id
    :ptype thread: str
    :return: the thread's config
    :rtype: dict[str, Any]
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread}}
    await graph.ainvoke({"log": ["turn-1"], "ask": False}, config)
    await graph.ainvoke({"log": ["turn-2"], "ask": False}, config)
    await graph.ainvoke({"log": ["turn-3"], "ask": True}, config)
    return config


async def _rows(executor: AsyncpgPoolAdapter, table: str, thread: str) -> list[dict[str, Any]]:
    """every key of one thread's rows in one table.

    :param executor: the database
    :ptype executor: AsyncpgPoolAdapter
    :param table: ``checkpoints`` or ``checkpoint_writes``
    :ptype table: str
    :param thread: the stored thread id
    :ptype thread: str
    :return: the rows' keys
    :rtype: list[dict[str, Any]]
    """
    columns = "checkpoint_ns, checkpoint_id" + (", task_id, idx" if table == "checkpoint_writes" else "")
    return await executor.fetch(f"SELECT {columns} FROM {table} WHERE thread_id = $1 ORDER BY {columns}", thread)


def _newest(rows: list[dict[str, Any]]) -> dict[str, str]:
    """each namespace's largest checkpoint id.

    :param rows: checkpoint keys
    :ptype rows: list[dict[str, Any]]
    :return: namespace -> newest checkpoint id
    :rtype: dict[str, str]
    """
    newest: dict[str, str] = {}
    for row in rows:
        newest[row["checkpoint_ns"]] = max(newest.get(row["checkpoint_ns"], ""), row["checkpoint_id"])
    return newest


class TestKeepLatestOnARealGraph:
    """a paused run, pruned, resumes as if nothing had been pruned."""

    async def test_only_the_newest_checkpoint_per_namespace_and_its_writes_remain(
        self, executor: AsyncpgPoolAdapter
    ) -> None:
        """root and subgraph namespaces each keep one checkpoint, the newest, with its writes."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _paused_thread(_graph(saver), "t-graph")
        before = await _rows(executor, "checkpoints", "t-graph")
        writes_before = await _rows(executor, "checkpoint_writes", "t-graph")
        newest = _newest(before)
        assert len(newest) >= 3, f"expected the root and at least two subgraph namespaces, got {sorted(newest)}"
        assert len(before) > len(newest), "the fixture left no history to prune"

        result = await saver.aprune(["t-graph"])

        after = await _rows(executor, "checkpoints", "t-graph")
        writes_after = await _rows(executor, "checkpoint_writes", "t-graph")
        assert {(row["checkpoint_ns"], row["checkpoint_id"]) for row in after} == set(newest.items())
        kept_writes = [row for row in writes_before if row["checkpoint_id"] == newest[row["checkpoint_ns"]]]
        assert writes_after == kept_writes
        assert kept_writes, "the paused run's interrupt write was not among the fixture's writes"
        assert result.checkpoints_deleted == len(before) - len(after)
        assert result.writes_deleted == len(writes_before) - len(writes_after)
        assert result.threads == 1
        assert result.namespaces_held == 0

    async def test_the_paused_run_resumes_from_the_kept_checkpoint(self, executor: AsyncpgPoolAdapter) -> None:
        """resuming the pruned thread gives the same state as resuming an unpruned twin."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        pruned = await _paused_thread(graph, "t-pruned")
        control = await _paused_thread(graph, "t-control")
        state = await graph.aget_state(pruned)
        assert state.next, "the fixture's third turn did not pause"

        await saver.aprune(["t-pruned"])
        assert (await graph.aget_state(pruned)).next == state.next

        resumed = await graph.ainvoke(Command(resume="yes"), pruned)
        expected = await graph.ainvoke(Command(resume="yes"), control)
        assert resumed == expected
        assert resumed["log"][-1] == "decision:yes"
        assert "turn-3" in resumed["log"]

    async def test_a_thread_already_pruned_loses_nothing_more(self, executor: AsyncpgPoolAdapter) -> None:
        """a second prune deletes nothing."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _paused_thread(_graph(saver), "t-twice")
        await saver.aprune(["t-twice"])

        again = await saver.aprune(["t-twice"])

        assert (again.checkpoints_deleted, again.writes_deleted) == (0, 0)

    async def test_other_threads_are_untouched(self, executor: AsyncpgPoolAdapter) -> None:
        """only the named thread loses rows."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        await _paused_thread(graph, "t-named")
        await _paused_thread(graph, "t-other")
        other = await _rows(executor, "checkpoints", "t-other")
        other_writes = await _rows(executor, "checkpoint_writes", "t-other")

        await saver.aprune(["t-named"])

        assert await _rows(executor, "checkpoints", "t-other") == other
        assert await _rows(executor, "checkpoint_writes", "t-other") == other_writes

    async def test_delete_removes_every_row(self, executor: AsyncpgPoolAdapter) -> None:
        """the ``delete`` strategy is a purge of each thread."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _paused_thread(_graph(saver), "t-gone")
        checkpoints = len(await _rows(executor, "checkpoints", "t-gone"))
        writes = len(await _rows(executor, "checkpoint_writes", "t-gone"))

        result = await saver.aprune(["t-gone"], strategy="delete")

        assert await _rows(executor, "checkpoints", "t-gone") == []
        assert await _rows(executor, "checkpoint_writes", "t-gone") == []
        assert (result.checkpoints_deleted, result.writes_deleted) == (checkpoints, writes)


def _merge(state: list[str], batches: Any) -> list[str]:
    """a batching-invariant reducer for a DeltaChannel.

    :param state: the accumulated list
    :ptype state: list[str]
    :param batches: the writes, each a list
    :ptype batches: Any
    :return: the new list
    :rtype: list[str]
    """
    return [*state, *(item for batch in batches for item in batch)]


class _DeltaState(TypedDict):
    """a log kept in a DeltaChannel: rebuilt from ancestors' writes, not stored in each checkpoint."""

    log: Annotated[list[str], DeltaChannel(_merge)]


def _note(state: _DeltaState) -> dict[str, Any]:
    """
    :param state: graph state
    :ptype state: _DeltaState
    :return: the update
    :rtype: dict[str, Any]
    """
    return {"log": ["noted"]}


class TestDeltaChannelHistoryIsKept:
    """LangGraph's DeltaChannel warning: a namespace whose state lives in its ancestors is left whole."""

    async def test_a_delta_channel_thread_keeps_its_ancestors_and_its_state(self, executor: AsyncpgPoolAdapter) -> None:
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        builder = StateGraph(_DeltaState)
        builder.add_node("note", _note)
        builder.add_edge(START, "note")
        builder.add_edge("note", END)
        graph = builder.compile(checkpointer=saver)
        config: dict[str, Any] = {"configurable": {"thread_id": "t-delta"}}
        await graph.ainvoke({"log": ["one"]}, config)
        await graph.ainvoke({"log": ["two"]}, config)
        state_before = (await graph.aget_state(config)).values
        newest = await saver.aget_tuple(config)
        assert newest is not None
        assert DELTA_COUNTERS_METADATA_KEY in newest.metadata, "LangGraph no longer marks unsnapshotted delta state"
        rows_before = await _rows(executor, "checkpoints", "t-delta")

        result = await saver.aprune(["t-delta"])

        assert result.namespaces_held == 1
        assert (result.checkpoints_deleted, result.writes_deleted) == (0, 0)
        assert await _rows(executor, "checkpoints", "t-delta") == rows_before
        assert (await graph.aget_state(config)).values == state_before == {"log": ["one", "noted", "two", "noted"]}


class TestCustomerScope:
    """aprune reconciles its customer exactly as adelete_thread does."""

    async def _history(self, saver: ThreeTierCheckpointSaver, thread: str, customer: UUID | None = None) -> None:
        """three checkpoints on one thread through ``saver``.

        :param saver: the saver
        :ptype saver: ThreeTierCheckpointSaver
        :param thread: the caller's thread id
        :ptype thread: str
        :param customer: the customer a config-resolved saver reads
        :ptype customer: UUID | None
        """
        configurable: dict[str, Any] = {"thread_id": thread, "checkpoint_ns": ""}
        if customer is not None:
            configurable["customer_id"] = customer
        for n in (1, 2, 3):
            await saver.aput({"configurable": dict(configurable)}, _checkpoint(f"cp-{n}"), {}, {})

    async def test_a_customer_scoped_prune_reaches_only_that_customers_thread(
        self, executor: AsyncpgPoolAdapter
    ) -> None:
        saver_a = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.for_customer(_CUSTOMER_A))
        saver_b = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.for_customer(_CUSTOMER_B))
        await self._history(saver_a, "shared")
        await self._history(saver_b, "shared")

        result = await saver_a.aprune(["shared"])

        assert result.checkpoints_deleted == 2
        assert [r["checkpoint_id"] for r in await _rows(executor, "checkpoints", f"{_CUSTOMER_A}/shared")] == ["cp-3"]
        assert len(await _rows(executor, "checkpoints", f"{_CUSTOMER_B}/shared")) == 3

    async def test_a_config_resolved_saver_prunes_the_named_customer(self, executor: AsyncpgPoolAdapter) -> None:
        saver = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.from_config())
        await self._history(saver, "shared", _CUSTOMER_A)
        await self._history(saver, "shared", _CUSTOMER_B)

        await saver.aprune(["shared"], customer=_CUSTOMER_B)

        assert len(await _rows(executor, "checkpoints", f"{_CUSTOMER_A}/shared")) == 3
        assert len(await _rows(executor, "checkpoints", f"{_CUSTOMER_B}/shared")) == 1

    @pytest.mark.parametrize(
        ("scope", "customer"),
        [
            (CheckpointScope.from_config(), None),
            (CheckpointScope.for_customer(_CUSTOMER_A), _CUSTOMER_B),
            (_UNSCOPED, _CUSTOMER_A),
        ],
        ids=["config-resolved-without-customer", "other-customer", "unscoped-given-customer"],
    )
    async def test_refusals_mirror_adelete_thread(
        self, executor: AsyncpgPoolAdapter, scope: CheckpointScope, customer: UUID | None
    ) -> None:
        """the same calls adelete_thread refuses are refused, naming aprune, and delete nothing."""
        writer = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await self._history(writer, "t-refused")
        saver = ThreeTierCheckpointSaver(executor, scope=scope)

        with pytest.raises(ValueError, match="adelete_thread"):
            await saver.adelete_thread("t-refused", customer=customer)
        with pytest.raises(ValueError, match="aprune"):
            await saver.aprune(["t-refused"], customer=customer)

        assert len(await _rows(executor, "checkpoints", "t-refused")) == 3

    async def test_restating_the_scopes_own_customer_is_accepted(self, executor: AsyncpgPoolAdapter) -> None:
        saver = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.for_customer(_CUSTOMER_A))
        await self._history(saver, "t-own")

        result = await saver.aprune(["t-own"], customer=_CUSTOMER_A)

        assert result.checkpoints_deleted == 2


def _checkpoint(checkpoint_id: str, value: str = "state") -> dict[str, Any]:
    """a LangGraph checkpoint carrying ``value`` in one channel.

    :param checkpoint_id: the checkpoint's id
    :ptype checkpoint_id: str
    :param value: the channel's value
    :ptype value: str
    :return: the checkpoint mapping
    :rtype: dict[str, Any]
    """
    return {
        "id": checkpoint_id,
        "v": 4,
        "ts": "2026-11-13T00:00:00Z",
        "channel_values": {"state": value},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


class TestCaches:
    """a pruned checkpoint is not served from L1 or L2; the kept one still reads."""

    async def _seed(self, saver: ThreeTierCheckpointSaver) -> None:
        """three checkpoints in the root namespace and in ``inner``, each with a write.

        :param saver: the saver
        :ptype saver: ThreeTierCheckpointSaver
        """
        for ns in ("", "inner"):
            for n in (1, 2, 3):
                config = {"configurable": {"thread_id": "t-cache", "checkpoint_ns": ns}}
                stored = await saver.aput(config, _checkpoint(f"cp-{n}", f"{ns}-{n}"), {}, {})
                await saver.aput_writes(stored, [("state", f"{ns}-write-{n}")], f"task-{n}")

    async def test_stale_cached_bundles_are_evicted_and_the_kept_one_reads(self, executor: AsyncpgPoolAdapter) -> None:
        l1, l2 = _DictL1Cache(), _SweepingL2Cache()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED, l1_cache=l1, l2_cache=l2, l2_bucket=_BUCKET)
        await self._seed(saver)
        # an L1 that missed another pod's writes, and an L2 holding an older bundle: both name pruned checkpoints
        stale = saver.serialize_checkpoint_tuple(_checkpoint("cp-1", "stale"), {}, None, [])  # type: ignore[arg-type]
        l1.store[("t-cache", "")] = stale
        l2.store["t-cache.inner"] = stale

        await saver.aprune(["t-cache"])

        assert l1.store == {}
        assert l2.store == {}
        for ns in ("", "inner"):
            config = {"configurable": {"thread_id": "t-cache", "checkpoint_ns": ns}}
            pinned = {"configurable": {"thread_id": "t-cache", "checkpoint_ns": ns, "checkpoint_id": "cp-1"}}
            kept = await saver.aget_tuple(config)  # type: ignore[arg-type]
            assert kept is not None
            assert kept.checkpoint["id"] == "cp-3"
            assert kept.checkpoint["channel_values"] == {"state": f"{ns}-3"}
            assert [write[2] for write in kept.pending_writes or []] == [f"{ns}-write-3"]
            assert await saver.aget_tuple(pinned) is None  # type: ignore[arg-type]
        # the read refilled L1 with the kept checkpoint, which the next read is served from
        cached = saver.deserialize_checkpoint_tuple(l1.store[("t-cache", "")])
        assert cached["checkpoint"]["id"] == "cp-3"

    async def test_a_failed_eviction_is_raised_after_the_rows_are_gone(self, executor: AsyncpgPoolAdapter) -> None:
        l1, l2 = _DictL1Cache(), _SweepingL2Cache()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED, l1_cache=l1, l2_cache=l2, l2_bucket=_BUCKET)
        await self._seed(saver)
        l2.fail_delete = True

        with pytest.raises(CheckpointPurgeIncompleteError, match="4 checkpoint"):
            await saver.aprune(["t-cache"])

        assert len(await _rows(executor, "checkpoints", "t-cache")) == 2
        # until a retry lands, this saver reads the thread from L3: L2 still holds a bundle
        l1.store.clear()
        l2.store["t-cache"] = saver.serialize_checkpoint_tuple(
            _checkpoint("cp-1", "stale"),  # type: ignore[arg-type]
            {},
            None,
            [],
        )
        read = await saver.aget_tuple({"configurable": {"thread_id": "t-cache", "checkpoint_ns": ""}})
        assert read is not None
        assert read.checkpoint["id"] == "cp-3"

        l2.fail_delete = False
        again = await saver.aprune(["t-cache"])
        assert again.checkpoints_deleted == 0
        assert l2.store == {}


# ---------------------------------------------------------------------------
# checkpoint_maintenance
# ---------------------------------------------------------------------------


async def _seed_estate(executor: AsyncpgPoolAdapter) -> ThreeTierCheckpointSaver:
    """three threads: ``owned-a`` with history in two namespaces, ``owned-b`` with none, ``gone`` with history.

    Every checkpoint carries one write. An ``owned`` table names the first two.

    :param executor: the database
    :ptype executor: AsyncpgPoolAdapter
    :return: an unscoped saver over it
    :rtype: ThreeTierCheckpointSaver
    """
    saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
    layout = {("owned-a", ""): 4, ("owned-a", "inner"): 3, ("owned-b", ""): 1, ("gone", ""): 2}
    for (thread, ns), count in layout.items():
        for n in range(1, count + 1):
            config = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
            stored = await saver.aput(config, _checkpoint(f"cp-{n}", "x" * 500 * n), {}, {})
            await saver.aput_writes(stored, [("state", "w" * 300 * n)], f"task-{n}")
    await executor.execute("CREATE TABLE owned (thread_id text PRIMARY KEY)")
    await executor.execute("INSERT INTO owned VALUES ('owned-a'), ('owned-b')")
    return saver


_OWNED_SQL = "SELECT thread_id FROM owned;"


async def _blob_bytes(executor: AsyncpgPoolAdapter, where_checkpoints: str, where_writes: str) -> int:
    """stored blob bytes of the rows each predicate selects, computed independently of the code under test.

    :param executor: the database
    :ptype executor: AsyncpgPoolAdapter
    :param where_checkpoints: predicate over ``checkpoints``
    :ptype where_checkpoints: str
    :param where_writes: predicate over ``checkpoint_writes``
    :ptype where_writes: str
    :return: the byte total
    :rtype: int
    """
    row = await executor.fetchrow(
        "SELECT (SELECT coalesce(sum(pg_column_size(checkpoint) + coalesce(pg_column_size(metadata_), 0)), 0) "
        f"FROM checkpoints WHERE {where_checkpoints}) + (SELECT coalesce(sum(pg_column_size(blob)), 0) "
        f"FROM checkpoint_writes WHERE {where_writes}) AS total"
    )
    assert row is not None
    return int(row["total"])


async def _counts(executor: AsyncpgPoolAdapter) -> tuple[int, int]:
    """
    :param executor: the database
    :ptype executor: AsyncpgPoolAdapter
    :return: rows in ``checkpoints`` and in ``checkpoint_writes``
    :rtype: tuple[int, int]
    """
    row = await executor.fetchrow(
        "SELECT (SELECT count(*) FROM checkpoints) AS c, (SELECT count(*) FROM checkpoint_writes) AS w"
    )
    assert row is not None
    return int(row["c"]), int(row["w"])


#: the rows the seeded estate's keep_latest removes: everything but cp-4 / cp-3 / cp-1 / cp-2 per namespace
_PRUNABLE = (
    "(thread_id = 'owned-a' AND checkpoint_ns = '' AND checkpoint_id <> 'cp-4') OR "
    "(thread_id = 'owned-a' AND checkpoint_ns = 'inner' AND checkpoint_id <> 'cp-3') OR "
    "(thread_id = 'gone' AND checkpoint_id <> 'cp-2')"
)


async def _analyze(database: _Database) -> None:
    """refresh both tables' statistics and publish them now, so the row estimates are current.

    :param database: the schema
    :ptype database: _Database
    """
    conn = await asyncpg.connect(database.dsn, server_settings={"search_path": database.schema})
    try:
        await conn.execute("ANALYZE checkpoints")
        await conn.execute("ANALYZE checkpoint_writes")
        # a backend publishes its statistics lazily; this makes it publish at the end of the next statement
        await conn.execute("SELECT pg_stat_force_next_flush()")
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


class TestCheckpointUsage:
    """the report's numbers, on a seeded table."""

    async def test_counts_and_prunable_rows(self, database: _Database) -> None:
        executor = database.executor
        await _seed_estate(executor)
        await _analyze(database)

        usage = await checkpoint_usage(executor, owned_threads_sql=_OWNED_SQL)

        # the row figures are the statistics' estimates, current after ANALYZE
        assert (usage.checkpoints.rows, usage.writes.rows) == (10, 10)
        assert usage.checkpoints.table == "checkpoints"
        assert usage.checkpoints.total_bytes >= usage.checkpoints.heap_bytes > 0
        assert usage.threads == 3
        assert usage.threads_with_history == 2
        assert (usage.prunable_checkpoint_rows, usage.prunable_write_rows) == (6, 6)
        assert usage.prunable_bytes == await _blob_bytes(executor, _PRUNABLE, _PRUNABLE)
        assert usage.held_namespaces == 0
        assert usage.unowned is not None
        assert usage.unowned.owned_threads == 2
        assert (usage.unowned.threads, usage.unowned.checkpoint_rows, usage.unowned.write_rows) == (1, 2, 2)
        assert usage.unowned.bytes == await _blob_bytes(executor, "thread_id = 'gone'", "thread_id = 'gone'")

    async def test_after_a_prune_nothing_is_prunable(self, database: _Database) -> None:
        """the report agrees with the prune, and without an owned query has no unowned figures."""
        executor = database.executor
        saver = await _seed_estate(executor)
        await saver.aprune(["owned-a", "gone"])
        await _analyze(database)

        usage = await checkpoint_usage(executor)

        assert usage.unowned is None
        assert (usage.prunable_checkpoint_rows, usage.prunable_write_rows, usage.prunable_bytes) == (0, 0, 0)
        assert (usage.checkpoints.rows, usage.writes.rows) == (4, 4)
        assert usage.threads_with_history == 0


class TestPruneThreads:
    """the batch prune: a dry run changes nothing and predicts the real run exactly."""

    @pytest.mark.parametrize("batch", [1, 200])
    async def test_dry_run_changes_nothing_and_equals_the_real_run(
        self, executor: AsyncpgPoolAdapter, batch: int
    ) -> None:
        saver = await _seed_estate(executor)
        prunable_bytes = await _blob_bytes(executor, _PRUNABLE, _PRUNABLE)
        before = await _counts(executor)

        dry = await prune_threads(saver, executor, batch=batch, dry_run=True)

        assert await _counts(executor) == before
        assert (dry.threads, dry.checkpoint_rows, dry.write_rows, dry.bytes) == (2, 6, 6, prunable_bytes)
        assert dry.dry_run

        real = await prune_threads(saver, executor, batch=batch)

        assert (real.threads, real.checkpoint_rows, real.write_rows, real.bytes, real.namespaces_held) == (
            dry.threads,
            dry.checkpoint_rows,
            dry.write_rows,
            dry.bytes,
            dry.namespaces_held,
        )
        assert real.batches == (2 if batch == 1 else 1)
        assert await _counts(executor) == (before[0] - 6, before[1] - 6)
        assert (await checkpoint_usage(executor)).prunable_checkpoint_rows == 0

    async def test_limit_stops_after_that_many_threads(self, executor: AsyncpgPoolAdapter) -> None:
        saver = await _seed_estate(executor)

        sweep = await prune_threads(saver, executor, batch=1, limit=1)

        assert (sweep.threads, sweep.checkpoint_rows) == (1, 1)
        assert len(await _rows(executor, "checkpoints", "owned-a")) == 7

    async def test_a_scoped_saver_is_refused(self, executor: AsyncpgPoolAdapter) -> None:
        await _seed_estate(executor)
        saver = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.for_customer(_CUSTOMER_A))

        with pytest.raises(ValueError, match="unscoped"):
            await prune_threads(saver, executor)

        assert await _counts(executor) == (10, 10)


class TestDeleteUnownedThreads:
    """only threads outside the app's owned set are deleted."""

    async def test_dry_run_then_real_run_delete_only_the_unowned_thread(self, executor: AsyncpgPoolAdapter) -> None:
        saver = await _seed_estate(executor)
        # a thread with writes and no checkpoint is still a thread
        await executor.execute(
            "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, blob) "
            "VALUES ('orphan', '', 'cp-1', 'task', 0, 'state', '\\x00'::bytea)"
        )
        owned_a = await _rows(executor, "checkpoints", "owned-a")
        owned_b = await _rows(executor, "checkpoint_writes", "owned-b")
        unowned_bytes = await _blob_bytes(
            executor, "thread_id IN ('gone', 'orphan')", "thread_id IN ('gone', 'orphan')"
        )

        dry = await delete_unowned_threads(saver, executor, owned_threads_sql=_OWNED_SQL, batch=1, dry_run=True)

        assert await _counts(executor) == (10, 11)
        assert (dry.threads, dry.checkpoint_rows, dry.write_rows, dry.bytes) == (2, 2, 3, unowned_bytes)

        real = await delete_unowned_threads(saver, executor, owned_threads_sql=_OWNED_SQL, batch=1)

        assert (real.threads, real.checkpoint_rows, real.write_rows, real.bytes) == (2, 2, 3, unowned_bytes)
        assert await _rows(executor, "checkpoints", "gone") == []
        assert await _rows(executor, "checkpoint_writes", "orphan") == []
        assert await _rows(executor, "checkpoints", "owned-a") == owned_a
        assert await _rows(executor, "checkpoint_writes", "owned-b") == owned_b

    async def test_an_owned_query_returning_nothing_is_refused(self, executor: AsyncpgPoolAdapter) -> None:
        saver = await _seed_estate(executor)

        with pytest.raises(ValueError, match="no rows"):
            await delete_unowned_threads(saver, executor, owned_threads_sql="SELECT thread_id FROM owned WHERE false")

        assert await _counts(executor) == (10, 10)


class TestCommandLine:
    """``python -m threetears.langgraph.checkpoint_maintenance`` against the test database."""

    async def _run(self, *args: str) -> tuple[int, str, str]:
        """run the module in a fresh interpreter.

        :param args: its arguments
        :ptype args: str
        :return: exit status, stdout, stderr
        :rtype: tuple[int, str, str]
        """
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "threetears.langgraph.checkpoint_maintenance",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        assert proc.returncode is not None
        return proc.returncode, out.decode(), err.decode()

    async def test_report_runs_against_the_database(self, database: _Database) -> None:
        await _seed_estate(database.executor)
        base = ["--dsn", database.dsn, "--schema", database.schema, "--owned-sql", _OWNED_SQL]

        status, out, err = await self._run("report", *base, "--json")
        assert status == 0, err
        report = json.loads(out)
        assert report["prunable_checkpoint_rows"] == 6
        assert report["unowned"]["threads"] == 1

        status, out, err = await self._run("report", *base)
        assert status == 0, err
        assert "would remove 6 checkpoint rows" in out
        assert "VACUUM FULL" in out

    async def test_prune_runs_only_with_yes_or_dry_run(self, database: _Database) -> None:
        await _seed_estate(database.executor)
        base = ["--dsn", database.dsn, "--schema", database.schema]

        status, _, err = await self._run("prune", *base)
        assert status == 2
        assert "--yes" in err
        assert await _counts(database.executor) == (10, 10)

        status, out, err = await self._run("prune", *base, "--yes", "--vacuum", "--json")
        assert status == 0, err
        assert json.loads(out)["checkpoint_rows"] == 6
        assert await _counts(database.executor) == (4, 4)
