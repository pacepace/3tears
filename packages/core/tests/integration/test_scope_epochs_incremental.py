"""Integration: a reader of the scope epochs reads only what changed, and answers what a whole read answers.

A copy follower asks the epochs twice a pass. Each ask read the whole table: on an election night, a whole-table read
a pass for a change of a few scopes. After its first read a :class:`ScopeEpochs` reads the writer's record, the
scopes still pending, and the scopes moved past the version it last read: the rows of the change, one statement. A
record it cannot trust (the table emptied, replaced or restored behind it) is read whole again, a cold start counted by
why; a scope whose epoch went back, or whose pending mark cleared unseen, is read again by key, never the table.

What a fake cannot prove: the statement, its paging past the rail's cap, and that real begins, touches, commits and
abandons (each its own transaction on a real Postgres) leave nothing a read of the changes misses. Uses the
session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import random
import uuid
from functools import partial
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from prometheus_client import REGISTRY
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import scope_epochs as scope_epochs_module
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, Unsettled
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.scope_epochs import (
    EpochSnapshot,
    ScopeEpochs,
    WriterLease,
    scope_epochs_collection,
    scope_epochs_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.lease import LeaseHolding
from threetears.core.keyset import read_keyset_pages
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem
from threetears.core.testing.plans import PlanRecorder, StatementPlan, plan_uses_index

pytestmark = pytest.mark.integration

_TABLE = "scope_epochs"


class _CountingL3(SqlL3Backend):
    """the L3 a reader reads through: counts the statements and rows it answers, and may run a step after one."""

    def __init__(self, pool: Any) -> None:
        super().__init__(pool)
        self.statements = 0
        self.rows = 0
        #: each statement answered: its text, its parameters and its rows
        self.seen: list[tuple[str, tuple[Any, ...], list[dict[str, Any]]]] = []
        #: run once, after the next statement answers (a commit landing while a read pages)
        self.after_next: Callable[[], Awaitable[Any]] | None = None

    def reset(self) -> None:
        self.statements = 0
        self.rows = 0
        self.seen = []

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        rows = await super().fetch(query, *params, namespace=namespace, **kwargs)
        self.statements += 1
        self.rows += len(rows)
        self.seen.append((query, params, rows))
        step, self.after_next = self.after_next, None
        if step is not None:
            await step()
        return rows


class _Lease:
    """the writers' lease as :meth:`KVLease.held` answers it: held by a live writer, or free."""

    def __init__(self, *, held: bool) -> None:
        self.holding = held

    async def held(self, key: str) -> LeaseHolding | None:
        return LeaseHolding(holder="pod-a", acquired_at=datetime.now(UTC)) if self.holding else None


async def _schema_pool(db_container: str) -> asyncpg.Pool:
    schema = f"epochs_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    # the table as the owner declares it (scope_epochs_schema), its indexes with it
    table = scope_epochs_schema(_TABLE).to_sqlalchemy_table(MetaData())
    await pool.execute(str(CreateTable(table).compile(dialect=postgresql.dialect())))
    for index in table.indexes:
        await pool.execute(str(CreateIndex(index).compile(dialect=postgresql.dialect())))
    return pool


def _reader(pool: asyncpg.Pool, *, page_size: int | None = None, writer: Any = None) -> tuple[ScopeEpochs, _CountingL3]:
    registry = CollectionRegistry()
    l1 = SQLiteBackend(db_name=f"epochs_{uuid.uuid4().hex[:8]}")
    collection_class = scope_epochs_collection(_TABLE)
    metadata = MetaData()
    collection_class.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    l3 = _CountingL3(pool)
    registry.configure(l1_backend=l1, l3_pool=l3, kv_key_scope="epochs-pod")
    collection = collection_class(registry, DefaultCoreConfig(), None)
    kwargs: dict[str, Any] = {} if page_size is None else {"page_size": page_size}
    return ScopeEpochs(collection, writer=writer, **kwargs), l3


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    held = await _schema_pool(db_container)
    try:
        yield held
    finally:
        await held.close()


async def _in_transaction(pool: asyncpg.Pool, step: Callable[[Any], Awaitable[Any]]) -> Any:
    async with pool.acquire() as conn, CallerTransaction(conn):
        return await step(conn)


async def _write(pool: asyncpg.Pool, epochs: ScopeEpochs, scopes: set[str]) -> int:
    version: int = await _in_transaction(pool, lambda conn: epochs.begin(conn=conn))
    await _in_transaction(pool, lambda conn: epochs.commit(version, scopes, conn=conn))
    return version


def _whole_reads(why: str) -> float:
    return REGISTRY.get_sample_value("scope_epochs_whole_reads_total", {"table": _TABLE, "why": why}) or 0.0


async def _whole(pool: asyncpg.Pool) -> EpochSnapshot:
    """what a whole read answers now: a fresh reader's first read."""
    fresh, _ = _reader(pool)
    return await fresh.snapshot()


async def test_a_settled_reader_reads_only_the_rows_of_a_change_and_the_whole_table_once(
    pool: asyncpg.Pool,
) -> None:
    """300 scopes held; each pass changes two. The old reader read all 301 rows a call, twice a pass."""
    writer, _ = _reader(pool)
    await _write(pool, writer, {f"race:r{index:03d}" for index in range(300)})
    reader, l3 = _reader(pool)
    first = _whole_reads("first")

    assert isinstance(await reader.settled(), EpochSnapshot)
    counted = _whole_reads("first") - first
    for step in range(5):
        changed = {f"race:r{step:03d}", f"state:S{step}"}
        version = await _write(pool, writer, changed)
        l3.reset()
        before = _whole_reads("first")

        settled = await reader.settled()

        assert _whole_reads("first") == before, "the reader read the whole table again"
        assert isinstance(settled, EpochSnapshot) and settled == await _whole(pool)
        assert all(settled.epoch(scope) == version for scope in changed)
        assert (l3.statements, l3.rows) == (1, 1 + len(changed)), (
            f"a read after a change of {len(changed)} scopes took {l3.statements} statements and {l3.rows} rows"
        )
        l3.reset()
        assert await reader.settled() == settled
        assert (l3.statements, l3.rows) == (1, 1), "a read with nothing changed read more than the writer's record"
    assert counted == 1, "a reader's first read was not counted as its whole read"


async def test_a_change_of_more_scopes_than_a_page_is_read_whole_in_pages(pool: asyncpg.Pool) -> None:
    writer, _ = _reader(pool)
    await _write(pool, writer, {"state:VA"})
    reader, _ = _reader(pool, page_size=3)
    await reader.snapshot()

    await _write(pool, writer, {f"race:r{index:02d}" for index in range(10)})

    assert await reader.snapshot() == await _whole(pool)


async def test_a_commit_landing_while_a_read_pages_is_taken_by_the_next_read(pool: asyncpg.Pool) -> None:
    """the writer's record is read in the first statement, so scopes a commit moves after it are past the version kept."""
    writer, _ = _reader(pool)
    await _write(pool, writer, {"state:VA"})
    reader, l3 = _reader(pool, page_size=2)
    await reader.snapshot()
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    scopes = {f"race:r{index}" for index in range(6)}
    await _in_transaction(pool, lambda conn: writer.touch(version, scopes, conn=conn))

    async def commit() -> None:
        await _in_transaction(pool, lambda conn: writer.commit(version, {"state:VA"}, conn=conn))

    l3.after_next = commit
    await reader.snapshot()  # its first page read the write in progress; the commit landed before the rest

    assert await reader.snapshot() == await _whole(pool)


async def test_a_reader_of_changes_answers_what_a_whole_read_answers_after_every_step(pool: asyncpg.Pool) -> None:
    """begins, touches, commits, superseded writes, rolled-back touches and abandons, interleaved: after each, the
    reader that reads only changes (paged three rows a statement) answers what a fresh reader's whole read answers."""
    rng = random.Random(20261109)
    scopes = [f"state:S{index:02d}" for index in range(12)]
    lease = _Lease(held=True)
    writer, _ = _reader(pool)
    reader, _ = _reader(pool, page_size=3, writer=WriterLease(lease, "writes"))  # type: ignore[arg-type]
    untrusted = {why: _whole_reads(why) for why in ("reset", "version_backwards", "epoch_backwards", "pending_lost")}
    current: int | None = None
    done: list[str] = []

    async def touch(conn: Any, version: int, touched: set[str]) -> None:
        await writer.touch(version, touched, conn=conn)

    async def commit(conn: Any, version: int, named: set[str]) -> None:
        await writer.commit(version, named, conn=conn)

    async def touch_then_fail(conn: Any, version: int) -> None:
        await writer.touch(version, set(rng.sample(scopes, 2)), conn=conn)
        raise ConnectionError("the data write failed")

    for _ in range(80):
        step = rng.choice(["begin", "touch", "touch", "commit", "rolled_back", "abandon", "stale_commit", "read"])
        if step == "begin" or current is None:
            step = "begin"
            current = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
        else:
            version: int = current
            if step == "touch":
                touched = set(rng.sample(scopes, rng.randint(1, 5)))
                await _in_transaction(pool, partial(touch, version=version, touched=touched))
            elif step == "commit":
                named = set(rng.sample(scopes, rng.randint(0, 3)))
                await _in_transaction(pool, partial(commit, version=version, named=named))
                current = None
            elif step == "rolled_back":
                with pytest.raises(ConnectionError):
                    await _in_transaction(pool, partial(touch_then_fail, version=version))
            elif step == "abandon":
                lease.holding = False
                assert isinstance(await reader.settled(), EpochSnapshot), (
                    "a write whose writer is gone was not abandoned"
                )
                lease.holding = True
                current = None
            elif step == "stale_commit":
                with pytest.raises(ValueError, match="in progress"):
                    await _in_transaction(pool, partial(commit, version=version - 1, named=set()))
        done.append(step)

        whole = await _whole(pool)
        settled = await reader.settled()
        assert await reader.snapshot() == whole, f"after {done}"
        if whole.writing is None:
            assert settled == whole, f"after {done}"
        else:
            assert isinstance(settled, Unsettled), f"after {done}"
            assert (dict(settled.epochs or {}), settled.pending, settled.version) == (
                dict(whole.epochs),
                whole.pending,
                whole.version,
            ), f"after {done}"
    assert {"begin", "touch", "commit", "rolled_back", "abandon", "stale_commit"} <= set(done)
    assert {why: _whole_reads(why) for why in untrusted} == untrusted, "a read of the changes was not trusted"


# ----------------------------------------------------------------------
# a record a read of its changes cannot follow is read whole again, counted by why, and answers what a whole read does
# ----------------------------------------------------------------------


async def _held(pool: asyncpg.Pool) -> tuple[ScopeEpochs, ScopeEpochs, _CountingL3]:
    writer, _ = _reader(pool)
    await _write(pool, writer, {"state:VA", "state:MD"})
    await _write(pool, writer, {"state:VA"})
    reader, l3 = _reader(pool)
    await reader.snapshot()
    return writer, reader, l3


async def _held_reader(pool: asyncpg.Pool) -> tuple[ScopeEpochs, ScopeEpochs]:
    writer, reader, _ = await _held(pool)
    return writer, reader


async def _read_whole_again(pool: asyncpg.Pool, reader: ScopeEpochs, why: str) -> None:
    before = _whole_reads(why)

    snapshot = await reader.snapshot()

    assert snapshot == await _whole(pool)
    assert _whole_reads(why) == before + 1, f"no whole read counted as {why}"


async def test_a_record_emptied_and_written_again_is_read_whole(pool: asyncpg.Pool) -> None:
    writer, reader = await _held_reader(pool)
    await pool.execute(f"DELETE FROM {_TABLE}")
    await _write(pool, writer, {"state:TX"})

    await _read_whole_again(pool, reader, "reset")


async def test_a_record_gone_is_read_whole(pool: asyncpg.Pool) -> None:
    _writer, reader = await _held_reader(pool)
    await pool.execute(f"DELETE FROM {_TABLE} WHERE scope = '*'")

    await _read_whole_again(pool, reader, "reset")


async def test_a_version_gone_back_is_read_whole(pool: asyncpg.Pool) -> None:
    _writer, reader = await _held_reader(pool)
    await pool.execute(f"UPDATE {_TABLE} SET epoch = epoch - 1 WHERE scope = '*'")

    await _read_whole_again(pool, reader, "version_backwards")


#: the whys a whole read is counted under
_WHYS = ("first", "reset", "version_backwards")


async def _held_among_many(pool: asyncpg.Pool) -> tuple[ScopeEpochs, ScopeEpochs, _CountingL3]:
    """a held reader of a table of 53 scopes, every one read."""
    writer, reader, l3 = await _held(pool)
    await _write(pool, writer, {f"race:r{index:03d}" for index in range(50)})
    await reader.snapshot()
    return writer, reader, l3


async def _read_again_by_key(pool: asyncpg.Pool, reader: ScopeEpochs, l3: _CountingL3) -> None:
    """the reader's next snapshot is a whole read's answer, read without the table: the change's statement, then one
    keyed read of the scopes it could not merge; no whole read counted under any why, the retired whys included.
    The count is taken before the comparison's own whole read (a fresh reader's first)."""
    whys = (*_WHYS, "epoch_backwards", "pending_lost")
    before = {why: _whole_reads(why) for why in whys}
    l3.reset()

    snapshot = await reader.snapshot()

    after = {why: _whole_reads(why) for why in whys}
    assert snapshot == await _whole(pool)
    assert after == before, "the table was read whole"
    assert l3.statements == 2 and l3.rows < 10, f"read {l3.rows} rows in {l3.statements} statements, not by key"


async def test_an_epoch_gone_back_is_read_again_by_key(pool: asyncpg.Pool) -> None:
    """Flips if a scope whose epoch went back is read with the whole table (coordinator ruling 2026-11-12), or the
    snapshot after it is not what a whole read answers."""
    writer, reader, l3 = await _held_among_many(pool)
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    await _in_transaction(pool, lambda conn: writer.touch(version, {"state:VA"}, conn=conn))
    await pool.execute(f"UPDATE {_TABLE} SET epoch = 1 WHERE scope = 'state:VA'")

    await _read_again_by_key(pool, reader, l3)


async def test_a_pending_mark_cleared_without_its_epoch_moving_is_read_again_by_key(pool: asyncpg.Pool) -> None:
    """the case a read of the changes cannot see: the scope is no longer pending, and its epoch did not move past.
    Flips if that scope is read with the whole table, or the snapshot after it is not what a whole read answers."""
    writer, reader, l3 = await _held_among_many(pool)
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    await _in_transaction(pool, lambda conn: writer.touch(version, {"state:MD"}, conn=conn))
    assert (await reader.snapshot()).pending == frozenset({"state:MD"})
    await pool.execute(f"UPDATE {_TABLE} SET pending = NULL WHERE scope = 'state:MD'")

    await _read_again_by_key(pool, reader, l3)


async def test_read_whole_is_reached_only_at_a_first_read_a_reset_or_a_restore(pool: asyncpg.Pool) -> None:
    """the proof of ScopeEpochs.snapshot's and ScopeEpochs._read_whole's whole_read(cold_start) declarations: the
    table is read whole at a reader's first snapshot, after it was emptied or its record is gone (reset), and after it
    was restored behind the version held; a change, an epoch gone back and a pending mark lost never reach it."""
    before = {why: _whole_reads(why) for why in _WHYS}
    writer, reader, _ = await _held(pool)
    assert _whole_reads("first") == before["first"] + 1, "a first snapshot did not reach _read_whole"
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    await _in_transaction(pool, lambda conn: writer.touch(version, {"state:VA", "state:MD"}, conn=conn))
    await reader.snapshot()
    await pool.execute(f"UPDATE {_TABLE} SET epoch = 1 WHERE scope = 'state:VA'")
    await pool.execute(f"UPDATE {_TABLE} SET pending = NULL WHERE scope = 'state:MD'")
    await _write(pool, writer, {"state:TX"})
    assert await reader.snapshot() == await _whole(pool)
    assert {why: _whole_reads(why) for why in ("reset", "version_backwards")} == {
        "reset": before["reset"],
        "version_backwards": before["version_backwards"],
    }, "a change, an epoch gone back or a pending mark lost reached _read_whole"
    await pool.execute(f"UPDATE {_TABLE} SET epoch = epoch - 1 WHERE scope = '*'")
    await _read_whole_again(pool, reader, "version_backwards")
    await pool.execute(f"DELETE FROM {_TABLE} WHERE scope = '*'")
    await _read_whole_again(pool, reader, "reset")


#: scopes seeded for the speed test: past an election night's scopes (states, races, counties)
_SPEED_SCOPES = 20_000


@pytest.mark.loop_timing
async def test_read_whole_speed_at_real_size(pool: asyncpg.Pool) -> None:
    """a reader's first snapshot (ScopeEpochs._read_whole: the writer's record, then every scope a keyset page at a
    time, held and built) within 2x a bare keyset read of the same pages, interleaved on the same database."""
    now = datetime.now(UTC)
    await pool.copy_records_to_table(
        _TABLE,
        records=[("*", 2, 1, None, None, now, now)]
        + [(f"race:r{index:05d}", 2, 1, None, None, now, now) for index in range(_SPEED_SCOPES)],
        columns=["scope", "epoch", "previous_epoch", "writing", "pending", "date_created", "date_updated"],
    )
    await pool.execute(f"ANALYZE {_TABLE}")
    selected = "scope, epoch, previous_epoch, writing, pending, date_created, date_updated"

    async def bare() -> None:
        read = await read_keyset_pages(
            SqlL3Backend(pool), f"SELECT {selected} FROM {_TABLE}", ("scope",), page_size=DEFAULT_PAGE_SIZE
        )
        assert len(read.rows) == _SPEED_SCOPES + 1

    async def first_snapshot() -> None:
        fresh, _ = _reader(pool)
        assert len((await fresh.snapshot()).epochs) == _SPEED_SCOPES

    baseline, took = await best_seconds_interleaved(bare, first_snapshot)
    print(f"first snapshot of {_SPEED_SCOPES} scopes {took:.2f} s, bare keyset read {baseline:.2f} s")
    problem = budget_problem(f"a first snapshot (_read_whole) of {_SPEED_SCOPES} scopes", took, baseline)
    assert problem is None, problem


# ----------------------------------------------------------------------
# the plans of the reads of a change and of a commit, through the epochs table's two indexes (scope_epochs_schema)
# ----------------------------------------------------------------------

#: the scopes seeded for the plan tests, past where the planner would rather read every row
_PLANNED_SCOPES = 20_000


async def _seeded(pool: asyncpg.Pool) -> ScopeEpochs:
    """a table of many scopes, every one moved at version 2, and a writer over it."""
    now = datetime.now(UTC)
    await pool.copy_records_to_table(
        _TABLE,
        records=[("*", 2, 1, None, None, now, now)]
        + [(f"race:r{index:05d}", 2, 1, None, None, now, now) for index in range(_PLANNED_SCOPES)],
        columns=["scope", "epoch", "previous_epoch", "writing", "pending", "date_created", "date_updated"],
    )
    await pool.execute(f"ANALYZE {_TABLE}")
    writer, _ = _reader(pool)
    return writer


async def _plan(pool: asyncpg.Pool, sql: str, params: tuple[Any, ...]) -> StatementPlan:
    """the plan of one statement a reader sent, as it sent it."""
    async with pool.acquire() as conn:
        recorder = PlanRecorder(conn)
        await recorder.fetch(sql, *params)
        [plan] = recorder.plans
        return plan


class _ReadsPlanned:
    """a commit's connection whose reads are planned (a :class:`PlanRecorder`'s); its writes (the fence) run
    unplanned on the connection."""

    def __init__(self, conn: Any) -> None:
        self._raw = conn
        self.recorder = PlanRecorder(conn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.recorder, name)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return await self._raw.fetchrow(sql, *args)


async def _change_read(pool: asyncpg.Pool, reader: ScopeEpochs, l3: _CountingL3) -> tuple[str, tuple[Any, ...]]:
    """the one statement a held reader's next snapshot reads its change by (ScopeEpochs._read_changed)."""
    l3.reset()
    await reader.snapshot()
    assert l3.statements == 1, [query for query, _, _ in l3.seen]
    query, params, _ = l3.seen[0]
    return query, params


async def test_the_pending_scopes_are_read_through_their_partial_index(pool: asyncpg.Pool) -> None:
    """Flips if a commit's read of the pending scopes (ScopeEpochs.commit) or a reader's read of them in a change
    (ScopeEpochs._read_changed) stops planning through scope_epochs_pending: five scopes pending among 20,000."""
    writer = await _seeded(pool)
    reader, l3 = _reader(pool)
    await reader.snapshot()
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    touched = {f"race:r{index:05d}" for index in range(100, 105)}
    await _in_transaction(pool, lambda conn: writer.touch(version, touched, conn=conn))

    query, params = await _change_read(pool, reader, l3)
    assert plan_uses_index(await _plan(pool, query, params), "scope_epochs_pending", partial=True)

    async with pool.acquire() as conn:
        recorder = _ReadsPlanned(conn)
        async with CallerTransaction(recorder):
            await writer.commit(version, (), conn=recorder)
    [plan] = [plan for plan in recorder.recorder.plans if '"pending" IS NOT NULL' in plan.sql]
    assert plan_uses_index(plan, "scope_epochs_pending", partial=True), plan.plan
    assert (await _whole(pool)).epochs == {
        **{f"race:r{index:05d}": 2 for index in range(_PLANNED_SCOPES)},
        **dict.fromkeys(touched, version),
    }


async def test_the_scopes_moved_past_a_version_are_read_through_the_epoch_index(pool: asyncpg.Pool) -> None:
    """Flips if a reader's read of the scopes a commit moved past the version it holds (ScopeEpochs._read_changed)
    stops planning through scope_epochs_epoch: three scopes moved among 20,000."""
    writer = await _seeded(pool)
    reader, l3 = _reader(pool)
    await reader.snapshot()
    await _write(pool, writer, {"race:r00007", "race:r00500", "race:r19000"})

    query, params = await _change_read(pool, reader, l3)
    assert plan_uses_index(await _plan(pool, query, params), "scope_epochs_epoch")


# ----------------------------------------------------------------------
# old vs new: the keyed reads answer the rows the whole-table statements they replaced answered (d1, 2026-11-14)
# ----------------------------------------------------------------------

_SELECTED = "scope, epoch, previous_epoch, writing, pending, date_created, date_updated"

#: ScopeEpochs._read_changed before its keyed legs: one OR over every scope's row
_OLD_CHANGED = f"SELECT {_SELECTED} FROM {_TABLE} WHERE scope = '*' OR pending IS NOT NULL OR epoch > $1"  # noqa: S608 - test table

#: ScopeEpochs.commit before the partial index: the look for a pending scope at or past the version, then the moved
_OLD_BEHIND = (
    f"SELECT scope FROM {_TABLE} WHERE pending IS NOT NULL AND scope <> $2 AND epoch >= $1 "  # noqa: S608 - test table
    "ORDER BY scope LIMIT 5"
)
_OLD_MOVED = f"SELECT scope FROM {_TABLE} WHERE pending IS NOT NULL AND scope <> $1 ORDER BY scope"  # noqa: S608


def _by_scope(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["scope"]): {column: row[column] for column in _SELECTED.split(", ")} for row in rows}


class _ComparingConn:
    """a commit's connection that, at the commit's first read of the pending scopes, answers in the same transaction
    what the old statements answered, and keeps what the commit's own reads answered."""

    def __init__(self, conn: Any, version: int) -> None:
        self._conn = conn
        self._version = version
        self.old: tuple[list[str], list[str]] | None = None
        self.new: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        if '"pending" IS NOT NULL' in sql and self.old is None:
            behind = await self._conn.fetch(_OLD_BEHIND, self._version, "*")
            moved = await self._conn.fetch(_OLD_MOVED, "*")
            self.old = ([str(row["scope"]) for row in behind], [str(row["scope"]) for row in moved])
        rows = list(await self._conn.fetch(sql, *params))
        if '"pending" IS NOT NULL' in sql:
            self.new += [str(row["scope"]) for row in rows]
        return rows


async def test_the_keyed_reads_answer_what_the_whole_table_statements_answered(
    pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reader's read of a change (ScopeEpochs._read_changed: its one statement of keyed legs, and the legs'
    keyset pages past a page) names other rows than the old OR statement over every scope, or a commit
    (ScopeEpochs.commit) moves other scopes, or refuses other than the old look refused: after every step of a random
    walk of begins, touches, commits, abandons, hand-set epochs and hand-marked records, on three-row pages."""
    monkeypatch.setattr(scope_epochs_module, "DEFAULT_PAGE_SIZE", 3)
    rng = random.Random(20261114)
    scopes = [f"state:S{index:02d}" for index in range(12)]
    lease = _Lease(held=True)
    writer, _ = _reader(pool)
    reader, l3 = _reader(pool, page_size=3, writer=WriterLease(lease, "writes"))  # type: ignore[arg-type]
    held = await reader.snapshot()
    current: int | None = None
    compared = {"changes": 0, "paged": 0, "commits": 0, "refused": 0}

    async def commit(version: int, named: set[str]) -> None:
        async with pool.acquire() as raw:
            conn = _ComparingConn(raw, version)
            try:
                async with CallerTransaction(conn):
                    await writer.commit(version, named, conn=conn)
            except ValueError as refused:
                assert conn.old is not None and conn.old[0], f"refused where the old look found none: {refused}"
                assert str(conn.old[0]) in str(refused), (conn.old, str(refused))
                compared["refused"] += 1
                return
            assert conn.old is not None and not conn.old[0], f"the old look refused {conn.old}"
            assert [scope for scope in conn.new if scope != "*"] == conn.old[1]
            compared["commits"] += 1

    for _ in range(120):
        step = rng.choice(["begin", "touch", "touch", "commit", "commit", "abandon", "behind", "marked", "read"])
        if step == "begin" or current is None:
            current = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
        elif step == "touch":
            touched = set(rng.sample(scopes, rng.randint(1, 7)))
            version = current
            await _in_transaction(pool, lambda conn: writer.touch(version, touched, conn=conn))
        elif step == "commit":
            await commit(current, set(rng.sample(scopes, rng.randint(0, 4))))
            current = None
        elif step == "abandon":
            lease.holding = False
            await reader.settled()
            lease.holding = True
            current = None
            held = await reader.snapshot()
        elif step == "behind":
            # a pending scope already at the version a commit takes: both refuse it, naming it
            touched = {rng.choice(scopes)}
            version = current
            await _in_transaction(pool, lambda conn: writer.touch(version, touched, conn=conn))
            await pool.execute(f"UPDATE {_TABLE} SET epoch = $1 WHERE scope = $2", version, next(iter(touched)))
            await commit(version, set())
            await pool.execute(f"UPDATE {_TABLE} SET epoch = 0 WHERE scope = $1", next(iter(touched)))
        elif step == "marked":
            # the record marked pending by hand, which no touch does: neither moves it
            await pool.execute(f"UPDATE {_TABLE} SET pending = 1 WHERE scope = '*'")
            if current is not None:
                await commit(current, set())
                current = None
            await pool.execute(f"UPDATE {_TABLE} SET pending = NULL WHERE scope = '*'")
        old = _by_scope(await pool.fetch(_OLD_CHANGED, held.version))
        l3.reset()
        held = await reader.snapshot()
        # the change's statement and its legs' keyset pages; not a reread by key or a whole read after them
        reads = [
            rows
            for query, _, rows in l3.seen
            if query.startswith("(SELECT 0 AS") or '"pending" IS NOT NULL' in query or '"epoch" >' in query
        ]
        new = {scope: row for rows in reads for scope, row in _by_scope(rows).items()}
        assert new == old, f"after {step}: old named {sorted(old)}, new {sorted(new)}"
        compared["changes"] += 1
        compared["paged"] += len(reads) > 1
    assert compared["paged"] and compared["commits"] and compared["refused"], compared
