"""Integration: a reader of the scope epochs reads only what changed, and answers what a whole read answers.

A copy follower asks the epochs twice a pass. Each ask read the whole table: on an election night, a whole-table read
a pass for a change of a few scopes. After its first read a :class:`ScopeEpochs` reads the writer's record, the
scopes still pending, and the scopes moved past the version it last read: the rows of the change, one statement. A
record it cannot trust is read whole again, counted by why.

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

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.complete_copy import Unsettled
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.scope_epochs import (
    EpochSnapshot,
    ScopeEpochs,
    WriterLease,
    scope_epochs_collection,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.lease import LeaseHolding

pytestmark = pytest.mark.integration

_TABLE = "scope_epochs"


class _CountingL3(SqlL3Backend):
    """the L3 a reader reads through: counts the statements and rows it answers, and may run a step after one."""

    def __init__(self, pool: Any) -> None:
        super().__init__(pool)
        self.statements = 0
        self.rows = 0
        #: run once, after the next statement answers (a commit landing while a read pages)
        self.after_next: Callable[[], Awaitable[Any]] | None = None

    def reset(self) -> None:
        self.statements = 0
        self.rows = 0

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        rows = await super().fetch(query, *params, namespace=namespace, **kwargs)
        self.statements += 1
        self.rows += len(rows)
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
    await pool.execute(
        f"""
        CREATE TABLE {_TABLE} (
            scope TEXT PRIMARY KEY,
            epoch BIGINT NOT NULL,
            previous_epoch BIGINT,
            writing BIGINT,
            pending BIGINT,
            date_created TIMESTAMPTZ NOT NULL,
            date_updated TIMESTAMPTZ
        )
        """
    )
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


async def _held_reader(pool: asyncpg.Pool) -> tuple[ScopeEpochs, ScopeEpochs]:
    writer, _ = _reader(pool)
    await _write(pool, writer, {"state:VA", "state:MD"})
    await _write(pool, writer, {"state:VA"})
    reader, _ = _reader(pool)
    await reader.snapshot()
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


async def test_an_epoch_gone_back_is_read_whole(pool: asyncpg.Pool) -> None:
    writer, reader = await _held_reader(pool)
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    await _in_transaction(pool, lambda conn: writer.touch(version, {"state:VA"}, conn=conn))
    await pool.execute(f"UPDATE {_TABLE} SET epoch = 1 WHERE scope = 'state:VA'")

    await _read_whole_again(pool, reader, "epoch_backwards")


async def test_a_pending_mark_cleared_without_its_epoch_moving_is_read_whole(pool: asyncpg.Pool) -> None:
    """the case a read of the changes cannot see: the scope is no longer pending, and its epoch did not move past."""
    writer, reader = await _held_reader(pool)
    version: int = await _in_transaction(pool, lambda conn: writer.begin(conn=conn))
    await _in_transaction(pool, lambda conn: writer.touch(version, {"state:MD"}, conn=conn))
    assert (await reader.snapshot()).pending == frozenset({"state:MD"})
    await pool.execute(f"UPDATE {_TABLE} SET pending = NULL WHERE scope = 'state:MD'")

    await _read_whole_again(pool, reader, "pending_lost")
