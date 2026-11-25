"""a schema-backed collection over the SQLite L3, with no L2 and the ordinary L1 cache in front.

What a single-machine deployment builds: ``registry.configure(l1_backend=SQLiteBackend(...),
l3_pool=SqliteL3Backend(path))`` and each collection with ``nats_client=NO_L2``. The collection's own paths -- save,
read through, the optimistic-lock fence, delete, a caller's transaction -- reach the file through the store, and an
L1 eviction (by its LRU budget, or by hand) costs a read, never a row. Durability across a killed process is
``tests/unit/backends/test_sqlite_l3_survives_kill.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from threetears.core.backends import SqliteL3Backend
from threetears.core.cache import SQLiteBackend
from threetears.core.cache.l1_lru import L1Ledger
from threetears.core.collections import NO_L2
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import ConcurrentModificationError
from threetears.core.testing.l1_tables import TEST_L1_TABLE

QUEUE = TableSchema(
    name="send_queue",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("session", STRING_TYPE),
        Column("body", JSONB_TYPE),
        Column("state", STRING_TYPE),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    # EVICTABLE: a cache over the file, which an eviction can no longer lose a row of
    l1_eviction=TEST_L1_TABLE,
)
QueueCollection = collection_for_schema(QUEUE)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "app.sqlite"


@asynccontextmanager
async def _deployment(path: Path, *, ledger: L1Ledger | None = None) -> AsyncIterator[Any]:
    """one process's wiring: a fresh L1 cache, the store over ``path``, the collection with no L2."""
    store = SqliteL3Backend(path)
    store.open()
    l1 = SQLiteBackend(f"sqlite_l3_{uuid.uuid4().hex[:8]}", budget=ledger)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=store)
    try:
        yield QueueCollection(registry, DefaultCoreConfig(), NO_L2)
    finally:
        l1.reset()
        store.close()


def _row(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "session": "s1",
        "body": {"text": "hello"},
        "state": "queued",
        "date_created": now,
        "date_updated": now,
    }
    row.update(overrides)
    return row


async def test_a_saved_row_is_read_by_the_next_process(path: Path) -> None:
    row = _row()
    async with _deployment(path) as queue:
        await queue.save_entity(queue.create(row))
    async with _deployment(path) as queue:
        got = await queue.get(row["id"])
        assert got is not None
        stored = got.to_dict()
        assert {key: stored[key] for key in ("id", "session", "body", "state")} == {
            key: row[key] for key in ("id", "session", "body", "state")
        }


async def test_a_row_evicted_from_l1_is_read_back_from_the_file(path: Path) -> None:
    row = _row()
    async with _deployment(path) as queue:
        await queue.save_entity(queue.create(row))
        assert queue.evict_from_cache_sync(row["id"])
        assert not queue.exists_in_cache_sync(row["id"])
        got = await queue.get(row["id"])
        assert got is not None
        assert got.state == "queued"


async def test_rows_the_lru_evicts_are_all_still_there(path: Path) -> None:
    """an L1 budget far under the rows written: the LRU evicts, and every row reads back."""
    ledger = L1Ledger(budget=4096, thread=False)
    rows = [_row(body={"text": "x" * 200, "n": n}) for n in range(60)]
    async with _deployment(path, ledger=ledger) as queue:
        for row in rows:
            await queue.save_entity(queue.create(row))
        ledger.drain(fold=True)
        held = sum(queue.exists_in_cache_sync(row["id"]) for row in rows)
        assert held < len(rows)
        for row in rows:
            got = await queue.get(row["id"])
            assert got is not None
            assert got.body == row["body"]


async def test_an_update_from_a_stale_read_is_refused_by_the_fence(path: Path) -> None:
    row = _row()
    async with _deployment(path) as queue:
        await queue.save_entity(queue.create(row))
        first = await queue.get(row["id"])
        second = await queue.get(row["id"])
        assert first is not None and second is not None
        first.state = "sent"
        await queue.save_entity(first)
        second.state = "dead"
        with pytest.raises(ConcurrentModificationError):
            await queue.save_entity(second)
        queue.evict_from_cache_sync(row["id"])
        stored = await queue.get(row["id"])
        assert stored is not None
        assert stored.state == "sent"


async def test_a_delete_reaches_the_file(path: Path) -> None:
    row = _row()
    async with _deployment(path) as queue:
        await queue.save_entity(queue.create(row))
        await queue.delete(row["id"])
    async with _deployment(path) as queue:
        assert await queue.get(row["id"]) is None


async def test_saves_in_a_callers_transaction_commit_together_or_not_at_all(path: Path) -> None:
    kept, dropped = _row(), _row()
    async with _deployment(path) as queue:
        store = queue.l3_pool
        async with store.acquire() as conn, CallerTransaction(conn):
            await queue.save_entity(queue.create(kept), conn=conn)
        with pytest.raises(RuntimeError, match="abandoned"):
            async with store.acquire() as conn, CallerTransaction(conn):
                await queue.save_entity(queue.create(dropped), conn=conn)
                raise RuntimeError("abandoned")
    async with _deployment(path) as queue:
        assert await queue.get(kept["id"]) is not None
        assert await queue.get(dropped["id"]) is None


async def test_the_files_rows_are_scanned_by_their_columns(path: Path) -> None:
    rows = [_row(session="a"), _row(session="b"), _row(session="a")]
    async with _deployment(path) as queue:
        for row in rows:
            await queue.save_entity(queue.create(row))
        found = await queue.l3_pool.scan("send_queue", {"session": "a"})
        assert sorted(r["id"] for r in found) == sorted(r["id"] for r in rows if r["session"] == "a")
