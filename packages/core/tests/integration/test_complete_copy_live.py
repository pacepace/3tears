"""Integration: an L1 that is a complete copy of its L3 table, proven before it is read.

What a fake cannot prove: that the copy pages a real Postgres table by its key and gets every row,
that the fingerprint taken in Postgres brackets the read, and that a row written to the table
while it is read, or evicted from L1 afterwards, leaves the copy refused until it is warmed again.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.complete_copy import CompleteCopy, IncompleteCopyError
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.fingerprint import key_fingerprint, postgres_fingerprint_sql

pytestmark = pytest.mark.integration

_TABLE = "results"

_SCHEMA = TableSchema(
    name=_TABLE,
    primary_key=("race", "reported_at"),
    columns=[
        Column("race", STRING_TYPE),
        Column("reported_at", DATETIMETZ_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE, nullable=True),
    ],
    on_conflict="update",
)

_COLLECTION = collection_for_schema(_SCHEMA)

_START = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)


@dataclass
class _Held:
    pool: asyncpg.Pool
    collection: Any
    l1: DuckDBBackend


async def _insert(pool: asyncpg.Pool, race: str, minutes: int, votes: int) -> None:
    now = datetime.now(UTC)
    await pool.execute(
        f"INSERT INTO {_TABLE} (race, reported_at, votes, date_created, date_updated) VALUES ($1, $2, $3, $4, $4)",
        race,
        _START + timedelta(minutes=minutes),
        votes,
        now,
    )


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    """a fresh schema holding ten rows, and its collection with a DuckDB L1.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the pool, the collection and its L1
    :rtype: AsyncIterator[_Held]
    """
    schema = f"copy_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute(
            f"""
            CREATE TABLE {_TABLE} (
                race TEXT NOT NULL,
                reported_at TIMESTAMPTZ NOT NULL,
                votes BIGINT,
                date_created TIMESTAMPTZ NOT NULL,
                date_updated TIMESTAMPTZ,
                PRIMARY KEY (race, reported_at)
            )
            """
        )
        for index in range(10):
            await _insert(pool, f"race-{index % 4}", index, 100 + index)
        l1 = DuckDBBackend()
        l1.initialize(_SCHEMA.to_sqlalchemy_table(MetaData()).metadata)
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        registry.bind_table(_TABLE, l1_backend=l1)
        yield _Held(pool, _COLLECTION(registry, DefaultCoreConfig(), None), l1)
        l1.reset()
    finally:
        await pool.close()


def _held_rows(l1: DuckDBBackend) -> list[tuple[Any, ...]]:
    return [(r["race"], r["votes"]) for r in l1.execute_query(f"SELECT race, votes FROM {_TABLE} ORDER BY votes")]


async def test_a_warmed_copy_holds_every_row_and_is_proven(held: _Held) -> None:
    copy = CompleteCopy(held.collection, page_size=3)

    proof = await copy.warm()

    assert proof.table == _TABLE
    assert proof.row_count == 10
    assert _held_rows(held.l1) == [(f"race-{i % 4}", 100 + i) for i in range(10)]
    assert copy.require() == proof


async def test_a_copy_never_warmed_is_refused(held: _Held) -> None:
    copy = CompleteCopy(held.collection)

    with pytest.raises(IncompleteCopyError, match=f"{_TABLE}: not proven complete"):
        copy.require()


async def test_a_row_leaving_l1_leaves_the_copy_refused_until_it_is_warmed_again(held: _Held) -> None:
    copy = CompleteCopy(held.collection, page_size=4)
    await copy.warm()
    await _insert(held.pool, "race-new", 30, 999)

    await held.collection.invalidate_cache(("race-new", _START + timedelta(minutes=30)))

    with pytest.raises(IncompleteCopyError, match="changed"):
        copy.require()
    proof = await copy.warm()
    assert proof.row_count == 11
    assert ("race-new", 999) in _held_rows(held.l1)


async def test_a_row_the_table_no_longer_holds_leaves_the_copy(held: _Held) -> None:
    copy = CompleteCopy(held.collection, page_size=4)
    await copy.warm()
    await held.pool.execute(f"DELETE FROM {_TABLE} WHERE votes = 100")

    proof = await copy.warm()

    assert proof.row_count == 9
    assert ("race-0", 100) not in _held_rows(held.l1)


async def test_a_table_written_while_it_is_read_is_refused(held: _Held, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = CompleteCopy(held.collection, page_size=3)
    l3 = held.collection.required_l3_pool
    read = l3.fetch
    pages = []

    async def fetch_then_write(query: str, *args: Any) -> Any:
        rows = await read(query, *args)
        pages.append(len(rows))
        if len(pages) == 1:
            await _insert(held.pool, "race-late", 40, 7)
        return rows

    monkeypatch.setattr(l3, "fetch", fetch_then_write)

    with pytest.raises(IncompleteCopyError, match="changed while it was read"):
        await copy.warm()
    with pytest.raises(IncompleteCopyError):
        copy.require()


async def test_a_row_evicted_while_the_copy_is_warmed_leaves_it_refused(
    held: _Held, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = CompleteCopy(held.collection, page_size=3)
    l3 = held.collection.required_l3_pool
    read = l3.fetch

    async def fetch_then_evict(query: str, *args: Any) -> Any:
        rows = await read(query, *args)
        held.collection.evict_from_cache_sync(("race-0", _START))
        return rows

    monkeypatch.setattr(l3, "fetch", fetch_then_evict)

    with pytest.raises(IncompleteCopyError, match="changed"):
        await copy.warm()
    with pytest.raises(IncompleteCopyError):
        copy.require()


async def test_a_copy_needs_an_l1_it_can_replace_whole(held: _Held) -> None:
    l1 = SQLiteBackend(db_name=f"copy_{uuid.uuid4().hex[:8]}")
    l1.initialize(_SCHEMA.to_sqlalchemy_table(MetaData()).metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=SqlL3Backend(held.pool))
    try:
        with pytest.raises(ValueError, match="replace_all"):
            CompleteCopy(_COLLECTION(registry, DefaultCoreConfig(), None))
    finally:
        l1.reset()


async def test_postgres_and_python_digest_text_keys_to_the_same_number(held: _Held) -> None:
    row = await held.pool.fetchrow(postgres_fingerprint_sql(_TABLE, ["race"]))
    races = await held.pool.fetch(f"SELECT race FROM {_TABLE}")

    python = key_fingerprint([(r["race"],) for r in races])

    assert (int(row["row_count"]), str(row["digest"])) == (python.row_count, python.digest)
