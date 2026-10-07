"""Integration: complete copies of several L3 tables, built beside the live set and swapped in whole.

What a fake cannot prove: that a new set is read from a real Postgres page by page while readers
keep reading the live one, that the live set a reader holds is never touched by the build that
replaces it, and that a build which cannot show one state of every table leaves the live set as it
was.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.collections.complete_copy import BufferedCopies, IncompleteCopyError, read_l3_rows
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration

_RESULTS = TableSchema(
    name="results",
    primary_key=("race", "county"),
    columns=[Column("race", STRING_TYPE), Column("county", STRING_TYPE), Column("votes", BIGINT_TYPE, nullable=True)],
    on_conflict="update",
)
_RACES = TableSchema(
    name="races",
    primary_key="race",
    columns=[Column("race", STRING_TYPE), Column("state", STRING_TYPE)],
    on_conflict="update",
)


class _Result(BaseEntity):
    primary_key_field = "county"


class _Race(BaseEntity):
    primary_key_field = "race"


def _metadata() -> MetaData:
    metadata = MetaData()
    _RESULTS.to_sqlalchemy_table(metadata)
    _RACES.to_sqlalchemy_table(metadata)
    return metadata


def _new_backend() -> DuckDBBackend:
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    return backend


class _Writer:
    """the writer's own record, as a seqlock: the last committed write, or None while one is in progress."""

    def __init__(self) -> None:
        self.version = 0
        self.writing = False
        self.asked = 0

    async def settled(self) -> int | None:
        self.asked += 1
        return None if self.writing else self.version


@dataclass
class _Held:
    pool: asyncpg.Pool
    collections: list[Any]
    writer: _Writer


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"buffered_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute("CREATE TABLE results (race TEXT, county TEXT, votes BIGINT, PRIMARY KEY (race, county))")
        await pool.execute("CREATE TABLE races (race TEXT PRIMARY KEY, state TEXT NOT NULL)")
        for index in range(7):
            await pool.execute("INSERT INTO results VALUES ($1, $2, $3)", f"race-{index % 2}", f"c{index}", 100 + index)
        await pool.execute("INSERT INTO races VALUES ('race-0', 'VA'), ('race-1', 'MD')")
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        collections = [
            collection_for_schema(_RESULTS, entity_class=_Result)(registry, DefaultCoreConfig(), None),
            collection_for_schema(_RACES, entity_class=_Race)(registry, DefaultCoreConfig(), None),
        ]
        yield _Held(pool, collections, _Writer())
    finally:
        await pool.close()


def _votes(backend: DuckDBBackend) -> list[int]:
    return [r["votes"] for r in backend.execute_query("SELECT votes FROM results ORDER BY votes")]


def _copies(held: _Held, **kwargs: Any) -> BufferedCopies[int]:
    return BufferedCopies(held.collections, _new_backend, held.writer.settled, page_size=3, **kwargs)


async def test_a_set_never_built_is_refused() -> None:
    copies: BufferedCopies[int] = BufferedCopies([], _new_backend, _Writer().settled)

    with pytest.raises(IncompleteCopyError, match="no complete copy has been built"):
        copies.require()


async def test_a_build_holds_every_row_of_every_table_and_is_the_live_set(held: _Held) -> None:
    copies = _copies(held)

    generation = await copies.build()

    assert copies.require() is generation
    assert generation.stamp == 0
    assert {table: proof.row_count for table, proof in generation.proofs.items()} == {"results": 7, "races": 2}
    assert _votes(generation.backend) == [100 + i for i in range(7)]


async def test_a_reader_keeps_the_set_it_took_while_the_next_is_built_and_swapped_in(held: _Held) -> None:
    copies = _copies(held)
    first = await copies.build()
    await held.pool.execute("UPDATE results SET votes = votes + 1000 WHERE race = 'race-0'")
    held.writer.version = 1
    seen_mid_build: list[list[int]] = []
    l3 = held.collections[0].required_l3_pool
    read = l3.fetch

    async def fetch_and_read_the_live_set(query: str, *args: Any) -> Any:
        rows = await read(query, *args)
        # a reader during the build: it gets the live set, whole, never the one being built
        seen_mid_build.append(_votes(copies.require().backend))
        return rows

    l3.fetch = fetch_and_read_the_live_set  # type: ignore[method-assign]
    second = await copies.build()

    assert seen_mid_build and all(seen == [100 + i for i in range(7)] for seen in seen_mid_build)
    assert copies.require() is second and second.stamp == 1
    assert _votes(second.backend) == sorted([101, 103, 105, *(1100 + i for i in (0, 2, 4, 6))])
    # the reader who took the first set before the swap still reads it whole
    assert _votes(first.backend) == [100 + i for i in range(7)]


async def test_a_build_while_a_write_is_in_progress_is_refused_and_the_live_set_stays(held: _Held) -> None:
    copies = _copies(held)
    live = await copies.build()
    held.writer.writing = True

    with pytest.raises(IncompleteCopyError, match="a write is in progress"):
        await copies.build()

    assert copies.require() is live


async def test_a_write_landing_between_two_tables_is_refused_and_the_live_set_stays(held: _Held) -> None:
    """each table held still while it was read, yet the set is not one state: a commit fell between."""
    copies = _copies(held)
    live = await copies.build()
    l3 = held.collections[0].required_l3_pool
    read = l3.fetch

    async def fetch_then_commit(query: str, *args: Any) -> Any:
        rows = await read(query, *args)
        if "races" in query:
            held.writer.version = 2  # a whole write committed after results was read
        return rows

    l3.fetch = fetch_then_commit  # type: ignore[method-assign]

    with pytest.raises(IncompleteCopyError, match="changed while the copies were built"):
        await copies.build()
    assert copies.require() is live


async def test_a_build_that_fails_part_way_leaves_the_live_set(held: _Held) -> None:
    copies = _copies(held)
    live = await copies.build()
    l3 = held.collections[1].required_l3_pool
    read = l3.fetch

    async def fails_on_races(query: str, *args: Any) -> Any:
        if "races" in query:
            raise ConnectionError("the L3 rail went away")
        return await read(query, *args)

    l3.fetch = fails_on_races  # type: ignore[method-assign]

    with pytest.raises(ConnectionError):
        await copies.build()
    assert copies.require() is live


async def test_a_set_already_at_the_writers_state_is_not_built_again(held: _Held) -> None:
    copies = _copies(held)
    live = await copies.build()

    assert await copies.build_if_behind() is live

    held.writer.version = 3
    rebuilt = await copies.build_if_behind()
    assert rebuilt is not live and rebuilt.stamp == 3


async def test_builds_asked_for_together_run_one_at_a_time(held: _Held) -> None:
    copies = _copies(held)
    in_build = 0
    most = 0
    l3 = held.collections[0].required_l3_pool
    read = l3.fetch

    async def counted(query: str, *args: Any) -> Any:
        nonlocal in_build, most
        in_build += 1
        most = max(most, in_build)
        try:
            await asyncio.sleep(0.01)
            return await read(query, *args)
        finally:
            in_build -= 1

    l3.fetch = counted  # type: ignore[method-assign]

    first, second = await asyncio.gather(copies.build_if_behind(), copies.build_if_behind())

    assert most == 1
    assert first is second, "the second ask built again although the first left the set current"


async def test_a_part_of_a_table_is_read_by_equality_filters_across_pages(held: _Held) -> None:
    l3 = held.collections[0].required_l3_pool

    rows = await read_l3_rows(
        l3, "results", ["race", "county", "votes"], ["race", "county"], where={"race": "race-0"}, page_size=2
    )

    assert [r["county"] for r in rows] == ["c0", "c2", "c4", "c6"]
