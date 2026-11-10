"""Integration: installing the change-note ring leaves every one of its indexes usable.

An index build that fails part way leaves the index behind marked invalid (``pg_index.indisvalid``): the planner
never reads it, and ``CREATE INDEX IF NOT EXISTS`` takes it as present and reports success. On YugabyteDB an
online build racing other DDL on the database was seen to end so with no error at all: every secondary index of a
freshly installed ring left invalid, and each page of a commit's noted keys read the whole note table (322,100 rows
for 5,000) instead of its own notes through ``(table_name, noted_at)``. These check that an install finds such an
index and builds it again, and that the reads then go through it.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.ddl.change_notes import note_table

pytestmark = pytest.mark.integration

_INVALID = (
    "SELECT c.relname::text FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND NOT i.indisvalid ORDER BY 1"
)


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str


class _NoLog:
    """the keeper's change log, never read by an install."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"an install reads no change log ({name})")


class _Pool:
    """one connection as the keeper's pool: ``fetch`` and ``execute`` on it, ``acquire`` handing it out."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        return await self._conn.execute(query, *args)

    def acquire(self) -> Any:
        conn = self._conn

        class _Held:
            async def __aenter__(self) -> Any:
                return conn

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _Held()


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """an installed ring whose current table's ``(table_name, noted_at)`` index a failed build left invalid."""
    schema = f"ringinst_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await ChangeNoteKeeper(_Pool(conn), notes_schema=schema, log=_NoLog()).install()
        assert await conn.fetch(_INVALID, schema) == []
        notes = f"{schema}.{note_table(0)}"
        index = f"{note_table(0)}_table"
        await conn.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text) "  # noqa: S608
            "SELECT 'lsf_runs', ARRAY[g::text], g::text FROM generate_series(1, 50) g"
        )
        # the build fails on the repeated table_name, and leaves its index behind under the ring's own name
        await conn.execute(f'DROP INDEX {schema}."{index}"')
        with pytest.raises(asyncpg.PostgresError, match="duplicate"):
            await conn.execute(f'CREATE UNIQUE INDEX CONCURRENTLY "{index}" ON {notes} (table_name)')
        assert [row[0] for row in await conn.fetch(_INVALID, schema)] == [index]
        yield _Ring(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def test_installing_the_ring_builds_an_invalid_index_of_it_again(ring: _Ring) -> None:
    """Flips if an install takes a ring index a failed build left invalid as present, and the reads scan the table."""
    await ChangeNoteKeeper(_Pool(ring.conn), notes_schema=ring.schema, log=_NoLog()).install()

    assert await ring.conn.fetch(_INVALID, ring.schema) == []
    definition = await ring.conn.fetchval(
        "SELECT pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 AND c.relname = $2",
        ring.schema,
        f"{note_table(0)}_table",
    )
    assert "UNIQUE" not in definition and "(table_name" in definition and "noted_at" in definition, definition
    # a table of 50 notes is cheaper to scan: the planner is told not to, which an invalid index cannot answer
    async with ring.conn.transaction():
        await ring.conn.execute("SET LOCAL enable_seqscan = off")
        plan = "\n".join(
            row[0]
            for row in await ring.conn.fetch(
                f"EXPLAIN SELECT noted_at FROM {ring.schema}.{note_table(0)} WHERE table_name = 'lsf_runs' "  # noqa: S608
                "ORDER BY noted_at"
            )
        )
    assert f"{note_table(0)}_table" in plan, plan
