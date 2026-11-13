"""Integration: resizing an embedding column costs what the same statements cost by hand, at real size.

``_resize_column`` (reached from ``_resize_on_one_session``) is a declared ``schema_change`` whole read: when the
configured width differs from the stored one it counts the embeddings the column holds (the warning says how many are
cleared), drops the column's indexes, changes its type and builds them again. The bare operation is those statements
by hand on the same table: the count, the drop, the ``ALTER`` and the build. Each round resizes one way by the path and
back by hand, so both start from a full column; the path is held within 2x the bare
(:func:`threetears.core.testing.calibrate.budget_problem`).

Timed runs belong under measure-lock. Against the session ``db_container`` (Postgres); set
``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there. ``THREETEARS_RESIZE_SPEED_ROWS`` sets the rows
(default 200,000).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.embedding_width import EmbeddingColumn, resize_embedding_columns
from threetears.core.testing.calibrate import budget_problem

pytestmark = pytest.mark.integration

_ROWS = int(os.environ.get("THREETEARS_RESIZE_SPEED_ROWS", "200000"))
#: rounds each side runs, turn about
_ROUNDS = 3


class _Store:
    """the store ``resize_embedding_columns`` takes, over one asyncpg connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *params)]

    async def execute(self, sql: str, *params: Any) -> str:
        return str(await self._conn.execute(sql, *params))


@pytest.fixture
async def notes(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    schema = f"resize_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1800)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path TO {schema}, public")
        await conn.execute(
            "CREATE TABLE notes (id bigint PRIMARY KEY, created bigint NOT NULL, embedding public.vector(4))"
        )
        await conn.execute(
            "INSERT INTO notes SELECT g, g, CASE WHEN g % 10 = 0 THEN NULL ELSE '[1,2,3,4]'::public.vector(4) END "
            f"FROM generate_series(1, {_ROWS}) g"
        )
        await conn.execute("CREATE INDEX ix_notes_with_embedding ON notes (created) WHERE embedding IS NOT NULL")
        await conn.execute("ANALYZE notes")
        yield conn, schema
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _by_hand(conn: asyncpg.Connection, schema: str, width: int) -> int:
    """the bare operation: the same count, drop, ALTER and build ``_resize_column`` sends."""
    held = int(await conn.fetchval(f'SELECT count(*) FROM "{schema}"."notes" WHERE "embedding" IS NOT NULL'))
    await conn.execute(f'DROP INDEX IF EXISTS "{schema}"."ix_notes_with_embedding"')
    await conn.execute(
        f'ALTER TABLE "{schema}"."notes" ALTER COLUMN "embedding" '
        f"TYPE public.vector({width}) USING NULL::public.vector({width})"
    )
    await conn.execute(f"CREATE INDEX ix_notes_with_embedding ON {schema}.notes (created) WHERE embedding IS NOT NULL")
    return held


async def _refill(conn: asyncpg.Connection, width: int) -> None:
    """give the column its embeddings back at ``width``, untimed, so each timed resize clears a full column."""
    vector = "[" + ",".join(["1"] * width) + "]"
    await conn.execute(f"UPDATE notes SET embedding = '{vector}'::public.vector({width}) WHERE id % 10 <> 0")


async def test_resize_column_speed_at_real_size(notes: tuple[asyncpg.Connection, str]) -> None:
    """``_resize_column``'s schema_change read and rebuild, reached through ``_resize_on_one_session``, against the
    same statements by hand, interleaved."""
    conn, schema = notes
    store = _Store(conn)
    column = EmbeddingColumn("notes", "embedding")
    best_path = best_bare = float("inf")
    for _ in range(_ROUNDS):
        began = time.perf_counter()
        cleared = await resize_embedding_columns(store, [column], 6)
        best_path = min(best_path, time.perf_counter() - began)
        assert cleared == _ROWS - _ROWS // 10
        await _refill(conn, 6)
        began = time.perf_counter()
        held = await _by_hand(conn, schema, 4)
        best_bare = min(best_bare, time.perf_counter() - began)
        assert held == cleared
        await _refill(conn, 4)
    problem = budget_problem(f"_resize_column over {_ROWS} rows", best_path, best_bare)
    assert problem is None, problem
