"""Integration: a keyed write sets a ``vector`` column to NULL and to a value, on a pool set up as every 3tears host
sets one up (:func:`~threetears.core.collections.init_connection`), and to NULL on a bare connection.

On YugabyteDB (the platform's database, ``THREETEARS_TEST_POSTGRES_URL``) ``vector`` is a built-in type that asyncpg
has no codec for: a statement binding one was refused when prepared, a None bind included. Spec from hub #37
(2026-11-12): a None assignment is the literal ``NULL``; a value binds through the vector text codec the canonical
pool init registers.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.collections import init_connection
from threetears.core.keyed import update_keyed

pytestmark = pytest.mark.integration


@pytest.fixture
async def schema(db_container: str) -> AsyncIterator[str]:
    """a schema of its own holding ``items (id, embedding vector(3))``: one row with a vector, one without."""
    name = f"vector_writes_{uuid.uuid4().hex[:10]}"
    held = await asyncpg.connect(db_container)
    try:
        await held.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await held.execute(f'CREATE SCHEMA "{name}"')
        await held.execute(f'CREATE TABLE "{name}".items (id BIGINT PRIMARY KEY, embedding vector(3))')
        await held.execute(f"INSERT INTO \"{name}\".items VALUES (1, '[1,2,3]'), (2, NULL)")
        yield name
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        await held.close()


async def test_a_host_pool_writes_a_vector_to_null_and_to_a_value(db_container: str, schema: str) -> None:
    """Flips if a pool set up by ``init_connection`` cannot clear a vector through ``update_keyed``, cannot write one
    (from its components or its text) through ``update_keyed`` or a hand-written INSERT, or reads one back as other
    than its text."""
    table = f'"{schema}".items'
    pool = await asyncpg.create_pool(db_container, init=init_connection, min_size=1, max_size=1)
    try:
        cleared = await update_keyed(pool, table, values={"embedding": None}, by={"id": 1})
        written = await update_keyed(pool, table, values={"embedding": [0.5, 0, 1]}, by={"id": 2})
        assert (cleared.count, written.count) == (1, 1)
        await pool.execute(f"INSERT INTO {table} (id, embedding) VALUES ($1, $2), ($3, $4)", 3, "[4,5,6]", 4, None)
        rows = await pool.fetch(f"SELECT id, embedding FROM {table} ORDER BY id")
        assert [(row["id"], row["embedding"]) for row in rows] == [(1, None), (2, "[0.5,0,1]"), (3, "[4,5,6]"), (4, None)]
    finally:
        await pool.close()


async def test_a_bare_connection_clears_a_vector(db_container: str, schema: str) -> None:
    """Flips if ``update_keyed`` setting a vector to None binds the None: a connection with no vector codec (a host
    pool not set up by ``init_connection``) is refused on YugabyteDB when the statement is prepared."""
    table = f'"{schema}".items'
    conn = await asyncpg.connect(db_container)
    try:
        cleared = await update_keyed(conn, table, values={"embedding": None}, by={"id": 1})
        assert cleared.count == 1
        assert await conn.fetchval(f"SELECT embedding IS NULL FROM {table} WHERE id = 1") is True
    finally:
        await conn.close()
