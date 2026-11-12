"""integration: a write's SET parameter takes the consumer's override, and a pgvector column needs none.

The generated write equivalence (:func:`threetears.core.testing.rewrites.assert_rewrite_same`, ``shape`` ``write``)
fills each SET parameter with a value no seeded row holds. A column of a type with no generic value (``macaddr``
here) cannot be filled unless the consumer's ``overrides`` reach the SET parameter as they reach the seeded rows; a
``vector(n)`` column (the hub's embedding backfill, ``SET embedding = $1::text::public.vector``) is filled as pgvector
text of its width. On the session's ``db_container`` (PostgreSQL), or YugabyteDB through
``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.testing.rewrites import assert_rewrite_same
from threetears.core.testing.same_results import Seeded

pytestmark = pytest.mark.integration


def _write(column: str, set_sql: str) -> dict[str, Any]:
    return {
        "shape": "write",
        "helper": "update_keyed",
        "table": "o45_hosts",
        "old_sql": f"UPDATE o45_hosts SET {column} = {set_sql} WHERE id = $2",
        "pins": [{"column": "id", "kind": "param", "param": 2, "literal": None}],
        "sets": (
            [{"column": column, "kind": "param", "param": 1, "literal": None}]
            if set_sql == "$1"
            else [{"column": column, "kind": "fragment", "param": None, "literal": set_sql, "params": [1]}]
        ),
    }


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding a table with a ``macaddr`` and a ``vector(3)`` column."""
    schema = f"o45_sets_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE o45_hosts (id uuid PRIMARY KEY, name text NOT NULL, mac macaddr, embedding public.vector(3))"
        )
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def test_a_set_parameter_of_a_type_with_no_generic_value_takes_the_override(conn: asyncpg.Connection) -> None:
    """Flips if a SET parameter ignores the table's override: a macaddr has no generic value, so the write cannot be
    proven without one (the hub's embedding writes were held back by exactly this)."""
    entry = _write("mac", "$1")
    with pytest.raises(AssertionError, match="no generic value; give it an override"):
        await assert_rewrite_same(conn, entry)

    def mac(seed: Seeded) -> str:
        return f"08:00:2b:{seed.index // 65536 % 256:02x}:{seed.index // 256 % 256:02x}:{seed.index % 256:02x}"

    await assert_rewrite_same(conn, entry, overrides={"o45_hosts": {"mac": mac}})


async def test_a_vector_set_parameter_needs_no_override(conn: asyncpg.Connection) -> None:
    """Flips if a vector(n) column has no generic value: the embedding backfill's SET (a text cast to the vector, the
    hub's spelling) is then unprovable without a consumer override."""
    await assert_rewrite_same(conn, _write("embedding", "$1::text::public.vector"))
