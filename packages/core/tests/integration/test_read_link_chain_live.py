"""integration: a chain of links read through ``read_link_chain`` gives what its hand-written recursive statement gave.

The entry is a ``chain`` rewrite as the codemod records it (:func:`threetears.enforcement.no_whole_reads.rewrite.
parse_chain`): the hub's origin chains (a concept's, a playbook entry's), proven by
:func:`threetears.core.testing.rewrites.assert_rewrite_same` over a chain, a row linking nowhere, a cycle walked round
under its bound, and a key no row holds. On the session's ``db_container`` (PostgreSQL), or YugabyteDB through
``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.testing.rewrites import assert_rewrite_same

pytestmark = pytest.mark.integration

_OLD_SQL = """
WITH RECURSIVE chain(id, origin_concept_id, depth) AS (
    SELECT id, origin_concept_id, 1 FROM l_concepts WHERE id = $1
  UNION ALL
    SELECT c.id, c.origin_concept_id, chain.depth + 1
      FROM l_concepts c
      JOIN chain ON c.id = chain.origin_concept_id
     WHERE chain.depth < $2
)
SELECT id, origin_concept_id FROM chain
"""

_CHAIN: dict[str, Any] = {
    "shape": "chain",
    "helper": "read_link_chain",
    "method": "fetch",
    "function": "origins",
    "table": "l_concepts",
    "old_sql": _OLD_SQL,
    "columns": ["id", "origin_concept_id"],
    "pins": [{"column": "id", "kind": "param", "param": 1}, {"column": "", "kind": "most", "param": 2}],
    "args": ["start", "most"],
    "chain": {"key": "id", "link": "origin_concept_id"},
}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding a table whose rows link to one another."""
    schema = f"vj_chain_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE l_concepts (id uuid PRIMARY KEY, origin_concept_id uuid, name text NOT NULL)"
        )
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def test_a_chain_of_links_gives_what_its_recursive_statement_gave(conn: asyncpg.Connection) -> None:
    """Flips if ``read_link_chain`` walks otherwise than the statement: a link not followed, a cycle not walked round
    to its bound, a row answered for a key no row holds."""
    await assert_rewrite_same(conn, _CHAIN)


async def test_the_proof_sees_a_chain_walked_otherwise(conn: asyncpg.Connection) -> None:
    """Flips if the proof passes a chain walked by another bound: one row fewer than the statement walks."""
    shorter = {**_CHAIN, "old_sql": _OLD_SQL.replace("chain.depth < $2", "chain.depth < $2 - 1")}
    with pytest.raises(AssertionError):
        await assert_rewrite_same(conn, shorter)
