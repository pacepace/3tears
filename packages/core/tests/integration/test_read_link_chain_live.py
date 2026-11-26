"""integration: a chain of links read through ``read_link_chain`` gives what its hand-written recursive statement gave.

The entry is a ``chain`` rewrite as the codemod records it (:func:`threetears.enforcement.no_whole_reads.rewrite.
parse_chain`): the hub's origin chains (a concept's, a playbook entry's), proven by
:func:`threetears.core.testing.rewrites.assert_rewrite_same` over a chain, a row linking nowhere, a cycle walked round
under its bound, and a key no row holds. A stamped table's chain (``ROW_STAMPED``, as the hub's knowledge tables are) is
walked by the same one statement over its live rows alone: it stops at a gone row, and a cycle ends where it comes
back. On the session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema
of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, UUID_TYPE
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.keyed import read_link_chain
from threetears.core.testing.l1_tables import TEST_L1_TABLE
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


#: a stamped table whose rows link to one another, its declaration registered as the hub's knowledge tables' are
_STAMPED = "l_stamped_concepts"

_STAMPED_SCHEMA = TableSchema(
    name=_STAMPED,
    primary_key=("id",),
    columns=[
        Column("id", UUID_TYPE),
        Column("origin_concept_id", UUID_TYPE, nullable=True),
        Column("name", STRING_TYPE),
    ],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)


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


async def test_a_stamped_chain_stops_at_a_gone_row_and_walks_a_cycle_once(conn: asyncpg.Connection) -> None:
    """Flips if a link chain of a stamped table is refused, answers a gone row or one past it, walks a cycle round
    under its bound, or is sent as more than the one recursive statement: a -> b -> c (gone) -> d, x -> y -> z -> x."""
    await conn.execute(
        f"CREATE TABLE {_STAMPED} (id uuid PRIMARY KEY, origin_concept_id uuid, name text NOT NULL,"
        " epoch bigint NOT NULL DEFAULT 0, gone boolean NOT NULL DEFAULT false)"
    )
    ids = {name: uuid.uuid5(uuid.NAMESPACE_OID, name) for name in "abcdxyz"}
    links = {"a": "b", "b": "c", "c": "d", "d": None, "x": "y", "y": "z", "z": "x"}
    await conn.executemany(
        f"INSERT INTO {_STAMPED} (id, origin_concept_id, name, gone) VALUES ($1, $2, $3, $4)",
        [(ids[name], None if to is None else ids[to], name, name == "c") for name, to in links.items()],
    )
    sent: list[str] = []

    class _Counted:
        """the real connection, each statement it is sent recorded."""

        async def fetch(self, query: str, *params: Any) -> list[Any]:
            sent.append(query)
            return list(await conn.fetch(query, *params))

    async def walked(start: str, most: int) -> list[str]:
        chained = await read_link_chain(
            conn, _STAMPED, key="id", link="origin_concept_id", start=ids[start], most=most, columns=["name"]
        )
        return [row["name"] for row in chained]

    assert await walked("a", 10) == ["a", "b"]
    assert await walked("c", 10) == []
    assert sorted(await walked("x", 10)) == ["x", "y", "z"]
    assert sorted(await walked("y", 2)) == ["y", "z"]
    assert await walked("d", 10) == ["d"]
    # one statement a chain, its live guard on the first row and on every link's
    await read_link_chain(
        _Counted(), _STAMPED, key="id", link="origin_concept_id", start=ids["x"], most=10, columns=["name"]
    )
    assert len(sent) == 1 and sent[0].count('NOT t."gone"') == 2, sent
