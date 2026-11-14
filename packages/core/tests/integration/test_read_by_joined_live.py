"""integration: a joined read gives what its hand-written join gave (``read_by_joined``), and the generated proof sees
a join answered otherwise.

Each entry is a ``join`` rewrite as the codemod records it (:func:`threetears.enforcement.no_whole_reads.rewrite.
parse_join`), proven by :func:`threetears.core.testing.rewrites.assert_rewrite_same`: the read's table seeded as a
point read's, each joined table from its rows (one match, two, none, rows matching but for a pin, strays), every
row and column compared. On the session's ``db_container`` (PostgreSQL), or YugabyteDB through
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

_DDL = (
    "CREATE TABLE j_customers (id uuid PRIMARY KEY, slug text NOT NULL, name text)",
    "CREATE TABLE j_keys (id uuid PRIMARY KEY, customer_id uuid, label text NOT NULL)",
    "CREATE TABLE j_namespaces (row_scope text NOT NULL, namespace_id uuid NOT NULL, name text NOT NULL, "
    "PRIMARY KEY (row_scope, namespace_id))",
    "CREATE TABLE j_links (ref text NOT NULL, link text NOT NULL, relation_id uuid NOT NULL, "
    "PRIMARY KEY (ref, link, relation_id))",
    "CREATE TABLE j_relations (id uuid PRIMARY KEY, name text NOT NULL)",
    "CREATE TABLE j_columns (table_id uuid NOT NULL, position integer NOT NULL, column_name text NOT NULL, "
    "note text, PRIMARY KEY (table_id, position))",
)


def _pin(column: str, kind: str, **more: Any) -> dict[str, Any]:
    return {"column": column, "kind": kind, "param": None, "literal": None, "cast": None, **more}


def _join(table: str, on: dict[str, str], columns: dict[str, str], **more: Any) -> dict[str, Any]:
    return {"table": table, "on": on, "columns": columns, "left": False, "pins": [], "matched": "", **more}


def _entry(
    old_sql: str, table: str, pins: list[dict[str, Any]], joins: list[dict[str, Any]], **more: Any
) -> dict[str, Any]:
    return {
        "shape": "join",
        "helper": "read_by_joined",
        "method": "fetch",
        "table": table,
        "old_sql": old_sql,
        "columns": None,
        "pins": pins,
        "args": ["a"],
        "joins": joins,
        **more,
    }


#: a key with its customer's slug: every column of the read, one renamed column of a table joined by its key
_WITH_SLUG = _entry(
    "SELECT ak.*, c.slug AS customer_slug FROM j_keys ak JOIN j_customers c ON c.id = ak.customer_id "
    "WHERE ak.label = $1",
    "j_keys",
    [_pin("label", "param", param=1)],
    [_join("j_customers", {"id": "customer_id"}, {"customer_slug": "slug"})],
)

#: a left join pinned in its ON, with whether it matched, beside an inner join
_LEFT_MATCHED = _entry(
    "SELECT ak.id, ak.label, c.name AS customer_name, (n.namespace_id IS NOT NULL) AS live FROM j_keys ak "
    "LEFT JOIN j_namespaces n ON n.row_scope = 'customer' AND n.namespace_id = ak.customer_id "
    "LEFT JOIN j_customers c ON c.id = ak.customer_id WHERE ak.label = $1",
    "j_keys",
    [_pin("label", "param", param=1)],
    [
        _join(
            "j_namespaces",
            {"namespace_id": "customer_id"},
            {},
            left=True,
            pins=[_pin("row_scope", "literal", literal="customer")],
            matched="live",
        ),
        _join("j_customers", {"id": "customer_id"}, {"customer_name": "name"}, left=True),
    ],
    columns=["id", "label"],
)

#: the parameters pin the joined side: it is read first, and the FROM table joined back, in its order
_SWITCHED = _entry(
    "SELECT r.id, r.name FROM j_relations r JOIN j_links l ON l.relation_id = r.id "
    "WHERE l.ref = $1 AND l.link = 'member' ORDER BY r.id",
    "j_links",
    [_pin("ref", "param", param=1), _pin("link", "literal", literal="member")],
    [_join("j_relations", {"id": "relation_id"}, {"id": "id", "name": "name"})],
    columns=[],
    order=[{"column": "relation_id", "descending": False, "nulls": None, "shown": "id"}],
)

#: a one-to-many left join: each read row once for each joined row, or once with none
_MANY = _entry(
    "SELECT ak.id, cl.column_name FROM j_keys ak LEFT JOIN j_columns cl ON cl.table_id = ak.customer_id "
    "WHERE ak.label = $1",
    "j_keys",
    [_pin("label", "param", param=1)],
    [_join("j_columns", {"table_id": "customer_id"}, {"column_name": "column_name"}, left=True)],
    columns=["id"],
)

#: correlated counts beside every column of the read: how many joined rows match it, and how many with a note
_COUNTED = _entry(
    "SELECT ak.*, (SELECT COUNT(*) FROM j_columns cl WHERE cl.table_id = ak.customer_id) AS column_count, "
    "(SELECT COUNT(*) FROM j_columns cl WHERE cl.table_id = ak.customer_id AND cl.note IS NOT NULL) AS noted "
    "FROM j_keys ak WHERE ak.label = $1",
    "j_keys",
    [_pin("label", "param", param=1)],
    [_join("j_columns", {"table_id": "customer_id"}, {}, counts={"column_count": None, "noted": "note"})],
)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding the joined tables."""
    schema = f"vj_joins_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        for statement in _DDL:
            await connection.execute(statement)
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@pytest.mark.parametrize(
    "entry",
    [_WITH_SLUG, _LEFT_MATCHED, _SWITCHED, _MANY, _COUNTED],
    ids=["star", "left-matched", "switched", "one-to-many", "counted"],
)
async def test_a_joined_read_gives_what_its_join_gave(conn: asyncpg.Connection, entry: dict[str, Any]) -> None:
    """Flips if ``read_by_joined`` answers any seed case otherwise than the hand-written join: a row a JOIN drops
    kept, a LEFT JOIN's unmatched row dropped, a joined row matched twice answered once, a joined table's pin
    ignored, a NULL value matched, the switched read out of its order."""
    await assert_rewrite_same(conn, entry)


async def test_the_proof_sees_a_join_answered_otherwise(conn: asyncpg.Connection) -> None:
    """Flips if the generated proof passes a joined read answered otherwise than its statement: a LEFT JOIN's
    unmatched rows dropped (recorded as a JOIN), a JOIN's unmatched rows kept (recorded as a LEFT JOIN), a count of the
    rows holding a note answered as a count of every row."""
    as_inner = {**_MANY, "joins": [{**_MANY["joins"][0], "left": False}]}
    with pytest.raises(AssertionError):
        await assert_rewrite_same(conn, as_inner)
    as_left = {**_WITH_SLUG, "joins": [{**_WITH_SLUG["joins"][0], "left": True}]}
    with pytest.raises(AssertionError):
        await assert_rewrite_same(conn, as_left)
    every_row = {**_COUNTED, "joins": [{**_COUNTED["joins"][0], "counts": {"column_count": None, "noted": None}}]}
    with pytest.raises(AssertionError):
        await assert_rewrite_same(conn, every_row)
