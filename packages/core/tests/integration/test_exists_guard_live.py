"""integration: a keyed write or read whose guard is a correlated EXISTS gives what its hand-written statement gave.

Each entry is a rewrite as the codemod records it for ``DELETE FROM <table> <alias> WHERE <pins> AND EXISTS (SELECT 1
FROM <table> <other> WHERE <other>.x = <alias>.x ...) RETURNING <alias>.id`` (the hub's data-upgrade moves: a copy
under a new name gives way to the curated row under the old one) and for a read ``... AND NOT EXISTS (...)``: the
guard respelled over the helper's row (the table's bare name in a write, ``t`` in a read), proven by
:func:`threetears.core.testing.rewrites.assert_rewrite_same`. On the session's ``db_container`` (PostgreSQL), or
YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
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

_PINS: list[dict[str, Any]] = [
    {"column": "ds", "kind": "any", "param": 1, "literal": None, "cast": "uuid", "params": [], "compared": []},
    {"column": "schema_name", "kind": "param", "param": 2, "literal": None, "cast": None, "params": [], "compared": []},
    {"column": "table_name", "kind": "param", "param": 4, "literal": None, "cast": None, "params": [], "compared": []},
]

#: the copy under the new table name gives way where the curated row under the old one holds the same column
_GIVE_WAY: dict[str, Any] = {
    "shape": "write",
    "helper": "delete_keyed",
    "method": "fetch",
    "table": "g_cols",
    "old_sql": "DELETE FROM g_cols fresh WHERE fresh.ds = ANY($1::uuid[]) AND fresh.schema_name = $2 AND "
    "fresh.table_name = $4 AND EXISTS ( SELECT 1 FROM g_cols kept WHERE kept.ds = fresh.ds AND kept.schema_name = $2 "
    "AND kept.table_name = $3 AND kept.column_name = fresh.column_name ) RETURNING fresh.id",
    "pins": _PINS,
    "sets": [],
    "returning": ["id"],
    "args": ["ids", "schema", "old", "new"],
    "where": [
        {
            "column": "",
            "kind": "guard",
            "param": None,
            "literal": "EXISTS ( SELECT 1 FROM g_cols kept WHERE kept.ds = g_cols.ds AND kept.schema_name = $1 AND "
            "kept.table_name = $2 AND kept.column_name = g_cols.column_name )",
            "cast": None,
            "params": [2, 3],
            "compared": ["schema_name", "table_name"],
        }
    ],
}

#: the read of the columns no other row of the same name holds under another table
_ALONE: dict[str, Any] = {
    "shape": "point",
    "helper": "read_by",
    "method": "fetch",
    "table": "g_cols",
    "old_sql": "SELECT id, column_name FROM g_cols c WHERE c.ds = $1 AND NOT EXISTS (SELECT 1 FROM g_cols o "
    "WHERE o.ds = c.ds AND o.column_name = c.column_name AND o.id <> c.id)",
    "columns": ["id", "column_name"],
    "pins": [
        {"column": "ds", "kind": "param", "param": 1, "literal": None, "cast": None, "params": [], "compared": []}
    ],
    "args": ["ds"],
    "where": [
        {
            "column": "",
            "kind": "guard",
            "param": None,
            "literal": "NOT EXISTS (SELECT 1 FROM g_cols o WHERE o.ds = t.ds AND o.column_name = t.column_name AND "
            "o.id <> t.id)",
            "cast": None,
            "params": [],
            "compared": [],
        }
    ],
}


def _paired(seed: Seeded) -> str:
    return f"c{seed.index // 2}"


#: two seeded rows a column name, so a row has another of its name and NOT EXISTS keeps some rows and drops others
_PAIRED: dict[str, Any] = {"g_cols": {"column_name": _paired}}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding the curated columns table."""
    schema = f"vj_exists_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE g_cols (id uuid PRIMARY KEY, ds uuid NOT NULL, schema_name text NOT NULL, "
            "table_name text NOT NULL, column_name text NOT NULL)"
        )
        await connection.execute("CREATE INDEX g_cols_by_name ON g_cols (ds, schema_name, table_name, column_name)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@pytest.mark.parametrize("entry", [_GIVE_WAY, _ALONE], ids=["write-exists", "read-not-exists"])
async def test_a_correlated_exists_guard_gives_what_the_statement_gave(
    conn: asyncpg.Connection, entry: dict[str, Any]
) -> None:
    """Flips if the guard respelled over the helper's row answers otherwise than the statement as written: the write
    deleting (and returning) other rows, the read keeping other rows."""
    await assert_rewrite_same(conn, entry, overrides=_PAIRED)


async def test_the_proof_sees_a_guard_reaching_the_wrong_row(conn: asyncpg.Connection) -> None:
    """Flips if the proof passes a guard correlated to its own inner row instead of the helper's (``o.id <> o.id``:
    never true, so NOT EXISTS keeps every row)."""
    (guard,) = _ALONE["where"]
    wrong = {**_ALONE, "where": [{**guard, "literal": guard["literal"].replace("o.id <> t.id", "o.id <> o.id")}]}
    with pytest.raises(AssertionError):
        await assert_rewrite_same(conn, wrong, overrides=_PAIRED)
