"""integration: the rewrite proof of a read of a stamped table seeds its rows live and compares ``*`` on its columns.

A stamped table (``ROW_STAMPED``, a declaration registered in the process, as the hub's are) answers its live rows alone
to every keyed read, and a read of every column answers its declared columns, never its gone flag, its restore bucket
or an undeclared column (a moved column's ``<col>_moved_from``). The proof
(:func:`threetears.core.testing.rewrites.assert_rewrite_same`) seeds every row live and compares a ``SELECT *`` (and
a join's ``t.*``) on the declared columns and the join's aliases. On the session's ``db_container`` (PostgreSQL), or
YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.schema_backed import STRING_TYPE, UUID_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.rewrites import assert_rewrite_same

pytestmark = pytest.mark.integration

_ITEMS = "rws_items"

_ITEMS_SCHEMA = TableSchema(
    name=_ITEMS,
    primary_key=("id",),
    columns=[Column("id", UUID_TYPE), Column("customer_id", UUID_TYPE), Column("name", STRING_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_BY_ID: dict[str, Any] = {
    "helper": "read_row",
    "function": "one",
    "table": _ITEMS,
    "columns": None,
    "old_sql": f"SELECT * FROM {_ITEMS} WHERE id = $1",
    "pins": [{"column": "id", "kind": "param", "param": 1, "literal": None}],
}

_BY_CUSTOMER: dict[str, Any] = {
    "helper": "read_by",
    "function": "many",
    "table": _ITEMS,
    "columns": None,
    "old_sql": f"SELECT * FROM {_ITEMS} WHERE customer_id = $1",
    "pins": [{"column": "customer_id", "kind": "param", "param": 1, "literal": None}],
}

_JOINED: dict[str, Any] = {
    "helper": "read_by_joined",
    "shape": "join",
    "function": "joined",
    "table": _ITEMS,
    "columns": None,
    "old_sql": (
        f"SELECT t.*, o.label AS owner_label FROM {_ITEMS} t JOIN rws_owners o ON o.id = t.customer_id"
        " WHERE t.customer_id = $1"
    ),
    "pins": [{"column": "customer_id", "kind": "param", "param": 1, "literal": None}],
    "joins": [
        {
            "columns": {"owner_label": "label"},
            "left": False,
            "matched": "",
            "on": {"id": "customer_id"},
            "pins": [],
            "table": "rws_owners",
        }
    ],
    "order": [],
}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding the stamped table as L3 holds one (its epoch, gone flag,
    restore bucket and a moved column's undeclared pair) and a plain table it joins."""
    schema = f"rws_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            f"CREATE TABLE {_ITEMS} (id uuid PRIMARY KEY, customer_id uuid NOT NULL, name text NOT NULL,"
            " epoch bigint NOT NULL, gone boolean NOT NULL DEFAULT false,"
            " epoch_bucket smallint GENERATED ALWAYS AS ((epoch % 16)::smallint) STORED,"
            " name_moved_from text, name_moved_epoch bigint)"
        )
        await connection.execute("CREATE TABLE rws_owners (id uuid PRIMARY KEY, label text NOT NULL)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@pytest.mark.parametrize("entry", [_BY_ID, _BY_CUSTOMER, _JOINED], ids=["read_row", "read_by", "joined"])
async def test_a_read_of_every_column_of_a_stamped_table_is_proven(
    conn: asyncpg.Connection, entry: dict[str, Any]
) -> None:
    """Flips if the proof seeds a stamped table's rows gone (every other row, a boolean's generic value: the helper
    rightly skips them) or compares ``*`` on columns no keyed read answers (the gone flag, the restore bucket, a
    moved column's ``_moved_from`` / ``_moved_epoch``)."""
    await assert_rewrite_same(conn, entry)
