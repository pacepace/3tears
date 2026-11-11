"""integration: core's point reads of run-time-named tables read what the statements they replaced read.

Each read names its table at run time (the owner's epochs table, a pod's fence table, a generated collection's table),
so it cannot sit in the use-the-helper manifest, whose source check reads a literal table. Each is held here instead,
by the manifest's own proof (:func:`threetears.core.testing.rewrites.assert_rewrite_same`): its table built from its
declaration, seeded from the catalog for every case, the old statement against the helper call (every row its key names,
then the one row or value), on the session's ``db_container`` (PostgreSQL), or YugabyteDB through
``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.

- ``ScopeEpochs._read_whole``'s version read -> ``read_value``; ``begin`` / ``_abandon``'s ``FOR UPDATE NOWAIT`` read and
  ``touch``'s ``FOR SHARE`` fence -> ``read_row`` with the same lock (the lock is the read's own; the rows are compared,
  each inside its case's transaction).
- ``SourceTokenFence._read`` -> ``read_row``.
- ``create_collection_from_table_def``'s ``fetch_from_store`` (``SELECT * ... WHERE pk = $1``) -> ``read_row``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.coordination.source_token import source_tokens_schema
from threetears.core.testing.rewrites import assert_rewrite_same

pytestmark = pytest.mark.integration


def _pin(column: str, param: int) -> dict[str, Any]:
    return {"column": column, "kind": "param", "param": param, "literal": None}


#: (the old statement, the helper, the columns read, the pins) per read; each table as named below
_READS: dict[str, dict[str, Any]] = {
    "ScopeEpochs._read_whole version": {
        "table": "o24_scope_epochs",
        "helper": "read_value",
        "columns": ["epoch"],
        "old_sql": "SELECT epoch FROM o24_scope_epochs WHERE scope = $1",
        "pins": [_pin("scope", 1)],
    },
    "ScopeEpochs.begin / _abandon FOR UPDATE NOWAIT": {
        "table": "o24_scope_epochs",
        "helper": "read_row",
        "columns": ["writing"],
        "old_sql": "SELECT writing FROM o24_scope_epochs WHERE scope = $1 FOR UPDATE NOWAIT",
        "pins": [_pin("scope", 1)],
    },
    "ScopeEpochs.touch FOR SHARE": {
        "table": "o24_scope_epochs",
        "helper": "read_row",
        "columns": ["writing"],
        "old_sql": "SELECT writing FROM o24_scope_epochs WHERE scope = $2 AND writing = $1 FOR SHARE",
        "pins": [_pin("scope", 2), _pin("writing", 1)],
    },
    "SourceTokenFence._read": {
        "table": "o24_source_tokens",
        "helper": "read_row",
        "columns": ["run_id", "as_of"],
        "old_sql": "SELECT run_id, as_of FROM o24_source_tokens WHERE signal = $1",
        "pins": [_pin("signal", 1)],
    },
    "collection_factory fetch_from_store": {
        "table": "o24_widgets",
        "helper": "read_row",
        "columns": None,
        "old_sql": "SELECT * FROM o24_widgets WHERE id = $1",
        "pins": [_pin("id", 1)],
    },
}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding the epochs and fence tables as declared, and a generated
    collection's table."""
    schema = f"corepts_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
        for declared in (scope_epochs_schema("o24_scope_epochs"), source_tokens_schema("o24_source_tokens")):
            table = declared.to_sqlalchemy_table(MetaData())
            await connection.execute(str(CreateTable(table).compile(dialect=dialect)))
        await connection.execute("CREATE TABLE o24_widgets (id uuid PRIMARY KEY, name text, size integer)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@pytest.mark.parametrize("read", list(_READS), ids=list(_READS))
async def test_the_helper_reads_what_the_statement_it_replaced_read(conn: asyncpg.Connection, read: str) -> None:
    """Flips if a converted core point read returns other rows, values or columns than its old statement."""
    await assert_rewrite_same(conn, _READS[read])
