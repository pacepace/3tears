"""integration: each agent-knowledge index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; this is the ``plan_test`` each names. The tables are the hub's; they are built here
from this package's declarations, seeded past the size where the planner would rather scan, and the statement the
collection really writes (run over a pool that records statements) is asked whether its plan found its rows through
the index by an index condition (:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.agent.knowledge.collections import ConceptCollection, PlaybookEntryCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import TableSchema
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.plans import explain_statement, plan_uses_index

pytestmark = pytest.mark.integration
#: rows seeded per table: past the size where the planner would rather scan than use an index
_ROWS = 5000

#: a value for a NOT NULL column with no default, by its type, from the series row ``g``
_BY_TYPE: dict[str, str] = {
    "uuid": "gen_random_uuid()",
    "text": "'v' || g",
    "character varying": "'v' || g",
    "timestamp with time zone": "now() - g * interval '1 second'",
    "timestamp without time zone": "now() - g * interval '1 second'",
    "integer": "g",
    "bigint": "g",
    "smallint": "1",
    "double precision": "g",
    "real": "g",
    "numeric": "g",
    "boolean": "false",
    "jsonb": "'{}'::jsonb",
    "json": "'{}'::json",
    "ARRAY": "'{}'",
}


async def _seed(
    conn: asyncpg.Connection, table: str, rows: int = _ROWS, *, _params: tuple[Any, ...] = (), **values: str
) -> None:
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
    :param _params: the parameters the expressions name ($1, ...)
    :ptype _params: tuple[Any, ...]
    :param values: a SQL expression of the series row ``g`` per column; every other NOT NULL column with no
        default gets one by its type
    :ptype values: str
    :return: nothing
    :rtype: None
    """
    columns = await conn.fetch(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = $1",
        table,
    )
    names: list[str] = []
    exprs: list[str] = []
    for column in columns:
        name = column["column_name"]
        if name in values:
            names.append(name)
            exprs.append(values[name])
        elif column["is_nullable"] == "NO" and column["column_default"] is None:
            names.append(name)
            exprs.append(_BY_TYPE[column["data_type"]])
    await conn.execute(
        f'INSERT INTO "{table}" ({", ".join(names)}) SELECT {", ".join(exprs)} FROM generate_series(1, {rows}) g',
        *_params,
    )
    await conn.execute(f'ANALYZE "{table}"')


class _Captured:
    """an L3 pool that runs nothing and records each statement asked of it, so a test explains the statement the
    code under test really writes."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any, **_: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any, **_: Any) -> None:
        self.asked.append((sql, args))

    async def fetchval(self, sql: str, *args: Any, **_: Any) -> int:
        self.asked.append((sql, args))
        return 0

    async def execute(self, sql: str, *args: Any, **_: Any) -> str:
        self.asked.append((sql, args))
        return "UPDATE 0"


async def _create(conn: asyncpg.Connection, schema: TableSchema) -> None:
    """create a declared table and its declared indexes as Postgres would hold them.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param schema: the declaration
    :ptype schema: TableSchema
    :return: nothing
    :rtype: None
    """
    table = schema.to_sqlalchemy_table(MetaData())
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    await conn.execute(str(CreateTable(table).compile(dialect=dialect)))
    for index in table.indexes:
        await conn.execute(str(CreateIndex(index).compile(dialect=dialect)))


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema holding the two knowledge tables, as declared."""
    schema = f"knowplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        for declared in (PlaybookEntryCollection.schema, ConceptCollection.schema):
            await _create(connection, declared)
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def test_draft_owner_indexes(conn: asyncpg.Connection) -> None:
    """Flips if a user's own drafts, entries or concepts, stop reading their table's draft-owner index."""
    users = [uuid.uuid4() for _ in range(100)]
    status = "CASE WHEN g % 50 = 0 THEN 'draft' ELSE 'active' END"
    await _seed(conn, "playbook_entries", user_id="($1::uuid[])[1 + g % 100]", status=status, _params=(users,))
    await _seed(conn, "concepts", user_id="($1::uuid[])[1 + g % 100]", status=status, _params=(users,))
    captured = _Captured()
    registry = CollectionRegistry()
    registry.configure(l3_pool=captured)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    await PlaybookEntryCollection(registry=registry, config=cfg).list_own_drafts(users[0], customer_scope=uuid.uuid4())
    await ConceptCollection(registry=registry, config=cfg).list_own_drafts(users[0], customer_scope=uuid.uuid4())
    (entries, entry_args), (concepts, concept_args) = captured.asked
    entry_plan = await explain_statement(conn, entries, *entry_args)
    assert plan_uses_index(entry_plan, "playbook_entries_draft_owner_idx"), entry_plan.plan
    concept_plan = await explain_statement(conn, concepts, *concept_args)
    assert plan_uses_index(concept_plan, "concepts_draft_owner_idx"), concept_plan.plan
