"""integration: a caller's visible knowledge (each collection's ``list_visible_to_user``) reads what the statement it
replaced read.

The listing admitted each row through the three-scope visibility clause
(:func:`threetears.agent.acl.three_scope_visibility_clause`, a correlated ``EXISTS`` over the caller's grants run for
every row) beside ``status = 'active'`` and the datasource gather across the origin link, the concepts ``LEFT JOIN``-ed
to their bound table's name. It now reads the caller's customers once and the rows a page at a time by key
(:func:`threetears.agent.acl.visible_rows`), the bound tables by the ids the concepts hold. The equivalence harness seeds
each table for every case and compares the old statement's rows with the rows the collection builds its snapshots
from, over the grants of every scope arm (a type grant on knowledge, a namespace grant, a type grant on another type,
none), platform / customer / user rows, the caller's and another user's, active and draft, each datasource gather. On
the session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its
own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.schema import CreateTable

from threetears.agent.acl import three_scope_visibility_clause
from threetears.agent.acl.migrations import register as register_acl
from packages.core.tests.dialect_support import pg_dialect
from threetears.agent.knowledge import collections as knowledge_collections
from threetears.agent.knowledge.collections import ConceptCollection, PlaybookEntryCollection
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_START = datetime(2026, 11, 1, tzinfo=UTC)


def _named(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"visible-{name}")


#: the caller, and another user of the same customers
_CALLER, _OTHER = _named("caller"), _named("other")

#: a customer the caller holds a knowledge type grant on, one it holds a namespace grant on, one it holds a type grant
#: on another type only, and one it holds nothing on
_TYPE_GRANTED, _NAMESPACE_GRANTED, _OTHER_TYPE, _UNGRANTED = (
    _named("type-granted"),
    _named("namespace-granted"),
    _named("other-type"),
    _named("ungranted"),
)

#: each seeded row's customer, in turn: a platform row first
_CUSTOMERS = (None, _TYPE_GRANTED, _NAMESPACE_GRANTED, _OTHER_TYPE, _UNGRANTED)

#: a customer datasource linked to a platform one, that platform one, and an unlinked datasource
_LINKED, _PLATFORM, _UNLINKED = _named("ds-linked"), _named("ds-platform"), _named("ds-unlinked")

#: each seeded row's anchor, in turn
_ANCHORS = (_LINKED, _PLATFORM, _UNLINKED, _named("ds-elsewhere"))

#: a bound table that exists, and one that does not (a concept bound to it reads no name)
_TABLE, _GONE_TABLE = _named("table"), _named("gone-table")


class _Store:
    """the migration runner's store over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


async def _grant(conn: asyncpg.Connection, scope: str, **values: Any) -> None:
    """one group holding the caller, with one grant of ``scope`` (its other columns from ``values``)."""
    group, role = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch) VALUES ('platform', $1, NULL, $2, $3)",
        group,
        f"g{group}",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO roles (role_id, customer_id, name, description, permissions, epoch)"
        " VALUES ($1, NULL, $2, 'r', '{\"*\": [\"read\"]}'::jsonb, $3)",
        role,
        f"r{role}",
        next_epoch(),
    )
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id, epoch) VALUES ($1, $2, 'user', $3, $4)",
        uuid.uuid4(),
        group,
        _CALLER,
        next_epoch(),
    )
    if scope == "type_customer":
        # acl v005: a type grant records its scope's customer
        values = {**values, "customer_id": values.get("scope_customer_id")}
    values = {**values, "epoch": next_epoch()}
    columns = ", ".join(values)
    marks = ", ".join(f"${index}" for index in range(5, 5 + len(values)))
    await conn.execute(
        f"INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, {columns})"
        f" VALUES ('platform', $1, $2, $3, $4, {marks})",
        uuid.uuid4(),
        role,
        group,
        scope,
        *values.values(),
    )


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the access tables (the acl package's own), the two knowledge tables as declared, the
    hub's ``datasources`` origin link and ``datasource_tables`` names, and the caller's grants of every arm."""
    schema = f"knowvis_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(connection))
        dialect = pg_dialect()
        for declared in (PlaybookEntryCollection.schema, ConceptCollection.schema):
            table = declared.to_sqlalchemy_table(MetaData())
            await connection.execute(str(CreateTable(table).compile(dialect=dialect)))
        await connection.execute("CREATE TABLE datasources (id uuid PRIMARY KEY, origin_datasource_id uuid)")
        await connection.executemany(
            "INSERT INTO datasources VALUES ($1, $2)",
            [(_LINKED, _PLATFORM), (_PLATFORM, None), (_UNLINKED, None)],
        )
        await connection.execute(
            "CREATE TABLE datasource_tables (id uuid PRIMARY KEY, schema_name text NOT NULL, table_name text NOT NULL)"
        )
        await connection.execute("INSERT INTO datasource_tables VALUES ($1, 'public', 'users')", _TABLE)
        await _grant(connection, "type_customer", scope_customer_id=_TYPE_GRANTED, scope_namespace_type="knowledge")
        await _grant(connection, "type_customer", scope_customer_id=_OTHER_TYPE, scope_namespace_type="datasource")
        namespace = uuid.uuid4()
        await connection.execute(
            "INSERT INTO namespaces (row_scope, namespace_id, customer_id, namespace_type, name, schema_name,"
            " date_created, date_updated, epoch) VALUES ('customer', $1, $2, 'datasource', $3, $4, now(), now(), $5)",
            namespace,
            _NAMESPACE_GRANTED,
            f"n{namespace.hex}",
            f"s{namespace.hex[-16:]}",
            next_epoch(),
        )
        await _grant(connection, "namespace", scope_namespace_id=namespace)
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _common(seed: Seeded) -> dict[str, Any]:
    """a row of each customer in turn, the caller's, another user's or no user's (every third NULL-owned in the NULLs
    case), active but every seventh a draft, anchored on each datasource in turn."""
    stamp = _START + timedelta(minutes=seed.tie)
    owners = (None, _CALLER, _OTHER)
    return {
        "id": uuid.uuid5(uuid.NAMESPACE_OID, f"visible-row-{seed.tenant}-{seed.index}"),
        "customer_id": _CUSTOMERS[(seed.index + seed.tenant) % len(_CUSTOMERS)],
        "user_id": None if seed.null else owners[seed.index % len(owners)],
        "status": "draft" if seed.index % 7 == 3 else "active",
        "datasource_id": _ANCHORS[seed.index % len(_ANCHORS)],
        "date_created": stamp,
        "date_updated": stamp,
    }


def _entry(seed: Seeded) -> Mapping[str, Any]:
    return {
        **_common(seed),
        "playbook_id": uuid.uuid5(uuid.NAMESPACE_OID, "pb"),
        "title": f"t{seed.index}",
        "body": "b",
        "origin_entry_id": None if seed.null else uuid.uuid5(uuid.NAMESPACE_OID, "origin"),
    }


def _concept(seed: Seeded) -> Mapping[str, Any]:
    return {
        **_common(seed),
        "name": f"n{seed.index}",
        "definition": "d",
        "datasource_table_id": (None, _TABLE, _GONE_TABLE)[seed.index % 3],
    }


def _old_sql(
    table: str, datasource_id: uuid.UUID | None, datasource_table_id: uuid.UUID | None = None
) -> tuple[str, list[Any]]:
    """the statement the listing sent before, as the collection built it."""
    alias = "pe" if table == "playbook_entries" else "co"
    clause, params = three_scope_visibility_clause(
        user_id=_CALLER,
        customer_id_column=f"{alias}.customer_id",
        user_id_column=f"{alias}.user_id",
        scope_namespace_type="knowledge",
        param_offset=1,
    )
    if table == "playbook_entries":
        sql = (
            "SELECT pe.id, pe.customer_id, pe.user_id, pe.origin_entry_id, pe.title, pe.body, pe.tags, "
            "pe.datasource_id, pe.always_inject, pe.date_created FROM playbook_entries pe "
            f"WHERE ({clause}) AND pe.status = 'active' "
        )
    else:
        sql = (
            "SELECT co.id, co.customer_id, co.user_id, co.origin_concept_id, co.name, co.aliases, co.definition, "
            "co.datasource_id, co.datasource_table_id, co.sql_fragment, co.caveats, co.tags, co.always_inject, "
            "co.date_created, dt.schema_name AS bound_schema_name, dt.table_name AS bound_table_name "
            "FROM concepts co LEFT JOIN datasource_tables dt ON dt.id = co.datasource_table_id "
            f"WHERE ({clause}) AND co.status = 'active' "
        )
    if datasource_id is not None:
        params.append(datasource_id)
        idx = len(params)
        sql += (
            f" AND {alias}.datasource_id IN (${idx}, (SELECT origin_datasource_id FROM datasources WHERE id = ${idx}))"
        )
    if datasource_table_id is not None:
        params.append(datasource_table_id)
        sql += f" AND co.datasource_table_id = ${len(params)}"
    return f"{sql} ORDER BY {alias}.date_created ASC", params


class _Proxy:
    """the rbac-read proxy pool the listing reads through (a customer scope on each statement), over the test
    connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.scopes: list[Any] = []

    async def fetch(self, query: str, *params: Any, customer_scope: Any) -> list[Any]:
        self.scopes.append(customer_scope)
        return list(await self._conn.fetch(query, *params))

    async def fetchrow(self, query: str, *params: Any, customer_scope: Any) -> Any:
        self.scopes.append(customer_scope)
        return await self._conn.fetchrow(query, *params)


_ENTRY_COLUMNS = [
    "id",
    "customer_id",
    "user_id",
    "origin_entry_id",
    "title",
    "body",
    "tags",
    "datasource_id",
    "always_inject",
    "date_created",
]
_CONCEPT_COLUMNS = [
    "id",
    "customer_id",
    "user_id",
    "origin_concept_id",
    "name",
    "aliases",
    "definition",
    "datasource_id",
    "datasource_table_id",
    "sql_fragment",
    "caveats",
    "tags",
    "always_inject",
    "date_created",
    "bound_schema_name",
    "bound_table_name",
]


async def _same_listing(
    conn: asyncpg.Connection,
    monkeypatch: pytest.MonkeyPatch,
    collection: Any,
    row: Any,
    builder: str,
    columns: list[str],
    **narrowed: Any,
) -> None:
    """compare the listing, narrowed by ``narrowed`` (``datasource_id=``, ``datasource_table_id=``), with the
    statement it replaced, on the rows the collection builds its snapshots from."""
    scope = uuid.uuid4()
    table = collection.schema.name
    built: list[dict[str, Any]] = []
    build = getattr(knowledge_collections, builder)

    def keep(found: Mapping[str, Any]) -> Any:
        built.append(dict(found))
        return build(found)

    monkeypatch.setattr(knowledge_collections, builder, keep)

    async def helper(c: asyncpg.Connection) -> list[dict[str, Any]]:
        built.clear()
        proxy = _Proxy(c)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        reading = collection(registry=CollectionRegistry(), config=config)
        reading.l3_pool = proxy
        await reading.list_visible_to_user(_CALLER, customer_scope=scope, **narrowed)
        assert proxy.scopes and set(proxy.scopes) == {scope}
        return list(built)

    old_sql, params = _old_sql(table, narrowed.get("datasource_id"), narrowed.get("datasource_table_id"))
    answered = await assert_same_results(
        conn,
        table=table,
        row=row,
        old_sql=old_sql,
        params=params,
        helper=helper,
        columns=columns,
        # oldest first; the old order left rows created at the same instant in the engine's order, the keyed one
        # breaks them by id
        order_by=["date_created"],
    )
    # the seeds reach every arm: some rows visible, some not, in the many-rows case
    assert any(answered), "no case admitted a row: the grants were not seeded"


@pytest.mark.parametrize("datasource", ["every", "linked", "unlinked"])
@pytest.mark.parametrize(
    ("collection", "row", "builder", "columns"),
    [
        (PlaybookEntryCollection, _entry, "_row_to_snapshot", _ENTRY_COLUMNS),
        (ConceptCollection, _concept, "_row_to_concept_snapshot", _CONCEPT_COLUMNS),
    ],
    ids=["playbook_entries", "concepts"],
)
async def test_the_visible_listing_reads_what_the_statement_it_replaced_read(
    conn: asyncpg.Connection,
    monkeypatch: pytest.MonkeyPatch,
    collection: Any,
    row: Any,
    builder: str,
    columns: list[str],
    datasource: str,
) -> None:
    """Flips if the keyed listing builds its snapshots from other rows than the three-scope clause's statement read:
    a customer the grants do not cover (another type's), a namespace grant's customer missed, another user's own row,
    a draft, a row off the datasource gather, a concept's bound table named otherwise, an order other than oldest
    first; or drops the customer scope from a statement."""
    datasource_id = {"every": None, "linked": _LINKED, "unlinked": _UNLINKED}[datasource]
    await _same_listing(conn, monkeypatch, collection, row, builder, columns, datasource_id=datasource_id)


async def test_the_visible_concepts_of_one_bound_table_are_the_statement_s(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the concepts listed for one bound table (on the linked datasource's gather) are not the ones the
    statement it replaced listed: another table's, or every table's."""
    await _same_listing(
        conn,
        monkeypatch,
        ConceptCollection,
        _concept,
        "_row_to_concept_snapshot",
        _CONCEPT_COLUMNS,
        datasource_id=_LINKED,
        datasource_table_id=_TABLE,
    )
