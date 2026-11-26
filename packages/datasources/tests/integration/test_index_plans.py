"""integration: each index the datasources package declares serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. The hub's migrations build these tables
and their indexes; each test builds the table and the index as the hub leaves them, seeds it past the size where the
planner would rather scan, runs the purpose's own statement through ``EXPLAIN ANALYZE``
(:class:`threetears.core.testing.plans.PlanRecorder`), and asks the plan whether it found its rows through the index by
an index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from threetears.core.collections.delete_cascade import read_delete_cascade
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import PlanRecorder, explain_statement, plan_uses_index
from threetears.datasources.collections import (
    CapabilitySourceCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.datasources.schema_priming import SchemaPrimingIntegration

pytestmark = pytest.mark.integration

#: ``datasource_tables`` as hub v001, v006, v010 and v033 leave it, the columns a cascade's read names
_DATASOURCE_TABLES_DDL = """
CREATE TABLE datasource_tables (
    id uuid NOT NULL PRIMARY KEY,
    datasource_id uuid NOT NULL,
    schema_name character varying(255) NOT NULL,
    table_name character varying(255) NOT NULL,
    template_id uuid,
    date_created timestamp with time zone NOT NULL DEFAULT now(),
    date_updated timestamp with time zone NOT NULL DEFAULT now(),
    epoch bigint NOT NULL DEFAULT 1,
    gone boolean NOT NULL DEFAULT false
)
"""

#: hub v146's index (``USING lsm (template_id HASH)`` on YugabyteDB, the database's default first column there)
_BY_TEMPLATE_DDL = (
    "CREATE INDEX datasource_tables_by_template ON datasource_tables (template_id) WHERE template_id IS NOT NULL"
)

#: the hub's natural-key index, among live rows only (datasources' LIVE_UNIQUE_INDEXES)
_NATURAL_KEY_DDL = (
    "CREATE UNIQUE INDEX datasource_tables_datasource_id_schema_name_table_name_key"
    " ON datasource_tables (datasource_id, schema_name, table_name) WHERE NOT gone"
)

#: rows seeded: past the size where the planner would rather scan than use an index
_ROWS = 5000

#: templates the rows are bound to (one row in ten is bound to none)
_TEMPLATES = 500


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema holding ``datasource_tables`` and its template index, seeded."""
    schema = f"dsplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(_DATASOURCE_TABLES_DDL)
        await connection.execute(_BY_TEMPLATE_DDL)
        templates = [uuid.uuid4() for _ in range(_TEMPLATES)]
        await connection.execute(
            "INSERT INTO datasource_tables (id, datasource_id, schema_name, table_name, template_id)"
            " SELECT gen_random_uuid(), gen_random_uuid(), 's', 't' || g,"
            " CASE WHEN g % 10 <> 0 THEN ($1::uuid[])[1 + g % $2] END FROM generate_series(1, $3) g",
            templates,
            _TEMPLATES,
            _ROWS,
        )
        await connection.execute("ANALYZE datasource_tables")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def test_a_template_delete_reads_the_tables_it_unbinds_through_datasource_tables_by_template(
    conn: asyncpg.Connection,
) -> None:
    """Flips if the tables a template's delete is about to unbind (read_delete_cascade: datasource_tables.template_id
    is ON DELETE SET NULL, DeleteCascade on TableTemplateCollection) stop reading datasource_tables_by_template by the
    template, or read past the template's own tables."""
    template = await conn.fetchval("SELECT template_id FROM datasource_tables WHERE template_id IS NOT NULL LIMIT 1")
    bound = await conn.fetchval("SELECT count(*) FROM datasource_tables WHERE template_id = $1", template)
    recorder = PlanRecorder(conn)
    templates = TableTemplateCollection(registry=CollectionRegistry(), config=DefaultCoreConfig())
    templates.l3_pool = recorder
    found = await read_delete_cascade(templates, [template])
    assert bound and [len(rows.keys) for rows in found] == [bound], found
    plan = recorder.plans[0]
    assert plan_uses_index(plan, "datasource_tables_by_template"), json.dumps(plan.plan)[:2500]
    assert plan.rows_scanned == bound, json.dumps(plan.plan)[:2500]


async def test_a_table_is_read_by_its_natural_key_through_its_live_unique_index(conn: asyncpg.Connection) -> None:
    """Flips if the introspector's read of one table by its natural key (datasource, schema, name) stops reading the
    hub's live unique index on it (datasources' LIVE_UNIQUE_INDEXES), or answers another row."""
    await conn.execute(_NATURAL_KEY_DDL)
    await conn.execute("ANALYZE datasource_tables")
    wanted = await conn.fetchrow("SELECT id, datasource_id FROM datasource_tables WHERE table_name = 't7'")
    recorder = PlanRecorder(conn)
    tables = DataSourceTableCollection(registry=CollectionRegistry(), config=DefaultCoreConfig())
    tables.l3_pool = recorder
    found = await tables.get_by_natural_key(wanted["datasource_id"], "s", "t7")
    assert found is not None and found.id == wanted["id"], found
    (plan,) = recorder.plans
    assert plan_uses_index(plan, "datasource_tables_datasource_id_schema_name_table_name_key"), json.dumps(plan.plan)[
        :2500
    ]


#: ``datasources`` as hub v001 leaves it, the columns the sweep's read names (and the name it is unique on)
_DATASOURCES_DDL = """
CREATE TABLE datasources (
    id uuid NOT NULL PRIMARY KEY,
    name character varying(255) NOT NULL,
    status character varying(50) NOT NULL,
    date_created timestamp with time zone NOT NULL DEFAULT now(),
    date_updated timestamp with time zone NOT NULL DEFAULT now(),
    epoch bigint NOT NULL DEFAULT 1,
    gone boolean NOT NULL DEFAULT false
)
"""

#: hub v001's index (``USING lsm (status HASH)`` on YugabyteDB, the database's default first column there)
_BY_STATUS_DDL = "CREATE INDEX idx_datasources_status ON datasources (status)"

#: the sweep's read as it was written before the keyed pages: the equivalence proof's reference
_OLD_ACTIVE_IDS = "SELECT id FROM datasources WHERE status = $1 ORDER BY date_created, id"

#: one datasource in this many is active (the rest disabled): selective enough that the planner would rather walk
#: the index than scan, as it is wherever disabled rows outnumber the active (a registry of tens of rows reads
#: either way in one statement)
_ACTIVE_EVERY = 8


@pytest.fixture
async def sources(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema holding ``datasources`` and its status index, seeded (some rows sharing a
    ``date_created``, so the order's tie-break by id is exercised)."""
    schema = f"dsplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(_DATASOURCES_DDL)
        await connection.execute(_BY_STATUS_DDL)
        await connection.execute(
            "INSERT INTO datasources (id, name, status, date_created)"
            " SELECT gen_random_uuid(), 'ds' || g, CASE WHEN g % $1 = 0 THEN 'active' ELSE 'disabled' END,"
            " timestamptz '2026-01-01' + (g / 3) * interval '1 minute' FROM generate_series(1, $2) g",
            _ACTIVE_EVERY,
            _ROWS,
        )
        await connection.execute("ANALYZE datasources")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


def _sources_on(pool: object) -> CapabilitySourceCollection:
    collection = CapabilitySourceCollection(registry=CollectionRegistry(), config=DefaultCoreConfig())
    collection.l3_pool = pool  # type: ignore[assignment]
    return collection


async def test_the_sweep_reads_the_active_datasources_through_idx_datasources_status(
    sources: asyncpg.Connection,
) -> None:
    """Flips if the scheduler sweep's active ids (CapabilitySourceCollection.iter_active_ids) stop reading
    idx_datasources_status by the status, or read past the active rows."""
    active = await sources.fetchval("SELECT count(*) FROM datasources WHERE status = 'active'")
    recorder = PlanRecorder(sources)
    found = await _sources_on(recorder).iter_active_ids()
    assert len(found) == active
    plan = recorder.plans[0]
    assert plan_uses_index(plan, "idx_datasources_status"), json.dumps(plan.plan)[:2500]
    assert plan.rows_scanned == active, json.dumps(plan.plan)[:2500]


async def test_the_sweep_answers_the_active_ids_the_unkeyed_read_answered(sources: asyncpg.Connection) -> None:
    """Flips if the keyed pages answer other ids, or another order, than the old whole statement, before or after a
    row changes status (one disabled, one enabled) and one is added past the last page."""
    collection = _sources_on(sources)
    for step in range(3):
        old = [row["id"] for row in await sources.fetch(_OLD_ACTIVE_IDS, "active")]
        new = await collection.iter_active_ids()
        assert new == old and len(new) > 500, step
        if step == 0:
            await sources.execute("UPDATE datasources SET status = 'disabled' WHERE id = $1", old[len(old) // 2])
            await sources.execute(
                "UPDATE datasources SET status = 'active' WHERE id = (SELECT id FROM datasources"
                " WHERE status = 'disabled' ORDER BY date_created, id LIMIT 1)"
            )
        elif step == 1:
            await sources.execute(
                "INSERT INTO datasources (id, name, status, date_created) VALUES ($1, 'late', 'active', $2)",
                uuid.uuid4(),
                await sources.fetchval("SELECT max(date_created) FROM datasources"),
            )


async def test_schema_priming_reads_its_names_through_the_unique_name_index(db_container: str) -> None:
    """Flips if schema priming's name resolution (a set of names, ``status = 'active'``, the visibility guard) stops
    reading idx_datasources_name, one row a name: hinted on YugabyteDB to the hub's status index it read every active
    datasource and filtered (D8, 2026-11-15)."""
    schema = f"dsprime_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        table = CapabilitySourceCollection.schema.to_sqlalchemy_table(MetaData())
        for column in table.columns:
            column.server_default = None
        await conn.execute(str(CreateTable(table).compile(dialect=postgresql.dialect())))  # type: ignore[no-untyped-call]
        # the hub's status index (the one the hint named before)
        await conn.execute("CREATE INDEX idx_datasources_status ON datasources (status)")
        # the hub's name index, among live rows only (datasources' LIVE_UNIQUE_INDEXES): the read names NOT gone
        await conn.execute("CREATE UNIQUE INDEX idx_datasources_name ON datasources (name) WHERE NOT gone")
        await conn.execute("CREATE TABLE datasource_customers (datasource_id uuid NOT NULL, customer_id uuid NOT NULL)")
        columns = await conn.fetch(
            "SELECT column_name, data_type FROM information_schema.columns"
            " WHERE table_schema = current_schema() AND table_name = 'datasources' AND is_nullable = 'NO'"
        )
        by_type = {
            "uuid": "gen_random_uuid()",
            "boolean": "false",
            "jsonb": "'{}'::jsonb",
            "integer": "0",
            "bigint": "1",
        }
        values = {"name": "'ds-' || g", "status": "CASE WHEN g % 50 = 0 THEN 'disabled' ELSE 'active' END"}
        names = [c["column_name"] for c in columns]
        exprs = [
            values.get(name)
            or ("now()" if c["data_type"].startswith("timestamp") else by_type.get(c["data_type"], "'v' || g"))
            for name, c in zip(names, columns, strict=True)
        ]
        await conn.execute(
            f"INSERT INTO datasources ({', '.join(names)}) SELECT {', '.join(exprs)} FROM generate_series(1, {_ROWS}) g"
        )
        await conn.execute("ANALYZE datasources")
        # the real connection answers which engine it is, so the read is spelled (and hinted) for it; nothing else runs
        captured = CapturedL3(real=conn, passes=lambda _sql: False)

        class _Digests:
            l3_pool = captured

        priming = SchemaPrimingIntegration(_Digests(), customer_id=uuid.uuid4(), datasource_names=["ds-1", "ds-2"])
        await priming.datasource_ids()
        (sql, args), *rest = [(sql, args) for sql, args in captured.asked if '"datasources"' in sql]
        assert not rest, captured.asked
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_datasources_name"), plan.plan
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
