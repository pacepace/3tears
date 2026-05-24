"""Integration parity: SQLAlchemy table factories vs canonical migration DDL.

The agent-wake :func:`agent_wake_schedules_table` /
:func:`wake_fires_table` / :func:`webhook_subscriptions_table`
factories construct standalone SQLAlchemy ``Table`` objects (the
collections hand-roll their SQL and are not
:class:`SchemaBackedCollection` subclasses, so there is no
``TableSchema`` single source of truth to delegate to -- see
``tables.py`` module docstring). A standalone factory risks drifting
from the canonical migration DDL.

This test pins the two against each other structurally. It:

1. applies the full v001-v005 wake migration chain (plus its
   conversations + agent-skills dependencies) to one fresh Postgres
   schema (the migration-DDL truth, in its FINAL post-v005 shape --
   v004 extends ``wake_fires.status``, v005 opens the
   ``webhook_subscriptions.verification_scheme`` CHECK);
2. emits each factory's ``CREATE TABLE`` + ``CREATE INDEX`` DDL into a
   second fresh schema (the factory truth), via the SQLAlchemy
   PostgreSQL dialect compiler;
3. introspects both schemas through the Postgres catalogs and asserts
   they are structurally identical: column names + types + nullability
   + server defaults, primary key, UNIQUE constraints, foreign keys +
   ON DELETE action, CHECK constraints, and indexes (name + columns +
   uniqueness + access method + partial predicate + DESC ordering).

The agent-skills factory schema is also emitted into the factory
schema so the wake FKs to ``agent_skills(skill_id)`` resolve at
DDL-emit time; only the three wake tables are compared.

If anyone edits a migration without editing the factory (or vice
versa), this test fails. The two cannot silently diverge.

The factory carries only columns + constraints + indexes -- it does
NOT carry any trigger / function DDL (none of the wake tables have a
trigger today; if one is added later, it lives in the migration and
the consumer installs it, exactly as agent-skills documents for its
FTS ``search_vector`` trigger).

Excluded from CI (``-m "not integration"`` -- no docker on the runner);
run locally / in the integration lane with a docker daemon available.
"""

from __future__ import annotations

import asyncpg
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.skills.tables import (
    agent_skill_invocations_table,
    agent_skills_table,
)
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

# The three wake tables this factory exposes, in dependency order:
# agent_wake_schedules (FK target of wake_fires) first,
# webhook_subscriptions (the other FK target of wake_fires) next,
# wake_fires (both FKs) last. Only these are compared; the agent_skills
# tables are emitted into the factory schema solely so the
# skill_id / default_skill_id FKs resolve.
_TABLES = (
    "agent_wake_schedules",
    "webhook_subscriptions",
    "wake_fires",
)

# Tables emitted into the factory schema in DDL order: agent_skills
# first (FK target for schedules + subscriptions), then the wake tables
# in their own dependency order. agent_skill_invocations is included to
# match the agent_skills factory's own composite-FK shape (harmless;
# the wake tables do not reference it).
_FACTORY_EMIT_ORDER = (
    "agent_skills",
    "agent_skill_invocations",
    "agent_wake_schedules",
    "webhook_subscriptions",
    "wake_fires",
)


def _build_runner() -> MigrationRunner:
    """Register conversations + agent-skills + agent-wake on a fresh runner.

    Conversations + agent-skills are required because wake declares
    ``depends_on=("conversations", "agent_skills")``.

    :return: a runner ready to apply the wake migration chain
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    return runner


def _factory_metadata() -> sa.MetaData:
    """Build a metadata carrying the agent-skills + wake factory tables.

    Registration order matters for FK resolution: agent_skills (FK
    target) before the wake tables; agent_wake_schedules +
    webhook_subscriptions (FK targets of wake_fires) before wake_fires.

    :return: metadata with all five tables registered
    :rtype: sa.MetaData
    """
    metadata = sa.MetaData()
    agent_skills_table(metadata)
    agent_skill_invocations_table(metadata)
    agent_wake_schedules_table(metadata)
    webhook_subscriptions_table(metadata)
    wake_fires_table(metadata)
    return metadata


async def _create_factory_schema(conn: asyncpg.Connection, schema: str) -> None:
    """Emit the factory DDL into ``schema`` via the PostgreSQL dialect.

    :param conn: connection with ``search_path`` already set to ``schema``
    :ptype conn: asyncpg.Connection
    :param schema: target schema name (search_path is set by the caller)
    :ptype schema: str
    :return: nothing
    :rtype: None
    """
    metadata = _factory_metadata()
    dialect = postgresql.dialect()
    # Tables in dependency order so every FK target exists before the
    # referencing table is created.
    for table_name in _FACTORY_EMIT_ORDER:
        table = metadata.tables[table_name]
        await conn.execute(str(CreateTable(table).compile(dialect=dialect)))
        for index in table.indexes:
            await conn.execute(str(CreateIndex(index).compile(dialect=dialect)))


async def _columns(conn: asyncpg.Connection, schema: str, table: str) -> dict[str, tuple]:
    """Return ``column_name -> (data_type, udt_name, is_nullable, default)``.

    ``udt_name`` distinguishes e.g. ``jsonb`` / ``bytea`` precisely and
    would distinguish array types if any existed, so the signature
    carries it.

    :param conn: introspection connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read from
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :return: column name -> structural signature tuple
    :rtype: dict[str, tuple]
    """
    rows = await conn.fetch(
        """
        SELECT column_name, data_type, udt_name, is_nullable, column_default
          FROM information_schema.columns
         WHERE table_schema = $1 AND table_name = $2
        """,
        schema,
        table,
    )
    return {
        r["column_name"]: (
            r["data_type"],
            r["udt_name"],
            r["is_nullable"],
            # normalise the type-qualified default form Postgres echoes
            # back (``'inline'::text`` vs ``'inline'``) so the
            # migration's bare-literal default and the factory's
            # dialect-emitted default compare equal.
            _normalise_default(r["column_default"]),
        )
        for r in rows
    }


def _normalise_default(value: str | None) -> str | None:
    """Strip the ``::type`` cast suffix Postgres appends to defaults.

    Postgres stores ``'inline'`` as ``'inline'::text``, ``'{}'::jsonb``
    as ``'{}'::jsonb``, and ``false`` as ``false``; the two schemas may
    render the same logical default with or without the cast depending
    on the emit path. Stripping the trailing cast suffix makes them
    comparable while keeping the literal value. For JSONB the literal
    itself is ``'{}'`` after the suffix strip, identical on both sides.

    :param value: raw ``column_default`` from information_schema
    :ptype value: str | None
    :return: default with any trailing ``::type`` cast removed
    :rtype: str | None
    """
    if value is None:
        return None
    base = value
    marker = base.rfind("::")
    if marker != -1:
        base = base[:marker]
    return base.strip()


async def _constraints(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
) -> dict[str, tuple]:
    """Return constraint signatures keyed by a stable, name-free key.

    Primary-key / unique constraints are matched by their structural
    shape (columns) rather than by name, because the migrations declare
    PK + UNIQUE anonymously (Postgres auto-names them) so the
    auto-generated names differ between the two schemas. CHECK + FK
    constraints ARE named in the DDL, so their names are included in the
    signature; the schema-qualification ``pg_get_constraintdef`` adds to
    FK referenced tables is stripped so only structural shape compares.

    :param conn: introspection connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read from
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :return: set-shaped dict of structural constraint signatures
    :rtype: dict[str, tuple]
    """
    rows = await conn.fetch(
        """
        SELECT
            c.contype,
            c.conname,
            pg_get_constraintdef(c.oid) AS def,
            c.confdeltype,
            (
                SELECT array_agg(att.attname ORDER BY k.ord)
                  FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
                  JOIN pg_attribute att
                    ON att.attrelid = c.conrelid AND att.attnum = k.attnum
            ) AS cols
          FROM pg_constraint c
          JOIN pg_class rel ON rel.oid = c.conrelid
          JOIN pg_namespace ns ON ns.oid = rel.relnamespace
         WHERE ns.nspname = $1 AND rel.relname = $2
        """,
        schema,
        table,
    )
    result: dict[str, tuple] = {}
    for r in rows:
        contype = r["contype"]
        cols = tuple(r["cols"] or ())
        if contype == b"p" or contype == "p":
            # primary key: match on columns only (anonymous in DDL)
            result[f"pk:{cols}"] = ("p", cols)
        elif contype == b"u" or contype == "u":
            # unique: match on columns only (anonymous in DDL)
            result[f"u:{cols}"] = ("u", cols)
        elif contype == b"c" or contype == "c":
            # check: named in DDL -> match on name + normalised def.
            check_def = _norm_def(r["def"]).replace(f"{schema}.", "")
            result[f"c:{r['conname']}"] = ("c", r["conname"], check_def)
        elif contype == b"f" or contype == "f":
            # foreign key: named in DDL -> match on name + def + on-delete.
            # ``pg_get_constraintdef`` qualifies the referenced table
            # with the current schema, which differs between the two
            # compared schemas; strip it so only the structural shape
            # (columns, referenced columns, ON DELETE action) compares.
            fk_def = _norm_def(r["def"]).replace(f"{schema}.", "")
            result[f"f:{r['conname']}"] = (
                "f",
                r["conname"],
                fk_def,
                r["confdeltype"],
            )
    return result


def _norm_def(value: str) -> str:
    """Collapse internal whitespace in a constraint definition.

    ``pg_get_constraintdef`` is deterministic for a given catalog state,
    but defensive whitespace normalisation keeps the comparison robust
    against trivial formatting differences between the migration's
    multi-line CHECK and the factory's emit.

    :param value: raw constraint definition
    :ptype value: str
    :return: whitespace-collapsed definition
    :rtype: str
    """
    return " ".join(value.split())


async def _indexes(conn: asyncpg.Connection, schema: str, table: str) -> dict[str, str]:
    """Return ``index_name -> normalised index definition``.

    The constraint-backed indexes (PK / UNIQUE) carry auto-generated
    names that differ between the schemas, so they are excluded here and
    covered by :func:`_constraints` instead. Only the explicitly named
    ``CREATE INDEX`` indexes are compared by name -- including the
    partial-index ``WHERE`` predicates and the ``DESC`` orderings, which
    Postgres preserves in ``indexdef``.

    :param conn: introspection connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read from
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :return: index name -> definition with the schema prefix stripped
    :rtype: dict[str, str]
    """
    rows = await conn.fetch(
        """
        SELECT i.indexname, i.indexdef
          FROM pg_indexes i
         WHERE i.schemaname = $1 AND i.tablename = $2
           AND NOT EXISTS (
               SELECT 1 FROM pg_constraint c
                WHERE c.conindid = (
                    SELECT cls.oid FROM pg_class cls
                      JOIN pg_namespace ns ON ns.oid = cls.relnamespace
                     WHERE cls.relname = i.indexname AND ns.nspname = $1
                )
           )
        """,
        schema,
        table,
    )
    result: dict[str, str] = {}
    for r in rows:
        # strip the schema-qualification so the two schemas' defs match
        # (``ON sched_mig.agent_wake_schedules`` vs the factory schema).
        normalised = r["indexdef"].replace(f"{schema}.", "")
        result[r["indexname"]] = _norm_def(normalised)
    return result


async def _apply_both_schemas(
    url: str,
    migration_schema: str,
    factory_schema: str,
) -> None:
    """Apply migrations to one schema and factory DDL to another.

    :param url: testcontainer Postgres URL
    :ptype url: str
    :param migration_schema: schema receiving the full migration chain
    :ptype migration_schema: str
    :param factory_schema: schema receiving the factory-emitted DDL
    :ptype factory_schema: str
    :return: nothing
    :rtype: None
    """
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{migration_schema}", public')
        store = AsyncpgStore(conn)
        runner = _build_runner()
        applied = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
        assert applied > 0

        await conn.execute(f'SET search_path TO "{factory_schema}", public')
        await _create_factory_schema(conn, factory_schema)
    finally:
        await conn.close()


@pytest.fixture
async def two_schemas(pg_url: str):  # noqa: ANN201 -- async-gen fixture
    """Create two fresh schemas and drop both on teardown.

    :param pg_url: testcontainer URL
    :ptype pg_url: str
    :return: tuple ``(url, migration_schema, factory_schema)``
    :rtype: tuple[str, str, str]
    """
    base = f"wk_par_{id(object())}".lower()
    migration_schema = f"{base}_mig"
    factory_schema = f"{base}_fac"
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{migration_schema}"')
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{factory_schema}"')
    finally:
        await conn.close()
    yield (pg_url, migration_schema, factory_schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{migration_schema}" CASCADE')
        await conn.execute(f'DROP SCHEMA IF EXISTS "{factory_schema}" CASCADE')
    finally:
        await conn.close()


class TestFactoryMigrationParity:
    """The factory DDL is structurally identical to the migration DDL."""

    @pytest.mark.parametrize("table", _TABLES)
    async def test_columns_match(
        self,
        two_schemas: tuple[str, str, str],
        table: str,
    ) -> None:
        """Every column (type + nullability + default) matches."""
        url, mig, fac = two_schemas
        await _apply_both_schemas(url, mig, fac)
        conn = await asyncpg.connect(url)
        try:
            mig_cols = await _columns(conn, mig, table)
            fac_cols = await _columns(conn, fac, table)
        finally:
            await conn.close()
        assert mig_cols == fac_cols

    @pytest.mark.parametrize("table", _TABLES)
    async def test_constraints_match(
        self,
        two_schemas: tuple[str, str, str],
        table: str,
    ) -> None:
        """PK, UNIQUE, CHECK, and FK (with ON DELETE) all match."""
        url, mig, fac = two_schemas
        await _apply_both_schemas(url, mig, fac)
        conn = await asyncpg.connect(url)
        try:
            mig_cons = await _constraints(conn, mig, table)
            fac_cons = await _constraints(conn, fac, table)
        finally:
            await conn.close()
        assert mig_cons == fac_cons

    @pytest.mark.parametrize("table", _TABLES)
    async def test_indexes_match(
        self,
        two_schemas: tuple[str, str, str],
        table: str,
    ) -> None:
        """Every named CREATE INDEX matches by name + definition."""
        url, mig, fac = two_schemas
        await _apply_both_schemas(url, mig, fac)
        conn = await asyncpg.connect(url)
        try:
            mig_idx = await _indexes(conn, mig, table)
            fac_idx = await _indexes(conn, fac, table)
        finally:
            await conn.close()
        assert mig_idx == fac_idx
