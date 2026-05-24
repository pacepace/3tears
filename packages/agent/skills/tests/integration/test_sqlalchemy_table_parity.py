"""Integration parity: SQLAlchemy table factories vs canonical migration DDL.

The agent-skills :func:`agent_skills_table` /
:func:`agent_skill_invocations_table` factories construct standalone
SQLAlchemy ``Table`` objects (the collections hand-roll their SQL and
are not :class:`SchemaBackedCollection` subclasses, so there is no
``TableSchema`` single source of truth to delegate to -- see
``tables.py`` module docstring). A standalone factory risks drifting
from the canonical v001/v002 migration DDL.

This test pins the two against each other structurally. It:

1. applies the v001 + v002 migrations to one fresh Postgres schema
   (the migration-DDL truth);
2. emits each factory's ``CREATE TABLE`` + ``CREATE INDEX`` DDL into a
   second fresh schema (the factory truth), via the SQLAlchemy
   PostgreSQL dialect compiler;
3. introspects both schemas through the Postgres catalogs and asserts
   they are structurally identical: column names + types + nullability
   + server defaults, primary key, UNIQUE constraints, foreign keys +
   ON DELETE action, CHECK constraints, and indexes (name + columns +
   uniqueness + access method).

If anyone edits a migration without editing the factory (or vice
versa), this test fails. The two cannot silently diverge.

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
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_TABLES = ("agent_skills", "agent_skill_invocations")


def _build_runner() -> MigrationRunner:
    """Register conversations + agent-skills on a fresh runner.

    Conversations is required because skills declares
    ``depends_on=("conversations",)``.

    :return: a runner ready to apply the skills migration chain
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    return runner


def _factory_metadata() -> sa.MetaData:
    """Build a metadata carrying both factory tables.

    :return: metadata with ``agent_skills`` + ``agent_skill_invocations``
    :rtype: sa.MetaData
    """
    metadata = sa.MetaData()
    agent_skills_table(metadata)
    agent_skill_invocations_table(metadata)
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
    # Tables in dependency order: agent_skills first so the composite FK
    # on agent_skill_invocations resolves against an existing target.
    for table_name in _TABLES:
        table = metadata.tables[table_name]
        await conn.execute(str(CreateTable(table).compile(dialect=dialect)))
        for index in table.indexes:
            await conn.execute(str(CreateIndex(index).compile(dialect=dialect)))


async def _columns(conn: asyncpg.Connection, schema: str, table: str) -> dict[str, tuple]:
    """Return ``column_name -> (data_type, is_nullable, column_default)``.

    ``udt_name`` distinguishes ``text`` from ``text[]`` (the latter
    reports ``data_type='ARRAY'`` + ``udt_name='_text'``), so the
    signature carries it.

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
            # back (``'additive'::text`` vs ``'additive'``) so the
            # migration's bare-literal default and the factory's
            # dialect-emitted default compare equal.
            _normalise_default(r["column_default"]),
        )
        for r in rows
    }


def _normalise_default(value: str | None) -> str | None:
    """Strip the ``::type`` cast suffix Postgres appends to defaults.

    Postgres stores ``'additive'`` as ``'additive'::text`` and ``true``
    as ``true``; the two schemas may render the same logical default
    with or without the cast depending on emit path. Stripping the cast
    suffix makes them comparable while keeping the literal value.

    :param value: raw ``column_default`` from information_schema
    :ptype value: str | None
    :return: default with any trailing ``::type`` cast removed
    :rtype: str | None
    """
    if value is None:
        return None
    # split off a trailing ``::sometype`` cast (only the last one; the
    # literal itself never contains ``::`` in our schema).
    base = value
    marker = base.rfind("::")
    if marker != -1:
        base = base[:marker]
    return base.strip()


async def _constraints(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    *,
    other_schema: str,
) -> dict[str, tuple]:
    """Return constraint signatures keyed by a stable, name-free key.

    Primary-key / unique / check / foreign-key constraints are matched
    by their structural shape rather than by name, because the v001/v002
    migrations declare PK + UNIQUE anonymously (Postgres auto-names
    them) so the auto-generated names differ between the two schemas.
    CHECK + FK constraints ARE named in the DDL, so their names are
    included in the signature.

    :param conn: introspection connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read from
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :param other_schema: the sibling schema name; unused here but kept
        for signature symmetry with callers that strip both schema
        qualifications (the FK def qualifies the referenced table with
        its own schema, which differs between the two compared schemas)
    :ptype other_schema: str
    :return: set-shaped dict of structural constraint signatures
    :rtype: dict[str, tuple]
    """
    del other_schema
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
            # strip the schema prefix the def may carry so the two
            # compared schemas yield identical text.
            check_def = _norm_def(r["def"]).replace(f"{schema}.", "")
            result[f"c:{r['conname']}"] = ("c", r["conname"], check_def)
        elif contype == b"f" or contype == "f":
            # foreign key: named in DDL -> match on name + def + on-delete.
            # ``pg_get_constraintdef`` qualifies the referenced table
            # with the current schema, which differs between the two
            # compared schemas; strip it so only the structural shape
            # (columns, referenced columns, ON DELETE action) is compared.
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
    multi-line CHECK and the factory's single-line emit.

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
    ``CREATE INDEX`` indexes are compared by name.

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
        # (``ON sk_a.agent_skills`` vs ``ON sk_b.agent_skills``).
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
    :param migration_schema: schema receiving the v001/v002 migrations
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
    base = f"sk_par_{id(object())}".lower()
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
            mig_cons = await _constraints(conn, mig, table, other_schema=fac)
            fac_cons = await _constraints(conn, fac, table, other_schema=mig)
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
