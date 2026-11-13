"""Migrate once per session: a template database per migration set, cloned for each test.

Most agent-block integration tests need a schema with a package's whole migration chain applied (memory's is 43
steps) and nothing else. Applying it per test cost about 0.2 s of every such test's setup, most of the block's
time. Instead, the first test that asks for a migration set builds a template database once -- the ``vector``
extension, a fixed schema :data:`SCHEMA`, the migrations applied into it -- and every test gets its own database
copied from it (``CREATE DATABASE ... TEMPLATE ... STRATEGY FILE_COPY``), dropped with ``WITH (FORCE)`` after.
A copy is a file copy of a small database, not a migration replay.

The ``(url, schema)`` a package's ``applied_schema`` fixture yields keeps its meaning: a URL whose database holds
the migrated schema, and the schema's name. Only the URL's database changes, to the test's own copy.

Tests that are ABOUT migrating -- a migration chain, a schema checked against its declarations, a migration applied
to a schema in a given state -- keep their fresh per-test schema (``pg_schema``) and migrate it themselves.

It works against any server the ``test`` user may create databases on: the pool's process-mode Postgres, a
testcontainer, CI's service container. Templates are per process and per server, and named uniquely, so two
processes sharing one server never meet. As the process exits it drops its templates, so a long-lived server
(one ``THREETEARS_TEST_POSTGRES_URL`` names) does not collect them; a server already gone (a child process, a
testcontainer) took them with it.

YugabyteDB copies no database from a template, and its DDL is slow: migrating memory's chain into a fresh schema took
about a minute a test (297 tests, five hours). There the process migrates one schema per migration set, once, and
every test is handed that schema with what the last test left taken out (:func:`_kept_schema`): after each test the
schema's catalog is read again -- its relations with their stats, columns, indexes, constraints, functions,
triggers, views, types -- and only when it is exactly what the migrations made are the rows the test wrote deleted,
the sequences set back and the rows the migrations themselves wrote checked unchanged. Any difference (a test that
made or altered anything, analyzed a table, or changed a migration's own rows), or any failure to reset, drops the
schema, and the next test migrates a fresh one: a test never sees another's leftovers, it only skips the minute when
the last test left nothing but rows.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from threetears.core.testing.schemas import drop_schema
from threetears.observe import contain, get_logger

log = get_logger(__name__)

#: the schema every template, and so every copy, holds the migrations in
SCHEMA = "it"

#: a migration set: given a connection whose ``search_path`` is the schema, and the schema's name, migrate it
Migrate = Callable[[asyncpg.Connection, str], Awaitable[None]]

#: the templates this process built: (server, migration set) -> template database
_TEMPLATES: dict[tuple[str, str], str] = {}
#: a database URL on each server a template was built on, to drop the templates through
_SERVERS: dict[str, str] = {}
#: whether each server asked copies a database from a template (YugabyteDB does not)
_COPIES_TEMPLATES: dict[str, bool] = {}


def _on_database(url: str, database: str) -> str:
    """``url`` with its database replaced.

    :param url: a ``postgresql://`` URL
    :ptype url: str
    :param database: the database
    :ptype database: str
    :return: the URL of ``database`` on the same server, as the same user
    :rtype: str
    """
    return urlunsplit(urlsplit(url)._replace(path=f"/{database}"))


def _server_of(url: str) -> str:
    """the server and user part of ``url``, which a template belongs to.

    :param url: a ``postgresql://`` URL
    :ptype url: str
    :return: the URL without its database
    :rtype: str
    """
    return urlunsplit(urlsplit(url)._replace(path="", query="", fragment=""))


async def _execute(url: str, sql: str) -> None:
    """run one statement on its own connection (``CREATE``/``DROP DATABASE`` refuse a transaction).

    :param url: the database to connect to
    :ptype url: str
    :param sql: the statement
    :ptype sql: str
    :return: None
    :rtype: None
    """
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _template(url: str, key: str, migrate: Migrate) -> str:
    """the template database for migration set ``key`` on ``url``'s server, built on first use.

    :param url: a database on the server
    :ptype url: str
    :param key: the migration set's name (``memory``, ``wake``), unique per set
    :ptype key: str
    :param migrate: applies the set
    :ptype migrate: Migrate
    :return: the template's name
    :rtype: str
    """
    found = _TEMPLATES.get((_server_of(url), key))
    if found is not None:
        return found
    template = f"tmpl_{key}_{uuid.uuid4().hex[:12]}"
    await _execute(url, f'CREATE DATABASE "{template}"')
    try:
        conn = await asyncpg.connect(_on_database(url, template))
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
            await conn.execute(f'CREATE SCHEMA "{SCHEMA}"')
            await conn.execute(f'SET search_path TO "{SCHEMA}", public')
            await migrate(conn, SCHEMA)
        finally:
            await conn.close()
    except BaseException:
        # a half-migrated template would hand every later test a broken schema
        await _execute(url, f'DROP DATABASE IF EXISTS "{template}" WITH (FORCE)')
        raise
    if not _SERVERS:
        atexit.register(_drop_templates)
    _SERVERS.setdefault(_server_of(url), url)
    _TEMPLATES[(_server_of(url), key)] = template
    return template


async def _drop_on(url: str, templates: list[str]) -> None:
    """drop ``templates`` on ``url``'s server.

    :param url: a database on the server
    :ptype url: str
    :param templates: the template databases
    :ptype templates: list[str]
    :return: None
    :rtype: None
    """
    conn = await asyncpg.connect(url, timeout=5)
    try:
        for template in templates:
            await conn.execute(f'DROP DATABASE IF EXISTS "{template}" WITH (FORCE)')
    finally:
        await conn.close()


def _drop_templates() -> None:
    """drop every template this process built, as it exits; best effort.

    :return: None
    :rtype: None
    """
    for server, url in _SERVERS.items():
        templates = [template for (on, _key), template in _TEMPLATES.items() if on == server]
        try:
            asyncio.run(_drop_on(url, templates))
        except OSError, TimeoutError:
            # NOSILENT: the server is gone (a child process or a testcontainer, stopped at the session's end),
            # and its templates went with it: nothing is left to drop
            pass
        except Exception as exc:  # noqa: BLE001 - contained: the process is exiting, its tests already reported
            contain(exc, log=log, what=f"dropping the test template databases on {server}")


async def _copies_templates(url: str) -> bool:
    """whether ``url``'s server copies a database from a template: every Postgres does, YugabyteDB does not.

    :param url: a database on the server
    :ptype url: str
    :return: True unless the server is YugabyteDB
    :rtype: bool
    """
    server = _server_of(url)
    if server not in _COPIES_TEMPLATES:
        conn = await asyncpg.connect(url)
        try:
            version = str(await conn.fetchval("SELECT version()"))
        finally:
            await conn.close()
        _COPIES_TEMPLATES[server] = "-YB-" not in version
    return _COPIES_TEMPLATES[server]


async def _migrated_schema(url: str, migrate: Migrate) -> str:
    """a fresh schema in ``url``'s database, migrated; dropped again if the migration fails.

    :param url: the database
    :ptype url: str
    :param migrate: applies the migration set
    :ptype migrate: Migrate
    :return: the migrated schema's name
    :rtype: str
    """
    schema = f"{SCHEMA}_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            await migrate(conn, schema)
        except BaseException:
            await drop_schema(conn, schema)
            raise
    finally:
        await conn.close()
    return schema


#: one line per thing the schema holds, so two reads of the catalog compare as text: each relation with its kind and
#: its stats (an ANALYZE changes what a plan test sees), each column, index, constraint, function, trigger, view and
#: type. ``$1`` is the schema.
_CATALOG_SQL = """
WITH ns AS (SELECT oid FROM pg_namespace WHERE nspname = $1)
SELECT line FROM (
    SELECT 'rel ' || c.relname || ' ' || c.relkind::text || ' ' || c.reltuples::text || ' ' || c.relpages::text AS line
    FROM pg_class c WHERE c.relnamespace = (SELECT oid FROM ns)
    UNION ALL
    SELECT 'col ' || c.relname || '.' || a.attname || ' ' || format_type(a.atttypid, a.atttypmod) || ' '
        || a.attnotnull::text || ' ' || coalesce(pg_get_expr(d.adbin, d.adrelid), '')
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
    WHERE c.relnamespace = (SELECT oid FROM ns) AND a.attnum > 0 AND NOT a.attisdropped
    UNION ALL
    SELECT 'idx ' || pg_get_indexdef(i.indexrelid)
    FROM pg_index i JOIN pg_class c ON c.oid = i.indrelid
    WHERE c.relnamespace = (SELECT oid FROM ns)
    UNION ALL
    SELECT 'con ' || c.relname || ' ' || k.conname || ' ' || pg_get_constraintdef(k.oid)
    FROM pg_constraint k JOIN pg_class c ON c.oid = k.conrelid
    WHERE k.connamespace = (SELECT oid FROM ns)
    UNION ALL
    SELECT 'fn ' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ') ' || md5(p.prosrc)
    FROM pg_proc p WHERE p.pronamespace = (SELECT oid FROM ns)
    UNION ALL
    SELECT 'trg ' || pg_get_triggerdef(t.oid)
    FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
    WHERE c.relnamespace = (SELECT oid FROM ns) AND NOT t.tgisinternal
    UNION ALL
    SELECT 'view ' || c.relname || ' ' || pg_get_viewdef(c.oid)
    FROM pg_class c WHERE c.relnamespace = (SELECT oid FROM ns) AND c.relkind IN ('v', 'm')
    UNION ALL
    SELECT 'type ' || t.typname || ' ' || t.typtype::text
    FROM pg_type t WHERE t.typnamespace = (SELECT oid FROM ns)
) lines
ORDER BY line
"""

#: the schema's tables a row can be deleted from (partitions go with their parent), and each foreign key between them
_TABLES_SQL = """
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind IN ('r', 'p') AND NOT c.relispartition
ORDER BY c.relname
"""
_FOREIGN_KEYS_SQL = """
SELECT child.relname AS child, parent.relname AS parent
FROM pg_constraint k
JOIN pg_class child ON child.oid = k.conrelid
JOIN pg_class parent ON parent.oid = k.confrelid
JOIN pg_namespace n ON n.oid = k.connamespace
WHERE n.nspname = $1 AND k.contype = 'f'
"""
_SEQUENCES_SQL = """
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind = 'S' ORDER BY c.relname
"""


@dataclass
class _Kept:
    """a schema migrated once and handed to every test of its migration set, and what the migrations left in it.

    :ivar schema: the schema's name
    :ivar catalog: its catalog as the migrations made it (:data:`_CATALOG_SQL`)
    :ivar emptied: the tables the migrations left empty, children before the tables they refer to: emptied after
        each test
    :ivar held: each table the migrations wrote rows into -> a digest of those rows, checked after each test
    :ivar sequences: each sequence -> its ``last_value`` and ``is_called`` as the migrations left it, set back after
        each test
    """

    schema: str
    catalog: list[str]
    emptied: list[str] = field(default_factory=list)
    held: dict[str, str] = field(default_factory=dict)
    sequences: dict[str, tuple[int, bool]] = field(default_factory=dict)


#: the kept schema per (database, migration set), on a server that copies no template
_KEPT: dict[tuple[str, str], _Kept] = {}
#: every database a schema was kept in -> the schemas kept there, to drop as the process exits
_KEPT_IN: dict[str, set[str]] = {}


def emptying_order(tables: list[str], references: list[tuple[str, str]]) -> list[str]:
    """``tables`` ordered so each comes before every table it refers to: a delete never trips a foreign key.

    :param tables: the tables
    :ptype tables: list[str]
    :param references: (child, parent) per foreign key; a table referring to itself needs no order
    :ptype references: list[tuple[str, str]]
    :return: the order
    :rtype: list[str]
    :raises ValueError: when the foreign keys make a cycle (no order deletes every table's rows)
    """
    children: dict[str, set[str]] = {table: set() for table in tables}
    for child, parent in references:
        if child != parent and parent in children and child in children:
            children[parent].add(child)
    ordered: list[str] = []
    placed: set[str] = set()
    remaining = list(tables)
    while remaining:
        ready = [table for table in remaining if children[table] <= placed]
        if not ready:
            raise ValueError(f"foreign keys make a cycle among {sorted(remaining)}")
        for table in ready:
            ordered.append(table)
            placed.add(table)
        remaining = [table for table in remaining if table not in placed]
    return ordered


def _quoted(schema: str, name: str) -> str:
    """``"schema"."name"``, both TRUSTED identifiers read from the catalog."""
    return f'"{schema}"."{name}"'


async def _rows_digest(conn: asyncpg.Connection, schema: str, table: str) -> str:
    """a digest of every row ``table`` holds, the same whatever order they are read in."""
    digest = await conn.fetchval(
        f"SELECT md5(coalesce(string_agg(t::text, E'\\n' ORDER BY t::text), '')) FROM {_quoted(schema, table)} t"
    )
    return str(digest)


async def _catalog(conn: asyncpg.Connection, schema: str) -> list[str]:
    """the schema's catalog, one line a thing it holds (:data:`_CATALOG_SQL`)."""
    return [str(row["line"]) for row in await conn.fetch(_CATALOG_SQL, schema)]


async def _keep(url: str, schema: str) -> _Kept:
    """what the migrations left in ``schema``: its catalog, its empty tables in emptying order, a digest of each
    table they wrote rows into, and its sequences' state.

    :param url: the database
    :ptype url: str
    :param schema: the freshly migrated schema
    :ptype schema: str
    :return: what to hand each test back to
    :rtype: _Kept
    """
    conn = await asyncpg.connect(url)
    try:
        kept = _Kept(schema=schema, catalog=await _catalog(conn, schema))
        tables = [str(row["relname"]) for row in await conn.fetch(_TABLES_SQL, schema)]
        references = [(str(row["child"]), str(row["parent"])) for row in await conn.fetch(_FOREIGN_KEYS_SQL, schema)]
        empty: list[str] = []
        for table in tables:
            if await conn.fetchval(f"SELECT EXISTS (SELECT 1 FROM {_quoted(schema, table)})"):
                kept.held[table] = await _rows_digest(conn, schema, table)
            else:
                empty.append(table)
        kept.emptied = emptying_order(empty, references)
        for row in await conn.fetch(_SEQUENCES_SQL, schema):
            name = str(row["relname"])
            state = await conn.fetchrow(f"SELECT last_value, is_called FROM {_quoted(schema, name)}")
            assert state is not None
            kept.sequences[name] = (int(state["last_value"]), bool(state["is_called"]))
    finally:
        await conn.close()
    return kept


async def _reset(url: str, kept: _Kept) -> str | None:
    """hand ``kept``'s schema back as the migrations left it, if only rows changed; else say what did.

    :param url: the database
    :ptype url: str
    :param kept: the schema and what the migrations left in it
    :ptype kept: _Kept
    :return: None when the schema is as the migrations left it again; what differs, when it cannot be
    :rtype: str | None
    """
    conn = await asyncpg.connect(url)
    try:
        catalog = await _catalog(conn, kept.schema)
        if catalog != kept.catalog:
            changed = sorted(set(catalog) ^ set(kept.catalog))
            return f"the catalog changed: {changed[:5]}"
        for table, digest in kept.held.items():
            if await _rows_digest(conn, kept.schema, table) != digest:
                return f"the rows the migrations wrote into {table} changed"
        async with conn.transaction():
            for table in kept.emptied:
                await conn.execute(f"DELETE FROM {_quoted(kept.schema, table)}")
            for name, (last_value, is_called) in kept.sequences.items():
                await conn.execute(
                    "SELECT setval($1::regclass, $2, $3)", _quoted(kept.schema, name), last_value, is_called
                )
    finally:
        await conn.close()
    return None


async def _drop_kept(url: str, schema: str) -> None:
    """drop a kept schema (on a reset that could not be made, or as the process exits)."""
    conn = await asyncpg.connect(url, timeout=5)
    try:
        await drop_schema(conn, schema)
    finally:
        await conn.close()
    _KEPT_IN.get(url, set()).discard(schema)


def _drop_kept_schemas() -> None:
    """drop every schema this process kept, as it exits; best effort.

    :return: None
    :rtype: None
    """
    for url, schemas in list(_KEPT_IN.items()):
        for schema in list(schemas):
            try:
                asyncio.run(_drop_kept(url, schema))
            except OSError, TimeoutError:
                # NOSILENT: the server is gone (a child process or a testcontainer, stopped at the session's end),
                # and its schemas went with it: nothing is left to drop
                pass
            except Exception as exc:  # noqa: BLE001 - contained: the process is exiting, its tests already reported
                contain(exc, log=log, what=f"dropping the kept test schema {schema}")


@contextlib.asynccontextmanager
async def _kept_schema(url: str, key: str, migrate: Migrate) -> AsyncIterator[tuple[str, str]]:
    """migration set ``key``'s schema in ``url``'s database, migrated once a process and handed back as the
    migrations left it after each test; yields ``(url, schema)``. When a test leaves more than rows (or the reset
    fails), the schema is dropped and the next test migrates a fresh one.

    :param url: the database
    :ptype url: str
    :param key: the migration set's name
    :ptype key: str
    :param migrate: applies the migration set
    :ptype migrate: Migrate
    :yield: ``url`` and the migrated schema's name
    :rtype: AsyncIterator[tuple[str, str]]
    """
    kept = _KEPT.get((url, key))
    if kept is None:
        schema = await _migrated_schema(url, migrate)
        if not _KEPT_IN:
            atexit.register(_drop_kept_schemas)
        _KEPT_IN.setdefault(url, set()).add(schema)
        try:
            kept = await _keep(url, schema)
        except BaseException:
            await _drop_kept(url, schema)
            raise
        _KEPT[(url, key)] = kept
    try:
        yield url, kept.schema
    finally:
        _KEPT.pop((url, key), None)
        try:
            refused = await _reset(url, kept)
        except Exception as exc:  # noqa: BLE001 - the schema is dropped below; the next test migrates a fresh one
            refused = f"the reset failed: {exc!r}"
        if refused is None:
            _KEPT[(url, key)] = kept
        else:
            log.info(
                "pg_template: the kept schema is dropped, the next test migrates a fresh one",
                extra={"extra_data": {"schema": kept.schema, "key": key, "why": refused}},
            )
            await _drop_kept(url, kept.schema)


def kept_schema(url: str, key: str) -> str | None:
    """the schema kept for migration set ``key`` in ``url``'s database, if one is kept now.

    :param url: the database
    :ptype url: str
    :param key: the migration set's name
    :ptype key: str
    :return: the schema's name, or None
    :rtype: str | None
    """
    kept = _KEPT.get((url, key))
    return None if kept is None else kept.schema


async def drop_kept(url: str, key: str) -> None:
    """drop the schema kept for migration set ``key`` in ``url``'s database, if any; the next test migrates afresh.

    :param url: the database
    :ptype url: str
    :param key: the migration set's name
    :ptype key: str
    :return: None
    :rtype: None
    """
    kept = _KEPT.pop((url, key), None)
    if kept is not None:
        await _drop_kept(url, kept.schema)


@contextlib.asynccontextmanager
async def migrated_copy(
    url: str, *, key: str, migrate: Migrate, copies_templates: bool | None = None
) -> AsyncIterator[tuple[str, str]]:
    """a database of the test's own, copied from migration set ``key``'s template; yields ``(url, schema)``. On
    YugabyteDB, which copies no template, the set's schema in ``url``'s database, migrated once a process and handed
    back as the migrations left it (:func:`_kept_schema`), instead.

    :param url: a database on the server to use (the session's ``db_container``)
    :ptype url: str
    :param key: the migration set's name, the same for every caller passing the same ``migrate``
    :ptype key: str
    :param migrate: applies the set into the schema of a connection it is given
    :ptype migrate: Migrate
    :param copies_templates: whether the server copies a database from a template; None asks the server (a test of
        the kept schema passes False on PostgreSQL)
    :ptype copies_templates: bool | None
    :yield: the copy's URL and the migrated schema's name
    :rtype: AsyncIterator[tuple[str, str]]
    """
    if not (await _copies_templates(url) if copies_templates is None else copies_templates):
        async with _kept_schema(url, key, migrate) as migrated:
            yield migrated
        return
    template = await _template(url, key, migrate)
    copy = f"t_{uuid.uuid4().hex}"
    await _execute(url, f'CREATE DATABASE "{copy}" TEMPLATE "{template}" STRATEGY FILE_COPY')
    try:
        yield _on_database(url, copy), SCHEMA
    finally:
        await _execute(url, f'DROP DATABASE IF EXISTS "{copy}" WITH (FORCE)')
