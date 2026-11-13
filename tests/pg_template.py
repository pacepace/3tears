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
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import asyncpg

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


@contextlib.asynccontextmanager
async def migrated_copy(url: str, *, key: str, migrate: Migrate) -> AsyncIterator[tuple[str, str]]:
    """a database of the test's own, copied from migration set ``key``'s template; yields ``(url, schema)``.

    :param url: a database on the server to use (the session's ``db_container``)
    :ptype url: str
    :param key: the migration set's name, the same for every caller passing the same ``migrate``
    :ptype key: str
    :param migrate: applies the set into the schema of a connection it is given
    :ptype migrate: Migrate
    :yield: the copy's URL and the migrated schema's name
    :rtype: AsyncIterator[tuple[str, str]]
    """
    template = await _template(url, key, migrate)
    copy = f"t_{uuid.uuid4().hex}"
    await _execute(url, f'CREATE DATABASE "{copy}" TEMPLATE "{template}" STRATEGY FILE_COPY')
    try:
        yield _on_database(url, copy), SCHEMA
    finally:
        await _execute(url, f'DROP DATABASE IF EXISTS "{copy}" WITH (FORCE)')
