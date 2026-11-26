"""the ring's cutover step on a session that already holds the DDL lock, against a real PostgreSQL.

A boot migration holds the database's DDL lock on its run's one connection. :func:`drop_ring_on` runs on that
session: the session lock is re-entrant there, so the step takes it again at once, drops a ring planted in the
database, and gives back only its own hold. The pool form on a second connection would wait for the run's lock
forever. Both consumer shapes: identity-core's (a ``ConnectionSession`` over its migration connection, holding the
lock) and the hub's (a migration body's store, under the runner's lock).

Each test gets a database of its own: the step sweeps every schema of its database, and the lock is per database.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from threetears.core.data.migrations import (
    DDL_LOCK_NAMESPACE,
    ConnectionSession,
    DdlLockPolicy,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
    database_ddl_lock,
    ddl_lock_key,
)
from threetears.core.data.migrations.ring_cutover import RING_TABLE, RING_TABLES, RingDropped, drop_ring_on
from threetears.core.ddl.keeper_marks import MARKS_TABLE

pytestmark = pytest.mark.integration

#: a step that waits for the lock would wait forever on its own run; the test's bound, far past one that never waits
_NEVER_WAITS_SECONDS: Final = 10.0

#: the platform schema a ring was kept in
_SCHEMA: Final = "platform"


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16 to match the rest of the core integration suite."""
    return "pgvector/pgvector:pg16"


def _with_database(url: str, database: str) -> str:
    """``url`` pointed at another database on the same server.

    :param url: asyncpg-compatible URL
    :ptype url: str
    :param database: database name to substitute
    :ptype database: str
    :return: the rewritten URL
    :rtype: str
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment))


@pytest.fixture
async def ring_db(db_container: str) -> AsyncIterator[str]:
    """a database of its own holding a planted ring, dropped afterwards.

    :param db_container: URL of the shared session container
    :ptype db_container: str
    :return: URL of the new database
    :rtype: AsyncIterator[str]
    """
    name = f"drop_ring_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    url = _with_database(db_container, name)
    conn = await asyncpg.connect(url)
    try:
        await _plant_ring(conn)
    finally:
        await conn.close()
    yield url
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


async def _plant_ring(conn: asyncpg.Connection) -> None:
    """the ring as a database cut over still holds it: its tables, a note function, a noted table's trigger, the
    keeper's marks, and a trigger of the table's own that is not the ring's.

    :param conn: a connection to the database
    :ptype conn: asyncpg.Connection
    """
    await conn.execute(f"CREATE SCHEMA {_SCHEMA}")
    for table in RING_TABLES:
        await conn.execute(f"CREATE TABLE {_SCHEMA}.{table} (id bigint PRIMARY KEY)")
    await conn.execute(f"CREATE TABLE {_SCHEMA}.{MARKS_TABLE} (id bigint PRIMARY KEY)")
    await conn.execute(f"CREATE TABLE {_SCHEMA}.roles (id bigint PRIMARY KEY)")
    for function in ("threetears_note_roles", "own_roles_check"):
        await conn.execute(
            f"CREATE FUNCTION {_SCHEMA}.{function}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$"
        )
    await conn.execute(
        f"CREATE TRIGGER threetears_change_note AFTER INSERT ON {_SCHEMA}.roles "
        f"FOR EACH ROW EXECUTE FUNCTION {_SCHEMA}.threetears_note_roles()"
    )
    await conn.execute(
        f"CREATE TRIGGER roles_own_check BEFORE INSERT ON {_SCHEMA}.roles "
        f"FOR EACH ROW EXECUTE FUNCTION {_SCHEMA}.own_roles_check()"
    )


async def _left(url: str) -> dict[str, Any]:
    """what of the ring and its neighbours the database still holds.

    :param url: database URL
    :ptype url: str
    :return: the ring's tables, the marks table, the triggers on ``roles`` and the schema's functions
    :rtype: dict[str, Any]
    """
    conn = await asyncpg.connect(url)
    try:
        tables = {
            str(row["relname"])
            for row in await conn.fetch(
                "SELECT c.relname::text AS relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind = 'r'",
                _SCHEMA,
            )
        }
        triggers = {
            str(row["tgname"])
            for row in await conn.fetch(
                "SELECT tgname::text AS tgname FROM pg_trigger WHERE tgrelid = to_regclass($1) AND NOT tgisinternal",
                f"{_SCHEMA}.roles",
            )
        }
        functions = {
            str(row["proname"])
            for row in await conn.fetch(
                "SELECT p.proname::text AS proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = $1",
                _SCHEMA,
            )
        }
    finally:
        await conn.close()
    return {"tables": tables, "triggers": triggers, "functions": functions}


async def _lock_is_free(url: str) -> bool:
    """whether a fresh session can take the database's DDL lock at once (given straight back).

    :param url: database URL
    :ptype url: str
    :return: True when the lock was free
    :rtype: bool
    """
    conn = await asyncpg.connect(url)
    try:
        key = ddl_lock_key(await conn.fetchval("SELECT current_database()"))
        free = bool(await conn.fetchval("SELECT pg_try_advisory_lock($1, $2)", DDL_LOCK_NAMESPACE, key))
        if free:
            await conn.fetchval("SELECT pg_advisory_unlock($1, $2)", DDL_LOCK_NAMESPACE, key)
    finally:
        await conn.close()
    return free


def _assert_ring_gone_rest_kept(left: dict[str, Any], dropped: RingDropped) -> None:
    """the ring's tables, trigger and note function are gone; the marks, the table and its own trigger stay.

    :param left: what :func:`_left` read
    :ptype left: dict[str, Any]
    :param dropped: what the step said it dropped
    :ptype dropped: RingDropped
    """
    assert left["tables"] == {MARKS_TABLE, "roles"}
    assert left["triggers"] == {"roles_own_check"}
    assert left["functions"] == {"own_roles_check"}
    assert set(dropped.tables) == {f"{_SCHEMA}.{table}" for table in RING_TABLES}
    assert f"{_SCHEMA}.{RING_TABLE}" in dropped.tables
    assert dropped.triggers == (f"{_SCHEMA}.roles: threetears_change_note",)


async def test_on_the_connection_holding_the_lock_the_ring_goes_at_once(ring_db: str) -> None:
    """identity-core's shape: its migration connection holds the DDL lock, and the step runs on that connection.
    Flips if the step waits for the lock its own session holds, or gives back the caller's hold with its own."""
    conn = await asyncpg.connect(ring_db)
    try:
        session = ConnectionSession(conn)
        async with database_ddl_lock(session):
            dropped = await asyncio.wait_for(drop_ring_on(session), _NEVER_WAITS_SECONDS)
            assert not await _lock_is_free(ring_db), "the caller still holds the lock after the step"
    finally:
        await conn.close()
    _assert_ring_gone_rest_kept(await _left(ring_db), dropped)
    assert await _lock_is_free(ring_db)


async def test_a_migration_body_drops_the_ring_on_the_run_s_store(ring_db: str) -> None:
    """the hub's shape: a boot migration's body hands the step its store, under the runner's lock. Flips if the step
    takes the lock on a connection other than the run's (it would wait on the run forever)."""
    dropped: list[RingDropped] = []

    async def _body(store: Any) -> None:
        """the hub's v193, as it reads once its bridge is gone."""
        dropped.append(await asyncio.wait_for(drop_ring_on(store), _NEVER_WAITS_SECONDS))

    pkg = PackageMigrations(name="drop_ring_it", scope=MigrationScope.PLATFORM)
    pkg.version(1)(_body)
    runner = MigrationRunner(lock_policy=DdlLockPolicy(poll_interval=0.05, max_wait=_NEVER_WAITS_SECONDS))
    runner.register(pkg)
    conn = await asyncpg.connect(ring_db)
    try:
        await conn.execute(f"SET search_path TO {_SCHEMA}")
        applied = await runner.apply_for_platform_schema(ConnectionSession(conn))
    finally:
        await conn.close()
    assert applied == 1
    left = await _left(ring_db)
    left["tables"].discard("_schema_migrations")
    _assert_ring_gone_rest_kept(left, dropped[0])
    assert await _lock_is_free(ring_db)
