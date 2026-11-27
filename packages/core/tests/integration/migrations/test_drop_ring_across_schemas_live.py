"""the ring's cutover step on a database from before 0.95: the ring in the platform schema, noted tables in agent schemas.

Hub v193 failed on a real database carrying data from before 0.95 (2026-11-27, the local stack): a noted table's note
function lives in the ring's schema (the platform schema), its three change-note triggers on the table, in the table's
own schema (an agent's ``ns_*``). Dropped a schema at a time, the platform schema's functions went before the agent
schema's triggers, and PostgreSQL refused: ``cannot drop function threetears_note_... because other objects depend on
it``. :func:`drop_ring_on` now drops every ring trigger of the database before any note function.

Each statement of the step commits on its own (the migration run holds no transaction around a body, on either engine),
so a run cut part way leaves what it dropped dropped; the rerun reads what is left and finishes.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
Each test gets a database of its own: the step sweeps every schema of its database.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from threetears.core.data.migrations import (
    ConnectionSession,
    DdlLockPolicy,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
    database_ddl_lock,
)
from threetears.core.data.migrations.ring_cutover import RING_TABLE, RING_TABLES, RingDropped, drop_ring_on
from threetears.core.ddl.keeper_marks import MARKS_TABLE

pytestmark = pytest.mark.integration

#: a step that waits for the lock would wait forever on its own run; the test's bound, far past one that never waits
_NEVER_WAITS_SECONDS: Final = 60.0

#: the ring's schema: it sorts before ``ns_*``, as the hub's platform schema does, so a schema-at-a-time drop reaches
#: the ring's functions before the agent schema's triggers
_PLATFORM: Final = "hub_platform"

#: an agent's schema, named as the hub names one, holding a noted product table
_AGENT: Final = "ns_ea0c2c5cf03e52e291fab3fe451de0ed"

#: the noted tables: the platform schema's own and the agent's product table
_NOTED: Final = ((_PLATFORM, "roles"), (_AGENT, "result_snapshots_by_id"))

#: the ring's own functions, in the ring's schema, as the ring's DDL created them
_RING_FUNCTIONS: Final = (
    "threetears_change_note_put",
    "threetears_change_note_whole",
    "threetears_change_note_buffer_open",
    "threetears_change_note_flush",
)

#: the rows the product table carries, kept whole through the cutover
_ROWS: Final = 25


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


def _note_function(schema: str, table: str) -> str:
    """a noted table's function name, as the ring's DDL named it: the prefix and a digest of its schema and table.

    :param schema: the table's schema
    :ptype schema: str
    :param table: the table
    :ptype table: str
    :return: the function's name
    :rtype: str
    """
    return "threetears_note_" + hashlib.sha256(f"{schema}.{table}".encode()).hexdigest()[:24]


@pytest.fixture
async def ring_era_db(db_container: str) -> AsyncIterator[str]:
    """a database of its own holding a planted ring-era layout, dropped afterwards.

    :param db_container: URL of the shared session container
    :ptype db_container: str
    :return: URL of the new database
    :rtype: AsyncIterator[str]
    """
    name = f"ring_era_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    url = _with_database(db_container, name)
    conn = await asyncpg.connect(url)
    try:
        await _plant_ring_era(conn)
    finally:
        await conn.close()
    yield url
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        await admin.close()


async def _plant_ring_era(conn: asyncpg.Connection) -> None:
    """the layout a database from before 0.95 holds: the ring's tables, marks and functions in the platform schema; a
    note function per noted table there too; each noted table's three triggers on the table, in its own schema; and the
    agent table's own trigger, which is not the ring's.

    :param conn: a connection to the database
    :ptype conn: asyncpg.Connection
    """
    for schema in (_PLATFORM, _AGENT):
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    for table in (*RING_TABLES, MARKS_TABLE):
        await conn.execute(f'CREATE TABLE "{_PLATFORM}"."{table}" (id bigint PRIMARY KEY)')
    for function in _RING_FUNCTIONS:
        await conn.execute(
            f'CREATE FUNCTION "{_PLATFORM}"."{function}"() RETURNS void LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$'
        )
    for schema, table in _NOTED:
        target = f'"{schema}"."{table}"'
        function = f'"{_PLATFORM}"."{_note_function(schema, table)}"'
        await conn.execute(f"CREATE TABLE {target} (id bigint PRIMARY KEY, body text NOT NULL)")
        await conn.execute(
            f"CREATE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$"
        )
        whole = "current_setting('threetears.note_whole', true)"
        await conn.execute(
            f"CREATE TRIGGER threetears_change_note AFTER INSERT OR UPDATE OR DELETE ON {target} "
            f"FOR EACH ROW WHEN ({whole} IS DISTINCT FROM '1') EXECUTE FUNCTION {function}()"
        )
        await conn.execute(
            f"CREATE TRIGGER threetears_change_note_truncate_table AFTER TRUNCATE ON {target} "
            f"FOR EACH STATEMENT EXECUTE FUNCTION {function}()"
        )
        await conn.execute(
            f"CREATE TRIGGER threetears_change_note_whole_write AFTER INSERT OR UPDATE OR DELETE ON {target} "
            f"FOR EACH STATEMENT WHEN ({whole} = '1') EXECUTE FUNCTION {function}()"
        )
    agent_table = f'"{_AGENT}"."result_snapshots_by_id"'
    await conn.execute(
        f'CREATE FUNCTION "{_AGENT}".own_body_check() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$'
    )
    await conn.execute(
        f'CREATE TRIGGER own_body_check BEFORE INSERT ON {agent_table} FOR EACH ROW EXECUTE FUNCTION "{_AGENT}".own_body_check()'
    )
    await conn.execute(
        f"INSERT INTO {agent_table} (id, body) SELECT g, 'snapshot ' || g FROM generate_series(1, {_ROWS}) g"
    )


async def _left(url: str) -> dict[str, Any]:
    """what of the ring and its neighbours the database still holds, in every schema.

    :param url: database URL
    :ptype url: str
    :return: ring functions and triggers left anywhere, the triggers and rows of the agent's table, the platform tables
    :rtype: dict[str, Any]
    """
    conn = await asyncpg.connect(url)
    try:
        functions = {
            f"{row['nspname']}.{row['proname']}"
            for row in await conn.fetch(
                "SELECT n.nspname::text AS nspname, p.proname::text AS proname FROM pg_proc p "
                "JOIN pg_namespace n ON n.oid = p.pronamespace "
                r"WHERE p.proname::text LIKE 'threetears\_%'"
            )
        }
        ring_triggers = {
            f"{row['nspname']}.{row['relname']}: {row['tgname']}"
            for row in await conn.fetch(
                "SELECT n.nspname::text AS nspname, c.relname::text AS relname, g.tgname::text AS tgname "
                "FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                r"WHERE NOT g.tgisinternal AND g.tgname::text LIKE 'threetears\_%'"
            )
        }
        agent_triggers = {
            str(row["tgname"])
            for row in await conn.fetch(
                "SELECT tgname::text AS tgname FROM pg_trigger WHERE tgrelid = to_regclass($1) AND NOT tgisinternal",
                f"{_AGENT}.result_snapshots_by_id",
            )
        }
        platform_tables = {
            str(row["relname"])
            for row in await conn.fetch(
                "SELECT c.relname::text AS relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind = 'r'",
                _PLATFORM,
            )
        }
        rows = await conn.fetch(f'SELECT id, body FROM "{_AGENT}"."result_snapshots_by_id" ORDER BY id')
    finally:
        await conn.close()
    return {
        "functions": functions,
        "ring_triggers": ring_triggers,
        "agent_triggers": agent_triggers,
        "platform_tables": platform_tables,
        "rows": [(int(row["id"]), str(row["body"])) for row in rows],
    }


def _assert_ring_gone_rest_kept(left: dict[str, Any]) -> None:
    """no ring function or trigger anywhere; the marks, the noted tables, the agent table's own trigger and its rows stay.

    :param left: what :func:`_left` read
    :ptype left: dict[str, Any]
    """
    assert left["functions"] == set()
    assert left["ring_triggers"] == set()
    assert left["agent_triggers"] == {"own_body_check"}
    assert left["platform_tables"] == {MARKS_TABLE, "roles"}
    assert left["rows"] == [(index, f"snapshot {index}") for index in range(1, _ROWS + 1)]


async def test_v193_takes_down_a_ring_whose_triggers_live_in_agent_schemas(ring_era_db: str) -> None:
    """the hub's v193 shape, a migration body handed the run's store, on the layout that failed it. Flips if a note
    function is dropped while an agent schema's trigger still calls it (the run fails with ``cannot drop function ...
    because other objects depend on it``), if any ring function or trigger is left in any schema, or if a product
    table's rows or its own trigger are touched."""
    dropped: list[RingDropped] = []

    async def _body(store: Any) -> None:
        """the hub's v193."""
        dropped.append(
            await asyncio.wait_for(drop_ring_on(store, policy=DdlLockPolicy(max_wait=0)), _NEVER_WAITS_SECONDS)
        )

    pkg = PackageMigrations(name="drop_ring_era_it", scope=MigrationScope.PLATFORM)
    pkg.version(1)(_body)
    runner = MigrationRunner(lock_policy=DdlLockPolicy(poll_interval=0.05, max_wait=_NEVER_WAITS_SECONDS))
    runner.register(pkg)
    conn = await asyncpg.connect(ring_era_db)
    try:
        await conn.execute(f'SET search_path TO "{_PLATFORM}"')
        applied = await runner.apply_for_platform_schema(ConnectionSession(conn))
    finally:
        await conn.close()
    assert applied == 1
    left = await _left(ring_era_db)
    left["platform_tables"].discard("_schema_migrations")
    _assert_ring_gone_rest_kept(left)
    (answer,) = dropped
    assert set(answer.triggers) == {
        f"{schema}.{table}: {trigger}"
        for schema, table in _NOTED
        for trigger in (
            "threetears_change_note",
            "threetears_change_note_truncate_table",
            "threetears_change_note_whole_write",
        )
    }
    assert len(answer.functions) == len(_RING_FUNCTIONS) + len(_NOTED)
    assert set(answer.tables) == {f"{_PLATFORM}.{table}" for table in RING_TABLES}
    assert f"{_PLATFORM}.{RING_TABLE}" in answer.tables


class _CutSession(ConnectionSession):
    """a session whose statements stop after ``cut`` of them: a run that fails part way, every statement before the
    cut committed (as the migration run commits each one).

    :param conn: the connection
    :param cut: the statements that run before the failure
    """

    def __init__(self, conn: asyncpg.Connection, cut: int) -> None:
        super().__init__(conn)
        self._left = cut

    async def execute(self, sql: str, *params: Any) -> str:
        if sql.startswith("DROP ") and self._left == 0:
            raise RuntimeError("the run failed part way")
        if sql.startswith("DROP "):
            self._left -= 1
        return await super().execute(sql, *params)


#: the cutover's drops on the planted layout: three triggers a noted table, then each function, then one table drop
_TRIGGER_DROPS: Final = 3 * len(_NOTED)
_FUNCTION_DROPS: Final = len(_RING_FUNCTIONS) + len(_NOTED)


@pytest.mark.parametrize(
    "cut",
    [
        pytest.param(_TRIGGER_DROPS // 2, id="among-the-triggers"),
        pytest.param(_TRIGGER_DROPS + _FUNCTION_DROPS // 2, id="among-the-functions"),
        pytest.param(_TRIGGER_DROPS + _FUNCTION_DROPS, id="before-the-tables"),
    ],
)
async def test_a_rerun_after_a_run_failed_part_way_finishes_the_job(ring_era_db: str, cut: int) -> None:
    """a run cut part way (its earlier drops committed, as each statement commits on its own) and then run again.
    Flips if the rerun trips on what the first run left (a trigger gone but its function kept, a function gone but
    its schema no longer holding a ring trigger), or leaves any ring function, trigger or table behind."""
    conn = await asyncpg.connect(ring_era_db)
    try:
        cut_session = _CutSession(conn, cut)
        async with database_ddl_lock(cut_session):
            with pytest.raises(RuntimeError, match="part way"):
                await asyncio.wait_for(drop_ring_on(cut_session), _NEVER_WAITS_SECONDS)
        session = ConnectionSession(conn)
        async with database_ddl_lock(session):
            rerun = await asyncio.wait_for(drop_ring_on(session), _NEVER_WAITS_SECONDS)
            again = await asyncio.wait_for(drop_ring_on(session), _NEVER_WAITS_SECONDS)
    finally:
        await conn.close()
    _assert_ring_gone_rest_kept(await _left(ring_era_db))
    assert len(rerun.triggers) + len(rerun.functions) == max(0, _TRIGGER_DROPS + _FUNCTION_DROPS - cut)
    assert set(rerun.tables) == {f"{_PLATFORM}.{table}" for table in RING_TABLES}
    assert again.triggers == again.functions == again.tables == ()
