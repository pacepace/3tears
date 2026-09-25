"""
the database-wide DDL lock against a real PostgreSQL.

what only a real engine can show:

- two migration runs against DIFFERENT schemas of ONE database take turns:
  the second body starts only after the first run has released the lock
- a waiter polls instead of sitting in a statement, so it does not deadlock
  against a holder running ``CREATE INDEX CONCURRENTLY`` on a populated table
  -- a waiter blocked in ``pg_advisory_lock`` holds a snapshot the build waits
  for, while it waits for the holder's lock
- ``max_wait`` gives up with the typed error and runs nothing
- a run that fails, or is cancelled mid-statement, leaves the lock free
- a pool-backed ``DataStore`` is pinned to one connection for a run: the
  bodies -- ``create_table`` included -- run on the session holding the lock,
  and dipp's shape (apply then downgrade through one DataStore) works
- ``create_table`` outside a migration waits for the lock
- the premise behind pinning: asyncpg's pool drops a session lock the moment
  the connection is handed back

each test gets its own database, so the lock these tests contend on is not
the one the rest of the suite's migrations take on the shared container.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.data.migrations import (
    DDL_LOCK_NAMESPACE,
    ConnectionSession,
    DdlLockPolicy,
    DdlLockTimeoutError,
    MigrationFailedError,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
    database_ddl_lock,
    ddl_lock_key,
)
from threetears.core.data.schema import ColumnDef, IndexDef, TableDef
from threetears.core.data.store import DataStore

pytestmark = pytest.mark.integration

_POLL = DdlLockPolicy(poll_interval=0.05, log_interval=5.0)

_BUILD_TIMEOUT_SECONDS = 60.0

_POPULATED_ROWS = 50_000

_HOLDS_THE_DDL_LOCK_SQL = (
    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND classid = $1 AND objid = $2 "
    "AND objsubid = 2 AND granted AND pid = pg_backend_pid()"
)


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16 to match the rest of the core integration suite."""
    return "pgvector/pgvector:pg16"


def _with_database(url: str, database: str) -> str:
    """
    return ``url`` pointed at another database on the same server.

    :param url: asyncpg-compatible URL
    :ptype url: str
    :param database: database name to substitute
    :ptype database: str
    :return: the rewritten URL
    :rtype: str
    """
    parts = urlsplit(url)
    result = urlunsplit((parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment))
    return result


@pytest.fixture
async def lock_db(db_container: str) -> AsyncIterator[str]:
    """
    create a database of its own for one test and drop it afterwards.

    :param db_container: URL of the shared session container
    :ptype db_container: str
    :return: URL of the new database
    :rtype: AsyncIterator[str]
    """
    name = f"ddl_lock_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    yield _with_database(db_container, name)
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


async def _connect_to_schema(url: str, schema: str) -> asyncpg.Connection:
    """
    open a connection whose search_path is a fresh schema.

    :param url: database URL
    :ptype url: str
    :param schema: schema to create and bind
    :ptype schema: str
    :return: the bound connection
    :rtype: asyncpg.Connection
    """
    conn: asyncpg.Connection = await asyncpg.connect(url)
    await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}"')
    return conn


def _runner(body: Callable[[Any], Awaitable[None]], *, policy: DdlLockPolicy = _POLL) -> MigrationRunner:
    """
    build a runner with one agent-scope package whose v1 is ``body``.

    :param body: the migration body
    :ptype body: Callable[[Any], Awaitable[None]]
    :param policy: how the run waits for the lock
    :ptype policy: DdlLockPolicy
    :return: the runner
    :rtype: MigrationRunner
    """
    pkg = PackageMigrations(name="ddl_lock_it", scope=MigrationScope.AGENT)
    pkg.version(1)(body)
    runner = MigrationRunner(lock_policy=policy)
    runner.register(pkg)
    return runner


async def _lock_is_free(url: str) -> bool:
    """
    report whether a fresh session can take the database's DDL lock at once.

    takes it and gives it straight back, so the answer leaves nothing held.

    :param url: database URL
    :ptype url: str
    :return: True when the lock was free
    :rtype: bool
    """
    conn = await asyncpg.connect(url)
    try:
        database = await conn.fetchval("SELECT current_database()")
        key = ddl_lock_key(database)
        free = bool(await conn.fetchval("SELECT pg_try_advisory_lock($1, $2)", DDL_LOCK_NAMESPACE, key))
        if free:
            await conn.fetchval("SELECT pg_advisory_unlock($1, $2)", DDL_LOCK_NAMESPACE, key)
    finally:
        await conn.close()
    return free


async def _sample_wait_events(observer: asyncpg.Connection, pid: int, samples: int) -> list[str | None]:
    """
    sample what a backend is waiting on, several times over half a second.

    :param observer: a connection to read pg_stat_activity through
    :ptype observer: asyncpg.Connection
    :param pid: the backend to watch
    :ptype pid: int
    :param samples: how many samples to take
    :ptype samples: int
    :return: ``wait_event_type`` of each sample (None when not waiting)
    :rtype: list[str | None]
    """
    events: list[str | None] = []
    for _ in range(samples):
        events.append(await observer.fetchval("SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1", pid))
        await asyncio.sleep(0.5 / samples)
    return events


async def test_runs_on_two_schemas_of_one_database_take_turns(lock_db: str) -> None:
    """the second run's body starts only after the first run has released the lock."""
    loop = asyncio.get_running_loop()
    first_entered = asyncio.Event()
    first_may_finish = asyncio.Event()
    times: dict[str, float] = {}

    async def _first(store: Any) -> None:
        """hold the lock until told to finish, proving it is held on this session."""
        assert await store.query(_HOLDS_THE_DDL_LOCK_SQL, DDL_LOCK_NAMESPACE, ddl_lock_key(database)) == [{"count": 1}]
        first_entered.set()
        await first_may_finish.wait()

    async def _second(store: Any) -> None:
        """record when the second body started."""
        times["second_started"] = loop.time()

    conn_first = await _connect_to_schema(lock_db, "agent_first")
    conn_second = await _connect_to_schema(lock_db, "agent_second")
    observer = await asyncpg.connect(lock_db)
    database = await observer.fetchval("SELECT current_database()")
    try:
        first = asyncio.create_task(_runner(_first).apply_for_agent_schema(ConnectionSession(conn_first)))
        await asyncio.wait_for(first_entered.wait(), 10)
        second = asyncio.create_task(_runner(_second).apply_for_agent_schema(ConnectionSession(conn_second)))

        waits = await _sample_wait_events(observer, conn_second.get_server_pid(), samples=10)
        assert "second_started" not in times
        # polling: the waiting session is never parked on a lock inside a statement
        assert "Lock" not in waits

        times["first_released"] = loop.time()
        first_may_finish.set()
        assert await asyncio.wait_for(first, 10) == 1
        assert await asyncio.wait_for(second, 10) == 1
    finally:
        await conn_first.close()
        await conn_second.close()
        await observer.close()

    assert times["second_started"] > times["first_released"]
    assert await _lock_is_free(lock_db)


async def test_waiter_does_not_deadlock_a_concurrent_index_build(lock_db: str) -> None:
    """a waiter polling beside a CREATE INDEX CONCURRENTLY lets the build finish."""
    building_may_start = asyncio.Event()
    holder_entered = asyncio.Event()

    async def _build(store: Any) -> None:
        """build an index online on a populated table while holding the lock."""
        holder_entered.set()
        await building_may_start.wait()
        await store.execute("CREATE INDEX CONCURRENTLY idx_populated_label ON populated (label)")

    async def _noop(store: Any) -> None:
        """nothing to do: this run only has to get the lock."""

    conn_holder = await _connect_to_schema(lock_db, "agent_builder")
    conn_waiter = await _connect_to_schema(lock_db, "agent_waiter")
    try:
        await conn_holder.execute("CREATE TABLE populated (id BIGINT PRIMARY KEY, label TEXT NOT NULL)")
        await conn_holder.execute(
            "INSERT INTO populated SELECT g, md5(g::text) FROM generate_series(1, $1) AS g",
            _POPULATED_ROWS,
        )
        holder = asyncio.create_task(_runner(_build).apply_for_agent_schema(ConnectionSession(conn_holder)))
        await asyncio.wait_for(holder_entered.wait(), 10)
        waiter = asyncio.create_task(_runner(_noop).apply_for_agent_schema(ConnectionSession(conn_waiter)))
        # let the waiter start polling before the build begins waiting on snapshots
        await asyncio.sleep(0.5)
        building_may_start.set()

        assert await asyncio.wait_for(holder, _BUILD_TIMEOUT_SECONDS) == 1
        assert await asyncio.wait_for(waiter, _BUILD_TIMEOUT_SECONDS) == 1
        valid = await conn_holder.fetchval(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = 'agent_builder.idx_populated_label'::regclass"
        )
    finally:
        await conn_holder.close()
        await conn_waiter.close()

    assert valid is True
    assert await _lock_is_free(lock_db)


async def test_max_wait_raises_the_typed_error_and_runs_nothing(lock_db: str) -> None:
    """a run that cannot get the lock within max_wait gives up before any body runs."""
    ran: list[str] = []

    async def _record(store: Any) -> None:
        """record that the body ran."""
        ran.append("ran")

    conn_holder = await asyncpg.connect(lock_db)
    conn_waiter = await _connect_to_schema(lock_db, "agent_impatient")
    try:
        async with database_ddl_lock(ConnectionSession(conn_holder), _POLL):
            runner = _runner(_record, policy=DdlLockPolicy(poll_interval=0.05, max_wait=0.5))
            with pytest.raises(DdlLockTimeoutError) as info:
                await runner.apply_for_agent_schema(ConnectionSession(conn_waiter))
        ledger = await conn_waiter.fetchval("SELECT to_regclass('agent_impatient._schema_migrations')")
    finally:
        await conn_holder.close()
        await conn_waiter.close()

    assert ran == []
    assert ledger is None
    assert info.value.waited_seconds >= 0.5
    assert info.value.database.startswith("ddl_lock_")
    assert await _lock_is_free(lock_db)


async def test_failed_run_leaves_the_lock_free(lock_db: str) -> None:
    """a migration body that raises still releases the lock."""

    async def _fail(store: Any) -> None:
        """fail after some DDL has run."""
        await store.execute("CREATE TABLE half_done (id INT)")
        msg = "migration body failed"
        raise RuntimeError(msg)

    conn = await _connect_to_schema(lock_db, "agent_failing")
    try:
        with pytest.raises(MigrationFailedError):
            await _runner(_fail).apply_for_agent_schema(ConnectionSession(conn))
        database = await conn.fetchval("SELECT current_database()")
        still_held_here = await conn.fetchval(_HOLDS_THE_DDL_LOCK_SQL, DDL_LOCK_NAMESPACE, ddl_lock_key(database))
    finally:
        await conn.close()

    assert still_held_here == 0
    assert await _lock_is_free(lock_db)


@pytest.mark.parametrize("where", ["inside a statement", "between statements"])
async def test_cancelled_run_leaves_the_lock_free(lock_db: str, where: str) -> None:
    """a run cancelled mid-statement or between statements still releases the lock."""
    entered = asyncio.Event()

    async def _stall(store: Any) -> None:
        """stall until cancelled, in the database or in the event loop."""
        entered.set()
        if where == "inside a statement":
            await store.execute("SELECT pg_sleep(30)")
        else:
            await asyncio.Event().wait()

    conn = await _connect_to_schema(lock_db, "agent_cancelled")
    try:
        run = asyncio.create_task(_runner(_stall).apply_for_agent_schema(ConnectionSession(conn)))
        await asyncio.wait_for(entered.wait(), 10)
        await asyncio.sleep(0.3)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 10)
        assert await _lock_is_free(lock_db)
        # the session that ran it is still usable and holds nothing
        assert await conn.fetchval("SELECT 1") == 1
    finally:
        await conn.close()


async def test_data_store_run_migrations_holds_the_lock_on_its_bodies_session(lock_db: str) -> None:
    """run_migrations pins one pooled connection: the body sees the lock on its own session."""
    seen: list[int] = []

    async def _check(store: Any) -> None:
        """record how many holds of the DDL lock the body's own session has."""
        rows = await store.query(_HOLDS_THE_DDL_LOCK_SQL, DDL_LOCK_NAMESPACE, ddl_lock_key(database))
        seen.append(int(rows[0]["count"]))
        await store.execute("CREATE TABLE IF NOT EXISTS pinned (id INT)")

    admin = await asyncpg.connect(lock_db)
    database = await admin.fetchval("SELECT current_database()")
    await admin.execute('CREATE SCHEMA "agent_pooled"')
    await admin.close()
    pool = await asyncpg.create_pool(lock_db, min_size=2, max_size=4, server_settings={"search_path": "agent_pooled"})
    try:
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        store = DataStore(uuid.uuid4(), registry)
        applied = await store.run_migrations(_runner(_check))
        created = await pool.fetchval("SELECT to_regclass('agent_pooled.pinned')")
    finally:
        await pool.close()

    assert applied == 1
    assert seen == [1]
    assert created is not None
    assert await _lock_is_free(lock_db)


async def _pooled_store(url: str, schema: str) -> tuple[asyncpg.Pool, DataStore]:
    """
    open a pool bound to a fresh schema and a DataStore over it.

    :param url: database URL
    :ptype url: str
    :param schema: schema to create and bind every pooled connection to
    :ptype schema: str
    :return: the pool (for the caller to close) and the store
    :rtype: tuple[asyncpg.Pool, DataStore]
    """
    admin = await asyncpg.connect(url)
    try:
        await admin.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(url, min_size=2, max_size=4, server_settings={"search_path": schema})
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return pool, DataStore(uuid.uuid4(), registry)


def _widgets() -> TableDef:
    """
    a table with one secondary index, as a create_table body builds it.

    :return: the table definition
    :rtype: TableDef
    """
    return TableDef(
        name="widgets",
        columns=[
            ColumnDef(name="id", column_type="uuid", primary_key=True),
            ColumnDef(name="label", column_type="text", nullable=False),
        ],
        indexes=[IndexDef(name="ix_widgets_label", columns=["label"])],
    )


async def test_a_body_create_table_runs_on_the_pinned_session_that_holds_the_lock(lock_db: str) -> None:
    """a migration body's create_table builds its table and index on the run's own locked session."""
    observed: dict[str, Any] = {}

    async def _create(store: Any) -> None:
        """create a table with an index, as dipp's v1 does, and look at the session doing it."""
        before = (await store.query("SELECT pg_backend_pid() AS pid"))[0]["pid"]
        await store.create_table(_widgets())
        after = (await store.query("SELECT pg_backend_pid() AS pid"))[0]["pid"]
        holds = await store.query(_HOLDS_THE_DDL_LOCK_SQL, DDL_LOCK_NAMESPACE, ddl_lock_key(database))
        observed.update(before=before, after=after, holds=int(holds[0]["count"]))

    admin = await asyncpg.connect(lock_db)
    database = await admin.fetchval("SELECT current_database()")
    await admin.close()
    pool, store = await _pooled_store(lock_db, "agent_created")
    try:
        applied = await _runner(_create).apply_for_agent_schema(store)
        index_valid = await pool.fetchval(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = 'agent_created.ix_widgets_label'::regclass"
        )
    finally:
        await pool.close()

    assert applied == 1
    assert observed["before"] == observed["after"]
    # one hold: create_table ran under the run's lock and did not take it again
    assert observed["holds"] == 1
    assert index_valid is True
    assert store["widgets"].table_name == "widgets"
    assert await _lock_is_free(lock_db)


async def test_dipp_shape_apply_then_downgrade_through_a_data_store(lock_db: str) -> None:
    """apply_for_platform_schema then downgrade_for_scope, each handed the same pool-backed DataStore."""

    async def _up(store: Any) -> None:
        """create the table."""
        await store.create_table(_widgets())

    async def _down(store: Any) -> None:
        """drop it."""
        await store.execute("DROP TABLE IF EXISTS widgets")

    pkg = PackageMigrations(name="dipp_shape", scope=MigrationScope.PLATFORM)
    pkg.version(1)(_up)
    pkg.downgrade(1)(_down)
    runner = MigrationRunner(lock_policy=_POLL)
    runner.register(pkg)
    pool, store = await _pooled_store(lock_db, "platform_dipp")
    try:
        applied = await runner.apply_for_platform_schema(store)
        created = await pool.fetchval("SELECT to_regclass('platform_dipp.widgets')")
        rolled = await runner.downgrade_for_scope(store, MigrationScope.PLATFORM)
        dropped = await pool.fetchval("SELECT to_regclass('platform_dipp.widgets')")
        ledger = await pool.fetchval("SELECT count(*) FROM platform_dipp._schema_migrations")
    finally:
        await pool.close()

    assert (applied, rolled) == (1, 1)
    assert created is not None
    assert dropped is None
    assert ledger == 0
    assert await _lock_is_free(lock_db)


async def test_create_table_outside_a_migration_waits_for_the_lock(lock_db: str) -> None:
    """create_table builds nothing while another session holds the database's DDL lock."""
    holder = await asyncpg.connect(lock_db)
    pool, store = await _pooled_store(lock_db, "app_tables")
    try:
        async with database_ddl_lock(ConnectionSession(holder), _POLL):
            creating = asyncio.create_task(store.create_table(_widgets()))
            await asyncio.sleep(0.5)
            built_while_held = await pool.fetchval("SELECT to_regclass('app_tables.widgets')")
            assert not creating.done()
        await asyncio.wait_for(creating, 30)
        built_after = await pool.fetchval("SELECT to_regclass('app_tables.widgets')")
    finally:
        await holder.close()
        await pool.close()

    assert built_while_held is None
    assert built_after is not None
    assert await _lock_is_free(lock_db)


async def test_premise_an_asyncpg_pool_drops_a_session_lock_on_release(lock_db: str) -> None:
    """a lock taken through Pool.execute is gone once the statement returns -- why a DataStore is pinned."""
    pool = await asyncpg.create_pool(lock_db, min_size=1, max_size=2)
    try:
        database = await pool.fetchval("SELECT current_database()")
        await pool.execute("SELECT pg_advisory_lock($1, $2)", DDL_LOCK_NAMESPACE, ddl_lock_key(database))
        free_afterwards = await _lock_is_free(lock_db)
    finally:
        await pool.close()

    assert free_afterwards
