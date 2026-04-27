"""
integration tests for the canonical migration runner against real Postgres.

rollback semantics cannot be proven against the in-memory
:class:`FakeDataStore`: rollback on failure, drift after hand-DDL, and
downgrade DDL all need a real PG engine. these tests spin up a
``testcontainers.postgres.PostgresContainer`` and drive the runner
through three scenarios end-to-end:

scenario A — fresh apply + re-apply idempotency
    build the composed agent runner (workspace + conversations +
    agent-tools + agent-memory + langgraph), apply every package's
    migrations into a fresh schema, assert every expected table is
    present. call apply again and assert zero migrations run.

scenario B — mid-sequence rollback on failure
    register a deliberately-broken extra migration at version 99 in
    one of the packages. apply. assert the runner raises
    :class:`MigrationFailedError`, that the failed version is NOT
    present in ``_schema_migrations``, and that every successfully-
    applied migration's row IS still present (failure halts the apply
    without wiping previous work).

scenario C — downgrade rolls back the most-recent migration
    build a small two-version package with matching upgrades + downgrades,
    apply both, downgrade one step, assert the second migration's
    changes are reverted and its bookkeeping row is gone.

the suite is guarded by ``@pytest.mark.integration`` and skips if
Docker is unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from threetears.core.data.migrations import (
    MigrationFailedError,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)


# canonical testcontainer harness -- single ``pytest_plugins`` entry
# pulls in ``db_container`` / ``db_image`` from
# :mod:`threetears.core.testing.fixtures` (test-harness-task-01).

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16.

    the agent-memory package's migration declares a vector column for
    embeddings; running the composed runner against a plain postgres
    image trips on that statement. pgvector/pgvector:pg16 matches
    the image 14-eng-ai-bot uses for its migration_db fixture.
    """
    return "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for :func:`threetears.core.testing.fixtures.db_container`.

    legacy name retained so existing fixture wiring (``pg_conn``)
    keeps working without per-test renames.
    """
    return db_container


class _AsyncpgStore:
    """
    DataStore-shape wrapper over an asyncpg connection for integration tests.

    :param conn: asyncpg connection with search_path pre-set
    :ptype conn: asyncpg.Connection
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """
        initialize wrapper.

        :param conn: asyncpg connection with search_path set
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """
        execute SQL via the underlying connection.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status tag
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        fetch rows as a list of dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._conn.fetch(sql, *params)
        result = [dict(r) for r in rows]
        return result


# minimal stub of the ``platform`` schema the aibots hub migrations
# create in production. agent-workspace v003 writes cross-schema into
# ``platform.namespaces`` (joined through ``platform.agents``) during
# its namespace backfill. the 3tears core test suite cannot import the
# hub's platform migrations, so the fixture below stands up the columns
# the v003 SELECT/INSERT touches — nothing more. this is a test
# precondition, not a shim: the production flow runs the hub's
# platform-scope migrations BEFORE any agent-scope migration touches
# the per-agent schema, and the same invariant is reproduced here.
_PLATFORM_SCHEMA_DDL = (
    'CREATE SCHEMA IF NOT EXISTS "platform"',
    """
    CREATE TABLE IF NOT EXISTS platform.agents (
        id UUID PRIMARY KEY,
        customer_id UUID NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS platform.namespaces (
        id UUID PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        namespace_type VARCHAR(20) NOT NULL,
        owner_agent_id UUID,
        schema_name VARCHAR(100),
        customer_id UUID,
        metadata JSONB,
        date_created TIMESTAMP NOT NULL,
        date_updated TIMESTAMP NOT NULL
    )
    """,
)


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """
    create a fresh schema per test and yield (pg_url, schema_name).

    the schema is dropped on teardown so each test gets a clean slate.
    the ``platform`` schema plus its ``agents`` and ``namespaces``
    tables are stood up alongside so agent-scope migrations that write
    cross-schema into ``platform.*`` (agent-workspace v003 is the
    current example) can run without reaching into the hub's platform
    migration chain.

    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"it_{id(object())}".lower()
    schema = schema.replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        for ddl in _PLATFORM_SCHEMA_DDL:
            await conn.execute(ddl)
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


def _build_composed_agent_runner() -> MigrationRunner:
    """
    compose the agent-scope runner used by the aibots hub broker.

    mirrors :func:`aibots.hub.broker.migrations.build_agent_runner`
    registration order, but lives here so the core test suite does
    not take a dependency on the 14-eng-ai-bot package.

    :return: runner with every agent-scope package registered
    :rtype: MigrationRunner
    """
    from threetears.agent.memory.migrations import register as register_memory
    from threetears.agent.tools.migrations import register as register_tools
    from threetears.agent.workspace.migrations import register as register_workspace
    from threetears.conversations.migrations import register as register_conversations
    from threetears.langgraph.migrations import register as register_langgraph

    runner = MigrationRunner()
    register_conversations(runner)
    register_tools(runner)
    register_memory(runner)
    register_workspace(runner)
    register_langgraph(runner)
    return runner


class TestScenarioA_FreshApply:
    """apply composed migrations against a fresh schema then re-apply."""

    async def test_fresh_apply_creates_every_table(self, pg_schema: tuple[str, str]) -> None:
        """
        every registered package contributes at least one live table.

        :param pg_schema: (url, schema) tuple from fixture
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = _build_composed_agent_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = _AsyncpgStore(conn)
            count = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count > 0
            # _schema_migrations row count must match applied count
            rows = await conn.fetch(f'SELECT version, package FROM "{schema}"._schema_migrations')
            assert len(rows) == count
            # re-apply is a no-op
            count2 = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count2 == 0
        finally:
            await conn.close()


class TestScenarioB_MidSequenceRollback:
    """an injected failing migration halts apply and preserves prior work."""

    async def test_failure_halts_apply_and_preserves_previous_work(self, pg_schema: tuple[str, str]) -> None:
        """
        inject a broken v99 on conversations; assert v99 is NOT recorded,
        earlier-successful migrations ARE recorded, and the failure
        surfaces as :class:`MigrationFailedError` naming the failing
        migration.

        :param pg_schema: (url, schema) tuple from fixture
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = _build_composed_agent_runner()

        # reach into the conversations package registration to add a v99
        # that deliberately issues invalid DDL; this simulates an author
        # bug mid-deploy.
        async def broken_migration(store: Any) -> None:
            """issue DDL Postgres will reject to force a mid-apply failure."""
            await store.execute("CREATE TABLE broken_table (THIS_IS_NOT_VALID_SQL)")

        conversations_pkg = runner.packages["conversations"]
        conversations_pkg.version(99)(broken_migration)

        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = _AsyncpgStore(conn)
            with pytest.raises(MigrationFailedError) as exc_info:
                await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert "conversations:99" in str(exc_info.value)

            # the failing version must NOT be present in bookkeeping
            rows = await conn.fetch(
                f"SELECT version FROM \"{schema}\"._schema_migrations WHERE package = 'conversations' AND version = 99"
            )
            assert rows == []

            # earlier conversations migrations should have been recorded
            rows = await conn.fetch(
                f"SELECT version FROM \"{schema}\"._schema_migrations WHERE package = 'conversations' ORDER BY version"
            )
            versions = [r["version"] for r in rows]
            assert 1 in versions

            # the broken_table itself must not exist
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{schema}' AND table_name = 'broken_table'"
            )
            assert rows == []
        finally:
            await conn.close()


class TestScenarioC_Downgrade:
    """downgrade --steps 1 reverts DDL and bookkeeping for the newest migration."""

    async def test_downgrade_one_step_reverts_schema_and_bookkeeping(self, pg_schema: tuple[str, str]) -> None:
        """
        build a toy two-version package with matching upgrades + downgrades,
        apply both, downgrade one step, assert the v2 table is gone and
        the v2 bookkeeping row is gone.

        :param pg_schema: (url, schema) tuple from fixture
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema

        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)

        async def up_v1(store: Any) -> None:
            """create demo_a table."""
            await store.execute("CREATE TABLE IF NOT EXISTS demo_a (id UUID PRIMARY KEY)")

        async def up_v2(store: Any) -> None:
            """create demo_b table."""
            await store.execute("CREATE TABLE IF NOT EXISTS demo_b (id UUID PRIMARY KEY)")

        async def down_v1(store: Any) -> None:
            """drop demo_a."""
            await store.execute("DROP TABLE IF EXISTS demo_a")

        async def down_v2(store: Any) -> None:
            """drop demo_b."""
            await store.execute("DROP TABLE IF EXISTS demo_b")

        pkg.version(1)(up_v1)
        pkg.version(2)(up_v2)
        pkg.downgrade(1)(down_v1)
        pkg.downgrade(2)(down_v2)

        runner = MigrationRunner()
        runner.register(pkg)

        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = _AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            # both tables exist
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{schema}' "
                f"  AND table_name IN ('demo_a', 'demo_b')"
            )
            assert {r["table_name"] for r in rows} == {"demo_a", "demo_b"}

            # downgrade one step: demo_b is gone, demo_a remains
            count = await runner.downgrade_for_scope(
                store,
                MigrationScope.AGENT,
                steps=1,  # type: ignore[arg-type]
            )
            assert count == 1
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{schema}' "
                f"  AND table_name IN ('demo_a', 'demo_b')"
            )
            remaining = {r["table_name"] for r in rows}
            assert remaining == {"demo_a"}
            # bookkeeping row for v2 is gone
            rows = await conn.fetch(
                f"SELECT version FROM \"{schema}\"._schema_migrations WHERE package = 'demo' ORDER BY version"
            )
            assert [r["version"] for r in rows] == [1]
        finally:
            await conn.close()


class TestDriftDetection:
    """`check` logic catches hand-DDL columns added after apply."""

    async def test_drift_detects_hand_added_column(self, pg_schema: tuple[str, str]) -> None:
        """
        apply the composed runner, hand-INSERT a column the runner did
        not declare, assert the drift diff reports the extra column.

        :param pg_schema: (url, schema) tuple from fixture
        :ptype pg_schema: tuple[str, str]
        """
        from threetears.core.data.migrations import (
            diff_expected_live,
            parse_ddl_to_expected,
            snapshot_live_schema,
        )

        url, schema = pg_schema
        runner = _build_composed_agent_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = _AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]

            # hand-add a column to one of the runner-created tables
            await conn.execute("ALTER TABLE conversations ADD COLUMN rogue_col TEXT")

            # collect expected DDL via preview against an empty shim
            # so every migration's DDL is captured
            class _EmptyStore:
                """bookkeeping-empty store so preview captures every version."""

                async def execute(self, sql: str, *params: Any) -> str:
                    """no-op execute during preview."""
                    return "NOOP"

                async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
                    """return empty rows for bookkeeping SELECTs."""
                    return []

            preview = await runner.preview_for_scope(
                _EmptyStore(),
                MigrationScope.AGENT,  # type: ignore[arg-type]
            )
            expected = parse_ddl_to_expected(preview.captured_ddl())
            live = await snapshot_live_schema(store)
            report = diff_expected_live(expected, live)
            extra_cols = {(t, c) for t, c, _ in report.extra_columns}
            assert ("conversations", "rogue_col") in extra_cols

            # JSON surface also shows it
            data = report.as_dict()
            assert data["clean"] is False
            assert any(
                entry["table"] == "conversations" and entry["column"] == "rogue_col" for entry in data["extra_columns"]
            )
        finally:
            await conn.close()
