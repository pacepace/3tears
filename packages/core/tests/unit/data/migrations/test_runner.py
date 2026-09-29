"""
unit tests for the canonical MigrationRunner.

covers the five contracts the runner must hold:

- registration of versioned async migration callables per package
- topological ordering across packages via ``depends_on`` declarations
- idempotent re-apply (second call applies zero migrations)
- rollback-on-failure (partial success reverts bookkeeping for the failed batch)
- scope separation (platform vs agent) with independent version tracking
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.data.migrations import (
    DDL_LOCK_NAMESPACE,
    DdlLockPolicy,
    DdlLockTimeoutError,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
    ddl_lock_key,
)
from threetears.core.data.migrations.errors import (
    DuplicateVersionError,
    MissingDependencyError,
    MigrationFailedError,
)
from threetears.core.data.schema import ColumnDef, IndexDef, TableDef
from threetears.core.data.store import DataStore

from ._fake_store import FakeDataStore, FakeLockingPool


async def _noop(store: object) -> None:
    """
    no-op migration body for tests that care only about ordering or bookkeeping.

    :param store: DataStore-like object, unused here
    :ptype store: object
    """
    return None


class TestRegistration:
    """package-scoped registration of versioned migration callables."""

    async def test_register_agent_package_assigns_versions(self) -> None:
        """PackageMigrations.version(n) registers callables keyed by version."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        pkg.version(2)(_noop)
        assert set(pkg.versions.keys()) == {1, 2}

    async def test_duplicate_version_raises(self) -> None:
        """registering two migrations at the same version raises DuplicateVersionError."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        with pytest.raises(DuplicateVersionError):
            pkg.version(1)(_noop)


class TestIdempotentApply:
    """second call to apply runs zero migrations."""

    async def test_agent_apply_twice_applies_once(self) -> None:
        """apply_for_agent_schema records v1 once, re-apply is a no-op."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore()
        first = await runner.apply_for_agent_schema(store)
        second = await runner.apply_for_agent_schema(store)

        assert first == 1
        assert second == 0
        assert store.migrations_table_created is True
        assert [row["version"] for row in store.migrations_rows] == [1]

    async def test_platform_apply_twice_applies_once(self) -> None:
        """apply_for_platform_schema records v1 once, re-apply is a no-op."""
        pkg = PackageMigrations(name="hub_platform", scope=MigrationScope.PLATFORM)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore()
        first = await runner.apply_for_platform_schema(store)
        second = await runner.apply_for_platform_schema(store)

        assert first == 1
        assert second == 0


class TestTopologicalOrdering:
    """across-package ordering follows declared depends_on graph."""

    async def test_dependency_precedes_dependent(self) -> None:
        """if memory depends_on workspace, workspace v1 runs before memory v1."""
        workspace = PackageMigrations(name="workspace", scope=MigrationScope.AGENT)
        memory = PackageMigrations(
            name="memory",
            scope=MigrationScope.AGENT,
            depends_on=("workspace",),
        )
        applied_order: list[str] = []

        async def workspace_v1(store: object) -> None:
            """record workspace v1 in the apply sequence."""
            applied_order.append("workspace:1")

        async def memory_v1(store: object) -> None:
            """record memory v1 in the apply sequence."""
            applied_order.append("memory:1")

        workspace.version(1)(workspace_v1)
        memory.version(1)(memory_v1)

        runner = MigrationRunner()
        runner.register(memory)
        runner.register(workspace)

        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)
        assert applied_order == ["workspace:1", "memory:1"]

    async def test_registration_order_does_not_matter(self) -> None:
        """registering memory before workspace still produces correct order."""
        workspace = PackageMigrations(name="workspace", scope=MigrationScope.AGENT)
        memory = PackageMigrations(
            name="memory",
            scope=MigrationScope.AGENT,
            depends_on=("workspace",),
        )
        applied_order: list[str] = []

        async def workspace_v1(store: object) -> None:
            """record workspace v1."""
            applied_order.append("workspace:1")

        async def memory_v1(store: object) -> None:
            """record memory v1."""
            applied_order.append("memory:1")

        workspace.version(1)(workspace_v1)
        memory.version(1)(memory_v1)

        runner = MigrationRunner()
        runner.register(memory)  # registered first, still applied last
        runner.register(workspace)

        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)
        assert applied_order == ["workspace:1", "memory:1"]

    async def test_missing_dependency_raises(self) -> None:
        """declaring depends_on a package that was not registered raises MissingDependencyError."""
        memory = PackageMigrations(
            name="memory",
            scope=MigrationScope.AGENT,
            depends_on=("workspace",),
        )
        memory.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(memory)

        store = FakeDataStore()
        with pytest.raises(MissingDependencyError) as exc_info:
            await runner.apply_for_agent_schema(store)
        assert "workspace" in str(exc_info.value)

    async def test_cycle_detection(self) -> None:
        """a dependency cycle between two packages raises MissingDependencyError."""
        a = PackageMigrations(name="a", scope=MigrationScope.AGENT, depends_on=("b",))
        b = PackageMigrations(name="b", scope=MigrationScope.AGENT, depends_on=("a",))
        a.version(1)(_noop)
        b.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(a)
        runner.register(b)

        store = FakeDataStore()
        with pytest.raises(MissingDependencyError):
            await runner.apply_for_agent_schema(store)


class TestRollbackOnFailure:
    """mid-apply failure records only the migrations that succeeded."""

    async def test_failure_mid_sequence_halts_and_reverts_failing(self) -> None:
        """v2 failure leaves v1 recorded and surfaces MigrationFailedError."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        applied: list[int] = []

        async def v1(store: object) -> None:
            """record successful v1 application."""
            applied.append(1)

        async def v2(store: object) -> None:
            """simulate failure during v2."""
            applied.append(2)
            msg = "simulated v2 failure"
            raise RuntimeError(msg)

        pkg.version(1)(v1)
        pkg.version(2)(v2)

        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore()
        with pytest.raises(MigrationFailedError) as exc_info:
            await runner.apply_for_agent_schema(store)

        assert "memory:2" in str(exc_info.value)
        # v1 executed fully and was recorded; v2 executed but its record was reverted
        assert applied == [1, 2]
        assert [row["version"] for row in store.migrations_rows] == [1]


class TestScopeSeparation:
    """platform and agent scopes track applied versions independently."""

    async def test_agent_apply_ignores_platform_packages(self) -> None:
        """apply_for_agent_schema does not run platform-scoped packages."""
        platform_pkg = PackageMigrations(name="hub_platform", scope=MigrationScope.PLATFORM)
        agent_pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        applied: list[str] = []

        async def platform_v1(store: object) -> None:
            """record platform v1."""
            applied.append("platform:1")

        async def agent_v1(store: object) -> None:
            """record agent v1."""
            applied.append("agent:1")

        platform_pkg.version(1)(platform_v1)
        agent_pkg.version(1)(agent_v1)

        runner = MigrationRunner()
        runner.register(platform_pkg)
        runner.register(agent_pkg)

        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)
        assert applied == ["agent:1"]

    async def test_platform_apply_ignores_agent_packages(self) -> None:
        """apply_for_platform_schema does not run agent-scoped packages."""
        platform_pkg = PackageMigrations(name="hub_platform", scope=MigrationScope.PLATFORM)
        agent_pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        applied: list[str] = []

        async def platform_v1(store: object) -> None:
            """record platform v1."""
            applied.append("platform:1")

        async def agent_v1(store: object) -> None:
            """record agent v1."""
            applied.append("agent:1")

        platform_pkg.version(1)(platform_v1)
        agent_pkg.version(1)(agent_v1)

        runner = MigrationRunner()
        runner.register(platform_pkg)
        runner.register(agent_pkg)

        store = FakeDataStore()
        await runner.apply_for_platform_schema(store)
        assert applied == ["platform:1"]


class TestPackageIsolation:
    """test harness can apply only one package's migrations, not the full set."""

    async def test_apply_package_runs_only_named_package(self) -> None:
        """apply_package applies migrations only for the named package."""
        workspace = PackageMigrations(name="workspace", scope=MigrationScope.AGENT)
        memory = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        applied: list[str] = []

        async def workspace_v1(store: object) -> None:
            """record workspace v1 apply."""
            applied.append("workspace:1")

        async def memory_v1(store: object) -> None:
            """record memory v1 apply."""
            applied.append("memory:1")

        workspace.version(1)(workspace_v1)
        memory.version(1)(memory_v1)

        runner = MigrationRunner()
        runner.register(workspace)
        runner.register(memory)

        store = FakeDataStore()
        count = await runner.apply_package(store, "memory")
        assert count == 1
        assert applied == ["memory:1"]

    async def test_apply_unknown_package_raises(self) -> None:
        """apply_package with unregistered name raises KeyError."""
        runner = MigrationRunner()
        store = FakeDataStore()
        with pytest.raises(KeyError):
            await runner.apply_package(store, "nonexistent")


def _lock_calls(store: FakeDataStore) -> list[tuple[str, tuple[object, ...]]]:
    """
    return the advisory lock/unlock statements the runner issued, in order.

    :param store: fake session that recorded the runner's statements
    :ptype store: FakeDataStore
    :return: ordered (sql, params) tuples naming an advisory-lock function
    :rtype: list[tuple[str, tuple[object, ...]]]
    """
    return [(sql, params) for sql, params in store.queried + store.executed if "advisory" in sql]


def _statement_log(store: FakeDataStore) -> list[str]:
    """
    return every statement the runner issued, queries and executes interleaved in order.

    :param store: fake session that recorded the runner's statements
    :ptype store: FakeDataStore
    :return: SQL text in issue order
    :rtype: list[str]
    """
    return [sql for sql, _params in store.statements]


class TestDatabaseWideDdlLock:
    """every run holds the one DDL lock of its database.

    two DDL jobs in one database -- in the same schema or in two different
    ones -- must not run at once: on YugabyteDB two index builds in one
    database both hang until the catalog-version wait times out. so the lock
    is keyed on ``current_database()``, taken by polling, and held from before
    the bookkeeping table is read until after the last row is written.
    """

    async def test_run_is_bracketed_by_the_lock(self) -> None:
        """the try-lock precedes every other statement and the unlock follows them all."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore(schema="agent_abc")
        await runner.apply_for_agent_schema(store)

        log = _statement_log(store)
        first_lock = next(i for i, sql in enumerate(log) if "pg_try_advisory_lock" in sql)
        unlock = next(i for i, sql in enumerate(log) if "pg_advisory_unlock" in sql)
        work = [i for i, sql in enumerate(log) if "_schema_migrations" in sql]
        assert work
        assert all(first_lock < i < unlock for i in work)
        assert unlock == len(log) - 1
        assert store.lock_holds == 0

    async def test_lock_is_polled_never_blocking(self) -> None:
        """the runner never issues the blocking pg_advisory_lock."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        assert [sql for sql, _ in _lock_calls(store) if "pg_advisory_lock(" in sql] == []

    async def test_two_schemas_of_one_database_share_one_lock(self) -> None:
        """the schema does not enter the key: two agent schemas serialise."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store_a = FakeDataStore(schema="agent_aaa", database="appdb")
        store_b = FakeDataStore(schema="agent_bbb", database="appdb")
        await runner.apply_for_agent_schema(store_a)
        await runner.apply_for_agent_schema(store_b)

        take_a = next(params for sql, params in _lock_calls(store_a) if "pg_try_advisory_lock" in sql)
        take_b = next(params for sql, params in _lock_calls(store_b) if "pg_try_advisory_lock" in sql)
        assert take_a == take_b == (DDL_LOCK_NAMESPACE, ddl_lock_key("appdb"))

    async def test_two_databases_take_different_locks(self) -> None:
        """different databases do not serialise against each other."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store_1 = FakeDataStore(database="db_one")
        store_2 = FakeDataStore(database="db_two")
        await runner.apply_for_agent_schema(store_1)
        await runner.apply_for_agent_schema(store_2)

        take_1 = next(params for sql, params in _lock_calls(store_1) if "pg_try_advisory_lock" in sql)
        take_2 = next(params for sql, params in _lock_calls(store_2) if "pg_try_advisory_lock" in sql)
        assert take_1 != take_2

    async def test_lock_released_when_migration_fails(self) -> None:
        """a raising migration still releases the lock."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)

        async def boom(store: object) -> None:
            """simulate a failing migration body."""
            msg = "boom"
            raise RuntimeError(msg)

        pkg.version(1)(boom)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore()
        with pytest.raises(MigrationFailedError):
            await runner.apply_for_agent_schema(store)

        assert any("pg_advisory_unlock" in sql for sql, _ in _lock_calls(store))
        assert store.lock_holds == 0

    @pytest.mark.parametrize("entry", ["platform", "agent", "package", "downgrade", "stamp"])
    async def test_every_writing_entry_point_takes_the_lock(self, entry: str) -> None:
        """apply (both scopes), apply_package, downgrade and stamp all run under the lock."""
        agent_pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        agent_pkg.version(1)(_noop)
        agent_pkg.downgrade(1)(_noop)
        platform_pkg = PackageMigrations(name="core", scope=MigrationScope.PLATFORM)
        platform_pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(agent_pkg)
        runner.register(platform_pkg)
        store = FakeDataStore()
        if entry == "downgrade":
            await runner.apply_for_agent_schema(store)
            store.queried.clear()
        entry_points = {
            "platform": lambda: runner.apply_for_platform_schema(store),
            "agent": lambda: runner.apply_for_agent_schema(store),
            "package": lambda: runner.apply_package(store, "memory"),
            "downgrade": lambda: runner.downgrade_for_scope(store, MigrationScope.AGENT),
            "stamp": lambda: runner.stamp_version(store, "memory", 7),
        }

        await entry_points[entry]()

        calls = [sql for sql, _ in _lock_calls(store)]
        assert any("pg_try_advisory_lock" in sql for sql in calls)
        assert any("pg_advisory_unlock" in sql for sql in calls)
        assert store.lock_holds == 0

    async def test_max_wait_from_the_runner_policy_applies(self) -> None:
        """a runner built with max_wait gives up with the typed error and runs nothing."""
        ran: list[int] = []

        async def _record(store: object) -> None:
            """record that the body ran."""
            ran.append(1)

        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_record)
        runner = MigrationRunner(lock_policy=DdlLockPolicy(poll_interval=0.001, max_wait=0.02))
        runner.register(pkg)

        store = FakeDataStore(lock_held_elsewhere=True)

        with pytest.raises(DdlLockTimeoutError):
            await runner.apply_for_agent_schema(store)
        assert ran == []
        assert store.migrations_rows == []


def _pool_store(pool: FakeLockingPool) -> DataStore:
    """
    build a pool-backed DataStore over a fake locking pool.

    :param pool: the fake pool
    :ptype pool: FakeLockingPool
    :return: the store
    :rtype: DataStore
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return DataStore(uuid.uuid4(), registry)


def _widgets() -> TableDef:
    """
    a table with one secondary index.

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


class TestPoolBackedDataStore:
    """a pool-backed DataStore is pinned to one connection for the whole run.

    the run's lock is a session lock, so every statement -- the lock, the
    bookkeeping and every body statement, create_table included -- must run
    on the connection that holds it. bodies receive a DataStore bound to it.
    """

    async def test_every_statement_runs_on_one_pinned_connection(self) -> None:
        """nothing reaches the pool directly; one connection runs the lock, the ledger and the body."""
        seen: list[DataStore] = []

        async def _body(store: DataStore) -> None:
            """run a statement and a query through the store the runner hands in."""
            seen.append(store)
            await store.execute("ALTER TABLE t ADD COLUMN IF NOT EXISTS c TEXT")
            await store.query("SELECT 1")

        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_body)
        runner = MigrationRunner()
        runner.register(pkg)
        pool = FakeLockingPool()

        assert await runner.apply_for_agent_schema(_pool_store(pool)) == 1

        used = [c for c in pool.connections if c.statements]
        assert pool.direct == []
        assert len(used) == 1
        statements = used[0].statements
        assert "pg_try_advisory_lock" in statements[1]
        assert "pg_advisory_unlock" in statements[-1]
        assert any("ALTER TABLE t" in sql for sql in statements)
        assert isinstance(seen[0], DataStore)
        assert seen[0].holds_ddl_lock
        assert pool.lock_holder is None

    async def test_create_table_in_a_body_runs_on_the_pinned_connection_without_a_second_lock(self) -> None:
        """a body's create_table uses the run's session and lock, and its collection outlives the run."""

        async def _create(store: DataStore) -> None:
            """create a table with an index, as dipp's v1 does."""
            await store.create_table(_widgets())

        pkg = PackageMigrations(name="dipp", scope=MigrationScope.PLATFORM)
        pkg.version(1)(_create)
        runner = MigrationRunner()
        runner.register(pkg)
        pool = FakeLockingPool()
        store = _pool_store(pool)

        await runner.apply_for_platform_schema(store)

        used = [c for c in pool.connections if c.statements]
        assert len(used) == 1
        statements = used[0].statements
        assert sum("pg_try_advisory_lock" in sql for sql in statements) == 1
        create = next(i for i, sql in enumerate(statements) if "CREATE TABLE IF NOT EXISTS widgets" in sql)
        index = next(i for i, sql in enumerate(statements) if "ix_widgets_label" in sql)
        unlock = next(i for i, sql in enumerate(statements) if "pg_advisory_unlock" in sql)
        assert create < index < unlock
        assert pool.direct == []
        # the collection is registered on the pool-backed store, not on the pinned session
        assert store["widgets"].table_name == "widgets"

    async def test_dipp_shape_apply_then_downgrade_with_a_data_store(self) -> None:
        """apply_for_platform_schema then downgrade_for_scope, both handed the same pool-backed DataStore."""

        async def _up(store: DataStore) -> None:
            """create the table."""
            await store.create_table(_widgets())

        async def _down(store: DataStore) -> None:
            """drop it again."""
            await store.execute("DROP TABLE IF EXISTS widgets")

        pkg = PackageMigrations(name="dipp", scope=MigrationScope.PLATFORM)
        pkg.version(1)(_up)
        pkg.downgrade(1)(_down)
        runner = MigrationRunner()
        runner.register(pkg)
        pool = FakeLockingPool()
        store = _pool_store(pool)

        assert await runner.apply_for_platform_schema(store) == 1
        assert await runner.downgrade_for_scope(store, MigrationScope.PLATFORM) == 1

        assert pool.rows == []
        assert pool.direct == []
        assert pool.lock_holder is None
        dropped_on = [c for c in pool.connections if any("DROP TABLE IF EXISTS widgets" in s for s in c.statements)]
        assert len(dropped_on) == 1
        assert any("pg_try_advisory_lock" in s for s in dropped_on[0].statements)

    async def test_run_migrations_is_the_agent_apply_on_a_pinned_connection(self) -> None:
        """DataStore.run_migrations delegates to the runner, which pins."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)
        pool = FakeLockingPool()

        assert await _pool_store(pool).run_migrations(runner) == 1
        assert pool.direct == []
        assert len([c for c in pool.connections if c.statements]) == 1


class TestCreateTableOutsideAMigration:
    """create_table takes the DDL lock itself, on one connection."""

    async def test_create_table_takes_the_lock_on_the_connection_that_runs_the_ddl(self) -> None:
        """try-lock, CREATE TABLE, CREATE INDEX, unlock -- in that order, on one connection."""
        pool = FakeLockingPool()
        store = _pool_store(pool)

        await store.create_table(_widgets())

        used = [c for c in pool.connections if c.statements]
        assert len(used) == 1
        statements = used[0].statements
        order = [
            next(i for i, sql in enumerate(statements) if marker in sql)
            for marker in ("pg_try_advisory_lock", "CREATE TABLE IF NOT EXISTS widgets", "ix_widgets_label")
        ]
        assert order == sorted(order)
        assert "pg_advisory_unlock" in statements[-1]
        assert pool.direct == []
        assert pool.lock_holder is None

    async def test_create_table_waits_while_another_session_holds_the_lock(self) -> None:
        """no DDL runs until the holder lets go."""
        pool = FakeLockingPool(size=3)
        holder = pool.take()
        pool.lock_holder = holder
        pool.lock_holds = 1
        store = _pool_store(pool)

        task = asyncio.create_task(store.create_table(_widgets()))
        await asyncio.sleep(0.05)
        ran_ddl = any("CREATE TABLE" in sql for c in pool.connections for sql in c.statements)
        pool.lock_holder = None
        pool.lock_holds = 0
        await asyncio.wait_for(task, 10)

        assert not ran_ddl
        assert any("CREATE TABLE IF NOT EXISTS widgets" in sql for c in pool.connections for sql in c.statements)


class TestReadPathsIssueNoDdl:
    """history, current versions and preview never issue DDL and never take the lock.

    a read path only needs to know what is applied; a schema without
    ``_schema_migrations`` has nothing applied, so a missing ledger is read as
    empty rather than created. that keeps a preview runnable against an
    in-memory shim, which is how the aibots hub's ``migrations check`` derives
    its expected DDL.
    """

    @pytest.mark.parametrize("ledger", ["present", "missing"])
    @pytest.mark.parametrize("entry", ["history", "current", "preview"])
    async def test_no_ddl_and_no_lock(self, entry: str, ledger: str) -> None:
        """whether or not the ledger exists, a read path creates nothing and locks nothing."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        store.migrations_table_created = ledger == "present"

        await _read_entry(runner, store, entry)

        assert [sql for sql, _ in store.executed if "CREATE" in sql.upper()] == []
        assert _lock_calls(store) == []

    async def test_a_missing_ledger_reads_as_nothing_applied(self) -> None:
        """history is empty, every package is at 0, and the preview captures every version."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        pkg.version(2)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()

        assert await runner.get_applied_history(store) == []
        assert await runner.current_versions(store, MigrationScope.AGENT) == {"memory": 0}
        preview = await runner.preview_for_scope(store, MigrationScope.AGENT)
        inserted = [s.params[0] for s in preview.captured_statements() if "INSERT INTO _schema_migrations" in s.sql]
        assert inserted == [1, 2]

    async def test_preview_runs_against_a_shim_that_is_not_a_database(self) -> None:
        """a store answering every query with no rows previews without touching a lock."""
        captured: list[str] = []

        async def _create(store: DataStore) -> None:
            """a body that creates a table, which the preview must capture."""
            await store.create_table(_widgets())

        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_create)
        runner = MigrationRunner()
        runner.register(pkg)

        preview = await runner.preview_for_scope(_FakeEmptyShim(captured), MigrationScope.AGENT)

        assert any("CREATE TABLE IF NOT EXISTS widgets" in sql for sql in preview.captured_ddl())
        assert captured == []


# parity-with: threetears.core.data.migrations.session.MigrationSession
class _FakeEmptyShim:
    """answers every query with no rows, as the hub's in-memory bookkeeping shim does."""

    def __init__(self, executed: list[str]) -> None:
        """
        record executes into ``executed``.

        :param executed: where executes are recorded
        :ptype executed: list[str]
        """
        self._executed = executed

    async def execute(self, sql: str, *params: object) -> str:
        """
        record an execute, which a preview must never reach.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: object
        :return: status tag
        :rtype: str
        """
        self._executed.append(sql)
        return "NOOP"

    async def query(self, sql: str, *params: object) -> list[dict[str, object]]:
        """
        answer nothing.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: object
        :return: no rows
        :rtype: list[dict[str, object]]
        """
        return []


async def _read_entry(runner: MigrationRunner, store: FakeDataStore, entry: str) -> None:
    """
    call one of the runner's read-only entry points.

    :param runner: the runner
    :ptype runner: MigrationRunner
    :param store: the fake session
    :ptype store: FakeDataStore
    :param entry: which entry point: history, current or preview
    :ptype entry: str
    """
    if entry == "history":
        await runner.get_applied_history(store)
    elif entry == "current":
        await runner.current_versions(store, MigrationScope.AGENT)
    else:
        await runner.preview_for_scope(store, MigrationScope.AGENT)


class TestPackagesView:
    """``packages`` exposes a read-only view of registered packages.

    the hub CLI iterates this for ``status``/``history``/``current``
    subcommands. read-only protects the runner's registration
    invariants from accidental caller-side mutation.
    """

    async def test_packages_is_empty_on_fresh_runner(self) -> None:
        runner = MigrationRunner()
        assert len(runner.packages) == 0

    async def test_packages_contains_registered_entries(self) -> None:
        runner = MigrationRunner()
        pkg_a = PackageMigrations(name="alpha", scope=MigrationScope.AGENT)
        pkg_b = PackageMigrations(name="beta", scope=MigrationScope.PLATFORM)
        runner.register(pkg_a)
        runner.register(pkg_b)
        view = runner.packages
        assert "alpha" in view
        assert "beta" in view
        assert view["alpha"] is pkg_a
        assert view["beta"] is pkg_b

    async def test_packages_view_rejects_setitem(self) -> None:
        """mutation via the public view is refused so callers cannot
        bypass :meth:`register`."""
        runner = MigrationRunner()
        pkg = PackageMigrations(name="alpha", scope=MigrationScope.AGENT)
        with pytest.raises(TypeError):
            runner.packages["alpha"] = pkg

    async def test_packages_view_rejects_delitem(self) -> None:
        runner = MigrationRunner()
        pkg = PackageMigrations(name="alpha", scope=MigrationScope.AGENT)
        runner.register(pkg)
        with pytest.raises(TypeError):
            del runner.packages["alpha"]

    async def test_packages_view_rejects_clear(self) -> None:
        runner = MigrationRunner()
        runner.register(PackageMigrations(name="alpha", scope=MigrationScope.AGENT))
        with pytest.raises(AttributeError):
            runner.packages.clear()  # type: ignore[attr-defined]

    async def test_packages_view_is_live(self) -> None:
        """a handle taken before registration still sees the new entry
        because MappingProxyType is a window onto the underlying dict,
        not a snapshot copy."""
        runner = MigrationRunner()
        view_before = runner.packages
        runner.register(PackageMigrations(name="alpha", scope=MigrationScope.AGENT))
        assert "alpha" in view_before
