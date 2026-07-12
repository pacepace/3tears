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

import pytest

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)
from threetears.core.data.migrations.errors import (
    DuplicateVersionError,
    MissingDependencyError,
    MigrationFailedError,
)

from ._fake_store import FakeDataStore


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
    return the advisory lock/unlock executes recorded by the fake store.

    :param store: fake store that recorded the runner's executes
    :ptype store: FakeDataStore
    :return: ordered (sql, params) tuples for pg_advisory_(un)lock calls
    :rtype: list[tuple[str, tuple[object, ...]]]
    """
    return [(sql, params) for sql, params in store.executed if "pg_advisory_" in sql]


class TestAdvisoryLocking:
    """apply runs are gated by a per-schema advisory lock.

    two pods starting concurrently must not both read an empty
    applied-set and double-apply DDL. the runner takes
    ``pg_advisory_lock`` around the whole apply sequence and releases it
    afterwards -- even when a migration body raises.
    """

    async def test_apply_wraps_run_in_advisory_lock(self) -> None:
        """apply_for_agent_schema locks before and unlocks after the DDL."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore(schema="agent_abc")
        await runner.apply_for_agent_schema(store)

        # the very first execute is the lock acquire; the very last is
        # the release. the migration INSERT lands strictly between them.
        first_sql = store.executed[0][0]
        last_sql = store.executed[-1][0]
        assert "pg_advisory_lock" in first_sql
        assert "pg_advisory_unlock" in last_sql

        insert_index = next(
            i for i, (sql, _params) in enumerate(store.executed) if "INSERT INTO _schema_migrations" in sql
        )
        assert 0 < insert_index < len(store.executed) - 1

    async def test_lock_released_when_migration_fails(self) -> None:
        """a raising migration still releases the advisory lock."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)

        async def boom(store: object) -> None:
            """simulate a failing migration body."""
            msg = "boom"
            raise RuntimeError(msg)

        pkg.version(1)(boom)
        runner = MigrationRunner()
        runner.register(pkg)

        store = FakeDataStore(schema="agent_abc")
        with pytest.raises(MigrationFailedError):
            await runner.apply_for_agent_schema(store)

        calls = _lock_calls(store)
        assert any("pg_advisory_lock" in sql for sql, _ in calls)
        assert any("pg_advisory_unlock" in sql for sql, _ in calls)

    async def test_lock_key_differs_per_schema(self) -> None:
        """distinct schemas produce distinct lock keys so they do not serialise."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store_a = FakeDataStore(schema="agent_aaa")
        store_b = FakeDataStore(schema="agent_bbb")
        await runner.apply_for_agent_schema(store_a)
        await runner.apply_for_agent_schema(store_b)

        acquire_a = next(params for sql, params in _lock_calls(store_a) if "pg_advisory_lock" in sql)
        acquire_b = next(params for sql, params in _lock_calls(store_b) if "pg_advisory_lock" in sql)
        # same namespace ($1), different schema key ($2)
        assert acquire_a[0] == acquire_b[0]
        assert acquire_a[1] != acquire_b[1]

    async def test_lock_key_stable_for_same_schema(self) -> None:
        """the same schema yields the same lock key across runs (cross-pod agreement)."""
        pkg = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        pkg.version(1)(_noop)
        runner = MigrationRunner()
        runner.register(pkg)

        store_1 = FakeDataStore(schema="platform")
        store_2 = FakeDataStore(schema="platform")
        await runner.apply_for_agent_schema(store_1)
        await runner.apply_for_agent_schema(store_2)

        key_1 = next(params for sql, params in _lock_calls(store_1) if "pg_advisory_lock" in sql)
        key_2 = next(params for sql, params in _lock_calls(store_2) if "pg_advisory_lock" in sql)
        assert key_1 == key_2


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
