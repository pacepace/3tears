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
        assert store._migrations_table_created is True
        assert [row["version"] for row in store._migrations_rows] == [1]

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
        assert [row["version"] for row in store._migrations_rows] == [1]


class TestScopeSeparation:
    """platform and agent scopes track applied versions independently."""

    async def test_agent_apply_ignores_platform_packages(self) -> None:
        """apply_for_agent_schema does not run platform-scoped packages."""
        platform_pkg = PackageMigrations(
            name="hub_platform", scope=MigrationScope.PLATFORM
        )
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
        platform_pkg = PackageMigrations(
            name="hub_platform", scope=MigrationScope.PLATFORM
        )
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
