"""
unit tests for preview mode and CLI-plumbing extensions on MigrationRunner.

preview mode wraps the DataStore in a :class:`PreviewStore` that
captures every ``execute`` call instead of running it. the runner walks
its normal apply sequence against the wrapper and returns the captured
DDL so operators can review what would happen before committing to the
real apply.

three contracts covered here:

- fresh apply captures expected DDL (every migration body produces one
  or more DDL entries, bookkeeping entries are classified separately)
- already-applied sequence captures empty DDL (runner reads applied
  versions from the underlying store and skips every one)
- partial apply (some applied, some pending) captures only the pending
  DDL, preserving apply order
"""

from __future__ import annotations

import pytest

from threetears.core.data.migrations import (
    CapturedStatement,
    MigrationError,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
    PreviewStore,
)

from ._fake_store import FakeDataStore


async def _create_widgets(store: object) -> None:
    """
    pretend to create the widgets table; records a recognizable DDL stmt.

    :param store: DataStore-like object
    :ptype store: object
    """
    await store.execute("CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)")  # type: ignore[attr-defined]


async def _create_gizmos(store: object) -> None:
    """
    pretend to create a second table; two statements to test ordering.

    :param store: DataStore-like object
    :ptype store: object
    """
    await store.execute("CREATE TABLE IF NOT EXISTS gizmos (id UUID PRIMARY KEY)")  # type: ignore[attr-defined]
    await store.execute("CREATE INDEX IF NOT EXISTS idx_gizmos_id ON gizmos (id)")  # type: ignore[attr-defined]


async def _drop_widgets(store: object) -> None:
    """
    inverse of _create_widgets, used for downgrade tests.

    :param store: DataStore-like object
    :ptype store: object
    """
    await store.execute("DROP TABLE IF EXISTS widgets")  # type: ignore[attr-defined]


async def _drop_gizmos(store: object) -> None:
    """
    inverse of _create_gizmos.

    :param store: DataStore-like object
    :ptype store: object
    """
    await store.execute("DROP INDEX IF EXISTS idx_gizmos_id")  # type: ignore[attr-defined]
    await store.execute("DROP TABLE IF EXISTS gizmos")  # type: ignore[attr-defined]


class TestPreviewFreshApply:
    """preview against an empty DB captures DDL for every pending migration."""

    async def test_preview_captures_all_pending_ddl(self) -> None:
        """every migration body's statements appear in the captured DDL."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        pkg.version(2)(_create_gizmos)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()

        preview = await runner.preview_for_scope(store, MigrationScope.AGENT)
        ddl = preview.captured_ddl()
        assert "CREATE TABLE IF NOT EXISTS widgets" in ddl[0]
        assert "CREATE TABLE IF NOT EXISTS gizmos" in ddl[1]
        assert "CREATE INDEX IF NOT EXISTS idx_gizmos_id" in ddl[2]
        # underlying store is untouched: no bookkeeping row was written
        assert store._migrations_rows == []


class TestPreviewAlreadyApplied:
    """preview against a fully-migrated DB captures empty DDL."""

    async def test_preview_empty_when_everything_applied(self) -> None:
        """after a real apply, preview returns no DDL entries."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        pkg.version(2)(_create_gizmos)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        preview = await runner.preview_for_scope(store, MigrationScope.AGENT)
        assert preview.captured_ddl() == []


class TestPreviewPartialApply:
    """preview skips already-applied versions, captures only the rest."""

    async def test_preview_returns_only_pending(self) -> None:
        """after applying v1 real, preview of v2 returns only v2 DDL."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        # now register v2 and preview
        pkg2 = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg2.version(1)(_create_widgets)
        pkg2.version(2)(_create_gizmos)
        runner2 = MigrationRunner()
        runner2.register(pkg2)

        preview = await runner2.preview_for_scope(store, MigrationScope.AGENT)
        ddl = preview.captured_ddl()
        # v1 is applied; only v2's two statements are captured
        assert len(ddl) == 2
        assert "gizmos" in ddl[0]
        assert "idx_gizmos_id" in ddl[1]


class TestPreviewRecognizesBookkeeping:
    """bookkeeping SQL is classified separately from DDL in the capture."""

    async def test_captured_statements_distinguishes_kinds(self) -> None:
        """the full capture labels each statement as DDL or BOOKKEEPING."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()

        preview = await runner.preview_for_scope(store, MigrationScope.AGENT)
        stmts = preview.captured_statements()
        kinds = {s.kind for s in stmts}
        assert "DDL" in kinds
        assert "BOOKKEEPING" in kinds


class TestTargetVersion:
    """apply with target= stops at the specified version per package."""

    async def test_target_caps_applied_version(self) -> None:
        """apply with target=1 leaves v2 unapplied."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        pkg.version(2)(_create_gizmos)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()

        count = await runner.apply_for_agent_schema(store, target=1)
        assert count == 1
        versions = {row["version"] for row in store._migrations_rows}
        assert versions == {1}


class TestDowngrade:
    """downgrade rolls back the most-recent N applied migrations."""

    async def test_downgrade_one_step_reverts_latest(self) -> None:
        """after applying v1 and v2, downgrade 1 leaves only v1 applied."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        pkg.version(2)(_create_gizmos)
        pkg.downgrade(1)(_drop_widgets)
        pkg.downgrade(2)(_drop_gizmos)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        rolled_back = await runner.downgrade_for_scope(store, MigrationScope.AGENT, steps=1)
        assert rolled_back == 1
        remaining = {row["version"] for row in store._migrations_rows}
        assert remaining == {1}

    async def test_downgrade_refuses_without_registered_down(self) -> None:
        """downgrade raises MigrationError if any targeted migration lacks a down."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        # no pkg.downgrade(1) registered on purpose
        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        with pytest.raises(MigrationError) as exc:
            await runner.downgrade_for_scope(store, MigrationScope.AGENT, steps=1)
        assert "no downgrade registered for demo:1" in str(exc.value)


class TestHistoryAndCurrent:
    """history + current versions read ``_schema_migrations`` faithfully."""

    async def test_history_returns_chronological_list(self) -> None:
        """get_applied_history returns rows ordered by apply time."""
        pkg = PackageMigrations(name="demo", scope=MigrationScope.AGENT)
        pkg.version(1)(_create_widgets)
        pkg.version(2)(_create_gizmos)

        runner = MigrationRunner()
        runner.register(pkg)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        history = await runner.get_applied_history(store)
        assert [row["version"] for row in history] == [1, 2]
        assert history[0]["package"] == "demo"

    async def test_current_versions_returns_max_per_package(self) -> None:
        """current_versions returns package -> max applied version."""
        workspace = PackageMigrations(name="workspace", scope=MigrationScope.AGENT)
        memory = PackageMigrations(name="memory", scope=MigrationScope.AGENT)
        workspace.version(1)(_create_widgets)
        workspace.version(2)(_create_gizmos)
        memory.version(1)(_create_widgets)

        runner = MigrationRunner()
        runner.register(workspace)
        runner.register(memory)
        store = FakeDataStore()
        await runner.apply_for_agent_schema(store)

        current = await runner.current_versions(store, MigrationScope.AGENT)
        assert current == {"workspace": 2, "memory": 1}


class TestStamp:
    """stamp_version writes bookkeeping without running DDL."""

    async def test_stamp_records_version_without_running_body(self) -> None:
        """stamp_version inserts bookkeeping without invoking the body."""
        runner = MigrationRunner()
        store = FakeDataStore()
        await runner.stamp_version(store, "ghost_pkg", 42)
        versions = [(r["package"], r["version"]) for r in store._migrations_rows]
        assert ("ghost_pkg", 42) in versions


class TestPreviewStoreType:
    """the CapturedStatement dataclass has the documented shape."""

    def test_captured_statement_is_dataclass(self) -> None:
        """CapturedStatement exposes sql, params, kind attributes."""
        stmt = CapturedStatement(sql="SELECT 1", params=(), kind="DDL")
        assert stmt.sql == "SELECT 1"
        assert stmt.params == ()
        assert stmt.kind == "DDL"

    def test_preview_store_captured_ddl_filters_bookkeeping(self) -> None:
        """captured_ddl excludes BOOKKEEPING kind entries."""
        ps = PreviewStore(underlying=object())
        ps.captured.append(CapturedStatement(sql="CREATE TABLE foo", params=(), kind="DDL"))
        ps.captured.append(
            CapturedStatement(sql="INSERT INTO _schema_migrations", params=(), kind="BOOKKEEPING")
        )
        ddl = ps.captured_ddl()
        assert ddl == ["CREATE TABLE foo"]
