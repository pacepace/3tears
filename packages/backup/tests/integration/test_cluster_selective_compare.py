"""Integration: cluster sets, selective restore, and the drift drill — against real PostgreSQL.

The three new capabilities proven end to end with the real dump tools, the real encrypted store,
and a real database: a cluster backup set covering databases nobody listed (with globals and a
per-table inventory in the manifest), rows brought back selectively by id / uuid7 range / date
range through the plan-apply split, and the drift comparator telling apart "the world moved on"
(tolerated, reported) from "the backup does not match" (failed).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid7

import asyncpg
import pytest
from pydantic import SecretStr
from threetears.core.testing.containers import check_docker_available

from threetears.backup.cluster import ClusterBackup, replace_database
from threetears.backup.compare import DriftComparator
from threetears.backup.config import BackupConfig
from threetears.backup.manifest import BackupManifest
from threetears.backup.selective import (
    RowSelection,
    SelectionTooLargeError,
    SelectiveRestore,
)
from threetears.object_store.filesystem import FilesystemObjectStore

_TOOLS = ("pg_dump", "pg_restore", "pg_dumpall", "psql")
_TOOLS_PRESENT = all(shutil.which(tool) for tool in _TOOLS)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TOOLS_PRESENT, reason="postgres client tools not on PATH"),
]

#: low scrypt work factor: these tests exercise plumbing, not the KDF's memory-hardness.
_TEST_WORK_FACTOR = 2**14


def _client_tool_major() -> int | None:
    out = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True, check=False)  # noqa: S603, S607
    match = re.search(r"(\d+)\.", out.stdout)
    return int(match.group(1)) if match else None


@pytest.fixture(scope="module")
def cluster_dsn() -> Any:
    """A module-private postgres container, client-tool matched, seeded with TWO databases.

    ``alpha`` carries a uuid7-keyed table with a ``date_created`` column (the platform shape);
    ``beta`` exists to prove enumeration finds databases nobody passed in. A ``drill_reader``
    role proves the globals dump captures cluster-level objects.
    """
    if not check_docker_available():
        pytest.skip("docker unavailable")
    major = _client_tool_major()
    if major is None:
        pytest.skip("cannot determine pg client version")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(f"postgres:{major}") as container:
        url = container.get_connection_url()
        if url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
        yield url


@pytest.fixture(scope="module")
def seeded(cluster_dsn: str) -> Any:
    """Seed the cluster; returns (admin_dsn, alpha_dsn, row ids oldest-first, seed moment)."""
    import asyncio

    ids: list[UUID] = []

    async def _seed() -> None:
        admin = await asyncpg.connect(cluster_dsn)
        try:
            await admin.execute("CREATE DATABASE alpha")
            await admin.execute("CREATE DATABASE beta")
            await admin.execute("CREATE ROLE drill_reader NOLOGIN")
        finally:
            await admin.close()
        alpha = await asyncpg.connect(replace_database(cluster_dsn, "alpha"))
        try:
            await alpha.execute(
                """
                CREATE TABLE things (
                    id uuid PRIMARY KEY,
                    name text NOT NULL,
                    date_created timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            await alpha.execute("CREATE TABLE keyed_only (id uuid PRIMARY KEY, payload text)")
            for i in range(20):
                row_id = uuid7()
                ids.append(row_id)
                await alpha.execute("INSERT INTO things (id, name) VALUES ($1, $2)", row_id, f"thing-{i}")
                await alpha.execute("INSERT INTO keyed_only (id, payload) VALUES ($1, $2)", row_id, f"payload-{i}")
        finally:
            await alpha.close()
        beta = await asyncpg.connect(replace_database(cluster_dsn, "beta"))
        try:
            await beta.execute("CREATE TABLE beta_rows (id uuid PRIMARY KEY)")
            await beta.execute("INSERT INTO beta_rows (id) VALUES ($1)", uuid7())
        finally:
            await beta.close()

    asyncio.run(_seed())
    return cluster_dsn, replace_database(cluster_dsn, "alpha"), ids, datetime.now(UTC)


@pytest.fixture(scope="module")
def backup_set(seeded: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One cluster backup of the seeded state; returns (cluster, manifest, admin_dsn, alpha_dsn, ids, store_root)."""
    import asyncio

    admin_dsn, alpha_dsn, ids, _ = seeded
    root: Path = tmp_path_factory.mktemp("backup-store")
    config = BackupConfig(passphrase=SecretStr("drill-passphrase"), encryption_work_factor=_TEST_WORK_FACTOR)
    cluster = ClusterBackup(config, FilesystemObjectStore(root), asyncpg.connect)
    manifest = asyncio.run(cluster.create_backup(admin_dsn))
    return cluster, manifest, admin_dsn, alpha_dsn, ids, root


async def _scratch_restore(cluster: ClusterBackup, manifest: BackupManifest, admin_dsn: str, database: str) -> str:
    """Restore one database of the set into a fresh scratch db; returns its dsn."""
    scratch_name = f"scratch_{uuid7().hex[:12]}"
    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{scratch_name}"')
    finally:
        await admin.close()
    scratch_dsn = replace_database(admin_dsn, scratch_name)
    await cluster.restore_database(manifest, database, scratch_dsn)
    return scratch_dsn


class TestClusterBackupSet:
    async def test_enumeration_covers_databases_nobody_listed(self, backup_set: Any) -> None:
        _, manifest, *_ = backup_set
        names = {dump.database for dump in manifest.databases}
        assert {"alpha", "beta"} <= names

    async def test_the_manifest_inventory_carries_exact_row_counts(self, backup_set: Any) -> None:
        _, manifest, *_ = backup_set
        alpha = next(d for d in manifest.databases if d.database == "alpha")
        counts = {(t.schema, t.table): t.row_count for t in alpha.tables}
        assert counts[("public", "things")] == 20
        assert counts[("public", "keyed_only")] == 20
        assert all(dump.sha256 for dump in manifest.databases)

    async def test_the_globals_dump_captures_cluster_roles(self, backup_set: Any) -> None:
        cluster, manifest, *_ = backup_set
        assert manifest.globals_key is not None
        from threetears.backup.gzip import gunzip_stream  # noqa: PLC0415

        store = cluster._store  # noqa: SLF001 -- decrypt through the engine's own wrapper, as a restore would
        chunks = [chunk async for chunk in gunzip_stream(store.open_read(manifest.globals_key))]
        assert b"drill_reader" in b"".join(chunks)

    async def test_a_manifest_is_addressable_by_its_stable_id(self, backup_set: Any) -> None:
        cluster, manifest, *_ = backup_set
        fetched = await cluster.get_manifest(manifest.backup_id)
        assert fetched == manifest
        listed = await cluster.list_manifests()
        assert manifest.backup_id in {m.backup_id for m in listed}

    async def test_restore_brings_every_row_back(self, backup_set: Any) -> None:
        cluster, manifest, admin_dsn, *_ = backup_set
        scratch_dsn = await _scratch_restore(cluster, manifest, admin_dsn, "alpha")
        conn = await asyncpg.connect(scratch_dsn)
        try:
            assert await conn.fetchval("SELECT count(*) FROM things") == 20
        finally:
            await conn.close()


class TestSelectiveRestore:
    @pytest.fixture()
    async def drifted(self, backup_set: Any) -> Any:
        """A scratch restore plus a mutated live alpha: one row updated, one deleted, one added."""
        cluster, manifest, admin_dsn, alpha_dsn, ids, _ = backup_set
        scratch_dsn = await _scratch_restore(cluster, manifest, admin_dsn, "alpha")
        live = await asyncpg.connect(alpha_dsn)
        try:
            await live.execute("UPDATE things SET name = 'mangled' WHERE id = $1", ids[0])
            await live.execute("DELETE FROM things WHERE id = $1", ids[1])
            await live.execute("INSERT INTO things (id, name) VALUES ($1, 'newborn')", uuid7())
        finally:
            await live.close()
        yield scratch_dsn, alpha_dsn, ids
        # selective tests repair what they broke via apply(); nothing to tear down.

    async def test_one_row_comes_back_by_id(self, drifted: Any) -> None:
        scratch_dsn, alpha_dsn, ids = drifted
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn)
        plan = await restore.plan(RowSelection(table="things", ids=(ids[0],)))
        assert len(plan.updates) == 1 and plan.updates[0].row["name"] == "thing-0"
        assert await restore.apply(plan) == 1
        live = await asyncpg.connect(alpha_dsn)
        try:
            assert await live.fetchval("SELECT name FROM things WHERE id = $1", ids[0]) == "thing-0"
        finally:
            await live.close()

    async def test_an_id_range_resurrects_a_deleted_row(self, drifted: Any) -> None:
        scratch_dsn, alpha_dsn, ids = drifted
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn)
        plan = await restore.plan(RowSelection(table="things", id_range=(ids[1], ids[3])))
        assert {p.action for p in plan.inserts} == {"insert"}
        assert len(plan.inserts) == 1  # the deleted row; its neighbours are identical or updated
        await restore.apply(plan)
        live = await asyncpg.connect(alpha_dsn)
        try:
            assert await live.fetchval("SELECT count(*) FROM things WHERE id = $1", ids[1]) == 1
        finally:
            await live.close()

    async def test_a_date_range_selects_by_time(self, drifted: Any) -> None:
        scratch_dsn, alpha_dsn, _ = drifted
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn)
        window = (
            datetime.now(UTC) - timedelta(hours=1),
            datetime.now(UTC) + timedelta(hours=1),
        )
        plan = await restore.plan(RowSelection(table="things", date_range=("date_created", window[0], window[1])))
        assert len(plan.inserts) + len(plan.updates) + len(plan.identical) == 20

    async def test_a_raw_where_predicate_selects_what_no_vocabulary_fits(self, drifted: Any) -> None:
        scratch_dsn, alpha_dsn, ids = drifted
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn)
        plan = await restore.plan(
            RowSelection(table="things", where="name = $1", where_params=("thing-0",))
        )
        assert plan.write_count == 1  # the mangled row, found by CONTENT rather than by id
        await restore.apply(plan)
        live = await asyncpg.connect(alpha_dsn)
        try:
            assert await live.fetchval("SELECT name FROM things WHERE id = $1", ids[0]) == "thing-0"
        finally:
            await live.close()

    async def test_a_fat_fingered_selection_is_refused(self, drifted: Any) -> None:
        scratch_dsn, alpha_dsn, _ = drifted
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn, max_rows=3)
        with pytest.raises(SelectionTooLargeError):
            await restore.plan(RowSelection(table="things", all_rows=True))


class TestDriftComparator:
    async def test_tolerated_drift_is_reported_not_failed(self, backup_set: Any) -> None:
        cluster, manifest, admin_dsn, alpha_dsn, ids, _ = backup_set
        scratch_dsn = await _scratch_restore(cluster, manifest, admin_dsn, "alpha")
        live = await asyncpg.connect(alpha_dsn)
        try:
            await live.execute("DELETE FROM things WHERE id = $1", ids[2])
        finally:
            await live.close()
        comparator = DriftComparator(asyncpg.connect, scratch_dsn, alpha_dsn)
        report = await comparator.compare(as_of=manifest.created_at)
        things = next(t for t in report.tables if t.table == "things")
        assert things.status in {"drifted", "matched"}  # one-row drift sits under the floor
        assert report.ok

    async def test_the_uuid7_cutoff_excludes_rows_born_after_the_backup(self, backup_set: Any) -> None:
        cluster, manifest, admin_dsn, alpha_dsn, _, _ = backup_set
        scratch_dsn = await _scratch_restore(cluster, manifest, admin_dsn, "alpha")
        live = await asyncpg.connect(alpha_dsn)
        try:
            await live.execute("INSERT INTO keyed_only (id, payload) VALUES ($1, 'post-backup')", uuid7())
        finally:
            await live.close()
        comparator = DriftComparator(asyncpg.connect, scratch_dsn, alpha_dsn)
        report = await comparator.compare(as_of=manifest.created_at)
        keyed = next(t for t in report.tables if t.table == "keyed_only")
        assert keyed.cutoff.startswith("uuid7:")
        assert keyed.status == "matched"  # the newborn row sits above the bound and is invisible

    async def test_gross_divergence_fails_the_report(self, backup_set: Any) -> None:
        cluster, manifest, admin_dsn, alpha_dsn, _, _ = backup_set
        scratch_dsn = await _scratch_restore(cluster, manifest, admin_dsn, "alpha")
        live = await asyncpg.connect(alpha_dsn)
        try:
            await live.execute("DELETE FROM things")
        finally:
            await live.close()
        comparator = DriftComparator(asyncpg.connect, scratch_dsn, alpha_dsn)
        report = await comparator.compare(as_of=manifest.created_at)
        things = next(t for t in report.tables if t.table == "things")
        assert things.status == "failed"
        assert not report.ok
        # repair for later tests in the module: put everything back through selective restore.
        restore = SelectiveRestore(asyncpg.connect, scratch_dsn, alpha_dsn, max_rows=100)
        await restore.apply(await restore.plan(RowSelection(table="things", all_rows=True)))
