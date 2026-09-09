"""Integration: one sick database must not cost the cluster its backup.

THE BUG THIS EXISTS FOR was observed in production, not imagined. A database
whose ``pg_namespace`` had lost its DocDB tablet made ``ysql_dump`` fail the
moment it tried to enumerate schemas. The dump loop had no handler, so that one
database aborted the whole set — and the cluster went with NO backups at all,
not five dumps and a gap, for as long as it took somebody to read the error.

Two properties, and the second is the one that makes the first safe:

- the healthy databases still dump, so a cluster with one broken database still
  has a backup of everything else;
- the failure is RECORDED in the manifest. A set that quietly omitted a database
  would present as a complete backup of a cluster it does not cover, and the
  first anyone would hear of it is a restore coming up short.

The failing database here is made to fail for a REAL reason — the dumping role
is denied CONNECT — rather than by patching the driver. A faked driver would
prove the ``except`` clause runs; it would not prove that a genuine dump-tool
failure lands there, which is the thing that was broken.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from pydantic import SecretStr
from threetears.core.testing.containers import check_docker_available

from threetears.backup.cluster import ClusterBackup, ClusterBackupError
from threetears.backup.config import BackupConfig
from threetears.object_store.filesystem import FilesystemObjectStore

_TOOLS_PRESENT = all(shutil.which(tool) for tool in ("pg_dump", "pg_dumpall"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TOOLS_PRESENT, reason="pg_dump/pg_dumpall not on PATH"),
]


def _config(root: str) -> BackupConfig:
    """build a backup config pointed at a throwaway prefix.

    :param root: storage prefix for this test's set
    :ptype root: str
    :return: the config
    :rtype: BackupConfig
    """
    return BackupConfig(
        passphrase=SecretStr("test-passphrase-not-a-real-one"), prefix=root, encryption_work_factor=2**4
    )


@pytest.fixture
async def cluster(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[tuple[str, ClusterBackup, str]]:
    """a container holding one healthy database and one that cannot be dumped.

    :yield: (admin dsn, the cluster backup, the name of the sick database)
    :rtype: AsyncIterator[tuple[str, ClusterBackup, str]]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer("pgvector/pgvector:pg16") as container:
        admin_dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        parsed = urlsplit(admin_dsn)
        healthy = f"healthy_{uuid4().hex[:8]}"
        sick = f"sick_{uuid4().hex[:8]}"
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(f'CREATE DATABASE "{healthy}"')
            await conn.execute(f'CREATE DATABASE "{sick}"')
            # A REAL refusal, not a patched driver. `REVOKE CONNECT` is NOT
            # enough and the first draft of this test was wrong for that reason:
            # the dumping role here is a superuser, and superusers ignore it --
            # the "sick" database dumped happily, empty. `datallowconn = false`
            # is the one that binds everybody, superusers included (it is how
            # template0 keeps people out), and it is a real state a database
            # lands in during maintenance.
            await conn.execute(f'ALTER DATABASE "{sick}" WITH ALLOW_CONNECTIONS false')
        finally:
            await conn.close()

        seeded = await asyncpg.connect(urlunsplit(parsed._replace(path=f"/{healthy}")))
        try:
            await seeded.execute("CREATE TABLE widgets (id int PRIMARY KEY)")
            await seeded.execute("INSERT INTO widgets (id) VALUES (1), (2), (3)")
        finally:
            await seeded.close()

        store = FilesystemObjectStore(str(tmp_path_factory.mktemp("sets")))
        yield admin_dsn, ClusterBackup(_config("itest"), store, asyncpg.connect), sick


class TestTransientDatabasesAreNotData:
    """a throwaway restore target has no business in a backup set.

    The set already contains the database a scratch was restored FROM, so
    dumping the scratch stores a second partial copy and pays twice to preserve
    nothing. And they are the databases most likely to be broken: half created,
    half dropped, or wedged mid-reap.

    In production a leftover `scratch_probe_hub2` -- wedged, its `pg_namespace`
    unreadable -- was enumerated by every backup and took the whole set down
    with it. Excluding it is why the WEDGED case below matters more than the
    healthy one.
    """

    async def test_a_scratch_database_is_not_dumped(self, cluster: tuple[str, ClusterBackup, str]) -> None:
        admin_dsn, backup, _sick = cluster
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute('CREATE DATABASE "scratch_probe_deadbeef"')
        finally:
            await conn.close()

        manifest = await backup.create_backup(admin_dsn)

        dumped = {dump.database for dump in manifest.databases}
        # `scratch_probe_`, deliberately: the prefix list used to name only the two
        # forms this toolchain generates, so a hand-made probe database matched
        # nothing and was dumped -- which is exactly how a wedged one took a live
        # cluster's backups down.
        assert "scratch_probe_deadbeef" not in dumped
        # and NOT by being recorded as a failure -- it was never attempted.
        assert "scratch_probe_deadbeef" not in {f.database for f in manifest.failed_databases}

    async def test_a_wedged_scratch_database_cannot_break_the_backup(
        self, cluster: tuple[str, ClusterBackup, str]
    ) -> None:
        """the production failure, reproduced: an unreachable scratch left lying around.

        Before the exclusion this database was enumerated, failed to dump, and --
        with no isolation -- aborted the entire cluster's backup. It must now be
        invisible to the backup entirely.
        """
        admin_dsn, backup, _sick = cluster
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute('CREATE DATABASE "scratch_probe_wedged"')
            await conn.execute('ALTER DATABASE "scratch_probe_wedged" WITH ALLOW_CONNECTIONS false')
        finally:
            await conn.close()

        manifest = await backup.create_backup(admin_dsn)

        assert "scratch_probe_wedged" not in {d.database for d in manifest.databases}
        assert "scratch_probe_wedged" not in {f.database for f in manifest.failed_databases}
        assert manifest.databases, "the healthy databases were lost to a scratch that should not have been touched"


class TestOneSickDatabaseDoesNotCostTheClusterItsBackup:
    async def test_the_healthy_databases_still_dump(self, cluster: tuple[str, ClusterBackup, str]) -> None:
        admin_dsn, backup, sick = cluster

        manifest = await backup.create_backup(admin_dsn)

        dumped = {dump.database for dump in manifest.databases}
        assert sick not in dumped, "the database that cannot be dumped must not be reported as dumped"
        assert dumped, "every healthy database was lost along with the sick one -- the blast radius bug"
        assert any(dump.tables for dump in manifest.databases), "a dump landed with no inventory at all"

    async def test_the_failure_is_recorded_rather_than_skipped(self, cluster: tuple[str, ClusterBackup, str]) -> None:
        """a silently short set is worse than a loud one: it looks like a backup."""
        admin_dsn, backup, sick = cluster

        manifest = await backup.create_backup(admin_dsn)

        assert not manifest.is_complete, "a set missing a database reported itself complete"
        failed = {failure.database for failure in manifest.failed_databases}
        assert sick in failed
        # the dump tool's own words, so the operator has the actual lead
        reason = next(f.error for f in manifest.failed_databases if f.database == sick)
        assert reason, "a failure was recorded with no reason at all"

    async def test_the_record_survives_the_manifest_round_trip(self, cluster: tuple[str, ClusterBackup, str]) -> None:
        """the manifest is the durable one: a restore reads THIS, not the log line."""
        admin_dsn, backup, sick = cluster

        written = await backup.create_backup(admin_dsn)
        [reread] = [m for m in await backup.list_manifests() if m.backup_id == written.backup_id]

        assert not reread.is_complete
        assert sick in {failure.database for failure in reread.failed_databases}

    async def test_a_set_where_nothing_dumped_is_refused_outright(
        self, cluster: tuple[str, ClusterBackup, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an empty manifest would be a backup in name that retention counts as one."""
        admin_dsn, backup, _sick = cluster

        def refuse(self: object, dsn: str, **kwargs: object) -> AsyncIterator[bytes]:
            raise RuntimeError("every database is unreadable")

        from threetears.backup.drivers import PostgresDriver  # noqa: PLC0415

        monkeypatch.setattr(PostgresDriver, "dump", refuse)

        with pytest.raises(ClusterBackupError, match="no database dumps"):
            await backup.create_backup(admin_dsn)
