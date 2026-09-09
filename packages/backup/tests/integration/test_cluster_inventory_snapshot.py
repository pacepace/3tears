"""Integration: a row written during a backup must not make the manifest wrong.

THE BUG THIS EXISTS FOR was found by running a dry run against a live cluster. The manifest's
row counts were taken on their own connection, and the dump ran afterwards under a snapshot of
its own -- two different instants. Anything written in between was in the bytes and not in the
count, so restoring the dump and comparing it to the manifest that names it reported a coverage
mismatch for ordinary write traffic. Two audit tables, each off by one, on an almost idle
cluster. Under real load the verdict is noise, and noise is how a genuinely short restore gets
waved through.

The write here is INJECTED at the exact moment that used to be unsafe -- after the inventory has
counted, before the dump has started -- rather than raced for. A concurrent writer would
reproduce this only sometimes, and a test that fails sometimes is a test that passes for the
wrong reason most of the time.

The comparison is the real one: restore the dump and count the restored copy. Asserting on the
manifest alone would prove only that the code recorded what it intended to.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from pydantic import SecretStr
from threetears.core.testing.containers import check_docker_available

from threetears.backup.cluster import ClusterBackup
from threetears.backup.config import BackupConfig
from threetears.object_store.filesystem import FilesystemObjectStore

_TOOLS_PRESENT = all(shutil.which(tool) for tool in ("pg_dump", "pg_dumpall", "pg_restore"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TOOLS_PRESENT, reason="pg_dump/pg_dumpall/pg_restore not on PATH"),
]

_SEEDED_ROWS = 3


def _server_image() -> str:
    """the pgvector image whose major version matches the LOCAL dump tools.

    This test restores, and a restore is the one direction where a version gap bites: pg_restore
    emits `SET` commands its own major understands, and an older server rejects them outright
    (`unrecognized configuration parameter "transaction_timeout"` on an 18-to-16 pairing). Pinning
    a fixed tag would make the suite pass or fail on which Postgres the developer happens to have
    installed, which is not a property of this code.

    :return: the container image to run
    :rtype: str
    """
    reported = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"(\d+)", reported)
    if match is None:
        raise RuntimeError(f"cannot read a major version out of {reported!r}")
    return f"pgvector/pgvector:pg{match.group(1)}"


class _WriteInjectingConnection:
    """a real connection that commits one extra row the instant the inventory finishes counting.

    Wraps rather than fakes: every statement reaches the real server, so the counts, the exported
    snapshot and the dump are all genuine. The only addition is the write, fired once, from a
    SEPARATE connection so it commits immediately and is visible to anything not already pinned
    to an earlier snapshot.
    """

    def __init__(self, inner: asyncpg.Connection, target_dsn: str, table: str) -> None:
        self._inner = inner
        self._target_dsn = target_dsn
        self._table = table
        self.injected = False

    async def execute(self, query: str) -> object:
        return await self._inner.execute(query)

    async def fetch(self, query: str) -> list[Any]:
        return list(await self._inner.fetch(query))

    async def fetchval(self, query: str) -> object:
        value = await self._inner.fetchval(query)
        if query.startswith("SELECT count(*)") and not self.injected:
            self.injected = True
            writer = await asyncpg.connect(self._target_dsn)
            try:
                await writer.execute(f"INSERT INTO {self._table} (id) VALUES (999)")
            finally:
                await writer.close()
        return value

    async def close(self) -> None:
        await self._inner.close()


@pytest.fixture
async def cluster(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[tuple[str, str, ClusterBackup, Any]]:
    """a container with one seeded database, and a connect that writes mid-inventory.

    :yield: (admin dsn, seeded database name, the cluster backup, the injecting connect)
    :rtype: AsyncIterator[tuple[str, str, ClusterBackup, Any]]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(_server_image()) as container:
        admin_dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        parsed = urlsplit(admin_dsn)
        database = f"seeded_{uuid4().hex[:8]}"

        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(f'CREATE DATABASE "{database}"')
        finally:
            await conn.close()

        database_dsn = urlunsplit(parsed._replace(path=f"/{database}"))
        seeded = await asyncpg.connect(database_dsn)
        try:
            await seeded.execute("CREATE TABLE widgets (id int PRIMARY KEY)")
            await seeded.execute("INSERT INTO widgets (id) VALUES (1), (2), (3)")
        finally:
            await seeded.close()

        async def injecting_connect(dsn: str) -> Any:
            inner = await asyncpg.connect(dsn)
            return _WriteInjectingConnection(inner, database_dsn, "widgets")

        config = BackupConfig(
            passphrase=SecretStr("test-passphrase-not-a-real-one"),
            prefix="itest",
            encryption_work_factor=2**4,
        )
        store = FilesystemObjectStore(str(tmp_path_factory.mktemp("sets")))
        yield admin_dsn, database, ClusterBackup(config, store, injecting_connect), parsed


class TestTheManifestDescribesTheDumpItNames:
    async def test_a_write_during_the_backup_does_not_desynchronize_the_inventory(
        self, cluster: tuple[str, str, ClusterBackup, Any]
    ) -> None:
        """the restored copy must hold exactly what the manifest said it would.

        Before the fix this failed by one: the injected row was invisible to the count and
        present in the dump.
        """
        admin_dsn, database, backup, parsed = cluster

        manifest = await backup.create_backup(admin_dsn)

        dump = next(d for d in manifest.databases if d.database == database)
        widgets = next(t for t in dump.tables if t.table == "widgets")

        restored_name = f"restored_{uuid4().hex[:8]}"
        admin = await asyncpg.connect(admin_dsn)
        try:
            await admin.execute(f'CREATE DATABASE "{restored_name}"')
        finally:
            await admin.close()

        restored_dsn = urlunsplit(parsed._replace(path=f"/{restored_name}"))
        await backup.restore_database(manifest, database, restored_dsn)

        restored = await asyncpg.connect(restored_dsn)
        try:
            actual = await restored.fetchval("SELECT count(*) FROM widgets")
        finally:
            await restored.close()

        assert actual == widgets.row_count

    async def test_the_write_really_happened(self, cluster: tuple[str, str, ClusterBackup, Any]) -> None:
        """guards the test itself: an injection that silently stopped firing would prove nothing.

        The live database must hold the extra row even though neither the manifest nor the dump
        does -- that asymmetry IS the window the bug lived in.
        """
        admin_dsn, database, backup, parsed = cluster

        manifest = await backup.create_backup(admin_dsn)

        live = await asyncpg.connect(urlunsplit(parsed._replace(path=f"/{database}")))
        try:
            live_rows = await live.fetchval("SELECT count(*) FROM widgets")
        finally:
            await live.close()

        dump = next(d for d in manifest.databases if d.database == database)
        widgets = next(t for t in dump.tables if t.table == "widgets")
        assert live_rows == _SEEDED_ROWS + 1
        assert widgets.row_count == _SEEDED_ROWS

    async def test_the_manifest_reports_a_synchronized_inventory(
        self, cluster: tuple[str, str, ClusterBackup, Any]
    ) -> None:
        """the flag must be earned against a real server, not just set by the happy path."""
        admin_dsn, database, backup, _parsed = cluster

        manifest = await backup.create_backup(admin_dsn)

        dump = next(d for d in manifest.databases if d.database == database)
        assert dump.inventory_snapshot_consistent
