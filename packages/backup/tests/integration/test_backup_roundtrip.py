"""Integration: the real dump -> encrypt -> store -> restore -> verify round-trip.

Proves the thing actually works against a live PostgreSQL (testcontainer) with the real
``pg_dump``/``pg_restore`` tools — the assertion a mock can't fake. Seeds a source database, backs
it up through the engine (encrypted on disk), restores into a throwaway temp database, and asserts
every row survived. Skips loudly when Docker or the pg client tools are absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from pydantic import SecretStr
from threetears.core.testing.containers import check_docker_available

from threetears.backup.config import BackupConfig
from threetears.backup.drivers import PostgresDriver, detect_driver
from threetears.backup.engine import BackupEngine
from threetears.backup.verify import (
    RestoreVerifier,
    count_tables,
    make_subprocess_hook,
    make_temp_db_provisioner,
)
from threetears.object_store.filesystem import FilesystemObjectStore

_TOOLS_PRESENT = all(shutil.which(tool) for tool in ("pg_dump", "pg_restore"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _TOOLS_PRESENT, reason="pg_dump/pg_restore not on PATH"),
]

_ROW_COUNT = 50


def _client_tool_major() -> int | None:
    """read the major version of the local ``pg_dump``.

    :return: major version, or ``None`` when it cannot be determined
    :rtype: int | None
    """
    found: int | None = None
    if _TOOLS_PRESENT:
        out = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True, check=False)  # noqa: S603, S607
        match = re.search(r"(\d+)\.", out.stdout)
        if match is not None:
            found = int(match.group(1))
    return found


@pytest.fixture(scope="module")
def backup_container() -> Iterator[str]:
    """a postgres container of this module's own, matching the local client tools.

    TWO reasons this module cannot use the shared session container, and each
    one alone is sufficient.

    **Version coupling.** This is the one suite that shells out to real
    ``pg_dump`` / ``pg_restore``, so the SERVER it runs against must be at
    least as new as those binaries. The shared container defaults to pg16
    while a developer's client tools are whatever their package manager
    installed -- 18.4 here. ``pg_dump`` 18 writes ``SET transaction_timeout``
    (a Postgres 17 GUC) into the dump and a pg16 server refuses to restore it,
    so the round-trip fails on a version skew that has nothing to do with the
    code under test. Pinning the server to the client's major removes the skew
    rather than tolerating it.

    **Co-tenancy.** The module dumps and restores WHOLE databases, creating and
    dropping tables in whatever it is pointed at. On the shared container it
    seeded ``public.widgets``, which broke ``packages/datasources``'
    search-path test -- that test proves an unqualified ``SELECT * FROM
    widgets`` cannot resolve, and a ``widgets`` in ``public`` makes it resolve
    fine. Both suites passed alone and failed together. Renaming the table
    would settle this collision and leave the next one waiting; not sharing a
    server settles the class.

    :yield: DSN of a module-private postgres container
    :rtype: Iterator[str]
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    major = _client_tool_major()
    if major is None:
        pytest.skip("could not determine the local pg_dump major version")

    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(f"postgres:{major}") as container:
        # testcontainers hands back a SQLAlchemy-style URL; asyncpg wants a
        # plain postgresql:// DSN.
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture(scope="module")
async def backup_database(backup_container: str) -> AsyncIterator[str]:
    """provision a database of this module's own, and yield its DSN.

    A fresh database per module rather than the container's default one, so a
    failed restore cannot leave debris that the next test in this file reads as
    state.

    :param backup_container: DSN of this module's postgres container
    :ptype backup_container: str
    :yield: DSN of a freshly created database, dropped on teardown
    :rtype: AsyncIterator[str]
    """
    parsed = urlsplit(backup_container)
    name = f"backup_itest_{uuid4().hex[:12]}"
    admin_dsn = urlunsplit(parsed._replace(path="/postgres"))

    admin = await asyncpg.connect(admin_dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    try:
        yield urlunsplit(parsed._replace(path=f"/{name}"))
    finally:
        admin = await asyncpg.connect(admin_dsn)
        try:
            # Terminate stragglers first: pg_restore's own connections can
            # outlive the test body, and DROP DATABASE refuses while any
            # session is attached.
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1",
                name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()


async def _seed_source(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS widgets")
        await conn.execute("CREATE TABLE widgets (id int PRIMARY KEY, name text NOT NULL)")
        await conn.executemany(
            "INSERT INTO widgets (id, name) VALUES ($1, $2)",
            [(i, f"widget-{i}") for i in range(_ROW_COUNT)],
        )
    finally:
        await conn.close()


def _engine(tmp_path: Path) -> BackupEngine:
    config = BackupConfig(passphrase=SecretStr("integration-passphrase"), encryption_work_factor=2**8)
    return BackupEngine(config, FilesystemObjectStore(tmp_path), PostgresDriver())


@pytest.mark.asyncio
async def test_backup_restores_into_temp_db_with_all_rows(backup_database: str, tmp_path: Path) -> None:
    await _seed_source(backup_database)
    engine = _engine(tmp_path)

    record = await engine.create_backup(backup_database)

    assert record.size_bytes > 0
    on_disk = (tmp_path / record.key).read_bytes()
    assert on_disk.startswith(b"3TB1")  # encrypted at rest, not raw dump

    async def assert_all_rows(dsn: str) -> dict[str, int]:
        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetchval("SELECT count(*) FROM widgets")
        finally:
            await conn.close()
        assert rows == _ROW_COUNT
        return {"widgets": int(rows)}

    verifier = RestoreVerifier(
        engine,
        make_temp_db_provisioner(backup_database, connect=asyncpg.connect),
        assertions=assert_all_rows,
    )
    result = await verifier.verify(record.key)

    assert result.ok is True
    assert result.checks == {"widgets": _ROW_COUNT}

    # the source is untouched, and the temp db was dropped after verification.
    conn = await asyncpg.connect(backup_database)
    try:
        assert await conn.fetchval("SELECT count(*) FROM widgets") == _ROW_COUNT
        temp_dbs = await conn.fetchval("SELECT count(*) FROM pg_database WHERE datname LIKE 'verify_restore_%'")
    finally:
        await conn.close()
    assert temp_dbs == 0


@pytest.mark.asyncio
async def test_detect_driver_identifies_postgres(backup_database: str) -> None:
    conn = await asyncpg.connect(backup_database)
    try:
        driver = await detect_driver(conn)
    finally:
        await conn.close()
    assert driver.name == "postgres"


@pytest.mark.skipif(not shutil.which("psql"), reason="psql not on PATH")
@pytest.mark.asyncio
async def test_default_table_count_assertion_and_subprocess_hook(backup_database: str, tmp_path: Path) -> None:
    await _seed_source(backup_database)
    engine = _engine(tmp_path)
    record = await engine.create_backup(backup_database)

    hook_output = tmp_path / "hook-count.txt"
    hook = make_subprocess_hook(
        ["sh", "-c", f'psql "$RESTORED_DATABASE_URL" -tAc "SELECT count(*) FROM widgets" > {hook_output}']
    )
    verifier = RestoreVerifier(
        engine,
        make_temp_db_provisioner(backup_database, connect=asyncpg.connect),
        assertions=count_tables(connect=asyncpg.connect),
        post_restore_hook=hook,
    )

    result = await verifier.verify(record.key)

    assert result.ok is True
    assert result.hook_ran is True
    assert result.checks["tables"] >= 1  # the built-in default assertion
    assert hook_output.read_text().strip() == str(_ROW_COUNT)  # the hook ran against the restored db
