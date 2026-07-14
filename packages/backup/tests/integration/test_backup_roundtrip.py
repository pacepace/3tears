"""Integration: the real dump -> encrypt -> store -> restore -> verify round-trip.

Proves the thing actually works against a live PostgreSQL (testcontainer) with the real
``pg_dump``/``pg_restore`` tools — the assertion a mock can't fake. Seeds a source database, backs
it up through the engine (encrypted on disk), restores into a throwaway temp database, and asserts
every row survived. Skips loudly when Docker or the pg client tools are absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import asyncpg
import pytest
from pydantic import SecretStr

from threetears.backup.config import BackupConfig
from threetears.backup.drivers import PostgresDriver, detect_driver
from threetears.backup.engine import BackupEngine
from threetears.backup.verify import (
    RestoreVerifier,
    count_public_tables,
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
async def test_backup_restores_into_temp_db_with_all_rows(db_container: str, tmp_path: Path) -> None:
    await _seed_source(db_container)
    engine = _engine(tmp_path)

    record = await engine.create_backup(db_container)

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
        make_temp_db_provisioner(db_container, connect=asyncpg.connect),
        assertions=assert_all_rows,
    )
    result = await verifier.verify(record.key)

    assert result.ok is True
    assert result.checks == {"widgets": _ROW_COUNT}

    # the source is untouched, and the temp db was dropped after verification.
    conn = await asyncpg.connect(db_container)
    try:
        assert await conn.fetchval("SELECT count(*) FROM widgets") == _ROW_COUNT
        temp_dbs = await conn.fetchval("SELECT count(*) FROM pg_database WHERE datname LIKE 'verify_restore_%'")
    finally:
        await conn.close()
    assert temp_dbs == 0


@pytest.mark.asyncio
async def test_detect_driver_identifies_postgres(db_container: str) -> None:
    conn = await asyncpg.connect(db_container)
    try:
        driver = await detect_driver(conn)
    finally:
        await conn.close()
    assert driver.name == "postgres"


@pytest.mark.skipif(not shutil.which("psql"), reason="psql not on PATH")
@pytest.mark.asyncio
async def test_default_table_count_assertion_and_subprocess_hook(db_container: str, tmp_path: Path) -> None:
    await _seed_source(db_container)
    engine = _engine(tmp_path)
    record = await engine.create_backup(db_container)

    hook_output = tmp_path / "hook-count.txt"
    hook = make_subprocess_hook(
        ["sh", "-c", f'psql "$RESTORED_DATABASE_URL" -tAc "SELECT count(*) FROM widgets" > {hook_output}']
    )
    verifier = RestoreVerifier(
        engine,
        make_temp_db_provisioner(db_container, connect=asyncpg.connect),
        assertions=count_public_tables(connect=asyncpg.connect),
        post_restore_hook=hook,
    )

    result = await verifier.verify(record.key)

    assert result.ok is True
    assert result.hook_ran is True
    assert result.checks["public_tables"] >= 1  # the built-in default assertion
    assert hook_output.read_text().strip() == str(_ROW_COUNT)  # the hook ran against the restored db
