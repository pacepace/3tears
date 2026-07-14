"""Unit tests for BackupEngine — orchestration over a real filesystem store + a fake driver.

The store is a real :class:`FilesystemObjectStore`, so encryption (by construction) and gzip
genuinely round-trip to disk; the driver is faked so no database is needed. The pg/yuga dump tools
are proven separately in the integration tier.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from threetears.backup.config import BackupConfig
from threetears.backup.drivers import DbDumpDriver
from threetears.backup.engine import BackupEngine, DeleteNotAllowedError
from threetears.object_store.filesystem import FilesystemObjectStore


class _FakeDriverBase(DbDumpDriver):
    def __init__(self, payload: bytes = b"DUMP-PAYLOAD-CONTENTS") -> None:
        self.payload = payload
        self.restored: bytes | None = None
        self.restored_dsn: str | None = None

    def dump_argv(self, dsn: str) -> list[str]:
        return ["true"]

    def restore_argv(self, dsn: str) -> list[str]:
        return ["true"]

    def dump(self, dsn: str, *, env: Mapping[str, str] | None = None) -> AsyncIterator[bytes]:
        return self._emit()

    async def _emit(self) -> AsyncIterator[bytes]:
        yield self.payload[:5]
        yield self.payload[5:]

    async def restore(self, dsn: str, source: AsyncIterator[bytes], *, env: Mapping[str, str] | None = None) -> None:
        buf = bytearray()
        async for chunk in source:
            buf += chunk
        self.restored = bytes(buf)
        self.restored_dsn = dsn


class FakePlainDriver(_FakeDriverBase):
    name = "fakeplain"
    compressed = False  # engine will gzip


class FakeCompressedDriver(_FakeDriverBase):
    name = "fakecomp"
    compressed = True  # engine skips gzip


def _config(**kw: object) -> BackupConfig:
    kw.setdefault("encryption_work_factor", 2**8)  # keep scrypt cheap for the suite
    return BackupConfig(passphrase=SecretStr("pw"), **kw)  # type: ignore[arg-type]


def _engine(tmp_path: Path, driver: DbDumpDriver, **cfg: object) -> BackupEngine:
    return BackupEngine(_config(**cfg), FilesystemObjectStore(tmp_path), driver, env={"scrypt": "fast"})


@pytest.mark.asyncio
async def test_create_then_restore_round_trip_plain(tmp_path: Path) -> None:
    driver = FakePlainDriver(payload=b"hello-database-dump" * 50)
    engine = _engine(tmp_path, driver)

    record = await engine.create_backup("postgresql://src/db")
    await engine.restore_into("postgresql://tmp/verify", record.key)

    assert driver.restored == driver.payload  # decrypt -> gunzip -> driver got the original
    assert driver.restored_dsn == "postgresql://tmp/verify"
    # what landed on disk is ciphertext, and the key marks it gzipped + encrypted
    on_disk = (tmp_path / record.key).read_bytes()
    assert on_disk.startswith(b"3TB1")
    assert driver.payload not in on_disk
    assert record.key.endswith(".fakeplain.dump.gz.enc")
    assert record.size_bytes > 0


@pytest.mark.asyncio
async def test_compressed_driver_skips_gzip_suffix(tmp_path: Path) -> None:
    driver = FakeCompressedDriver(payload=b"already-compressed-archive")
    engine = _engine(tmp_path, driver)

    record = await engine.create_backup("postgresql://src/db")
    await engine.restore_into("postgresql://tmp/verify", record.key)

    assert driver.restored == driver.payload
    assert record.key.endswith(".fakecomp.dump.enc")  # no .gz


@pytest.mark.asyncio
async def test_list_backups_newest_first_with_key_timestamps(tmp_path: Path) -> None:
    engine = _engine(tmp_path, FakePlainDriver())
    r1 = await engine.create_backup("dsn", when=datetime(2026, 7, 1, 3, 0, tzinfo=UTC))
    r2 = await engine.create_backup("dsn", when=datetime(2026, 7, 9, 3, 0, tzinfo=UTC))
    r3 = await engine.create_backup("dsn", when=datetime(2026, 7, 5, 3, 0, tzinfo=UTC))

    listed = await engine.list_backups()

    assert [r.key for r in listed] == [r2.key, r3.key, r1.key]  # newest first
    assert listed[0].created_at == datetime(2026, 7, 9, 3, 0, tzinfo=UTC)  # parsed from the key


@pytest.mark.asyncio
async def test_apply_retention_requires_allow_delete(tmp_path: Path) -> None:
    engine = _engine(tmp_path, FakePlainDriver(), allow_delete=False)
    await engine.create_backup("dsn")
    with pytest.raises(DeleteNotAllowedError):
        await engine.apply_retention()


@pytest.mark.asyncio
async def test_apply_retention_prunes_beyond_the_policy(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        FakePlainDriver(),
        allow_delete=True,
        retention_daily=1,
        retention_weekly=1,
        retention_monthly=1,
    )
    keep_daily = await engine.create_backup("dsn", when=datetime(2026, 7, 8, tzinfo=UTC))  # newest daily
    drop_daily = await engine.create_backup("dsn", when=datetime(2026, 7, 7, tzinfo=UTC))  # older daily
    keep_weekly = await engine.create_backup("dsn", when=datetime(2026, 7, 5, tzinfo=UTC))  # Sunday
    drop_weekly = await engine.create_backup("dsn", when=datetime(2026, 6, 28, tzinfo=UTC))  # older Sunday
    keep_monthly = await engine.create_backup("dsn", when=datetime(2026, 7, 1, tzinfo=UTC))  # 1st

    decision = await engine.apply_retention()

    assert {r.key for r in decision.delete} == {drop_daily.key, drop_weekly.key}
    surviving = {r.key for r in await engine.list_backups()}
    assert surviving == {keep_daily.key, keep_weekly.key, keep_monthly.key}


@pytest.mark.asyncio
async def test_plan_retention_is_non_destructive(tmp_path: Path) -> None:
    engine = _engine(tmp_path, FakePlainDriver(), retention_daily=1, retention_weekly=1, retention_monthly=1)
    await engine.create_backup("dsn", when=datetime(2026, 7, 7, tzinfo=UTC))
    await engine.create_backup("dsn", when=datetime(2026, 7, 8, tzinfo=UTC))

    decision = await engine.plan_retention()

    assert len(decision.delete) == 1  # would prune one
    assert len(await engine.list_backups()) == 2  # but nothing was deleted


@pytest.mark.asyncio
async def test_delete_backup_guard_and_success(tmp_path: Path) -> None:
    guarded = _engine(tmp_path, FakePlainDriver(), allow_delete=False)
    record = await guarded.create_backup("dsn")
    with pytest.raises(DeleteNotAllowedError):
        await guarded.delete_backup(record.key)

    allowed = BackupEngine(_config(allow_delete=True), FilesystemObjectStore(tmp_path), FakePlainDriver())
    await allowed.delete_backup(record.key)
    assert await allowed.list_backups() == []
