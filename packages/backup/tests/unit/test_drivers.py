"""Unit tests for dump drivers + autodetection (argv + version logic, no database)."""

from __future__ import annotations

import pytest

from threetears.backup.drivers import (
    PostgresDriver,
    YugabyteDriver,
    detect_driver,
    driver_for_version,
)

_PG_VERSION = "PostgreSQL 16.3 on aarch64-apple-darwin, compiled by clang"
_YB_VERSION = "PostgreSQL 11.2-YB-2.20.1.0-b0 on x86_64-pc-linux-gnu, compiled by gcc"


def test_postgres_argv() -> None:
    driver = PostgresDriver()
    assert driver.dump_argv("postgresql://u@h/db") == [
        "pg_dump",
        "--dbname",
        "postgresql://u@h/db",
        "--format=custom",
        "--no-owner",
        "--no-privileges",
    ]
    assert driver.restore_argv("postgresql://u@h/tmp") == [
        "pg_restore",
        "--dbname",
        "postgresql://u@h/tmp",
        "--no-owner",
        "--no-privileges",
        "--exit-on-error",
    ]


def test_yugabyte_argv() -> None:
    driver = YugabyteDriver()
    assert driver.dump_argv("postgresql://u@h/db")[0] == "ysql_dump"
    assert driver.restore_argv("postgresql://u@h/tmp")[0] == "ysqlsh"
    assert "ON_ERROR_STOP=1" in driver.restore_argv("postgresql://u@h/tmp")


@pytest.mark.parametrize("driver", [PostgresDriver(), YugabyteDriver()])
def test_dump_argv_omits_snapshot_when_none_was_exported(driver: PostgresDriver | YugabyteDriver) -> None:
    """no snapshot must mean no flag, not an empty one.

    `--snapshot=` with nothing after it is not "use the default"; the dump tool takes it as a
    snapshot id of the empty string and refuses to start.
    """
    assert not any(arg.startswith("--snapshot") for arg in driver.dump_argv("postgresql://u@h/db"))


@pytest.mark.parametrize("driver", [PostgresDriver(), YugabyteDriver()])
def test_dump_argv_joins_the_exported_snapshot(driver: PostgresDriver | YugabyteDriver) -> None:
    """the dump must read the instant the inventory counted, not one of its own.

    This is the whole fix: without the flag the tool picks its own snapshot whenever it happens
    to start, so every row written between the count and the dump lands in the bytes and not in
    the manifest -- and the dry run that compares them reports a mismatch for ordinary traffic.
    """
    argv = driver.dump_argv("postgresql://u@h/db", snapshot="0000ABCD-1-1")
    assert "--snapshot=0000ABCD-1-1" in argv


@pytest.mark.parametrize(
    ("version", "expected"),
    [(_PG_VERSION, "postgres"), (_YB_VERSION, "yugabyte")],
)
def test_driver_for_version(version: str, expected: str) -> None:
    assert driver_for_version(version).name == expected


# parity-with: threetears.backup.drivers._VersionSource
class _FakeConn:
    def __init__(self, version: str) -> None:
        self._version = version
        self.queries: list[str] = []

    async def fetchval(self, query: str) -> object:
        self.queries.append(query)
        return self._version


@pytest.mark.asyncio
async def test_detect_driver_postgres() -> None:
    conn = _FakeConn(_PG_VERSION)
    driver = await detect_driver(conn)
    assert driver.name == "postgres"
    assert conn.queries == ["SELECT version()"]


@pytest.mark.asyncio
async def test_detect_driver_yugabyte() -> None:
    driver = await detect_driver(_FakeConn(_YB_VERSION))
    assert driver.name == "yugabyte"
