"""Unit tests for dump drivers + autodetection (argv + version logic, no database)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from threetears.backup import drivers as drivers_module

from threetears.backup.drivers import (
    PostgresDriver,
    YugabyteDriver,
    detect_driver,
    driver_for_version,
)

_PG_VERSION = "PostgreSQL 16.3 on aarch64-apple-darwin, compiled by clang"
def _empty_stream() -> AsyncIterator[bytes]:
    """a dump stream the fake never reads; the environment is what these assert on.

    A plain function returning the async generator, not a coroutine that returns one: the
    latter has to be awaited, and forgetting to is a warning rather than a failure.
    """

    async def _gen() -> AsyncIterator[bytes]:
        return
        yield b""  # pragma: no cover -- unreachable, and what makes this a generator

    return _gen()


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


def test_yugabyte_leaves_serializable_deferrable_when_importing_a_snapshot() -> None:
    """without this every dump on Yugabyte fails, and no Postgres test can see it.

    Yugabyte's ysql_dump defaults serializable-deferrable ON where upstream pg_dump makes it
    opt-in, and Yugabyte refuses `SET TRANSACTION SNAPSHOT` in a serializable transaction:
    "cannot export/import snapshot in SERIALIZABLE Isolation Level". Confirmed against a live
    2026.1.0.0 cluster, where a valid exported id was refused until the flag was added.
    """
    argv = YugabyteDriver().dump_argv("postgresql://u@h/db", snapshot="abc-def")
    assert "--no-serializable-deferrable" in argv


def test_yugabyte_keeps_its_default_when_no_snapshot_is_imported() -> None:
    """the flag is bought for the snapshot, so it is not spent without one.

    Serializable-deferrable waits for a view free of serialization anomalies. Dropping it
    unconditionally would weaken every dump to buy something only the snapshot path needs.
    """
    argv = YugabyteDriver().dump_argv("postgresql://u@h/db")
    assert "--no-serializable-deferrable" not in argv


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


class TestTheRestoreBoundsItsCopyBatch:
    """an unbounded bulk COPY is a transaction the server refuses, not a slow one.

    Yugabyte batches COPY by ROW COUNT with no regard for row size. On a live 3 GB set whose
    LangGraph `checkpoints` rows averaged 115 KB and peaked near 196 KB, the default of 20000
    asked for a 2.3 GB transaction against a tserver inbound RPC buffer of about 365 MB. The
    server answered "Call rejected due to memory pressure", and the client saw only the
    follow-on "Predecessor request for N was not applied" against whichever table it reached --
    which is why it read as non-deterministic and named nothing useful.
    """

    def test_yugabyte_bounds_the_batch_when_given_one(self) -> None:
        options = YugabyteDriver().restore_pg_options(copy_rows_per_transaction=100)
        assert options == "-c yb_default_copy_from_rows_per_transaction=100"

    def test_yugabyte_leaves_the_server_default_alone_when_given_none(self) -> None:
        """None means "do not touch it", which is different from meaning zero."""
        assert YugabyteDriver().restore_pg_options() == ""

    def test_a_nonsense_bound_is_ignored_rather_than_sent(self) -> None:
        """0 means UNLIMITED to Yugabyte, so passing it through would restore the very bug."""
        assert YugabyteDriver().restore_pg_options(copy_rows_per_transaction=0) == ""

    def test_postgres_needs_no_such_option(self) -> None:
        """vanilla Postgres has no such GUC, and pg_restore drives its own transactions."""
        assert PostgresDriver().restore_pg_options(copy_rows_per_transaction=100) == ""


class TestTheRestoreEnvironmentIsComposedNotReplaced:
    """`feed_stdin` hands a mapping to the child as its WHOLE environment.

    So building one from the option fragment alone would drop PATH and every PG* variable the
    caller set -- including the password. This is the trap that makes the naive version fail
    only in deployment, where the environment actually carries something.
    """

    async def test_the_callers_environment_survives(self) -> None:
        captured: dict[str, Any] = {}

        async def fake_feed_stdin(argv: list[str], source: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        driver = YugabyteDriver()
        with patch.object(drivers_module, "feed_stdin", fake_feed_stdin):
            await driver.restore(
                "postgresql://u@h/db",
                _empty_stream(),
                env={"PGPASSWORD": "secret", "PATH": "/usr/bin"},
                copy_rows_per_transaction=100,
            )

        env = captured["env"]
        assert env["PGPASSWORD"] == "secret"
        assert env["PATH"] == "/usr/bin"
        assert "yb_default_copy_from_rows_per_transaction=100" in env["PGOPTIONS"]

    async def test_an_existing_pgoptions_is_added_to_rather_than_overwritten(self) -> None:
        """PGOPTIONS is a space-separated list, and the caller's entries are not ours to drop."""
        captured: dict[str, Any] = {}

        async def fake_feed_stdin(argv: list[str], source: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        with patch.object(drivers_module, "feed_stdin", fake_feed_stdin):
            await YugabyteDriver().restore(
                "postgresql://u@h/db",
                _empty_stream(),
                env={"PGOPTIONS": "-c statement_timeout=0"},
                copy_rows_per_transaction=100,
            )

        assert "statement_timeout=0" in captured["env"]["PGOPTIONS"]
        assert "yb_default_copy_from_rows_per_transaction=100" in captured["env"]["PGOPTIONS"]

    async def test_no_bound_leaves_the_environment_completely_untouched(self) -> None:
        """a driver that adds nothing must pass `env` through as it was, `None` included."""
        captured: dict[str, Any] = {}

        async def fake_feed_stdin(argv: list[str], source: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        with patch.object(drivers_module, "feed_stdin", fake_feed_stdin):
            await YugabyteDriver().restore("postgresql://u@h/db", _empty_stream())

        assert captured["env"] is None
