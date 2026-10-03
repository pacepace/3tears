"""Unit tests: a dump that lost a race with concurrent DDL is retried, and nothing else is.

Found live on a cold start: the hub's first scheduled cluster backup dumped database ``aibots``
while an agent schema was being created in it, and ``ysql_dump`` failed with ``schema with OID
17839 does not exist`` -- the dump read the catalog list of schemas, then asked about one the
concurrent DDL had changed underneath it. The other databases dumped, the set was marked
INCOMPLETE, and the cluster waited a whole interval for its next chance.

The fix retries ONE database's dump, inventory and all, when the dump tool's failure names an
object that vanished mid-dump -- bounded, with backoff, each retry logged -- and never on any other
failure. The dump subprocess is the only thing faked, as in ``test_cluster_snapshot``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import SecretStr

from threetears.backup import cluster as cluster_module
from threetears.backup import drivers as drivers_module
from threetears.backup.cluster import ClusterBackup, is_concurrent_ddl_failure
from threetears.backup.config import BackupConfig
from threetears.backup.manifest import BackupManifest
from threetears.backup.process import BackupToolError
from threetears.object_store.filesystem import FilesystemObjectStore

_YB_VERSION = "PostgreSQL 15.12-YB-2025.1.0.0-b0 on x86_64-pc-linux-gnu"
_DATABASE = "aibots"
_RACE = "ysql_dump: error: schema with OID 17839 does not exist\n"


# parity-with: threetears.backup.cluster._Connection
class _Connection:
    """answers the cluster's queries; counts the inventory transactions it was asked to open."""

    def __init__(self) -> None:
        self.inventories = 0

    async def execute(self, query: str, *args: object) -> object:
        if query.startswith("BEGIN"):
            self.inventories += 1
        return None

    async def fetch(self, query: str, *args: object) -> list[Any]:
        if "pg_database" in query:
            return [{"datname": _DATABASE}]
        return [{"table_schema": "public", "table_name": "widgets"}]

    async def fetchval(self, query: str, *args: object) -> object:
        if query == "SELECT version()":
            return _YB_VERSION
        if query == "SELECT pg_export_snapshot()":
            return "00000003-0000001B-1"
        return 3

    async def close(self) -> None:
        return None


class _ScriptedDumps:
    """the database dump fails with each scripted error in turn, then succeeds."""

    def __init__(self, failures: list[BackupToolError]) -> None:
        self.failures = failures
        self.database_dumps = 0

    def stream_stdout(self, argv: list[str], **_: object) -> AsyncIterator[bytes]:
        is_database_dump = "--globals-only" not in argv
        failure = self.failures.pop(0) if is_database_dump and self.failures else None
        if is_database_dump:
            self.database_dumps += 1

        async def chunks() -> AsyncIterator[bytes]:
            yield b"-- dump\n"
            if failure is not None:
                raise failure

        return chunks()


@pytest.fixture
def scripted() -> Iterator[_ScriptedDumps]:
    dumps = _ScriptedDumps([])
    original_drivers = drivers_module.stream_stdout
    original_cluster = cluster_module.stream_stdout
    drivers_module.stream_stdout = dumps.stream_stdout  # type: ignore[assignment]
    cluster_module.stream_stdout = dumps.stream_stdout  # type: ignore[assignment]
    try:
        yield dumps
    finally:
        drivers_module.stream_stdout = original_drivers  # type: ignore[assignment]
        cluster_module.stream_stdout = original_cluster  # type: ignore[assignment]


async def _run(tmp_path: Any, connection: _Connection, *, retries: int = 3) -> BackupManifest:
    async def connect(_dsn: str) -> _Connection:
        return connection

    config = BackupConfig(
        passphrase=SecretStr("test-passphrase-not-a-real-one"),
        prefix="utest",
        encryption_work_factor=2**4,
        dump_concurrent_ddl_retries=retries,
        dump_concurrent_ddl_retry_delay_seconds=0.01,
    )
    backup = ClusterBackup(config, FilesystemObjectStore(str(tmp_path)), connect)
    return await backup.create_backup("postgresql://u@h/yugabyte")


class TestADumpThatLostARaceWithDdlIsRetried:
    async def test_it_is_dumped_again_and_the_set_is_complete(
        self, tmp_path: Any, scripted: _ScriptedDumps, caplog: pytest.LogCaptureFixture
    ) -> None:
        scripted.failures.append(BackupToolError("ysql_dump", 1, _RACE))
        connection = _Connection()

        with caplog.at_level(logging.WARNING, logger="threetears.backup.cluster"):
            manifest = await _run(tmp_path, connection)

        assert manifest.is_complete
        assert [d.database for d in manifest.databases] == [_DATABASE]
        assert scripted.database_dumps == 2
        assert connection.inventories == 2, "the retry takes a fresh inventory under a fresh snapshot"
        retries = [r for r in caplog.records if "concurrent DDL" in r.getMessage()]
        assert len(retries) == 1 and r"OID 17839" in retries[0].getMessage()

    async def test_the_retries_are_bounded_and_the_last_failure_is_recorded(
        self, tmp_path: Any, scripted: _ScriptedDumps
    ) -> None:
        scripted.failures.extend(BackupToolError("ysql_dump", 1, _RACE) for _ in range(10))

        with pytest.raises(cluster_module.ClusterBackupError, match="schema with OID 17839"):
            await _run(tmp_path, _Connection(), retries=2)

        assert scripted.database_dumps == 3, "one attempt and two retries, then the database is recorded failed"


class TestTheBackoffDoubles:
    async def test_each_later_retry_waits_twice_as_long(
        self, tmp_path: Any, scripted: _ScriptedDumps, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripted.failures.extend(BackupToolError("ysql_dump", 1, _RACE) for _ in range(3))
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(cluster_module.asyncio, "sleep", _sleep)

        await _run(tmp_path, _Connection(), retries=3)

        assert slept == [0.01, 0.02, 0.04]


class TestNothingElseIsRetried:
    @pytest.mark.parametrize(
        "failure",
        [
            BackupToolError("ysql_dump", 1, "ysql_dump: error: permission denied for table widgets\n"),
            BackupToolError("ysql_dump", 1, "ysql_dump: error: connection to server failed\n"),
            # a timeout's message carries the stderr too; the sentinel exit code is what rules it out
            BackupToolError("ysql_dump", -1, f"timed out after 3600s. {_RACE}"),
            RuntimeError(_RACE),
        ],
    )
    async def test_a_failure_that_is_not_a_ddl_race_is_attempted_once(
        self, tmp_path: Any, scripted: _ScriptedDumps, failure: Exception
    ) -> None:
        scripted.failures.append(failure)  # type: ignore[arg-type]

        with pytest.raises(cluster_module.ClusterBackupError):
            await _run(tmp_path, _Connection())

        assert scripted.database_dumps == 1


class TestTheClassifier:
    @pytest.mark.parametrize(
        "stderr",
        [
            "ysql_dump: error: schema with OID 17839 does not exist",
            "pg_dump: error: schema with OID 2200 does not exist",
            "pg_dump: error: query failed: ERROR:  could not open relation with OID 16384",
            "pg_dump: error: query failed: ERROR:  cache lookup failed for relation 16384",
        ],
    )
    def test_a_vanished_catalog_object_is_a_ddl_race(self, stderr: str) -> None:
        assert is_concurrent_ddl_failure(BackupToolError("ysql_dump", 1, stderr))

    @pytest.mark.parametrize(
        "stderr",
        [
            "ysql_dump: error: permission denied for schema private",
            'ysql_dump: error: relation "widgets" does not exist',
            "pg_dump: error: server version mismatch",
        ],
    )
    def test_anything_else_is_not(self, stderr: str) -> None:
        assert not is_concurrent_ddl_failure(BackupToolError("ysql_dump", 1, stderr))

    def test_a_restore_tool_is_never_a_dump_race(self) -> None:
        assert not is_concurrent_ddl_failure(BackupToolError("ysqlsh", 1, _RACE))


class TestTheConfig:
    def test_negative_retries_are_refused(self) -> None:
        with pytest.raises(ValueError, match="dump_concurrent_ddl_retries"):
            BackupConfig(passphrase=SecretStr("p"), dump_concurrent_ddl_retries=-1)

    def test_a_non_positive_delay_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dump_concurrent_ddl_retry_delay_seconds"):
            BackupConfig(passphrase=SecretStr("p"), dump_concurrent_ddl_retry_delay_seconds=0)

    def test_the_environment_sets_both(self) -> None:
        config = BackupConfig.from_env(
            {
                "THREETEARS_BACKUP_PASSPHRASE": "p",
                "THREETEARS_BACKUP_DUMP_CONCURRENT_DDL_RETRIES": "5",
                "THREETEARS_BACKUP_DUMP_CONCURRENT_DDL_RETRY_DELAY_SECONDS": "2.5",
            }
        )
        assert (config.dump_concurrent_ddl_retries, config.dump_concurrent_ddl_retry_delay_seconds) == (5, 2.5)
