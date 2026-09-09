"""Unit tests: the inventory and the dump must describe ONE instant.

THE BUG THIS EXISTS FOR shipped, and a live cluster found it rather than a test. The row counts
were taken on their own connection and the dump started afterwards under a snapshot of its own,
so the manifest described the database at count time and the bytes described it at dump time.
A dry run compares a restored copy against the manifest that names it, so every row written in
between read as a coverage mismatch -- two audit tables, each off by one, on a cluster doing
almost nothing. Under load that verdict is noise, and noise is how a genuinely short restore
gets waved through.

These tests pin the wiring without a database: that the snapshot is exported inside the
transaction, that the counts are taken while it is held, that the SAME id reaches the dump tool,
and that a cluster which cannot export one is still backed up -- with the manifest saying so.

The dump subprocess is the only thing faked. Both modules that launch one are patched, because
`cluster` imports `stream_stdout` directly for the globals dump while `drivers` uses its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import SecretStr

from threetears.backup import cluster as cluster_module
from threetears.backup import drivers as drivers_module
from threetears.backup.cluster import ClusterBackup
from threetears.backup.config import BackupConfig
from threetears.backup.manifest import BackupManifest
from threetears.object_store.filesystem import FilesystemObjectStore

_PG_VERSION = "PostgreSQL 16.3 on aarch64-apple-darwin, compiled by clang"
_SNAPSHOT_ID = "00000003-0000001B-1"
_DATABASE = "app"
_ROWS = 3


class _SnapshotUnavailable(RuntimeError):
    """what a server without snapshot export raises, from this package's point of view.

    The real one is the injected driver's error class, which this package cannot name: the
    connection arrives as a callable so asyncpg stays out of the hard dependencies.
    """


# parity-with: threetears.backup.cluster._Connection
class _RecordingConnection:
    """answers the cluster's queries and records the order they arrived in."""

    def __init__(
        self,
        *,
        export_fails: bool = False,
        begin_fails: bool = False,
        rollback_fails: bool = False,
        export_returns_none: bool = False,
        close_fails: bool = False,
    ) -> None:
        self.close_fails = close_fails
        self.export_fails = export_fails
        self.export_returns_none = export_returns_none
        self.begin_fails = begin_fails
        self.rollback_fails = rollback_fails
        #: every statement this connection saw, in order -- the assertion surface for
        #: "counted INSIDE the transaction" is ordering, not any single call.
        self.log: list[str] = []

    async def execute(self, query: str) -> object:
        self.log.append(query)
        if query == "BEGIN ISOLATION LEVEL REPEATABLE READ" and self.begin_fails:
            raise _SnapshotUnavailable("this pooler does not allow explicit transaction control")
        if query == "ROLLBACK" and self.rollback_fails:
            raise _SnapshotUnavailable("connection closed by the server")
        return None

    async def fetch(self, query: str) -> list[Any]:
        self.log.append(query)
        if "pg_database" in query:
            return [{"datname": _DATABASE}]
        return [{"table_schema": "public", "table_name": "widgets"}]

    async def fetchval(self, query: str) -> object:
        self.log.append(query)
        if query == "SELECT version()":
            return _PG_VERSION
        if query == "SELECT pg_export_snapshot()":
            if self.export_fails:
                raise _SnapshotUnavailable("pg_export_snapshot is not enabled on this server")
            if self.export_returns_none:
                return None
            return _SNAPSHOT_ID
        return _ROWS

    async def close(self) -> None:
        self.log.append("CLOSE")
        # Only the INVENTORY connection, which is the one that held a transaction across the
        # dump and so the one a reaper closes. This fake answers for every connection the backup
        # opens, including the short-lived version and database-list ones -- failing those would
        # test that a backup dies before it starts, which is both true and beside the point.
        if self.close_fails and any(q.startswith("SELECT count(*)") for q in self.log):
            raise _SnapshotUnavailable("connection already gone")


@pytest.fixture
def dump_argv() -> Iterator[list[list[str]]]:
    """capture every dump argv instead of launching a dump tool.

    :yield: the argv of each subprocess the backup would have run
    :rtype: Iterator[list[list[str]]]
    """
    captured: list[list[str]] = []

    def fake_stream_stdout(argv: list[str], **_: object) -> AsyncIterator[bytes]:
        captured.append(argv)

        async def chunks() -> AsyncIterator[bytes]:
            yield b"-- dump\n"

        return chunks()

    original_drivers = drivers_module.stream_stdout
    original_cluster = cluster_module.stream_stdout
    drivers_module.stream_stdout = fake_stream_stdout  # type: ignore[assignment]
    cluster_module.stream_stdout = fake_stream_stdout  # type: ignore[assignment]
    try:
        yield captured
    finally:
        drivers_module.stream_stdout = original_drivers  # type: ignore[assignment]
        cluster_module.stream_stdout = original_cluster  # type: ignore[assignment]


async def _run_backup(tmp_path: Any, connection: _RecordingConnection) -> BackupManifest:
    """take a backup against the recording connection.

    :param tmp_path: pytest temp directory for the object store
    :param connection: the connection every step will be answered by
    :ptype connection: _RecordingConnection
    :return: the written manifest
    :rtype: BackupManifest
    """

    async def connect(_dsn: str) -> _RecordingConnection:
        return connection

    config = BackupConfig(
        passphrase=SecretStr("test-passphrase-not-a-real-one"), prefix="utest", encryption_work_factor=2**4
    )
    backup = ClusterBackup(config, FilesystemObjectStore(str(tmp_path)), connect)
    return await backup.create_backup("postgresql://u@h/postgres")


class TestTheDumpJoinsTheInventorySnapshot:
    async def test_the_exported_snapshot_reaches_the_dump_tool(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        connection = _RecordingConnection()
        await _run_backup(tmp_path, connection)

        database_dump = next(argv for argv in dump_argv if "--dbname" in argv and "--globals-only" not in argv)
        assert f"--snapshot={_SNAPSHOT_ID}" in database_dump

    async def test_the_counts_are_taken_inside_the_transaction(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        """ordering IS the contract -- a count after the rollback describes a different instant."""
        connection = _RecordingConnection()
        await _run_backup(tmp_path, connection)

        begin = connection.log.index("BEGIN ISOLATION LEVEL REPEATABLE READ")
        export = connection.log.index("SELECT pg_export_snapshot()")
        count = connection.log.index('SELECT count(*) FROM "public"."widgets"')
        rollback = len(connection.log) - 1 - connection.log[::-1].index("ROLLBACK")
        assert begin < export < count < rollback

    async def test_the_snapshot_outlives_the_dump(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        """an exported snapshot dies with its transaction, so the rollback must come last.

        Rolling back before the dump has finished streaming leaves the dump tool holding an id
        the server has already forgotten, and it fails outright.
        """
        connection = _RecordingConnection()
        await _run_backup(tmp_path, connection)

        assert connection.log[-2:] == ["ROLLBACK", "CLOSE"]

    async def test_the_manifest_records_that_the_inventory_is_synchronized(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        manifest = await _run_backup(tmp_path, _RecordingConnection())

        assert [d.inventory_snapshot_consistent for d in manifest.databases] == [True]


class TestAClusterWithoutSnapshotExportIsStillBackedUp:
    """losing a database's backup to protect a row count would be the wrong trade.

    The dump is the artifact; the count only describes it. So the backup proceeds and the
    manifest carries the fact that its inventory was taken beside the dump rather than within
    it -- the data carrying what it is missing, so no reader has to infer it.
    """

    async def test_the_backup_still_happens(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        manifest = await _run_backup(tmp_path, _RecordingConnection(export_fails=True))

        assert [d.database for d in manifest.databases] == [_DATABASE]
        assert manifest.is_complete

    async def test_the_manifest_says_the_inventory_is_unsynchronized(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        manifest = await _run_backup(tmp_path, _RecordingConnection(export_fails=True))

        assert [d.inventory_snapshot_consistent for d in manifest.databases] == [False]

    async def test_no_snapshot_flag_is_passed_to_the_dump_tool(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        """`--snapshot=` with nothing after it is a snapshot id of the empty string, not a default."""
        await _run_backup(tmp_path, _RecordingConnection(export_fails=True))

        database_dump = next(argv for argv in dump_argv if "--dbname" in argv and "--globals-only" not in argv)
        assert not any(arg.startswith("--snapshot") for arg in database_dump)

    async def test_the_failed_transaction_is_rolled_back_before_counting(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        """a failed statement leaves the transaction aborted, and every count in it would fail too."""
        connection = _RecordingConnection(export_fails=True)
        await _run_backup(tmp_path, connection)

        export = connection.log.index("SELECT pg_export_snapshot()")
        rollback = connection.log.index("ROLLBACK")
        count = connection.log.index('SELECT count(*) FROM "public"."widgets"')
        assert export < rollback < count


class TestCleanupNeverDestroysAFinishedDump:
    """by the time the transaction is ended the dump is hashed and in the store.

    A connection the server closed underneath us -- an idle-in-transaction timeout, a pooler
    cutoff -- is a cleanup problem, not a backup problem. Letting it raise would turn a backup
    that SUCCEEDED into a recorded failure and orphan its key in the object store, which is the
    worst of both: the bytes are paid for and the manifest disowns them.
    """

    async def test_a_pooler_that_refuses_begin_still_gets_a_backup(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        """the trade this refuses: losing a database's dump to protect a row count."""
        manifest = await _run_backup(tmp_path, _RecordingConnection(begin_fails=True))

        assert [d.database for d in manifest.databases] == [_DATABASE]
        assert [d.inventory_snapshot_consistent for d in manifest.databases] == [False]

    async def test_a_rollback_that_fails_does_not_lose_the_dump(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        manifest = await _run_backup(tmp_path, _RecordingConnection(rollback_fails=True))

        assert [d.database for d in manifest.databases] == [_DATABASE]
        assert manifest.is_complete
        assert not manifest.failed_databases

    async def test_a_close_that_fails_does_not_lose_the_dump(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        """the other half of the same guarantee, and it was the untested one.

        Ending the transaction had a test and closing the connection did not, so deleting the
        swallow around `close` left the suite green while a dead connection could still turn a
        stored dump into a recorded failure.
        """
        manifest = await _run_backup(tmp_path, _RecordingConnection(close_fails=True))

        assert [d.database for d in manifest.databases] == [_DATABASE]
        assert manifest.is_complete
        assert not manifest.failed_databases


class TestOnlyWhatWasOpenedIsTornDown:
    async def test_a_degraded_path_does_not_roll_back_twice(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        """the export handler already ended the transaction, so the teardown must not repeat it.

        A second ROLLBACK lands outside any transaction. Postgres answers with a warning rather
        than an error, so it costs nothing but noise -- on exactly the degraded path where the
        logs are being read closely.
        """
        connection = _RecordingConnection(export_fails=True)
        await _run_backup(tmp_path, connection)

        assert connection.log.count("ROLLBACK") == 1

    async def test_the_hold_is_protected_before_the_snapshot_is_exported(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        """the session is deliberately idle for the dump, which is what a reaper kills."""
        connection = _RecordingConnection()
        await _run_backup(tmp_path, connection)

        begin = connection.log.index("BEGIN ISOLATION LEVEL REPEATABLE READ")
        hold = connection.log.index("SET LOCAL idle_in_transaction_session_timeout = 0")
        export = connection.log.index("SELECT pg_export_snapshot()")
        assert begin < hold < export


class TestAMissingIdIsNotAnId:
    """`str(None)` is "None" -- a perfectly truthy id that the dump tool would be handed.

    The synchronized/unsynchronized decision is reconstructed from `snapshot is not None`, so
    anything that manufactures a non-None value out of a missing one reports a set as verified
    exactly when nothing was synchronized at all.
    """

    async def test_a_null_snapshot_id_takes_the_unsynchronized_path(
        self, tmp_path: Any, dump_argv: list[list[str]]
    ) -> None:
        manifest = await _run_backup(tmp_path, _RecordingConnection(export_returns_none=True))

        assert [d.inventory_snapshot_consistent for d in manifest.databases] == [False]

    async def test_the_string_none_never_reaches_the_dump_tool(self, tmp_path: Any, dump_argv: list[list[str]]) -> None:
        await _run_backup(tmp_path, _RecordingConnection(export_returns_none=True))

        database_dump = next(argv for argv in dump_argv if "--dbname" in argv and "--globals-only" not in argv)
        assert not any(arg.startswith("--snapshot") for arg in database_dump)
