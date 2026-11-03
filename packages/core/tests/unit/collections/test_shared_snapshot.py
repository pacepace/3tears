"""A scoped snapshot's copy published as shared-memory Arrow, only the scopes that moved written again.

The contracts: a version holds exactly the rows of the read it was published from, every table; a version after a
change rewrites only the scopes whose epoch moved, every other scope's part being the last version's own file; a
scope the copy no longer holds is gone from the next version; a read naming a version present already writes
nothing; a store whose current version this publisher did not make is written whole.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402

from threetears.core.collections.scoped_snapshot import SnapshotTable, VersionedRead  # noqa: E402
from threetears.core.collections.shared_snapshot import SharedSnapshot  # noqa: E402
from threetears.core.shared_arrow import SharedArrowStore  # noqa: E402

TABLES = (
    SnapshotTable(name="results", scope_column="state", key=("race", "county")),
    SnapshotTable(name="races", scope_column="state", key=("race",)),
)


@pytest.fixture
def copy() -> Iterator[Any]:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE results (race VARCHAR, county VARCHAR, state VARCHAR, votes BIGINT)")
    connection.execute("CREATE TABLE races (race VARCHAR, state VARCHAR, name VARCHAR)")
    for state in ("VA", "GA", "TX"):
        _set_state(connection, state, 1)
    yield connection
    connection.close()


def _set_state(connection: Any, state: str, votes: int) -> None:
    connection.execute("DELETE FROM results WHERE state = ?", [state])
    connection.execute("DELETE FROM races WHERE state = ?", [state])
    connection.execute(
        "INSERT INTO results SELECT 'r' || (i % 3), 'c' || i, ?, ? * i FROM range(1000) t(i)", [state, votes]
    )
    connection.execute("INSERT INTO races SELECT 'r' || i, ?, 'race ' || i FROM range(3) t(i)", [state])


def _read(connection: Any, epochs: dict[str, int]) -> VersionedRead:
    return VersionedRead(cursor=connection.cursor(), epochs=MappingProxyType(epochs))


def _rows(cursor: Any, table: str) -> list[tuple[object, ...]]:
    return sorted(cursor.execute(f'SELECT * FROM "{table}"').fetchall())


def _inode(store: SharedArrowStore, version: str, table: str, scope: str) -> int:
    # a plain scope names its part as it is
    return (store.root / "versions" / version / f"{table}@{scope}.arrow").stat().st_ino


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=256 << 20)


class TestPublishingTheCopy:
    def test_a_version_holds_exactly_the_reads_rows(self, store: SharedArrowStore, copy: Any) -> None:
        shared = SharedSnapshot(store, TABLES)
        version = shared.publish(_read(copy, {"VA": 1, "GA": 1, "TX": 1}))
        assert version.tables == ("races", "results")
        reader = store.reader()
        with reader.read() as read:
            for table in ("results", "races"):
                assert _rows(read.cursor, table) == _rows(copy, table)
        reader.close()

    def test_a_change_rewrites_only_the_scopes_that_moved(self, store: SharedArrowStore, copy: Any) -> None:
        shared = SharedSnapshot(store, TABLES)
        first = shared.publish(_read(copy, {"VA": 1, "GA": 1, "TX": 1}))
        reader = store.reader()
        with reader.read():
            _set_state(copy, "GA", 7)
            copy.execute("DELETE FROM results WHERE state = 'TX'")
            copy.execute("DELETE FROM races WHERE state = 'TX'")
            second = shared.publish(_read(copy, {"VA": 1, "GA": 2}))
            for table in ("results", "races"):
                assert _inode(store, first.name, table, "VA") == _inode(store, second.name, table, "VA")
                assert _inode(store, first.name, table, "GA") != _inode(store, second.name, table, "GA")
                assert not (store.root / "versions" / second.name / f"{table}@TX.arrow").exists()
        assert shared.scopes_written == 3 + 1
        with reader.read() as read:
            assert read.version == second.name
            for table in ("results", "races"):
                assert _rows(read.cursor, table) == _rows(copy, table)
        reader.close()
        assert store.versions() == (second.name,)

    def test_a_version_present_already_writes_nothing(self, store: SharedArrowStore, copy: Any) -> None:
        shared = SharedSnapshot(store, TABLES)
        first = shared.publish(_read(copy, {"VA": 1, "GA": 1, "TX": 1}))
        again = shared.publish(_read(copy, {"VA": 1, "GA": 1, "TX": 1}))
        assert again == first
        assert shared.published == 1

    def test_a_store_moved_by_another_publisher_is_written_whole(self, store: SharedArrowStore, copy: Any) -> None:
        shared = SharedSnapshot(store, TABLES)
        shared.publish(_read(copy, {"VA": 1, "GA": 1, "TX": 1}))
        store.publish("other", {"x": pa.table({"a": [1]})})
        _set_state(copy, "GA", 3)
        shared.publish(_read(copy, {"VA": 1, "GA": 2, "TX": 1}))
        assert shared.scopes_written == 3 + 3, "no base this publisher made: every scope written"
        reader = store.reader()
        with reader.read() as read:
            assert set(read.tables) == {"results", "races"}
            assert _rows(read.cursor, "results") == _rows(copy, "results")
        reader.close()

    def test_extra_tables_ride_in_every_version_and_names_follow_the_caller(
        self, store: SharedArrowStore, copy: Any
    ) -> None:
        shared = SharedSnapshot(
            store,
            TABLES,
            version_name=lambda read: "v-" + "-".join(f"{s}{e}" for s, e in sorted(read.epochs.items())),
            extra_tables=lambda read: {
                "epochs": pa.table({"scope": list(read.epochs), "epoch": list(read.epochs.values())})
            },
        )
        version = shared.publish(_read(copy, {"VA": 1, "GA": 1}))
        assert version.name == "v-GA1-VA1"
        reader = store.reader()
        with reader.read() as read:
            assert sorted(read.cursor.execute("SELECT scope, epoch FROM epochs").fetchall()) == [("GA", 1), ("VA", 1)]
        reader.close()

    def test_a_scope_no_epoch_names_is_written_every_time_and_dropped_once_gone(
        self, store: SharedArrowStore, copy: Any
    ) -> None:
        shared = SharedSnapshot(store, TABLES)
        first = shared.publish(_read(copy, {"VA": 1, "GA": 1}))
        reader = store.reader()
        with reader.read():
            second = shared.publish(_read(copy, {"VA": 1, "GA": 2}))
            assert _inode(store, first.name, "results", "VA") == _inode(store, second.name, "results", "VA")
            assert _inode(store, first.name, "results", "TX") != _inode(store, second.name, "results", "TX"), (
                "TX has no epoch to prove it unchanged"
            )
        copy.execute("DELETE FROM results WHERE state = 'TX'")
        third = shared.publish(_read(copy, {"VA": 1, "GA": 3}))
        assert not (store.root / "versions" / third.name / "results@TX.arrow").exists()
        assert (store.root / "versions" / third.name / "races@TX.arrow").exists(), "races still holds TX"
        with reader.read() as read:
            for table in ("results", "races"):
                assert _rows(read.cursor, table) == _rows(copy, table)
        reader.close()

    def test_a_scope_named_like_a_reserved_name_publishes(self, store: SharedArrowStore) -> None:
        connection = duckdb.connect(":memory:")
        connection.execute("CREATE TABLE results (race VARCHAR, county VARCHAR, state VARCHAR, votes BIGINT)")
        connection.execute("CREATE TABLE races (race VARCHAR, state VARCHAR, name VARCHAR)")
        for state in ("CURRENT", "x" * 200):
            _set_state(connection, state, 1)
        version = SharedSnapshot(store, TABLES).publish(_read(connection, {"CURRENT": 1, "x" * 200: 1}))
        reader = store.reader()
        with reader.read() as read:
            assert _rows(read.cursor, "results") == _rows(connection, "results")
        reader.close()
        assert version.tables == ("races", "results")

    @pytest.mark.parametrize(
        ("scope", "plain"),
        [("VA", True), ("new york", False), ("h-x", False), ("CURRENT", False), ("x" * 129, False)],
    )
    def test_a_scope_names_its_part_as_it_is_only_when_plain(
        self, store: SharedArrowStore, scope: str, plain: bool
    ) -> None:
        connection = duckdb.connect(":memory:")
        connection.execute("CREATE TABLE results (race VARCHAR, county VARCHAR, state VARCHAR, votes BIGINT)")
        connection.execute("CREATE TABLE races (race VARCHAR, state VARCHAR, name VARCHAR)")
        _set_state(connection, scope, 1)
        version = SharedSnapshot(store, TABLES).publish(_read(connection, {scope: 1}))
        parts = {path.name for path in version.path.iterdir() if path.name.startswith("results@")}
        assert (f"results@{scope}.arrow" in parts) is plain
        assert len(parts) == 2, "the schema file and the scope's one part"
