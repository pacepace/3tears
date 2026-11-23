"""Delta files: a table that opts in saves each commit's rows and deletes as one file, before the commit, or the commit
is refused; an orphan of a commit that never happened is swept; a file being written is never visible by its name."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pa_ipc = pytest.importorskip("pyarrow.ipc")

from threetears.core.collections.caller_transaction import CallerTransaction  # noqa: E402
from threetears.core.collections.delta_files import (  # noqa: E402
    _DELETED_COLUMN,
    _DELTA_FILES_DIR_ENV,
    _KEY_METADATA,
    _TABLE_METADATA,
    _WRITE_METADATA,
    DeltaFileRefusedError,
    TableDelta,
    _sweep_orphan_delta_files,
    write_delta_file,
)
from threetears.core.collections.registry import CollectionRegistry  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig  # noqa: E402
from threetears.core.entities.base import BaseEntity  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE  # noqa: E402

from packages.core.tests.durable_support import ConnOnlyPool  # noqa: E402


def _schema(name: str, *, opted_in: bool) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key=("office_key", "geo_id"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("geo_id", STRING_TYPE),
            Column("votes", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        l1_eviction=TEST_L1_TABLE,
        delta_files=opted_in,
    )


class _ResultRow(BaseEntity):
    """a result row, keyed within its race by its geography."""

    primary_key_field = "geo_id"


class _Conn:
    """an asyncpg-shaped connection whose transaction keeps its statements only when it commits; at its commit it
    notes which delta files are on disk, and a ``lost`` commit raises as a commit whose reply never came back."""

    def __init__(self, root: Path | None = None, *, lost: bool = False) -> None:
        self.root = root
        self.lost = lost
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.committed: list[tuple[str, tuple[Any, ...]]] = []
        self.files_at_commit: list[str] = []
        self.rolled_back = False

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        try:
            yield
        except BaseException:
            self.rolled_back = True
            raise
        if self.root is not None:
            self.files_at_commit = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*.arrow"))
        if self.lost:
            raise ConnectionError("the commit's reply never came back")
        self.committed = list(self.executed)

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"


def _collection(name: str = "results", *, opted_in: bool = True) -> Any:
    collection = collection_for_schema(_schema(name, opted_in=opted_in), entity_class=_ResultRow)(
        CollectionRegistry(), DefaultCoreConfig(), None
    )
    collection.l3_pool = ConnOnlyPool()
    return collection


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"office_key": "sn_NC1", "geo_id": f"{i:05d}", "votes": i} for i in range(count)]


def _read(root: Path, name: str) -> Any:
    with pa.memory_map(str(root / name)) as source:
        return pa_ipc.open_file(source).read_all()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(_DELTA_FILES_DIR_ENV, str(tmp_path))
    return tmp_path


async def test_an_opted_in_write_saves_one_file_a_table_with_its_rows_and_delete_markers_before_the_commit(
    root: Path,
) -> None:
    """Flips if a table that opts in commits without one file of its own holding every row written, in order, and a
    delete marker for each key deleted, saved before the commit, or if its metadata stops naming the table, the write
    and the key."""
    results, totals, conn = _collection(), _collection("totals"), _Conn(root)
    async with CallerTransaction(conn) as transaction:
        await results.save_rows(_rows(3), conn=conn)
        await results.delete_rows([("sn_NC1", "00009")], conn=conn)
        await totals.save_rows(_rows(1), conn=conn)
    files = transaction.delta_files
    assert set(files) == {"results", "totals"}
    write_id = Path(files["results"]).stem
    assert files == {"results": f"results/{write_id}.arrow", "totals": f"totals/{write_id}.arrow"}
    # saved before the commit, under the name the epoch record will hold
    assert conn.files_at_commit == sorted(files.values())
    assert conn.committed
    table = _read(root, files["results"])
    assert table.column_names == ["office_key", "geo_id", "votes", "date_created", "date_updated", _DELETED_COLUMN]
    rows = table.to_pylist()
    assert [(r["geo_id"], r["votes"], r[_DELETED_COLUMN]) for r in rows] == [
        ("00000", 0, False),
        ("00001", 1, False),
        ("00002", 2, False),
        ("00009", None, True),
    ]
    assert rows[0]["date_updated"] is not None and rows[3]["date_updated"] is None
    metadata = table.schema.metadata
    assert metadata[_TABLE_METADATA] == b"results"
    assert metadata[_WRITE_METADATA] == write_id.encode()
    assert metadata[_KEY_METADATA] == b'["office_key", "geo_id"]'
    assert _read(root, files["totals"]).num_rows == 1


async def test_a_table_not_opted_in_saves_no_file(root: Path) -> None:
    """Flips if a table that does not opt in gets a delta file, or its write changes at all for the opt-in."""
    collection, conn = _collection(opted_in=False), _Conn(root)
    async with CallerTransaction(conn) as transaction:
        await collection.save_rows(_rows(2), conn=conn)
        await collection.delete_rows([("sn_NC1", "00000")], conn=conn)
    assert transaction.delta_files == {}
    assert list(root.iterdir()) == []
    assert conn.committed


async def test_a_write_with_no_delta_files_directory_is_refused_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a table that opts in commits while no file can be saved for it (no directory configured)."""
    monkeypatch.delenv(_DELTA_FILES_DIR_ENV, raising=False)
    collection, conn = _collection(), _Conn()
    with pytest.raises(DeltaFileRefusedError, match="no delta files directory"):
        async with CallerTransaction(conn):
            await collection.save_rows(_rows(2), conn=conn)
    assert conn.rolled_back
    assert conn.committed == []


async def test_a_key_written_without_its_rows_recorded_is_refused_and_rolled_back(root: Path) -> None:
    """Flips if the transaction commits a key of an opted-in table that no write recorded for its file: the check is
    the transaction's, so a write path that forgets to record is refused, not trusted."""
    collection, conn = _collection(), _Conn(root)
    with pytest.raises(DeltaFileRefusedError, match="wrote 3 keys"):
        async with CallerTransaction(conn) as transaction:
            await collection.save_rows(_rows(2), conn=conn)
            # a write that enrolled its key and recorded nothing for the file
            transaction.enroll(collection, ("sn_NC1", "00007"))
    assert conn.rolled_back
    assert conn.committed == []
    assert list(root.rglob("*.arrow")) == []


async def test_writes_that_cannot_record_their_rows_are_refused_on_an_opted_in_table(root: Path) -> None:
    """Flips if a write path that cannot put its rows in a file (no caller's transaction to save it before, a store
    that copies files itself, a statement of the caller's own, a key-prefix delete) is let through on an opted-in
    table."""
    collection = _collection()
    with pytest.raises(DeltaFileRefusedError):
        await collection.save_to_store({"office_key": "sn_NC1", "geo_id": "00000", "votes": 1})
    with pytest.raises(DeltaFileRefusedError):
        await collection.delete_from_store(("sn_NC1", "00000"))
    with pytest.raises(DeltaFileRefusedError):
        await collection.load_files([], columns=["office_key", "geo_id", "votes"])
    with pytest.raises(DeltaFileRefusedError):
        async with collection.bypassing_write(("sn_NC1", "00000")):
            pass
    conn = _Conn(root)
    with pytest.raises(DeltaFileRefusedError, match="key-prefix delete"):
        async with CallerTransaction(conn) as transaction:
            transaction.enroll_prefixes(collection, [{"office_key": "sn_NC1"}])
    assert conn.rolled_back


async def test_a_crash_before_the_commit_leaves_an_orphan_the_sweep_removes_and_after_it_leaves_the_file(
    root: Path,
) -> None:
    """Flips if a file saved for a commit that never happened survives the sweep, if a committed version's file does
    not survive it, or if the sweep removes a file of the writer's work still in flight."""
    collection = _collection()
    lost = _Conn(root, lost=True)
    with pytest.raises(ConnectionError):
        async with CallerTransaction(lost):
            await collection.save_rows(_rows(1), conn=lost)
    [orphan] = lost.files_at_commit
    committed = _Conn(root)
    async with CallerTransaction(committed) as transaction:
        await collection.save_rows(_rows(1), conn=committed)
    kept = transaction.delta_files["results"]
    # the writer's work in flight began before both: nothing is swept yet
    assert _sweep_orphan_delta_files(root, named={kept}, before=0.0) == []
    assert _sweep_orphan_delta_files(root, named={kept}, before=time.time() + 1) == [orphan]
    assert not (root / orphan).exists()
    assert (root / kept).exists()


def test_the_rename_keeps_a_partly_written_file_invisible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a file can be seen under its own name before it is whole, if a write that fails leaves anything under
    that name, or if the sweep leaves the temporary file of a writer that died before its rename."""
    delta = TableDelta(name="results", types={"geo_id": pa.string()}, key=("geo_id",))
    delta.rows([{"geo_id": "00001"}])
    seen: list[tuple[bool, bool]] = []
    real_replace = os.replace

    def replace(source: Any, target: Any) -> None:
        # the moment before the rename: the file is whole under its temporary name, absent under its own
        seen.append((Path(source).exists(), Path(target).exists()))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    name = write_delta_file(tmp_path, delta, "w1")
    assert seen == [(True, False)]
    assert (tmp_path / name).exists()

    def fails(source: Any, target: Any) -> None:
        raise OSError("the volume went away")

    monkeypatch.setattr(os, "replace", fails)
    with pytest.raises(OSError):
        write_delta_file(tmp_path, delta, "w2")
    assert not (tmp_path / "results" / "w2.arrow").exists()
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["w1.arrow"]
    # a writer that died mid-write leaves its temporary name, which no record holds; the sweep removes it
    (tmp_path / "results" / ".w3.partial").write_bytes(b"ARROW1 half")
    assert _sweep_orphan_delta_files(tmp_path, named={name}, before=time.time() + 1) == ["results/.w3.partial"]


def test_rows_handed_as_arrow_are_typed_as_the_rail_answers_them() -> None:
    """Flips if an Arrow write's columns reach the file in the caller's types rather than the table's (a copy reading
    the file would stage other types than it reads from L3), or a column the write lacks is not null."""
    delta = TableDelta(
        name="results",
        types={"geo_id": pa.string(), "votes": pa.int64(), "share": pa.string(), "note": pa.string()},
        key=("geo_id",),
    )
    delta.arrow(
        pa.table(
            {
                "geo_id": pa.array(["a"], pa.large_string()),
                "votes": pa.array([7], pa.int32()),
                "share": pa.array([Decimal("0.25")], pa.decimal128(4, 2)),
            }
        )
    )
    table = delta.table("w")
    assert table.schema.field("votes").type == pa.int64()
    assert table.to_pylist() == [{"geo_id": "a", "votes": 7, "share": "0.25", "note": None, _DELETED_COLUMN: False}]


def test_a_table_opting_in_with_a_column_no_file_can_type_is_refused_where_declared() -> None:
    """Flips if an opted-in table whose column the rail does not answer as one Arrow type is accepted, to refuse every
    write later."""
    with pytest.raises(ValueError, match="payload"):
        TableSchema(
            name="blobs",
            primary_key="id",
            columns=[Column("id", STRING_TYPE), Column("payload", JSONB_TYPE)],
            l1_eviction=TEST_L1_TABLE,
            delta_files=True,
        )
    # NUMERIC is text on the rail, so it opts in
    TableSchema(
        name="shares",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("share", NUMERIC_TYPE, precision=6, scale=4)],
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
    )
