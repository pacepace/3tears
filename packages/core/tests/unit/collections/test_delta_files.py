"""Delta files: a table that opts in saves each commit's rows and deletes as one file, before the commit, or the commit
is refused; an orphan of a commit that never happened is swept; a file being written is never visible by its name."""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import UTC, datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest

pa = pytest.importorskip("pyarrow")
pa_ipc = pytest.importorskip("pyarrow.ipc")

from prometheus_client import REGISTRY  # noqa: E402

from threetears.core.collections import delta_files  # noqa: E402
from threetears.core.collections.caller_transaction import CallerTransaction, save_delta_files  # noqa: E402
from threetears.core.backends.schema_sql import (  # noqa: E402
    build_bulk_insert_sql,
    build_insert_sql,
    bulk_update_columns,
    insert_columns_for_data,
    upsert_set_columns,
)
from threetears.core.collections.delta_files import (  # noqa: E402
    DeltaChange,
    DeltaFileRefusedError,
    DeltaFileUnreadableError,
    LoadDelta,
    TableDelta,
    read_volume_file,
    read_delta_file_keys,
    OWN_DELTA_SAVES,
    sweep_orphan_delta_files,
    write_delta_file,
)
from threetears.core.testing.delta_files import apply_delta_row, read_delta_file
from threetears.core.collections.registry import CollectionRegistry  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
    l2_order_columns,
)
from threetears.core.config import DefaultCoreConfig  # noqa: E402
from threetears.core.entities.base import BaseEntity  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE  # noqa: E402

from packages.core.tests.durable_support import ConnOnlyPool  # noqa: E402


_OnConflict = Literal["raise", "ignore", "update"]


def _schema(name: str, *, opted_in: bool, on_conflict: _OnConflict = "update") -> TableSchema:
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
        on_conflict=on_conflict,
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


def _collection(name: str = "results", *, opted_in: bool = True, on_conflict: _OnConflict = "update") -> Any:
    collection = collection_for_schema(
        _schema(name, opted_in=opted_in, on_conflict=on_conflict), entity_class=_ResultRow
    )(CollectionRegistry(), DefaultCoreConfig(), None)
    collection.l3_pool = ConnOnlyPool()
    return collection


def _large_rows(count: int) -> Any:
    """``count`` rows of one office, each with a 60-character ``geo_id``.

    :param count: the rows
    :ptype count: int
    :return: the rows
    :rtype: pyarrow.Table
    """
    return pa.table(
        {
            "office_key": pa.array(["sn_NC1"] * count),
            "geo_id": pa.array([f"{i:060d}" for i in range(count)]),
            "votes": pa.array(range(count), pa.int32()),
        }
    )


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"office_key": "sn_NC1", "geo_id": f"{i:05d}", "votes": i} for i in range(count)]


#: the variable naming the delta files' directory, a pod's deploy config
_DIR_ENV = "THREETEARS_DELTA_FILES_DIR"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(_DIR_ENV, str(tmp_path))
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
    saved = read_delta_file(root / files["results"])
    assert [list(change.row) for change in saved.changes] == [
        ["office_key", "geo_id", "votes", "date_created", "date_updated"]
    ] * 4
    assert [(c.row["geo_id"], c.row["votes"], c.deleted) for c in saved.changes] == [
        ("00000", 0, False),
        ("00001", 1, False),
        ("00002", 2, False),
        ("00009", None, True),
    ]
    assert saved.changes[0].row["date_updated"] is not None and saved.changes[3].row["date_updated"] is None
    assert (saved.table, saved.write, saved.key) == ("results", write_id, ("office_key", "geo_id"))
    assert len(read_delta_file(root / files["totals"]).changes) == 1


def _saves(where: str) -> float:
    return REGISTRY.get_sample_value("delta_file_saves_total", {"where": where}) or 0.0


def _save_seconds(where: str) -> tuple[float, float]:
    """the save seconds histogram's count and sum for ``where``."""
    labels = {"where": where}
    count = REGISTRY.get_sample_value("delta_file_save_seconds_count", labels) or 0.0
    return count, REGISTRY.get_sample_value("delta_file_save_seconds_sum", labels) or 0.0


async def test_a_write_s_delta_files_are_built_and_saved_in_a_worker_process(
    root: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a write's delta file is built, compressed, hashed or written in the serving process (on its thread, as
    it was: the serving process's own file writer refuses here), or reads back other than what was written: a
    transaction's small file crosses to the worker as Arrow IPC bytes, and a load's large part (past the inline bound)
    through the shared-memory hand-off, released after."""

    def refused(*_: Any, **__: Any) -> None:
        raise AssertionError("a delta file was written in the serving process")

    monkeypatch.setattr(delta_files, "write_volume_file", refused)
    handoff = tmp_path_factory.mktemp("handoff")
    monkeypatch.setenv("THREETEARS_DELTA_HANDOFF_DIR", str(handoff))
    collection, conn = _collection(), _Conn(root)
    before = _saves("worker")
    timed, spent = _save_seconds("worker")
    async with CallerTransaction(conn) as transaction:
        await collection.save_rows(_rows(3), conn=conn)
    small = read_delta_file(root / transaction.delta_files["results"], table="results")
    assert [change.row["geo_id"] for change in small.changes] == ["00000", "00001", "00002"]
    # a load's rows past the inline bound: 200,000 rows of a 60-character text, about 13 MiB, built and read back on
    # a thread: a quarter second of the test's own Python held the loop past its bound (markpace-mbp, 2026-11-25)
    count = 200_000
    rows = await asyncio.to_thread(_large_rows, count)
    assert rows.nbytes > 8 << 20, "past the inline bound (8 MiB)"
    load = collection.table_delta()
    load.load([rows], [count], {})
    files = await save_delta_files({"results": load}, version=None)
    held = await asyncio.to_thread(read_volume_file, root / files["results"])
    assert held.num_rows == count and held.column("geo_id")[count - 1].as_py() == f"{count - 1:060d}"
    assert _saves("worker") == before + 2
    # each save's seconds observed as it completes (delta_file_save_seconds{where="worker"})
    assert _save_seconds("worker")[0] == timed + 2 and _save_seconds("worker")[1] > spent
    assert not any(handoff.rglob("*.arrow")), "the hand-off is released once the worker is done"


async def test_a_load_whose_constant_is_a_timestamp_is_saved_in_the_worker_with_it(root: Path) -> None:
    """Flips if a load's constants cross to the worker as Python values: a ``datetime`` constant (an LSF run's
    timestamp) is not plain data, so the worker refused the save and every such load from files failed (2026-11-24).
    Saved, each row carries the constant as the column's instant."""
    at = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)
    rows = pa.table({"office_key": pa.array(["sn_NC1", "sn_NC2"]), "geo_id": pa.array(["00000", "00001"])})
    load = TableDelta(
        name="results",
        types={"office_key": pa.string(), "geo_id": pa.string(), "loaded_at": pa.timestamp("us", tz="UTC")},
        key=("office_key", "geo_id"),
    )
    load.load([rows], [2], {"loaded_at": at})
    files = await save_delta_files({"results": load}, version=None)
    held = read_volume_file(root / files["results"])
    assert held.column("loaded_at").to_pylist() == [at, at]


async def test_a_transaction_fenced_at_a_write_names_its_files_by_that_write_s_version(root: Path) -> None:
    """Flips if a transaction whose data writes took their turn at a write's version (``ScopeEpochs.touch`` notes it,
    decision 53) saves its files under any other name than ``v<version>-<unique>``, or if a transaction fenced at two
    writes' versions is let save files named by either."""
    results, conn = _collection(), _Conn(root)
    async with CallerTransaction(conn) as transaction:
        transaction.note_write_version(12)
        transaction.note_write_version(12)
        await results.save_rows(_rows(2), conn=conn)
    (file,) = transaction.delta_files.values()
    assert delta_files.delta_file_version(file) == 12 and file.startswith("results/v12-"), file
    with pytest.raises(ValueError, match="one write's version"):
        async with CallerTransaction(_Conn(root)) as other:
            other.note_write_version(12)
            other.note_write_version(13)


async def test_a_save_under_a_write_s_version_is_named_by_it_and_an_older_name_carries_none(root: Path) -> None:
    """Flips if a save given its write's version (``ScopeEpochs.begin``'s, decision 53) is not named ``v<version>-...``,
    or the name does not read back as that version; or if a file of the older, unversioned name (a bare id, a
    recovery's) reads as having one, which would let a full claim to cover it."""
    rows = pa.table({"office_key": pa.array(["sn_NC1"]), "geo_id": pa.array(["00000"])})
    load = TableDelta(
        name="results", types={"office_key": pa.string(), "geo_id": pa.string()}, key=("office_key", "geo_id")
    )
    load.load([rows], [1], {})
    files = await save_delta_files({"results": load}, version=7)
    assert re.fullmatch(r"results/v7-[0-9a-f]{32}\.arrow", files["results"]), files
    assert delta_files.delta_file_version(files["results"]) == 7
    assert read_volume_file(root / files["results"]).num_rows == 1, "read back by the name its record holds"
    for older in (
        "results/0190a1b2c3d4e5f60718293a4b5c6d7e.arrow",
        "results/recovery-0190a1b2.arrow",
        "results/w0001.arrow",
    ):
        assert delta_files.delta_file_version(older) is None
    assert delta_files.versioned_write_id(12, "recovery-ab") == "v12-recovery-ab"
    with pytest.raises(ValueError):
        delta_files.versioned_write_id(-1, "ab")


async def test_without_a_hand_off_directory_a_large_part_is_saved_on_a_thread_counted(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a save with a part too large to cross inline is refused (or silently saved elsewhere) when no
    shared-memory hand-off directory is named: it is saved on a thread of the serving process as before, counted."""
    monkeypatch.delenv("THREETEARS_DELTA_HANDOFF_DIR", raising=False)
    count = 200_000
    rows = pa.table(
        {"office_key": pa.array(["sn_NC1"] * count), "geo_id": pa.array([f"{i:060d}" for i in range(count)])}
    )
    load = _collection().table_delta()
    load.load([rows], [count], {})
    before = _saves("thread")
    timed, spent = _save_seconds("thread")
    files = await save_delta_files({"results": load}, version=None)
    assert read_volume_file(root / files["results"]).num_rows == count
    assert _saves("thread") == before + 1
    assert _save_seconds("thread")[0] == timed + 1 and _save_seconds("thread")[1] > spent, "its seconds observed too"


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
    monkeypatch.delenv(_DIR_ENV, raising=False)
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
    that copies files itself with no rows given for its file, a statement of the caller's own, a key-prefix delete
    whose prefixes no write recorded) is let through on an opted-in table."""
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
    with pytest.raises(DeltaFileRefusedError, match="deleted 1 key prefixes .* recorded 0"):
        async with CallerTransaction(conn) as transaction:
            transaction.enroll_prefixes(collection, [{"office_key": "sn_NC1"}])
    assert conn.rolled_back


async def test_a_crash_before_the_commit_leaves_an_orphan_the_sweep_removes_and_after_it_leaves_the_file(
    root: Path,
) -> None:
    """Flips if a file saved for a commit that never happened survives the sweep, if a committed version's file does
    not survive it, or if the sweep removes a file of the writer's work still in flight or a file another process
    saved: it goes by the names this process saved and how each commit ended, never by a file's time on the volume
    (whose clock is not the writer's)."""
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
    assert orphan in OWN_DELTA_SAVES.orphans and kept not in OWN_DELTA_SAVES.orphans
    assert kept not in OWN_DELTA_SAVES.in_flight, "a commit that landed leaves the registry"
    # this process's commit still open, its file saved; and another process's file, which this process never saved
    [in_flight] = OWN_DELTA_SAVES.saving(["results"], "inflight")
    (root / in_flight).write_bytes(b"saved, its commit open")
    (root / "results" / "other.arrow").write_bytes(b"another writer's")
    past = time.time() - 3600  # files the volume stamps long ago: a clock compare would take every one
    for name in (in_flight, "results/other.arrow", orphan, kept):
        os.utime(root / name, (past, past))
    swept = sweep_orphan_delta_files(root, orphans=OWN_DELTA_SAVES.orphans, named={kept}, retention="delete")
    OWN_DELTA_SAVES.swept(swept)
    OWN_DELTA_SAVES.settled([in_flight], landed=True)
    assert orphan in swept and not (root / orphan).exists()
    assert (root / kept).exists() and (root / in_flight).exists() and (root / "results" / "other.arrow").exists()
    assert orphan not in OWN_DELTA_SAVES.orphans


def test_the_rename_keeps_a_partly_written_file_invisible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a file can be seen under its own name before it is whole, if a write that fails leaves anything under
    that name, or if the sweep leaves the temporary file of a writer that died before its rename."""
    delta = TableDelta(name="results", types={"geo_id": pa.string()}, key=("geo_id",))
    delta.rows([{"geo_id": "00001"}], [])
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
    # a write that died mid-write (its worker killed) leaves its temporary name, which no record holds; once its
    # commit is known not to have landed the sweep removes it by name
    (tmp_path / "results" / ".w3.partial").write_bytes(b"ARROW1 half")
    [w3] = OWN_DELTA_SAVES.saving(["results"], "w3")
    OWN_DELTA_SAVES.settled([w3], landed=False)
    assert sweep_orphan_delta_files(tmp_path, orphans=[w3], named={name}, retention="delete") == ["results/w3.arrow"]
    OWN_DELTA_SAVES.swept([w3])
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["w1.arrow"]


def test_rows_handed_as_arrow_are_typed_as_the_rail_answers_them(tmp_path: Path) -> None:
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
        ),
        ["votes", "share"],
    )
    table = delta.table("w")
    assert table.schema.field("votes").type == pa.int64()
    [change] = read_delta_file(tmp_path / write_delta_file(tmp_path, delta, "w")).changes
    assert change == DeltaChange(
        row={"geo_id": "a", "votes": 7, "share": "0.25", "note": None},
        deleted=False,
        written=frozenset({"votes", "share"}),
    )


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


#: the zstd frame's magic number, which starts every compressed buffer of an IPC file written with zstd inside
_ZSTD_MAGIC = bytes.fromhex("28b52ffd")


def test_a_file_is_arrow_ipc_with_zstd_inside_and_reads_back_the_same_rows(tmp_path: Path) -> None:
    """Flips if a delta file is written plain rather than with zstd inside its IPC buffers (decision 18), if it stops
    reading back as the rows written."""
    delta = TableDelta(name="results", types={"geo_id": pa.string(), "votes": pa.int64()}, key=("geo_id",))
    rows = [{"geo_id": f"{i:05d}", "votes": i % 7} for i in range(2_000)]
    delta.rows(rows, ["votes"])
    name = write_delta_file(tmp_path, delta, "w1")
    raw = (tmp_path / name).read_bytes()
    assert raw.startswith(b"ARROW1")
    assert _ZSTD_MAGIC in raw
    plain = pa.BufferOutputStream()
    with pa_ipc.new_file(plain, delta.table("w1").schema) as writer:
        writer.write_table(delta.table("w1"))
    assert len(raw) < plain.getvalue().size / 2
    assert [change.row for change in read_delta_file(tmp_path / name).changes] == rows


def _set_list(sql: str) -> list[str]:
    """the columns an upsert statement's ``DO UPDATE SET`` takes from the write (``c = EXCLUDED.c``); a column set to
    its own held value (the clause a row naming no mutable column gets) changes nothing, so it is not one."""
    if "DO UPDATE SET" not in sql:
        return []
    parts = [part.split(" = ") for part in sql.split("DO UPDATE SET ", 1)[1].split(", ")]
    return [column.strip() for column, value in parts if value.strip().startswith("EXCLUDED.")]


async def test_each_row_records_the_columns_its_statement_set_as_the_statement_names_them(root: Path) -> None:
    """Flips if a row's recorded columns differ from the ``DO UPDATE SET`` list L3 ran for it: a single-row upsert
    names only the columns its row names (a nullable column left out is not set, ``date_created`` never is), a bulk
    statement every mutable column it emits, a delete marker none."""
    collection, conn = _collection(), _Conn(root)
    stamp = datetime(2026, 11, 23, tzinfo=UTC)
    single = {"office_key": "sn_NC1", "geo_id": "00001", "date_created": stamp, "date_updated": stamp}
    async with CallerTransaction(conn) as transaction:
        await collection.save_to_store(single, conn=conn)
        transaction.enroll(collection, ("sn_NC1", "00001"))
        await collection.save_rows(_rows(2), conn=conn)
        await collection.delete_rows([("sn_NC1", "00009")], conn=conn)
    changes = read_delta_file(root / transaction.delta_files["results"]).changes
    recorded = [change.written for change in changes]
    [single_sql, bulk_sql, *_] = [sql for sql, _ in conn.committed]
    assert recorded[0] == frozenset(_set_list(single_sql)) == frozenset({"date_updated"})
    assert recorded[1] == recorded[2] == frozenset(_set_list(bulk_sql)) == frozenset({"votes", "date_updated"})
    assert recorded[3] == frozenset()
    # the nullable column the single row left out shows null in the file; the set says it was not written
    assert changes[0].row["votes"] is None


def test_the_recorded_sets_are_the_statements_own() -> None:
    """Flips if the columns a delta records for a write stop being the ones the statement 3tears generates sets: the
    single-row upsert, the bulk statement, a table that ignores conflicts (none) and one that raises on them (none)."""
    schema = _schema("results", opted_in=True)
    for data in (
        {"office_key": "a", "geo_id": "b"},
        {"office_key": "a", "geo_id": "b", "votes": 1},
        {"office_key": "a", "geo_id": "b", "votes": None, "date_created": "x", "date_updated": "y"},
    ):
        assert upsert_set_columns(schema, data) == _set_list(build_insert_sql(schema, data))
    emitted = insert_columns_for_data(schema, {"office_key": "a", "geo_id": "b"})
    assert bulk_update_columns(schema, emitted) == _set_list(build_bulk_insert_sql(schema, rows=1, columns=emitted))
    conflicts: tuple[_OnConflict, ...] = ("ignore", "raise")
    for on_conflict in conflicts:
        assert upsert_set_columns(_schema("results", opted_in=True, on_conflict=on_conflict), {"votes": 1}) == []


_HELD = {
    "office_key": "sn_NC1",
    "geo_id": "00001",
    "votes": 10,
    "date_created": "2026-11-01T00:00:00+00:00",
    "date_updated": "2026-11-01T00:00:00+00:00",
}


def _change(written: set[str], *, deleted: bool = False, **values: Any) -> DeltaChange:
    row = {"office_key": "sn_NC1", "geo_id": "00001", "votes": None, "date_created": None, "date_updated": None}
    return DeltaChange(row={**row, **values}, deleted=deleted, written=frozenset(written))


def test_a_delta_row_applies_to_the_held_row_as_l3_merged_the_upsert() -> None:
    """Flips if applying a delta row differs from the row L3 holds after the upsert: a column the write left out
    keeps its held value, ``date_created`` keeps the held row's, a column set takes the file's (null included), a
    write that set nothing leaves the row, a delete leaves no row, and a row L3 did not hold is the row inserted."""
    stamp = "2026-11-23T00:00:00+00:00"
    # a single-row upsert naming no votes: votes and date_created kept, date_updated taken
    left_out = _change({"date_updated"}, date_created=stamp, date_updated=stamp)
    assert apply_delta_row(_HELD, left_out) == {**_HELD, "date_updated": stamp}
    # a bulk upsert sets votes even to null, as its statement did
    bulk = _change({"votes", "date_updated"}, votes=None, date_created=stamp, date_updated=stamp)
    assert apply_delta_row(_HELD, bulk) == {**_HELD, "votes": None, "date_updated": stamp}
    # a table ignoring conflicts: the held row stands
    assert apply_delta_row(_HELD, _change(set(), votes=3, date_updated=stamp)) == _HELD
    assert apply_delta_row(_HELD, _change(set(), deleted=True)) is None
    assert apply_delta_row(None, left_out) == {
        "office_key": "sn_NC1",
        "geo_id": "00001",
        "votes": None,
        "date_created": stamp,
        "date_updated": stamp,
    }


def test_every_byte_flipped_in_a_delta_file_is_read_as_unreadable_never_as_other_rows(tmp_path: Path) -> None:
    """Flips if one byte flipped anywhere in a delta file reads as a valid file (decision 29 waits for and recreates
    only a file it can tell is damaged). The scan the decision-29 build ran (2026-11-23: a 2,000-row file, a byte
    flipped every 23 bytes) found about two in five flips read as a file of other values before the checksum; now each
    flip, in the IPC bytes or the trailer, is DeltaFileUnreadableError for the whole read, the keys' read and a part's
    read alike."""
    delta = TableDelta(
        name="results", types={"geo_id": pa.string(), "votes": pa.int64(), "note": pa.string()}, key=("geo_id",)
    )
    delta.rows([{"geo_id": f"{i:05d}", "votes": i * 37, "note": f"n{i % 13}"} for i in range(2_000)], ["votes", "note"])
    name = write_delta_file(tmp_path, delta, "w1")
    good = (tmp_path / name).read_bytes()
    probe = tmp_path / "results" / "w1.arrow"
    flips = valid = 0
    for offset in range(0, len(good), 23):
        flipped = bytearray(good)
        flipped[offset] ^= 0x5A
        probe.write_bytes(bytes(flipped))
        flips += 1
        for read in (read_delta_file, read_delta_file_keys, read_volume_file):
            try:
                read(probe)
            except DeltaFileUnreadableError:
                continue
            valid += 1
    assert flips > 100
    assert valid == 0, f"{valid} reads of {flips} flipped files read as valid"
    probe.write_bytes(good)
    assert len(read_delta_file(probe).changes) == 2_000


def test_a_reader_refuses_a_file_of_another_format(tmp_path: Path) -> None:
    """Flips if a file of another format version (the plain one with no written columns) is read as this one's, its
    written columns guessed."""
    path = tmp_path / "results.arrow"
    table = pa.table({"geo_id": ["a"]}).replace_schema_metadata({b"threetears.delta.format": b"1"})
    with pa_ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)
    with pytest.raises(ValueError, match="format"):
        read_delta_file(path)


def _defaulted_schema(**overrides: Any) -> TableSchema:
    """a result table whose status, flag and share take fixed database defaults when a write leaves them out."""
    columns = [
        Column("office_key", STRING_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("status", STRING_TYPE, server_default="'active'::text"),
        Column("flag", BOOL_TYPE, server_default="false"),
        Column("share", NUMERIC_TYPE, precision=6, scale=2, server_default="0.5"),
        Column("note", STRING_TYPE, nullable=True, server_default="NULL"),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
        *overrides.pop("extra", []),
    ]
    return TableSchema(
        name=overrides.pop("name", "defaulted"),
        primary_key=("office_key", "geo_id"),
        columns=columns,
        l1_eviction=TEST_L1_TABLE,
        delta_files=True,
        **overrides,
    )


async def test_a_fixed_default_lands_in_the_delta_row_of_a_new_row_and_a_held_row_keeps_its_value(root: Path) -> None:
    """Flips if a column whose database default is a fixed value, left out of a write, shows null in the delta file
    rather than the value the database gave a new row (decision 20), if it is counted among the columns the write set
    (a row L3 already held keeps its value, as the ``DO UPDATE`` that never names it keeps it), or if a column a row
    names as None is given the default (the ``INSERT`` sent the null)."""
    collection = collection_for_schema(_defaulted_schema(), entity_class=_ResultRow)(
        CollectionRegistry(), DefaultCoreConfig(), None
    )
    collection.l3_pool = ConnOnlyPool()
    conn = _Conn(root)
    stamp = datetime(2026, 11, 23, tzinfo=UTC)
    single = {"office_key": "sn_NC1", "geo_id": "00001", "date_created": stamp, "date_updated": stamp}
    async with CallerTransaction(conn) as transaction:
        await collection.save_to_store(single, conn=conn)
        transaction.enroll(collection, ("sn_NC1", "00001"))
        await collection.save_rows(_rows(1), conn=conn)
        await collection.save_rows([{**_rows(2)[1], "status": None, "flag": True}], conn=conn)
    # the statements left the defaulted columns out, so the database filled them in
    assert all("status" not in sql for sql, _ in conn.committed[:2])
    changes = read_delta_file(root / transaction.delta_files["defaulted"]).changes
    fixed = {"status": "active", "flag": False, "share": "0.50", "note": None}
    for change in changes[:2]:
        assert {name: change.row[name] for name in fixed} == fixed
        assert not set(fixed) & change.written
    assert (changes[2].row["status"], changes[2].row["flag"], changes[2].row["share"]) == (None, True, "0.50")
    # a new row: the copy holds no row for the key, as the INSERT found none, and takes the default
    inserted = apply_delta_row(None, changes[0])
    assert inserted is not None and inserted["status"] == "active"
    # a row L3 held: the column the write left out keeps the held value
    held = {**changes[0].row, "status": "closed", "flag": True, "share": "0.75"}
    merged = apply_delta_row(held, changes[0])
    assert merged is not None and (merged["status"], merged["flag"], merged["share"]) == ("closed", True, "0.75")


def test_a_fixed_default_lands_in_rows_handed_as_arrow(tmp_path: Path) -> None:
    """Flips if an Arrow write that lacks a column with a fixed database default (its ``INSERT`` leaves the column
    out) shows null in the file rather than the default."""
    delta = TableDelta(
        name="defaulted",
        types={"geo_id": pa.string(), "status": pa.string(), "flag": pa.bool_()},
        key=("geo_id",),
        defaults={"status": "active", "flag": False},
    )
    delta.arrow(pa.table({"geo_id": ["a", "b"], "flag": [True, None]}), ["flag"])
    changes = read_delta_file(tmp_path / write_delta_file(tmp_path, delta, "w")).changes
    assert [change.row for change in changes] == [
        {"geo_id": "a", "status": "active", "flag": True},
        {"geo_id": "b", "status": "active", "flag": None},
    ]


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (Column("stamped", DATETIMETZ_TYPE, server_default="now()"), "'stamped' whose server_default 'now()'"),
        (Column("stamped", DATETIMETZ_TYPE, server_default="CURRENT_TIMESTAMP"), "database computes it"),
        (Column("token", UUID_TYPE, server_default="gen_random_uuid()"), "'token' whose server_default"),
        (Column("seq", INT_TYPE, server_default="nextval('defaulted_seq'::regclass)"), "database computes it"),
        (Column("short", STRING_TYPE, server_default="'abcdef'::varchar(2)"), "database computes it"),
        (Column("counted", INT_TYPE, server_default="'1'::text"), "cast to 'text'"),
        (Column("when", DATETIMETZ_TYPE, server_default="'2026-11-03'"), "fixed datetimetz value"),
        (Column("search", TSVECTOR_TYPE, immutable=True, nullable=True), "'search': a tsvector is maintained"),
    ],
)
def test_a_default_the_database_computes_refuses_the_opt_in_where_declared(column: Column, message: str) -> None:
    """Flips if a table opting in to delta files is accepted with a column whose value the database decides at the
    commit (a function, a sequence, an expression, a cast that can change the value, a literal 3tears does not read
    exactly, a trigger-maintained tsvector), which a file written before the commit could only guess."""
    with pytest.raises(ValueError, match="delta_files=True") as refused:
        _defaulted_schema(extra=[column])
    assert message in str(refused.value)
    assert "cannot know the database's answer" in str(refused.value)
    # the same table not opted in is declared as before
    TableSchema(
        name="defaulted",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), column],
        l1_eviction=TEST_L1_TABLE,
    )


def test_a_guarded_write_refuses_the_opt_in_where_declared() -> None:
    """Flips if a table opting in to delta files is accepted with a write the database applies only if a condition
    holds (a compare-and-swap fence, the L2 order's fenced upsert), whose outcome a file written before the commit
    cannot know."""
    with pytest.raises(ValueError, match="cas_column='date_updated': a compare-and-swap"):
        _defaulted_schema(cas_column="date_updated")
    with pytest.raises(ValueError, match="cas_column='date_updated'"):
        _defaulted_schema(cas_column="date_updated", cas_null_safe=True)
    with pytest.raises(ValueError, match="L2 order columns.*an ordered upsert fenced"):
        _defaulted_schema(extra=l2_order_columns())


def test_a_table_with_only_fixed_defaults_and_no_guard_opts_in() -> None:
    """Flips if a table with no computed default, no tsvector and no guarded write is refused, or if its fixed
    defaults are not read as the database stores them and the rail answers them."""
    schema = _defaulted_schema(
        extra=[
            Column("kind", STRING_TYPE, server_default="'it''s'"),
            Column("count", BIGINT_TYPE, server_default="-3"),
            Column("ref", UUID_TYPE, server_default="'0123456789ABCDEF0123456789ABCDEF'::uuid"),
        ]
    )
    assert dict(schema.delta_file_defaults) == {
        "status": "active",
        "flag": False,
        "share": "0.50",
        "note": None,
        "kind": "it's",
        "count": -3,
        "ref": "01234567-89ab-cdef-0123-456789abcdef",
    }
    assert _schema("plain", opted_in=True).delta_file_defaults == {}
    assert _schema("plain_off", opted_in=False).delta_file_defaults == {}


def test_a_load_delta_from_one_read_cuts_it_into_each_files_rows_by_the_manifest_counts() -> None:
    read = pa.table({"id": list(range(6)), "votes": [10, 11, 12, 13, 14, 15]})

    delta = LoadDelta.from_read(read, [2, 0, 4], replaces=[{"state": "VA"}])

    assert [table.to_pydict()["id"] for table in delta.rows] == [[0, 1], [], [2, 3, 4, 5]]
    assert delta.replaces == [{"state": "VA"}]
    # the load checks each file's rows against its count where its file is made: the cut passes that check as is
    load = TableDelta(name="votes", types={"id": pa.int64(), "votes": pa.int64(), "state": pa.string()}, key=("id",))
    load.load(delta.rows, [2, 0, 4], {"state": "VA"})
    assert load.table("w1").num_rows == 6


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([2, 3], "count 5 rows, where 6 were read"),
        ([2, 5], "count 7 rows, where 6 were read"),
        ([2, None, 4], "file 1 has no row count"),
        ([-1, 7], "file 0 has no row count"),
    ],
)
def test_a_load_delta_from_a_read_its_counts_do_not_cut_exactly_is_refused(
    counts: list[int | None], message: str
) -> None:
    read = pa.table({"id": list(range(6))})

    with pytest.raises(ValueError, match=message):
        LoadDelta.from_read(read, counts)
