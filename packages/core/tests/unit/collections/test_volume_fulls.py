"""a full on the delta-files volume: parts and a manifest published by rename, fenced by term, and a copy's start.

The contract this pins (:mod:`threetears.core.collections.volume_fulls`, ``design-pod-delta-files.md`` decisions 8, 14,
15 and 16):

- a manifest is seen whole or not at all: one being written is never read;
- readers take the highest term, then the newest sequence: a stale writer's later, older-term manifest is ignored;
- a reader of the previous manifest keeps loading its parts after a new publish and the writer's sweep;
- a copy starts from the current full, each table's parts staged and each part's version reported;
- with no full published, a copy waits, its status saying so, and reads nothing from L3;
- the deletable list never names the current or the previous manifest's files, nor a delta named as a part.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")
pa_ipc = pytest.importorskip("pyarrow.ipc")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, FollowerPhase  # noqa: E402
from threetears.core.collections.delta_files import read_volume_file, write_volume_file  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable  # noqa: E402
from threetears.core.collections.epoch_change import EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullAbsent,
    FullPart,
    FullTable,
    StagedFull,
    StaleTermError,
    VolumeFulls,
    current_full_manifest,
    deletable_full_files,
    publish_full_manifest,
    write_full_part,
)

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_FOLLOWED = FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))

_OPENED: list[Any] = []


class _EmptyLog:
    """the NATS epoch log of a writer's first start: no record names a delta yet."""

    async def change(self, table_name: str, token: str) -> None:
        return None

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    """close every DuckDB the test opened: each runs a worker thread a core (see test_copy_follower)."""
    yield
    while _OPENED:
        _OPENED.pop()()


def _backend() -> DuckDBBackend:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    backend = DuckDBBackend()
    backend.initialize(metadata)
    _OPENED.append(backend.reset)
    return backend


def _rows(state: str, count: int, votes: int = 1) -> Any:
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "geo_id": pa.array(range(count), pa.int64()),
            "votes": pa.array([votes] * count, pa.int64()),
        }
    )


def _held(backend: DuckDBBackend) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        return sorted(cursor.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())


def _full(root: Path, term: int, version: str = "inc:5", **states: int) -> Any:
    """a full of ``votes``, a part per state of that many rows, published at ``term``."""
    parts = tuple(write_full_part(root, _TABLE, f"state:{s}", _rows(s, n), version) for s, n in states.items())
    return publish_full_manifest(root, term=term, tables={_TABLE: FullTable(version=version, parts=parts)})


async def _stage(root: Path, backend: DuckDBBackend) -> StagedFull:
    staged = await VolumeFulls(root, [_FOLLOWED]).stage(backend)
    assert isinstance(staged, StagedFull), staged
    backend.apply_staged(staged.changes, mark=1)
    return staged


# --- publishing ---------------------------------------------------------------------------------


async def test_a_manifest_being_written_is_never_read_and_one_renamed_into_place_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a manifest is written under its own name (a reader could open it part written) instead of a temporary
    name renamed into place, or if a reader takes a temporary file for a manifest."""
    _full(tmp_path, 1, VA=3)
    seen: list[Any] = []
    rename = os.rename

    def look_then_rename(src: Any, dst: Any) -> None:
        # the new manifest is whole on disk under its temporary name: a reader now still sees only the old one
        seen.append(current_full_manifest(tmp_path))
        rename(src, dst)

    monkeypatch.setattr(os, "rename", look_then_rename)
    _full(tmp_path, 1, VA=4)
    assert seen[-1] is not None and seen[-1].seq == 1
    monkeypatch.setattr(os, "rename", rename)
    assert current_full_manifest(tmp_path).seq == 2  # type: ignore[union-attr]
    # a write that died before its rename leaves only a temporary name, which no reader takes
    manifests = tmp_path / ".fulls" / "manifests"
    (manifests / ".full-00000000000000000009-00000000000000000001.json.dead.partial").write_bytes(b"{")
    assert current_full_manifest(tmp_path).seq == 2  # type: ignore[union-attr]


async def test_the_highest_term_wins_over_an_older_terms_newer_sequence(tmp_path: Path) -> None:
    """Flips if readers order manifests by sequence or by time rather than term first: a stale writer (term 1) that
    publishes after the new writer (term 2) took over would replace its full."""
    _full(tmp_path, 1, VA=1)
    _full(tmp_path, 1, VA=2)
    current = _full(tmp_path, 2, VA=3)
    # the stale writer wakes and publishes: refused up front
    with pytest.raises(StaleTermError):
        _full(tmp_path, 1, VA=9)
    # had it raced the check: its manifest (term 1, seq 7) lands beside the current one
    elsewhere = tmp_path / "stale-writer"
    for _ in range(7):
        stale = _full(elsewhere, 1, VA=9)
    shutil.copy(elsewhere / ".fulls" / "manifests" / stale.name, tmp_path / ".fulls" / "manifests" / stale.name)
    assert current_full_manifest(tmp_path) == current
    assert f".fulls/manifests/{stale.name}" in deletable_full_files(tmp_path)


# --- a copy's start -----------------------------------------------------------------------------


async def test_a_start_loads_every_part_and_reports_each_parts_version(tmp_path: Path) -> None:
    """Flips if a part's rows do not reach the copy, or a part's version is not reported (the follower would apply
    deltas a part already holds), or the table's mark is not its oldest version."""
    va = write_full_part(tmp_path, _TABLE, "state:VA", _rows("VA", 3), "inc:7")
    il = write_full_part(tmp_path, _TABLE, "state:IL", _rows("IL", 2), "inc:4")
    publish_full_manifest(tmp_path, term=3, tables={_TABLE: FullTable(version="inc:6", parts=(va, il))})
    backend = _backend()
    staged = await _stage(tmp_path, backend)
    assert _held(backend) == sorted([("VA", i, 1) for i in range(3)] + [("IL", i, 1) for i in range(2)])
    assert staged.versions == {_TABLE: {None: "inc:6", "state:VA": "inc:7", "state:IL": "inc:4"}}
    assert staged.marks == {_TABLE: "inc:4"}
    assert staged.rows == 5
    # a part is Arrow IPC with zstd inside, closed by its checksum
    part = read_volume_file(tmp_path / va.file)
    assert part.schema.metadata[b"threetears.full.version"] == b"inc:7"
    assert part.num_rows == 3


def _part_checks(outcome: str) -> tuple[float, float]:
    """the part checks' count and their seconds' count for ``outcome``."""
    labels = {"outcome": outcome}
    return (
        REGISTRY.get_sample_value("volume_fulls_part_checks_total", labels) or 0.0,
        REGISTRY.get_sample_value("volume_fulls_part_check_seconds_count", labels) or 0.0,
    )


async def test_each_part_a_copy_loads_is_checked_and_counted_by_outcome_with_its_seconds(tmp_path: Path) -> None:
    """Flips if a copy's check of a part it loads (checksum, then the row count its manifest says) stops being counted
    by outcome (``volume_fulls_part_checks_total``) with its seconds (``volume_fulls_part_check_seconds``): ok for each
    part loaded, rows_differ, unreadable and missing for a part that fails the start."""
    ok = _part_checks("ok")
    _full(tmp_path, 1, VA=3, IL=2)
    await _stage(tmp_path, _backend())
    assert _part_checks("ok") == (ok[0] + 2, ok[1] + 2)
    va = write_full_part(tmp_path, _TABLE, "state:VA", _rows("VA", 3), "inc:6")
    wrong = FullPart(scope=va.scope, file=va.file, version=va.version, rows=4)
    publish_full_manifest(tmp_path, term=2, tables={_TABLE: FullTable(version="inc:6", parts=(wrong,))})
    differ = _part_checks("rows_differ")
    with pytest.raises(ValueError, match="its manifest says 4"):
        await VolumeFulls(tmp_path, [_FOLLOWED]).stage(_backend())
    assert _part_checks("rows_differ") == (differ[0] + 1, differ[1] + 1)
    damaged = tmp_path / va.file
    body = bytearray(damaged.read_bytes())
    body[len(body) // 2] ^= 0xFF
    damaged.write_bytes(bytes(body))
    publish_full_manifest(tmp_path, term=3, tables={_TABLE: FullTable(version="inc:6", parts=(va,))})
    unreadable = _part_checks("unreadable")
    with pytest.raises(ValueError):
        await VolumeFulls(tmp_path, [_FOLLOWED]).stage(_backend())
    assert _part_checks("unreadable") == (unreadable[0] + 1, unreadable[1] + 1)
    damaged.unlink()
    missing = _part_checks("missing")
    with pytest.raises(FileNotFoundError):
        await VolumeFulls(tmp_path, [_FOLLOWED]).stage(_backend())
    assert _part_checks("missing") == (missing[0] + 1, missing[1] + 1)


async def test_a_delta_named_as_a_part_loads_its_rows_never_its_delete_markers(tmp_path: Path) -> None:
    """Flips if a delta file named as a part (decision 16) loads its delete markers as rows, or its marker column."""
    delta = tmp_path / _TABLE / "w1.arrow"
    delta.parent.mkdir(parents=True)
    rows = _rows("TX", 3).append_column("_3tears_deleted", pa.array([False, False, True]))
    with delta.open("wb") as sink:
        write_volume_file(sink, rows)
    part = FullPart(scope="state:TX", file=f"{_TABLE}/w1.arrow", version="inc:8", rows=2)
    publish_full_manifest(tmp_path, term=1, tables={_TABLE: FullTable(version="inc:8", parts=(part,))})
    backend = _backend()
    await _stage(tmp_path, backend)
    assert _held(backend) == [("TX", 0, 1), ("TX", 1, 1)]


async def test_a_reader_of_the_old_manifest_keeps_reading_its_parts_after_a_new_publish(tmp_path: Path) -> None:
    """Flips if a publish changes a file in place, or the writer's sweep deletes a part the previous manifest names:
    a copy that took the old manifest before the publish could not finish loading it."""
    _full(tmp_path, 1, VA=3, IL=2)
    old = current_full_manifest(tmp_path)
    assert old is not None
    # the writer folds VA: a new part for it, IL's part named again, then deletes what it may
    il = next(p for p in old.tables[_TABLE].parts if p.scope == "state:IL")
    va = write_full_part(tmp_path, _TABLE, "state:VA", _rows("VA", 4, votes=2), "inc:6")
    publish_full_manifest(tmp_path, term=1, tables={_TABLE: FullTable(version="inc:6", parts=(va, il))})
    for file in deletable_full_files(tmp_path):
        (tmp_path / file).unlink()
    # the reader that took the old manifest loads every part it names
    backend = _backend()
    staged = await VolumeFulls(tmp_path, [_FOLLOWED]).stage(backend, manifest=old)
    assert isinstance(staged, StagedFull)
    backend.apply_staged(staged.changes, mark=1)
    assert _held(backend) == sorted([("VA", i, 1) for i in range(3)] + [("IL", i, 1) for i in range(2)])
    # and a new reader takes the new full
    fresh = _backend()
    await _stage(tmp_path, fresh)
    assert _held(fresh) == sorted([("VA", i, 2) for i in range(4)] + [("IL", i, 1) for i in range(2)])


async def test_deletable_excludes_the_current_and_previous_manifests_files_and_a_reused_delta(tmp_path: Path) -> None:
    """Flips if the deletable list names a file the current or the previous manifest names, or a covered delta that a
    manifest names as a part, or leaves out a part or manifest neither names."""
    first = _full(tmp_path, 1, VA=1)
    second = _full(tmp_path, 1, VA=2)
    reused = tmp_path / _TABLE / "w2.arrow"
    reused.parent.mkdir(parents=True)
    with reused.open("wb") as sink:
        write_volume_file(sink, _rows("VA", 3))
    part = FullPart(scope="state:VA", file=f"{_TABLE}/w2.arrow", version="inc:9", rows=3)
    third = publish_full_manifest(tmp_path, term=1, tables={_TABLE: FullTable(version="inc:9", parts=(part,))})
    assert third.previous == second.name
    deletable = set(deletable_full_files(tmp_path, covered_deltas=[f"{_TABLE}/w1.arrow", f"{_TABLE}/w2.arrow"]))
    assert deletable == {f".fulls/manifests/{first.name}", *first.files(), f"{_TABLE}/w1.arrow"}
    assert not deletable & (second.files() | third.files())


# --- waiting for the full -----------------------------------------------------------------------


class _L3:
    """L3 that counts every statement asked of it: a copy waiting for the full asks none."""

    def __init__(self) -> None:
        self.statements = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements += 1
        return []


async def test_no_manifest_waits_saying_so_and_never_reads_l3_then_loads_the_full_once_published(
    tmp_path: Path,
) -> None:
    """Flips if a follower whose tables opt in reads L3 (whole or at all) when the volume holds no full, instead of
    waiting with its status saying so; or if it does not load the full once the writer publishes it."""
    l3, backend, registry = _L3(), _backend(), CollectionRegistry()
    registry.follow_generation(_TABLE)

    async def generation(table: str) -> str:
        return "inc:5"

    follower = CopyFollower(
        registry,
        [_FOLLOWED],
        backend,
        l3,
        settled=lambda: asyncio.sleep(0, result=0),
        generation=generation,
        changes=None,
        stamp_table="scope_epochs",
        fulls=VolumeFulls(tmp_path, [_FOLLOWED], poll_seconds=0.01),
    )
    start = asyncio.create_task(follower.start())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if follower.status().phase is FollowerPhase.WAITING:
            break
    await asyncio.sleep(0.05)  # several looks at the volume
    assert not start.done()
    assert follower.status().phase is FollowerPhase.WAITING
    assert follower.status().detail.startswith("waiting for the full")
    assert l3.statements == 0
    _full(tmp_path, 1, VA=3)
    assert await asyncio.wait_for(start, 5) is False
    assert follower.status().phase is FollowerPhase.FOLLOWING
    assert _held(backend) == [("VA", i, 1) for i in range(3)]
    assert registry.generation_marks.recorded(_TABLE) == "inc:5"
    assert l3.statements == 0


async def test_a_full_without_every_opted_in_table_is_waited_on(tmp_path: Path) -> None:
    """Flips if a copy loads a full that lacks one of its opted-in tables (that table would then have to be read whole
    from L3)."""
    publish_full_manifest(tmp_path, term=1, tables={"other": FullTable(version="inc:1")})
    found = await VolumeFulls(tmp_path, [_FOLLOWED]).stage(_backend())
    assert isinstance(found, FullAbsent)
    assert _TABLE in found.reason


class _WriterL3:
    """the writer's L3: ``votes`` as a reset left it, read whole once by the writer's rebuild."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.reads = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if sql.startswith("SELECT version()"):
            return [{"v": "PostgreSQL 16.4"}]
        self.reads += 1
        return sorted(self.rows, key=lambda row: (row["state"], row["geo_id"]))


class _Generations:
    async def current(self, table_name: str) -> str:
        return "inc:5"

    async def advance(self, table_name: str) -> str | None:
        raise AssertionError("a rebuild advances nothing")


async def test_a_waiting_copy_loads_the_writer_s_rebuild_and_never_reads_l3(tmp_path: Path) -> None:
    """Flips if a copy of an opted-in table reads L3 (whole or at all) while the volume holds no full, or does not load
    the full the writer's rebuild from L3 publishes (decision 14): the writer reads L3 once, the copy never."""
    copy_l3, backend, registry = _L3(), _backend(), CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [_FOLLOWED],
        backend,
        copy_l3,
        settled=lambda: asyncio.sleep(0, result=0),
        generation=_Generations().current,
        changes=None,
        stamp_table="scope_epochs",
        fulls=VolumeFulls(tmp_path, [_FOLLOWED], poll_seconds=0.01),
    )
    start = asyncio.create_task(follower.start())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if follower.status().phase is FollowerPhase.WAITING:
            break
    assert follower.status().phase is FollowerPhase.WAITING and copy_l3.statements == 0
    schema = TableSchema(
        name=_TABLE, primary_key=_KEY, columns=list(_COLUMNS), l1_eviction=TEST_L1_TABLE, delta_files=True
    )
    writer_l3 = _WriterL3([{"state": "VA", "geo_id": g, "votes": 1} for g in range(3)])
    folds = DeltaFolds(tmp_path, [FoldedTable.of(schema, scope=("state", "state:"))], retention="delete")
    try:
        # a writer with no full yet has no delta to re-learn: the epoch log is read, and holds nothing for it
        assert await folds.start(writer_l3, _Generations(), changes=_EmptyLog(), version=3) == (_TABLE,)
    finally:
        await folds.close()
    assert await asyncio.wait_for(start, 5) is False
    assert follower.status().phase is FollowerPhase.FOLLOWING
    assert _held(backend) == [("VA", i, 1) for i in range(3)]
    assert registry.generation_marks.recorded(_TABLE) == "inc:5"
    assert writer_l3.reads == 1 and copy_l3.statements == 0


# --- a table whose columns changed in place (decision 12) ---------------------------------------


_ADDED = TableSchema(
    name=_TABLE,
    primary_key=_KEY,
    columns=[*_COLUMNS, Column(name="source", column_type=STRING_TYPE, nullable=True)],
    l1_eviction=TEST_L1_TABLE,
    delta_files=True,
)
_FOLLOWED_ADDED = dataclasses.replace(FollowedTable.of(_ADDED), scope=("state", "state:"))


def _backend_added() -> DuckDBBackend:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        SaColumn("source", String(8)),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    backend = DuckDBBackend()
    backend.initialize(metadata)
    _OPENED.append(backend.reset)
    return backend


def _other_columns() -> float:
    return REGISTRY.get_sample_value("volume_fulls_other_columns_total", {"table": _TABLE}) or 0.0


async def test_a_full_over_other_columns_than_the_copy_s_schema_is_waited_on_and_counted(tmp_path: Path) -> None:
    """Flips if a copy whose table list added a column in place stages a full whose parts were made over the old
    columns (it would hold the table without the new column, or fail its start), instead of waiting for the writer's
    fresh full, counted; or if a full over its columns is refused."""
    _full(tmp_path, 1, VA=3)
    looks = _other_columns()
    found = await VolumeFulls(tmp_path, [_FOLLOWED_ADDED]).stage(_backend_added())
    assert isinstance(found, FullAbsent) and "other columns" in found.reason
    assert _other_columns() == looks + 1
    assert isinstance(await VolumeFulls(tmp_path, [_FOLLOWED]).stage(_backend()), StagedFull), "its own columns load"


async def test_a_column_added_in_place_makes_the_writer_start_fresh_once_and_the_copy_load_the_new_column(
    tmp_path: Path,
) -> None:
    """Flips if, after a newer table list adds a column in place (no recreate, no reset record), the writer keeps the
    full over the old columns, or rebuilds more than once; or if a copy starting meanwhile reads L3 or takes the old
    full instead of waiting for the fresh one and loading the new column."""
    old_schema = TableSchema(
        name=_TABLE, primary_key=_KEY, columns=list(_COLUMNS), l1_eviction=TEST_L1_TABLE, delta_files=True
    )
    writer_l3 = _WriterL3([{"state": "VA", "geo_id": g, "votes": 1} for g in range(3)])
    folds = DeltaFolds(tmp_path, [FoldedTable.of(old_schema, scope=("state", "state:"))], retention="delete")
    try:
        assert await folds.start(writer_l3, _Generations(), changes=_EmptyLog(), version=3) == (_TABLE,)
    finally:
        await folds.close()
    # the hub adds `source` in place; a copy of the new table list starts before the writer restarts
    copy_l3, backend, registry = _L3(), _backend_added(), CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [_FOLLOWED_ADDED],
        backend,
        copy_l3,
        settled=lambda: asyncio.sleep(0, result=0),
        generation=_Generations().current,
        changes=None,
        stamp_table="scope_epochs",
        fulls=VolumeFulls(tmp_path, [_FOLLOWED_ADDED], poll_seconds=0.01),
    )
    looks = _other_columns()
    start = asyncio.create_task(follower.start())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if follower.status().phase is FollowerPhase.WAITING:
            break
    await asyncio.sleep(0.05)  # several looks at the volume
    assert not start.done() and follower.status().phase is FollowerPhase.WAITING
    assert "other columns" in follower.status().detail
    assert _other_columns() > looks and copy_l3.statements == 0
    # the writer restarts on the new table list: the table starts fresh, read from L3 once
    writer_l3.rows = [{"state": "VA", "geo_id": g, "votes": 1, "source": f"s{g}"} for g in range(3)]
    folds = DeltaFolds(tmp_path, [FoldedTable.of(_ADDED, scope=("state", "state:"))], retention="delete")
    try:
        assert await folds.start(writer_l3, _Generations(), changes=_EmptyLog(), version=4) == (_TABLE,)
        assert await folds.start(writer_l3, _Generations(), changes=_EmptyLog(), version=5) == ()
    finally:
        await folds.close()
    assert writer_l3.reads == 2, "the first rebuild and the fresh start, nothing more"
    assert await asyncio.wait_for(start, 5) is False
    assert follower.status().phase is FollowerPhase.FOLLOWING
    with backend.read_snapshot() as cursor:
        held = sorted(cursor.execute(f"SELECT state, geo_id, votes, source FROM {_TABLE}").fetchall())
    assert held == [("VA", g, 1, f"s{g}") for g in range(3)]
    assert copy_l3.statements == 0
