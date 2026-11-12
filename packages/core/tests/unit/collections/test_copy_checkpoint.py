"""a copy's checkpoint loads each table on its own and writes only what changed.

The contract this pins (:mod:`threetears.core.collections.copy_checkpoint`):

- a table whose parts do not read back loads from the previous checkpoint, with that one's mark; with no previous it
  is left out as ``cold_start`` and every other table loads;
- a table whose schema digest differs is left out as ``schema_change``; the rest load;
- a write rewrites only the (table, scope) parts changed since this replica's last checkpoint and names the others'
  objects again; a table with no scope has its one part patched by the touched keys alone; a change of unknown tables
  writes every part;
- a table is read whole from the copy only at its cold start, a schema change or a whole read of it, each counted by
  why; a replica whose parts the newest no longer names writes nothing, never rereading whole what the newest holds;
- a derived table is rewritten only in the scopes the followed tables' changes touched;
- a failed or beaten write keeps what it was to write, for the next;
- no retire takes an object the newest or the previous checkpoint names;
- two code versions keep each other's tables in the one pointer, and each loads what it reads the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
ipc = pytest.importorskip("pyarrow.ipc")

from prometheus_client import REGISTRY  # noqa: E402

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.collections import copy_checkpoint  # noqa: E402
from threetears.core.collections.copy_checkpoint import (  # noqa: E402
    COLD_START,
    SCHEMA_CHANGE,
    CopyCheckpoints,
    StagedCheckpoint,
)
from threetears.core.collections.copy_follower import FollowedTable  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402

_VOTES = "votes"
_RACES = "races"
_NEEDLES = "needles"
_TOTALS = "totals"
_VOTES_KEY = ("state", "geo_id")
_RACES_KEY = ("race_id",)
#: a whole read because the follower read the table whole (copy_checkpoint_whole_writes_total's why)
_WHOLE_TOUCH = "whole_touch"
_KEYS = {_VOTES: _VOTES_KEY, _RACES: _RACES_KEY, _NEEDLES: _VOTES_KEY, _TOTALS: _RACES_KEY}


def _metadata(*, share: bool = False) -> MetaData:
    metadata = MetaData()
    columns: list[SaColumn[Any]] = [
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
    ]
    if share:
        columns.append(SaColumn("share", BigInteger))
    Table(_VOTES, metadata, *columns)
    Table(_RACES, metadata, SaColumn("race_id", BigInteger, primary_key=True), SaColumn("name", String(32)))
    # two tables an owner derives from the followed ones: one by state, one with no scope
    Table(
        _NEEDLES,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
    )
    Table(_TOTALS, metadata, SaColumn("race_id", BigInteger, primary_key=True), SaColumn("name", String(32)))
    return metadata


def _tables(*, share: bool = False) -> list[FollowedTable]:
    votes = [
        Column(name="state", column_type=STRING_TYPE),
        Column(name="geo_id", column_type=BIGINT_TYPE),
        Column(name="votes", column_type=BIGINT_TYPE),
    ]
    if share:
        votes.append(Column(name="share", column_type=BIGINT_TYPE))
    return [
        FollowedTable(name=_VOTES, key=_VOTES_KEY, columns=tuple(votes), scope=("state", "state:")),
        FollowedTable(
            name=_RACES,
            key=_RACES_KEY,
            columns=(Column(name="race_id", column_type=BIGINT_TYPE), Column(name="name", column_type=STRING_TYPE)),
        ),
    ]


def _needles(*, scoped: bool = True) -> FollowedTable:
    votes = _tables()[0]
    return FollowedTable(name=_NEEDLES, key=_VOTES_KEY, columns=votes.columns, scope=votes.scope if scoped else None)


def _totals() -> FollowedTable:
    return FollowedTable(name=_TOTALS, key=_RACES_KEY, columns=_tables()[1].columns)


class _Store:
    """an Object Store in memory; a put can be made to fail."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put_at: int | None = None
        self.puts = 0
        self.retired: list[str] = []

    async def put(self, name: str, data: bytes) -> None:
        self.puts += 1
        if self.fail_put_at is not None and self.puts == self.fail_put_at:
            raise ConnectionError("the store went away mid-write")
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

    async def list_names(self, *, prefix: str = "") -> list[str]:
        return [n for n in self.objects if n.startswith(prefix)]

    async def retire(self, names: list[str]) -> None:
        self.retired.extend(names)
        for name in names:
            self.objects.pop(name, None)


class _Pointers:
    """a pointer bucket in memory: create-if-absent and compare-and-set by revision."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self.revision = 0

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: Any = None) -> int | None:
        if key in self.entries:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def update(self, *, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        held = self.entries.get(key)
        if held is None or held[1] != revision:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def delete(self, *, key: str, revision: int | None = None, marker_ttl: Any = None) -> bool:
        return self.entries.pop(key, None) is not None

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        return [k for k in self.entries if k.startswith(prefix)]


def _votes(rows: Sequence[tuple[str, int, int]], *, share: bool = False) -> Any:
    columns: dict[str, Any] = {
        "state": pa.array([r[0] for r in rows], pa.string()),
        "geo_id": pa.array([r[1] for r in rows], pa.int64()),
        "votes": pa.array([r[2] for r in rows], pa.int64()),
    }
    if share:
        columns["share"] = pa.array([1] * len(rows), pa.int64())
    return pa.table(columns)


def _races(rows: Sequence[tuple[int, str]]) -> Any:
    return pa.table(
        {"race_id": pa.array([r[0] for r in rows], pa.int64()), "name": pa.array([r[1] for r in rows], pa.string())}
    )


class _Copy:
    """a follower's copy as a checkpoint sees it: a DuckDB, each change applied with a mark, and a read giving the
    stamp and the table marks of the change it holds."""

    def __init__(self, *, share: bool = False) -> None:
        self.share = share
        self.backend = DuckDBBackend()
        self.backend.initialize(_metadata(share=share))
        self.mark = 0
        self.marks: dict[str, str | None] = {}
        self.stamps: dict[int, tuple[Any, Mapping[str, str | None]]] = {}
        #: every statement a checkpoint write ran on the copy, and its parameters
        self.statements: list[tuple[str, Any]] = []

    def put(self, table: str, rows: Any, mark: str, stamp: int) -> None:
        key = _KEYS[table]
        stage = f"{STAGE_PREFIX}{table}__rows"
        self.backend.drop_stage(stage)
        self.backend.stage_rows(stage, table, rows, key)
        self.mark += 1
        self.backend.apply_staged([StagedChange(table=table, key=key, rows=stage, whole=True)], mark=self.mark)
        self.marks[table] = mark
        self.stamps[self.mark] = (stamp, dict(self.marks))

    def load(self, staged: StagedCheckpoint) -> None:
        self.mark += 1
        self.backend.apply_staged(staged.changes, mark=self.mark)
        self.marks.update(staged.marks)
        self.stamps[self.mark] = (staged.stamp, dict(staged.marks))

    @contextmanager
    def read(self) -> Iterator[tuple[Any, Any, Mapping[str, str | None]]]:
        with self.backend.read_snapshot() as cursor:
            mark = self.backend.staged_mark(cursor)
            stamp, marks = self.stamps.get(mark, (None, {})) if mark is not None else (None, {})
            yield _Recording(cursor, self.statements), stamp, marks

    def held(self, table: str) -> list[tuple[Any, ...]]:
        columns = "state, geo_id, votes" if _KEYS[table] == _VOTES_KEY else "race_id, name"
        with self.backend.read_snapshot() as cursor:
            return sorted(cursor.execute(f"SELECT {columns} FROM {table}").fetchall())  # noqa: S608


class _Recording:
    """a read's cursor that records each statement run on it."""

    def __init__(self, cursor: Any, statements: list[tuple[str, Any]]) -> None:
        self._cursor = cursor
        self._statements = statements

    def execute(self, sql: str, params: Any = None) -> Any:
        self._statements.append((sql, params))
        return self._cursor.execute(sql) if params is None else self._cursor.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def _whole_reads(copy: _Copy, table: str) -> list[str]:
    """the statements a write ran that read ``table`` of the copy whole."""
    return [sql for sql, _ in copy.statements if sql == f'SELECT * FROM "{table}"' or sql == f"SELECT * FROM {table}"]


def _whole_writes(table: str, why: str) -> float:
    return REGISTRY.get_sample_value("copy_checkpoint_whole_writes_total", {"table": table, "why": why}) or 0.0


def _checkpoints(copy: _Copy, store: _Store, pointers: _Pointers, owner: str, *, part_rows: int = 40) -> Any:
    return CopyCheckpoints(
        name="copy",
        tables=_tables(share=copy.share),
        backend=copy.backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner=owner,
        part_rows=part_rows,
    )


def _state(state: str, rows: int, votes: int = 1) -> list[tuple[str, int, int]]:
    return [(state, i, votes) for i in range(rows)]


async def _written_world() -> tuple[_Copy, Any, _Store, _Pointers]:
    """a copy of VA (50 rows: two parts), IL (10) and three races, and its first checkpoint (seq 1)."""
    store, pointers = _Store(), _Pointers()
    copy = _Copy()
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10)), "v1", 1)
    copy.put(_RACES, _races([(1, "gov"), (2, "sen"), (3, "house")]), "r1", 1)
    checkpoints = _checkpoints(copy, store, pointers, "a")
    assert await checkpoints.write(copy.read) is not None
    return copy, checkpoints, store, pointers


async def _stage_fresh(store: _Store, pointers: _Pointers, owner: str = "b", *, share: bool = False) -> Any:
    copy = _Copy(share=share)
    checkpoints = _checkpoints(copy, store, pointers, owner)
    staged = await checkpoints.stage()
    copy.load(staged)
    return copy, checkpoints, staged


def _part_names(store: _Store, table: str) -> set[str]:
    return {name for name in store.objects if f"/{table}/" in name}


def _writes(kind: str, outcome: str) -> float:
    return REGISTRY.get_sample_value("copy_checkpoint_writes_total", {"kind": kind, "outcome": outcome}) or 0.0


async def test_a_cold_start_write_of_rows_whose_scopes_interleave_stores_each_scope_apart_and_loads_them_all() -> None:
    """Flips if a whole write's cut by scope puts two scopes' rows in one part, or loses or repeats a row, when the
    copy's rows of a scope do not run together."""
    store, pointers = _Store(), _Pointers()
    copy = _Copy()
    interleaved = [row for pair in zip(_state("VA", 30), _state("IL", 30), strict=True) for row in pair]
    copy.put(_VOTES, _votes(interleaved + _state("TX", 5)), "v1", 1)
    copy.put(_RACES, _races([(1, "gov")]), "r1", 1)
    assert await _checkpoints(copy, store, pointers, "a", part_rows=50).write(copy.read) is not None
    for name in _part_names(store, _VOTES):
        states = set(ipc.open_stream(store.objects[name]).read_all().column("state").to_pylist())
        assert len(states) == 1, (name, states)
    loaded, _, staged = await _stage_fresh(store, pointers)
    assert staged.missing == {}
    assert loaded.held(_VOTES) == copy.held(_VOTES)


# --- loading ---------------------------------------------------------------------------------------


async def test_a_table_whose_parts_do_not_read_back_is_left_out_and_every_other_table_loads() -> None:
    copy, _, store, pointers = await _written_world()
    for name in _part_names(store, _VOTES):
        store.objects[name] = b"broken"
    loaded, _, staged = await _stage_fresh(store, pointers)
    assert staged.missing == {_VOTES: COLD_START}
    assert dict(staged.marks) == {_RACES: "r1"}
    assert staged.stamp == 1
    assert loaded.held(_RACES) == copy.held(_RACES)
    assert loaded.held(_VOTES) == []
    assert loaded.backend.staged_stages() == []


async def test_a_table_the_newest_cannot_give_loads_from_the_previous_checkpoint_with_its_mark() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    first_votes = copy.held(_VOTES)
    copy.put(_VOTES, _votes(_state("VA", 50, votes=2) + _state("IL", 10)), "v2", 2)
    copy.put(_RACES, _races([(1, "gov"), (2, "sen")]), "r2", 2)
    checkpoints.applied(53, {_VOTES: frozenset({"state:VA"}), _RACES: frozenset()}, {_RACES: [("3",)]})
    assert await checkpoints.write(copy.read) is not None
    newest = (await checkpoints.newest())[0]
    for name, _, _ in newest.tables[_VOTES].parts:
        if name not in newest.previous.names():
            store.objects[name] = b"broken"
    loaded, _, staged = await _stage_fresh(store, pointers)
    assert staged.missing == {}
    assert dict(staged.marks) == {_VOTES: "v1", _RACES: "r2"}
    assert staged.stamp == 1  # the oldest checkpoint a table came from
    assert loaded.held(_VOTES) == first_votes
    assert loaded.held(_RACES) == copy.held(_RACES)


async def test_a_table_whose_schema_digest_differs_is_a_schema_change_and_the_rest_load() -> None:
    copy, _, store, pointers = await _written_world()
    loaded, _, staged = await _stage_fresh(store, pointers, share=True)
    assert staged.missing == {_VOTES: SCHEMA_CHANGE}
    assert dict(staged.marks) == {_RACES: "r1"}
    assert loaded.held(_RACES) == copy.held(_RACES)


async def test_no_checkpoint_is_a_cold_start_of_every_table() -> None:
    copy, store, pointers = _Copy(), _Store(), _Pointers()
    staged = await _checkpoints(copy, store, pointers, "a").stage()
    assert not staged.loaded
    assert staged.missing == {_VOTES: COLD_START, _RACES: COLD_START}


# --- writing ---------------------------------------------------------------------------------------


async def test_a_write_rewrites_only_the_touched_scopes_and_names_every_other_part_again() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    first = (await checkpoints.newest())[0]
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=9) + _state("OH", 3)), "v2", 2)
    checkpoints.applied(13, {_VOTES: frozenset({"state:IL", "state:OH"})})
    puts, before = store.puts, _writes("incremental", "written")
    second = await checkpoints.write(copy.read)
    assert second is not None
    assert store.puts - puts == 2  # IL and OH, one part each
    assert _writes("incremental", "written") == before + 1
    va = [part for part in second.tables[_VOTES].parts if part[1] == "state:VA"]
    assert va == [part for part in first.tables[_VOTES].parts if part[1] == "state:VA"]
    assert second.tables[_RACES].parts == first.tables[_RACES].parts
    assert second.tables[_VOTES].mark == "v2"
    loaded, _, staged = await _stage_fresh(store, pointers)
    assert staged.missing == {}
    assert loaded.held(_VOTES) == copy.held(_VOTES)
    assert loaded.held(_RACES) == copy.held(_RACES)


async def test_a_touched_table_with_no_scope_is_patched_by_its_keys_and_untouched_tables_are_not() -> None:
    copy, checkpoints, store, _ = await _written_world()
    first = (await checkpoints.newest())[0]
    copy.put(_RACES, _races([(1, "gov"), (4, "ag")]), "r2", 2)
    checkpoints.applied(2, {_RACES: frozenset()}, {_RACES: [("2",), ("3",), ("4",)]})
    second = await checkpoints.write(copy.read)
    assert second is not None
    assert second.tables[_VOTES].parts == first.tables[_VOTES].parts
    assert not set(n for n, _, _ in second.tables[_RACES].parts) & first.names()


async def test_a_scope_whose_rows_all_left_drops_out_of_the_checkpoint() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    copy.put(_VOTES, _votes(_state("VA", 50)), "v2", 2)
    checkpoints.applied(10, {_VOTES: frozenset({"state:IL"})})
    second = await checkpoints.write(copy.read)
    assert {part[1] for part in second.tables[_VOTES].parts} == {"state:VA"}
    loaded, _, _ = await _stage_fresh(store, pointers)
    assert loaded.held(_VOTES) == copy.held(_VOTES)


async def test_a_failed_write_keeps_what_it_was_to_write_for_the_next() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    copy.put(_VOTES, _votes(_state("VA", 50, votes=3) + _state("IL", 10)), "v2", 2)
    checkpoints.applied(50, {_VOTES: frozenset({"state:VA"})})
    assert checkpoints.due()
    store.fail_put_at = store.puts + 2
    before = _writes("incremental", "failed")
    with pytest.raises(ConnectionError):
        await checkpoints.write(copy.read)
    assert _writes("incremental", "failed") == before + 1
    assert checkpoints.due()
    store.fail_put_at = None
    puts = store.puts
    assert await checkpoints.write(copy.read) is not None
    assert store.puts - puts == 2  # VA again, both its parts; nothing else
    assert not checkpoints.due()
    loaded, _, _ = await _stage_fresh(store, pointers)
    assert loaded.held(_VOTES) == copy.held(_VOTES)


async def test_a_write_beaten_to_the_pointer_keeps_what_it_was_to_write() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=4)), "v2", 2)
    checkpoints.applied(10, {_VOTES: frozenset({"state:IL"})})
    real_update = pointers.update

    async def beaten(*, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        if key == checkpoints.base:  # another replica moves the pointer first, once
            pointers.update = real_update  # type: ignore[method-assign]
            held = pointers.entries[key]
            await real_update(key=key, value=held[0], revision=held[1])
        return await real_update(key=key, value=value, revision=revision, ttl=ttl)

    pointers.update = beaten  # type: ignore[method-assign]
    named = set(store.objects)
    assert await checkpoints.write(copy.read) is None
    assert set(store.objects) == named
    puts = store.puts
    assert await checkpoints.write(copy.read) is not None
    assert store.puts - puts == 1  # IL again, which the beaten write took; nothing else
    loaded, _, _ = await _stage_fresh(store, pointers)
    assert loaded.held(_VOTES) == copy.held(_VOTES)


async def test_no_retire_takes_an_object_the_newest_or_the_previous_checkpoint_names() -> None:
    copy, checkpoints, store, _ = await _written_world()
    seen: list[set[str]] = [(await checkpoints.newest())[0].names()]
    for version in range(2, 6):
        copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=version)), f"v{version}", version)
        checkpoints.applied(10, {_VOTES: frozenset({"state:IL"})})
        written = await checkpoints.write(copy.read)
        assert written is not None
        seen.append(written.names())
        assert written.live() <= set(store.objects)
        # what only the checkpoint before the previous named is gone
        if len(seen) >= 3:
            assert not (seen[-3] - seen[-2] - seen[-1]) & set(store.objects)
    assert set(store.objects) == seen[-1] | seen[-2]


async def test_two_writers_of_one_sequence_never_write_the_same_object() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    names = (await checkpoints.newest())[0].names()
    assert all(name.split("/")[2] == "a" for name in names)


async def test_a_table_left_out_of_the_load_makes_a_checkpoint_due() -> None:
    _, _, store, pointers = await _written_world()
    for name in _part_names(store, _VOTES):
        store.objects[name] = b"broken"
    _, checkpoints, staged = await _stage_fresh(store, pointers)
    assert staged.missing == {_VOTES: COLD_START}
    assert checkpoints.due()


# --- two code versions -----------------------------------------------------------------------------


async def test_two_code_versions_keep_each_others_tables_and_each_loads_what_it_reads_the_same_way() -> None:
    old, _, store, pointers = await _written_world()
    # a newer version, whose votes has another column, starts: votes is a schema change; races loads
    new, checkpoints, staged = await _stage_fresh(store, pointers, share=True)
    assert staged.missing == {_VOTES: SCHEMA_CHANGE}
    assert checkpoints.due()
    # a checkpoint before it has read votes again keeps the old version's votes as it was
    new.put(_RACES, _races([(1, "gov")]), "r2", 2)
    checkpoints.applied(1, {_RACES: frozenset()}, {_RACES: [("2",), ("3",)]})
    kept = await checkpoints.write(new.read)
    assert kept is not None
    assert kept.tables[_VOTES] == kept.previous.tables[_VOTES]
    # once it has read votes whole, its checkpoint holds its own
    new.put(_VOTES, _votes(_state("VA", 5), share=True), "v2", 2)
    checkpoints.applied(5, {_VOTES: None}, whys={_VOTES: "reset"})
    assert checkpoints.due()
    before = _whole_writes(_VOTES, SCHEMA_CHANGE)
    mine = await checkpoints.write(new.read)
    assert _whole_writes(_VOTES, SCHEMA_CHANGE) == before + 1
    assert mine is not None and mine.tables[_VOTES].digest != kept.tables[_VOTES].digest
    # an old-version replica starting now loads votes from the previous checkpoint, which it reads the same way
    loaded, _, again = await _stage_fresh(store, pointers, "c")
    assert again.missing == {}
    assert loaded.held(_VOTES) == old.held(_VOTES)
    assert dict(again.marks) == {_VOTES: "v1", _RACES: "r2"}


async def test_a_writer_keeps_the_entry_of_a_table_it_does_not_follow() -> None:
    _, _, store, pointers = await _written_world()
    copy = _Copy()
    only_races = CopyCheckpoints(
        name="copy",
        tables=[t for t in _tables() if t.name == _RACES],
        backend=copy.backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner="d",
        part_rows=40,
    )
    copy.load(await only_races.stage())
    copy.put(_RACES, _races([(9, "mayor")]), "r2", 2)
    only_races.applied(1, {_RACES: frozenset()}, {_RACES: [("1",), ("2",), ("3",), ("9",)]})
    written = await only_races.write(copy.read)
    assert written is not None and written.previous is not None
    assert written.tables[_VOTES] == written.previous.tables[_VOTES]


# --- reading no more than changed ------------------------------------------------------------------


async def test_a_one_key_change_to_a_table_with_no_scope_patches_its_part_by_that_key_reading_only_it() -> None:
    copy, checkpoints, store, pointers = await _written_world()
    first = (await checkpoints.newest())[0]
    copy.put(_RACES, _races([(1, "gov"), (2, "senate"), (3, "house")]), "r2", 2)
    checkpoints.applied(1, {_RACES: frozenset()}, {_RACES: [("2",)]})
    copy.statements.clear()
    second = await checkpoints.write(copy.read)
    assert second is not None
    (old_name, _, _), (new_name, _, _) = first.tables[_RACES].parts[0], second.tables[_RACES].parts[0]
    old = ipc.open_stream(store.objects[old_name]).read_all()
    new = ipc.open_stream(store.objects[new_name]).read_all()
    old_rows, new_rows = set(zip(*old.to_pydict().values())), set(zip(*new.to_pydict().values()))
    assert old_rows - new_rows == {(2, "sen")} and new_rows - old_rows == {(2, "senate")}
    # the copy was read by the one key alone; nothing of it whole
    copy_reads = [(sql, params) for sql, params in copy.statements if '"races"' in sql]
    assert len(copy_reads) == 1 and "JOIN (SELECT unnest" in copy_reads[0][0] and copy_reads[0][1] == [["2"]]
    assert not _whole_reads(copy, _RACES) and not _whole_reads(copy, _VOTES)
    loaded, _, _ = await _stage_fresh(store, pointers)
    assert loaded.held(_RACES) == copy.held(_RACES)


async def test_a_touched_scope_is_read_by_its_scope_column_alone() -> None:
    copy, checkpoints, _, _ = await _written_world()
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=2)), "v2", 2)
    checkpoints.applied(10, {_VOTES: frozenset({"state:IL"})})
    copy.statements.clear()
    assert await checkpoints.write(copy.read) is not None
    assert [params for _, params in copy.statements if '"votes"' in _] == [[["IL"]]]
    assert not _whole_reads(copy, _VOTES) and not any("SELECT DISTINCT" in sql for sql, _ in copy.statements)


async def test_a_table_with_no_scope_touched_without_its_keys_is_refused() -> None:
    _, checkpoints, _, _ = await _written_world()
    with pytest.raises(ValueError, match="races"):
        checkpoints.applied(1, {_RACES: frozenset()})


async def test_a_whole_read_is_counted_by_why_and_only_cold_start_schema_change_or_a_whole_touch_reads_whole() -> None:
    before = {table: _whole_writes(table, COLD_START) for table in (_VOTES, _RACES)}
    copy, checkpoints, _, _ = await _written_world()
    assert {table: _whole_writes(table, COLD_START) for table in (_VOTES, _RACES)} == {
        table: count + 1 for table, count in before.items()
    }
    # the follower read votes whole (a reset, say): written whole, counted so
    copy.put(_VOTES, _votes(_state("VA", 3)), "v2", 2)
    checkpoints.applied(3, {_VOTES: None}, whys={_VOTES: "reset"})
    touched = _whole_writes(_VOTES, _WHOLE_TOUCH)
    copy.statements.clear()
    assert await checkpoints.write(copy.read) is not None
    assert _whole_writes(_VOTES, _WHOLE_TOUCH) == touched + 1
    assert _whole_reads(copy, _VOTES) and not _whole_reads(copy, _RACES)
    # a keyed change after it reads nothing whole
    copy.put(_VOTES, _votes(_state("VA", 3) + _state("OH", 1)), "v3", 3)
    checkpoints.applied(1, {_VOTES: frozenset({"state:OH"})})
    copy.statements.clear()
    assert await checkpoints.write(copy.read) is not None
    assert not _whole_reads(copy, _VOTES) and not _whole_reads(copy, _RACES)


async def _rewrite_everything(copy: _Copy, checkpoints: Any, version: int) -> None:
    copy.put(_VOTES, _votes(_state("VA", 50, votes=version) + _state("IL", 10, votes=version)), f"v{version}", version)
    copy.put(_RACES, _races([(1, f"gov{version}"), (2, "sen"), (3, "house")]), f"r{version}", version)
    checkpoints.applied(61, {_VOTES: frozenset({"state:VA", "state:IL"}), _RACES: frozenset()}, {_RACES: [("1",)]})
    assert await checkpoints.write(copy.read) is not None


async def test_a_replica_whose_parts_the_newest_no_longer_names_writes_nothing_and_reads_nothing_whole() -> None:
    copy, writer, store, pointers = await _written_world()
    other, late, _ = await _stage_fresh(store, pointers)
    # the first replica writes twice, every part anew: what the second loaded is retired
    await _rewrite_everything(copy, writer, 2)
    await _rewrite_everything(copy, writer, 3)
    other.put(_RACES, _races([(1, "gov"), (2, "sen")]), "r9", 9)
    late.applied(1, {_RACES: frozenset()}, {_RACES: [("3",)]})
    skipped = _writes("incremental", "skipped")
    other.statements.clear()
    newest = (await late.newest())[0]
    assert await late.write(other.read) is None
    assert _writes("incremental", "skipped") == skipped + 1
    assert not other.statements  # not a read of the copy, whole or keyed
    assert (await late.newest())[0].seq == newest.seq
    assert not late.due()  # it writes none until it loads again


async def test_a_replica_whose_parts_the_previous_still_names_writes_on_them() -> None:
    copy, writer, store, pointers = await _written_world()
    other, late, _ = await _stage_fresh(store, pointers)
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=5)), "v2", 2)
    writer.applied(10, {_VOTES: frozenset({"state:IL"})})
    assert await writer.write(copy.read) is not None
    # the second replica, on the first checkpoint's parts (the previous now), writes what it changed since
    other.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=5) + _state("OH", 2)), "v3", 3)
    other.put(_RACES, _races([(1, "gov"), (2, "sen"), (3, "house")]), "r1", 3)
    late.applied(12, {_VOTES: frozenset({"state:IL", "state:OH"})})
    other.statements.clear()
    written = await late.write(other.read)
    assert written is not None
    assert not _whole_reads(other, _VOTES) and not _whole_reads(other, _RACES)
    loaded, _, _ = await _stage_fresh(store, pointers, "c")
    assert loaded.held(_VOTES) == other.held(_VOTES)
    assert loaded.held(_RACES) == other.held(_RACES)


def _derived_checkpoints(copy: _Copy, store: _Store, pointers: _Pointers, owner: str, *derived: FollowedTable) -> Any:
    checkpoints = CopyCheckpoints(
        name="copy",
        tables=[*_tables(), *derived],
        backend=copy.backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner=owner,
        part_rows=40,
    )
    checkpoints.follow([_VOTES, _RACES])
    return checkpoints


async def _derived_world(*derived: FollowedTable) -> tuple[_Copy, Any, _Store, _Pointers]:
    store, pointers = _Store(), _Pointers()
    copy = _Copy()
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10)), "v1", 1)
    copy.put(_RACES, _races([(1, "gov"), (2, "sen")]), "r1", 1)
    copy.put(_NEEDLES, _votes(_state("VA", 50) + _state("IL", 10)), "n", 1)
    copy.put(_TOTALS, _races([(1, "gov"), (2, "sen")]), "t", 1)
    checkpoints = _derived_checkpoints(copy, store, pointers, "a", *derived)
    assert await checkpoints.write(copy.read) is not None
    return copy, checkpoints, store, pointers


def _derive(copy: _Copy, needles: Iterable[tuple[str, int, int]]) -> None:
    copy.put(_NEEDLES, _votes(list(needles)), "n", copy.mark)


async def test_a_derived_table_is_rewritten_only_in_the_scopes_the_followed_tables_touched() -> None:
    copy, checkpoints, store, pointers = await _derived_world(_needles())
    first = (await checkpoints.newest())[0]
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=7)), "v2", 2)
    _derive(copy, _state("VA", 50) + _state("IL", 10, votes=7))
    checkpoints.applied(10, {_VOTES: frozenset({"state:IL"})})
    copy.statements.clear()
    second = await checkpoints.write(copy.read)
    assert second is not None
    va = [p for p in second.tables[_NEEDLES].parts if p[1] == "state:VA"]
    assert va == [p for p in first.tables[_NEEDLES].parts if p[1] == "state:VA"]
    assert [params for sql, params in copy.statements if '"needles"' in sql] == [[["IL"]]]
    assert not _whole_reads(copy, _NEEDLES)
    # nothing followed changed: every derived part is named again
    copy.statements.clear()
    checkpoints.applied(1, {})
    third = await checkpoints.write(copy.read)
    assert third is not None and third.tables[_NEEDLES].parts == second.tables[_NEEDLES].parts
    assert not [sql for sql, _ in copy.statements if '"needles"' in sql]


async def test_a_derived_table_is_read_whole_when_a_followed_table_was() -> None:
    copy, checkpoints, _, _ = await _derived_world(_needles())
    copy.put(_VOTES, _votes(_state("VA", 5)), "v2", 2)
    _derive(copy, _state("VA", 5))
    checkpoints.applied(5, {_VOTES: None}, whys={_VOTES: "reset"})
    before = _whole_writes(_NEEDLES, _WHOLE_TOUCH)
    assert await checkpoints.write(copy.read) is not None
    assert _whole_writes(_NEEDLES, _WHOLE_TOUCH) == before + 1


async def test_a_derived_table_is_read_whole_when_a_followed_table_with_no_scope_changed() -> None:
    copy, checkpoints, _, _ = await _derived_world(_needles())
    copy.put(_RACES, _races([(1, "gov"), (2, "senate")]), "r2", 2)
    checkpoints.applied(1, {_RACES: frozenset()}, {_RACES: [("2",)]})
    # counted apart from a whole read the follower made: this one is no cold start (a finding until narrowed)
    before = _whole_writes(_NEEDLES, "unscoped_derived")
    assert await checkpoints.write(copy.read) is not None
    assert _whole_writes(_NEEDLES, "unscoped_derived") == before + 1


async def test_a_derived_table_with_no_scope_is_refused_by_name() -> None:
    copy = _Copy()
    with pytest.raises(ValueError, match="totals"):
        _derived_checkpoints(copy, _Store(), _Pointers(), "a", _totals())


async def test_a_table_with_no_scope_over_one_part_is_refused_by_name() -> None:
    store, pointers = _Store(), _Pointers()
    copy = _Copy()
    copy.put(_VOTES, _votes(_state("VA", 5)), "v1", 1)
    copy.put(_RACES, _races([(i, "r") for i in range(90)]), "r1", 1)
    checkpoints = _checkpoints(copy, store, pointers, "a")
    with pytest.raises(ValueError, match="races holds 90 rows"):
        await checkpoints.write(copy.read)
    assert not pointers.entries and not store.objects


async def test_a_whole_write_is_reached_only_at_a_cold_start_a_schema_change_or_a_whole_read() -> None:
    """the proof of _batches_cold_start's, _batches_schema_change's and _batches_rolling_upgrade's declarations (and
    where _batches_unscoped_derived, a finding, is reached): a write reads a table whole from the copy only when no checkpoint
    holds it (the first write), its schema changed, or the follower read it whole from L3; every steady-state write
    (scopes, a patch by key, a replica writing on the previous checkpoint's parts, a stale replica, a derived table's
    scopes) reads none. A whole read is a statement on the read's cursor reading a table with no filter, the one
    each of them runs."""
    store, pointers = _Store(), _Pointers()
    copy = _Copy()
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10)), "v1", 1)
    copy.put(_RACES, _races([(1, "gov"), (2, "sen")]), "r1", 1)
    copy.put(_NEEDLES, _votes(_state("VA", 50) + _state("IL", 10)), "n", 1)
    checkpoints = _derived_checkpoints(copy, store, pointers, "a", _needles())
    assert await checkpoints.write(copy.read) is not None

    def reached() -> list[str]:
        return sorted(table for table in (_VOTES, _RACES, _NEEDLES) if _whole_reads(copy, table))

    assert reached() == sorted([_NEEDLES, _RACES, _VOTES])  # the first checkpoint: every table's cold start
    copy.statements.clear()
    other = _Copy()
    late = _derived_checkpoints(other, store, pointers, "b", _needles())
    other.load(await late.stage())
    # scopes, and a patch by key, on the writer
    copy.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10, votes=3)), "v2", 2)
    _derive(copy, _state("VA", 50) + _state("IL", 10, votes=3))
    copy.put(_RACES, _races([(1, "gov"), (2, "senate")]), "r2", 2)
    checkpoints.applied(11, {_VOTES: frozenset({"state:IL"})})
    assert await checkpoints.write(copy.read) is not None
    assert reached() == []
    checkpoints.applied(1, {_RACES: frozenset()}, {_RACES: [("2",)]})
    # a change of a table with no scope reaches every derived scope: the derived table alone is read whole, by the
    # one whole read that is no declared situation (a finding until the derived table can be narrowed)
    unscoped = _whole_writes(_NEEDLES, "unscoped_derived")
    assert await checkpoints.write(copy.read) is not None
    assert reached() == [_NEEDLES]
    assert _whole_writes(_NEEDLES, "unscoped_derived") == unscoped + 1
    copy.statements.clear()
    # the second replica writes on the previous checkpoint's parts, then goes stale
    other.put(_VOTES, _votes(_state("VA", 50) + _state("IL", 10) + _state("OH", 1)), "v9", 9)
    late.applied(1, {_VOTES: frozenset({"state:OH"})})
    assert await late.write(other.read) is None  # the newest no longer names its parts of every table: stale
    assert not other.statements
    # the follower read votes whole (a reset): that table alone, and what derives from it
    copy.put(_VOTES, _votes(_state("VA", 4)), "v3", 3)
    _derive(copy, _state("VA", 4))
    checkpoints.applied(4, {_VOTES: None}, whys={_VOTES: "reset"})
    assert await checkpoints.write(copy.read) is not None
    assert reached() == [_NEEDLES, _VOTES]


async def test_a_table_read_whole_names_why_and_is_written_by_that_situations_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a table read whole with no why is taken, or a checkpoint writing it whole again reads it by another
    situation's declared read than the follower's why (a reset is a cold start, a migration a schema change, an older
    writer's whole record a rolling upgrade)."""
    copy, checkpoints, _, _ = await _written_world()
    with pytest.raises(ValueError, match="read whole"):
        checkpoints.applied(1, {_VOTES: None})
    reached: list[str] = []
    for kind, read in list(copy_checkpoint.WHOLE_WRITE_READS.items()):

        def recorded(cursor: Any, table: str, rows: int, *, kind: str = kind, read: Any = read) -> Any:
            reached.append(kind)
            return read(cursor, table, rows)

        monkeypatch.setitem(copy_checkpoint.WHOLE_WRITE_READS, kind, recorded)
    for why, kind in (
        ("reset", "cold_start"),
        ("schema_change", "schema_change"),
        ("legacy_whole_record", "rolling_upgrade"),
    ):
        reached.clear()
        checkpoints.applied(1, {_VOTES: None}, whys={_VOTES: why})
        assert await checkpoints.write(copy.read) is not None
        assert reached == [kind], (why, reached)
