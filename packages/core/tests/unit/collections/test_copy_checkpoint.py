"""a copy's checkpoint loads each table on its own and writes only what changed.

The contract this pins (:mod:`threetears.core.collections.copy_checkpoint`):

- a table whose parts do not read back loads from the previous checkpoint, with that one's mark; with no previous it
  is left out as ``cold_start`` and every other table loads;
- a table whose schema digest differs is left out as ``schema_change``; the rest load;
- a write rewrites only the (table, scope) parts changed since this replica's last checkpoint and names the others'
  objects again; a change of unknown tables writes every part;
- a failed or beaten write keeps what it was to write, for the next;
- no retire takes an object the newest or the previous checkpoint names;
- two code versions keep each other's tables in the one pointer, and each loads what it reads the same way.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from prometheus_client import REGISTRY  # noqa: E402

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
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
_VOTES_KEY = ("state", "geo_id")
_RACES_KEY = ("race_id",)


def _metadata(*, share: bool = False) -> MetaData:
    metadata = MetaData()
    columns = [
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
    ]
    if share:
        columns.append(SaColumn("share", BigInteger))
    Table(_VOTES, metadata, *columns)
    Table(_RACES, metadata, SaColumn("race_id", BigInteger, primary_key=True), SaColumn("name", String(32)))
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

    def put(self, table: str, rows: Any, mark: str, stamp: int) -> None:
        key = _VOTES_KEY if table == _VOTES else _RACES_KEY
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
            yield cursor, stamp, marks

    def held(self, table: str) -> list[tuple[Any, ...]]:
        columns = "state, geo_id, votes" if table == _VOTES else "race_id, name"
        with self.backend.read_snapshot() as cursor:
            return sorted(cursor.execute(f"SELECT {columns} FROM {table}").fetchall())  # noqa: S608


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
    checkpoints.applied(53, {_VOTES: frozenset({"state:VA"}), _RACES: frozenset()})
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


async def test_a_touched_table_with_no_scope_is_written_whole_and_untouched_tables_are_not() -> None:
    copy, checkpoints, store, _ = await _written_world()
    first = (await checkpoints.newest())[0]
    copy.put(_RACES, _races([(1, "gov"), (4, "ag")]), "r2", 2)
    checkpoints.applied(2, {_RACES: frozenset()})
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


async def test_a_change_of_unknown_tables_writes_every_part() -> None:
    copy, checkpoints, _, _ = await _written_world()
    first = (await checkpoints.newest())[0]
    checkpoints.applied(1)
    before = _writes("full", "written")
    second = await checkpoints.write(copy.read)
    assert not second.names() & first.names()
    assert _writes("full", "written") == before + 1


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
    checkpoints.applied(1, {_RACES: frozenset()})
    kept = await checkpoints.write(new.read)
    assert kept is not None
    assert kept.tables[_VOTES] == kept.previous.tables[_VOTES]
    # once it has read votes whole, its checkpoint holds its own
    new.put(_VOTES, _votes(_state("VA", 5), share=True), "v2", 2)
    checkpoints.applied(5, {_VOTES: None})
    assert checkpoints.due()
    mine = await checkpoints.write(new.read)
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
    only_races.applied(1, {_RACES: frozenset()})
    written = await only_races.write(copy.read)
    assert written is not None
    assert written.tables[_VOTES] == written.previous.tables[_VOTES]
