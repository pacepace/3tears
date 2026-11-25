"""The copy's apply in a worker process (:mod:`threetears.core.collections.shared_copy`): it never runs in the serving
process, a change rewrites only the parts it lands in and the new version names every other part's file as the
version before held it, the version holds exactly what the same change applied by DuckDB holds, and the apply makes
no Python object a row."""

from __future__ import annotations

import os
import sys
import threading
import tracemalloc
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_hooks
from threetears.core import arrow_keys
from threetears.core.collections.copy_parts import WHOLE_PART, PartRule, tier_part
from threetears.core.collections.shared_copy import (
    KEYS,
    LOCATED_PREFIX,
    MARK_TABLE,
    ROWS,
    ApplyPlan,
    ChangePage,
    SharedCopyBackend,
    apply_job,
    held_parts,
)
from threetears.core.shared_arrow import JobContext, SharedArrowStore, read_version

_STATES = pa.array(["AZ", "GA", "NC", "PA", "VA"], pa.string())

SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
RUN_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("run_id", pa.int64()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)

_KEY = ("state", "office_key", "candidate_id")
_RACE_KEY = ("office_key", "candidate_id")
_RUN_KEY = ("state", "run_id", "candidate_id")

BY_STATE = PartRule("results", _KEY, scope_column="state")
LOCATED = PartRule("results", _RACE_KEY, scope_column="state", locator="office_key")
WHOLE = PartRule("results", _KEY)
BY_RUN = PartRule("runs", _RUN_KEY, scope_column="state", run_column="run_id")


def _rows(count: int, seed: int, *, races: int | None = None) -> pa.Table:
    """rows over five states, unsorted; each race (``office_key``) in one state, a key at most once."""
    rng = np.random.default_rng(seed)
    race = rng.integers(0, races or max(count // 4, 1), count)
    candidate = rng.integers(0, 1 << 40, count)
    rows = pa.table(
        {
            "state": _STATES.take(pa.array(race % len(_STATES))),
            "office_key": pc.cast(pa.array(race), pa.string()),
            "candidate_id": pa.array(candidate, pa.int64()),
            "votes": pa.array(rng.integers(0, 1_000, count), pa.int64()),
        }
    )
    return rows.take(pa.array(arrow_keys.last_rows(rows, _KEY)))


def _version_rows(store: SharedArrowStore, table: str, schema: pa.Schema) -> pa.Table:
    """a table's rows as the current version holds them, every part's together."""
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        rows = mapped.tables.get(table, schema.empty_table())
        return pa.table({name: rows[name].combine_chunks() for name in schema.names}, schema=schema)


def _sorted(rows: pa.Table, key: Sequence[str]) -> pa.Table:
    return rows.sort_by([(column, "ascending") for column in key]).combine_chunks()


def _files(store: SharedArrowStore, table: str) -> dict[str, tuple[Path, int]]:
    """each part of ``table`` the current version holds: its file and the file's inode."""
    current = store.current
    assert current is not None
    return {part: (path, os.stat(path).st_ino) for part, path in held_parts(store.root, current.name, table).items()}


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


async def _backend(
    store: SharedArrowStore, *tables: tuple[str, pa.Schema, PartRule], **kwargs: Any
) -> SharedCopyBackend:
    return SharedCopyBackend(
        store, {name: schema for name, schema, _ in tables}, {name: rule for name, _, rule in tables}, **kwargs
    )


@pytest.fixture
async def by_state(store: SharedArrowStore) -> AsyncIterator[SharedCopyBackend]:
    backend = await _backend(store, ("results", SCHEMA, BY_STATE))
    yield backend
    await backend.close()


class _Calls:
    """every call this process makes, on any thread, to a function named in ``names`` (by its code's name)."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self.seen: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code.co_name in self.names:
            with self._lock:
                self.seen.append((frame.f_code.co_name, threading.get_ident()))

    def __enter__(self) -> _Calls:
        sys.setprofile(self)
        threading.setprofile_all_threads(self)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.setprofile_all_threads(None)
        sys.setprofile(None)


class TestTheApplyRunsInTheWorker:
    async def test_an_apply_never_runs_in_the_serving_process(self, by_state: SharedCopyBackend) -> None:
        await by_state.apply([ChangePage(ROWS, "results", _rows(2_000, 1))])
        change = [ChangePage(ROWS, "results", _rows(300, 2)), ChangePage(KEYS, "results", _rows(50, 1).select(_KEY))]
        # the apply's own work: the job, its settle of the pages, its match of held rows by key. Any of them called in
        # this process, on any thread, is the apply run here
        with _Calls(frozenset({"apply_job", "last_rows", "rows_named", "key_codes"})) as calls:
            applied = await by_state.apply(change)
        assert calls.seen == []
        assert applied.worker != os.getpid()
        assert applied.rows > 0

    async def test_the_change_is_published_with_its_mark(
        self, by_state: SharedCopyBackend, store: SharedArrowStore
    ) -> None:
        applied = await by_state.apply([ChangePage(ROWS, "results", _rows(100, 1))], mark=41)
        current = store.current
        assert current is not None and current.name == applied.version
        with read_version(store.root, current.name) as mapped:
            assert mapped.tables[MARK_TABLE]["mark"].to_pylist() == [41]


class TestAChangeRewritesOnlyItsParts:
    async def test_untouched_parts_are_the_version_befores_files(
        self, by_state: SharedCopyBackend, store: SharedArrowStore
    ) -> None:
        await by_state.apply([ChangePage(ROWS, "results", _rows(5_000, 1))])
        before = _files(store, "results")
        assert sorted(before) == _STATES.to_pylist()
        held = _version_rows(store, "results", SCHEMA)
        va = held.filter(pc.equal(held["state"], "VA"))
        changed = va.slice(0, 3).set_column(3, "votes", pa.array([7, 8, 9], pa.int64()))
        applied = await by_state.apply(
            [ChangePage(ROWS, "results", changed), ChangePage(KEYS, "results", va.slice(3, 2).select(_KEY))]
        )
        after = _files(store, "results")
        assert applied.parts == {"results": {"VA": True}}
        for state in ("AZ", "GA", "NC", "PA"):
            assert after[state] == before[state]
        assert after["VA"][1] != before["VA"][1]

    async def test_a_part_the_change_empties_is_dropped(
        self, by_state: SharedCopyBackend, store: SharedArrowStore
    ) -> None:
        rows = _rows(1_000, 1)
        await by_state.apply([ChangePage(ROWS, "results", rows)])
        before = _files(store, "results")
        az = rows.filter(pc.equal(rows["state"], "AZ")).select(_KEY)
        applied = await by_state.apply([ChangePage(KEYS, "results", az)])
        assert applied.parts == {"results": {"AZ": False}}
        after = _files(store, "results")
        assert sorted(after) == ["GA", "NC", "PA", "VA"]
        assert all(after[state] == before[state] for state in after)


def _duckdb_apply(
    db: duckdb.DuckDBPyConnection, table: str, key: Sequence[str], pages: Sequence[ChangePage], whole: bool
) -> None:
    """the same change in DuckDB, in this process: each page in order, a key's last row kept."""
    keys = ", ".join(key)
    types = dict(db.execute(f"SELECT column_name, column_type FROM (DESCRIBE {table})").fetchall())  # noqa: S608
    same = " AND ".join(f"{table}.{column} = CAST(p.{column} AS {types[column]})" for column in key)
    if whole:
        db.execute(f"DELETE FROM {table}")  # noqa: S608
    for page in pages:
        db.register("page", page.rows)
        db.execute(f"DELETE FROM {table} USING page AS p WHERE {same}")  # noqa: S608
        if page.kind == ROWS:
            db.execute(
                f"INSERT INTO {table} SELECT * EXCLUDE (n) FROM (SELECT *, row_number() OVER () AS n FROM page) "  # noqa: S608
                f"QUALIFY n = max(n) OVER (PARTITION BY {keys})"
            )
        db.unregister("page")


def _as_text(keys: pa.Table) -> pa.Table:
    """keys as a copy follower stages them: every column as text."""
    return pa.table({name: pc.cast(keys[name], pa.string()) for name in keys.column_names})


class TestTheVersionEqualsDuckDB:
    @pytest.mark.parametrize("rule", [BY_STATE, LOCATED, WHOLE], ids=["by-state", "located", "whole"])
    async def test_a_run_of_changes_leaves_what_duckdb_leaves(self, store: SharedArrowStore, rule: PartRule) -> None:
        backend = await _backend(store, ("results", SCHEMA, rule))
        db = duckdb.connect()
        db.execute(
            f"CREATE TABLE results (state VARCHAR, office_key VARCHAR, candidate_id BIGINT, votes BIGINT, "
            f"PRIMARY KEY ({', '.join(rule.key)}))"
        )
        try:
            rng = np.random.default_rng(7)
            held = _rows(3_000, 1, races=400)
            for step in range(8):
                fresh = _rows(400, 100 + step, races=400)
                current = _version_rows(store, "results", SCHEMA) if step else held
                some = current.take(pa.array(rng.choice(current.num_rows, 200, replace=False)))
                again = some.slice(0, 100).set_column(3, "votes", pa.array(rng.integers(0, 9, 100), pa.int64()))
                # a page naming a key twice keeps its last row; a key deleted, then written again, is back
                twice = pa.concat_tables(
                    [again.slice(0, 10), again.slice(0, 10).set_column(3, "votes", pa.array([1] * 10, pa.int64()))]
                )
                pages = [
                    ChangePage(ROWS, "results", held if step == 0 else fresh),
                    ChangePage(ROWS, "results", again),
                    ChangePage(KEYS, "results", _as_text(some.slice(100, 100).select(list(rule.key)))),
                    ChangePage(KEYS, "results", _as_text(again.slice(0, 20).select(list(rule.key)))),
                    ChangePage(ROWS, "results", twice),
                ]
                whole = step == 4
                await backend.apply(pages, whole={"results"} if whole else ())
                _duckdb_apply(db, "results", rule.key, pages, whole)
                expected = _sorted(db.execute("SELECT * FROM results").fetch_arrow_table().cast(SCHEMA), _KEY)
                assert _sorted(_version_rows(store, "results", SCHEMA), _KEY).equals(expected), f"step {step}"
            if rule.locator is not None:
                current = store.current
                assert current is not None
                assert held_parts(store.root, current.name, f"{LOCATED_PREFIX}results").keys() == {WHOLE_PART}
        finally:
            db.close()
            await backend.close()

    async def test_per_run_tiers_and_their_merge_leave_what_duckdb_leaves(self, store: SharedArrowStore) -> None:
        backend = await _backend(store, ("runs", RUN_SCHEMA, BY_RUN))
        db = duckdb.connect()
        db.execute("CREATE TABLE runs (state VARCHAR, run_id BIGINT, candidate_id BIGINT, votes BIGINT)")
        try:
            for run in range(1, 13):
                rows = pa.table(
                    {
                        "state": pa.array(["VA"] * 50 + ["NC"] * 50, pa.string()),
                        "run_id": pa.array([run] * 100, pa.int64()),
                        "candidate_id": pa.array(np.arange(100), pa.int64()),
                        "votes": pa.array(np.full(100, run), pa.int64()),
                    }
                )
                pages = [ChangePage(ROWS, "runs", rows)]
                await backend.apply(pages)
                _duckdb_apply(db, "runs", _RUN_KEY, pages, False)
            before = _files(store, "runs")
            assert len(before) == 24
            merged = await backend.merge()
            assert merged is not None
            # the lowest level due, the first scope by name: NC's runs 1-10, its newest (12) and 11 left alone
            assert (merged.table, merged.into) == ("runs", tier_part("NC", 1, 1, 10))
            assert merged.parts == tuple(tier_part("NC", 0, run, run) for run in range(1, 11))
            after = _files(store, "runs")
            assert sorted(set(after) - set(before)) == [merged.into]
            assert all(after[part] == before[part] for part in after if part != merged.into)
            # a change to a merged run rewrites the tier holding it
            fix = pa.table(
                {
                    "state": ["NC"],
                    "run_id": pa.array([3], pa.int64()),
                    "candidate_id": pa.array([5], pa.int64()),
                    "votes": pa.array([-1], pa.int64()),
                }
            )
            nc_into = merged.into
            pages = [ChangePage(ROWS, "runs", fix)]
            applied = await backend.apply(pages)
            _duckdb_apply(db, "runs", _RUN_KEY, pages, False)
            assert applied.parts == {"runs": {nc_into: True}}
            expected = _sorted(db.execute("SELECT * FROM runs").fetch_arrow_table().cast(RUN_SCHEMA), _RUN_KEY)
            assert _sorted(_version_rows(store, "runs", RUN_SCHEMA), _RUN_KEY).equals(expected)
        finally:
            db.close()
            await backend.close()


class TestShapeAndHooks:
    async def test_a_table_crossing_into_split_is_cut_whole_as_its_change_lands(self, store: SharedArrowStore) -> None:
        backend = await _backend(store, ("results", SCHEMA, WHOLE))
        try:
            rows = _rows(2_000, 1)
            await backend.apply([ChangePage(ROWS, "results", rows)])
            assert sorted(_files(store, "results")) == [WHOLE_PART]
            backend.use_rules({"results": BY_STATE})
            more = _rows(10, 5)
            applied = await backend.apply([ChangePage(ROWS, "results", more)])
            assert applied.parts["results"][WHOLE_PART] is False
            assert sorted(_files(store, "results")) == _STATES.to_pylist()
            both = pa.concat_tables([rows, more])
            expected = _sorted(both.take(pa.array(arrow_keys.last_rows(both, _KEY))), _KEY)
            assert _sorted(_version_rows(store, "results", SCHEMA), _KEY).equals(expected)
        finally:
            await backend.close()

    async def test_derived_tables_land_in_the_changes_own_publish(self, store: SharedArrowStore) -> None:
        backend = await _backend(
            store,
            ("results", SCHEMA, BY_STATE),
            (copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA, PartRule(copy_hooks.TOTALS, ("state",))),
            hooks=copy_hooks.StateTotals(),
            preload=(copy_hooks.__name__,),
        )
        try:
            rows = _rows(2_000, 1)
            first = await backend.apply([ChangePage(ROWS, "results", rows)])
            va = rows.filter(pc.equal(rows["state"], "VA"))
            second = await backend.apply([ChangePage(KEYS, "results", va.slice(0, 10).select(_KEY))])
            assert int(second.version.split("-")[1]) == int(first.version.split("-")[1]) + 1
            totals = _version_rows(store, copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA)
            held = _version_rows(store, "results", SCHEMA)
            by_state = {
                state: pc.sum(held.filter(pc.equal(held["state"], state))["votes"]).as_py()
                for state in _STATES.to_pylist()
            }
            assert dict(zip(totals["state"].to_pylist(), totals["votes"].to_pylist(), strict=True)) == by_state
        finally:
            await backend.close()


class TestArrowEndToEnd:
    """Arrow's buffers live in its own memory pool, which ``tracemalloc`` does not see; every Python object does, and
    every NumPy buffer. A 4k-row change to 200k held rows in five parts makes NumPy arrays of row numbers and key codes
    (measured 2026-11-25: 1.16 MB at its peak, about 5 bytes a held row, the held side freed part by part) and a few
    objects a part or a page; one Python object a held row (a 28-byte int at least, its list's 8-byte slot) would be
    7 MB more. The change's own rows cost NumPy about 50 bytes a row (its key codes, their sort, the row numbers), so
    the change is kept small here: the bound is the held rows'."""

    _ROWS = 200_000

    async def test_an_apply_makes_no_python_object_a_row(
        self, by_state: SharedCopyBackend, store: SharedArrowStore
    ) -> None:
        held = _rows(self._ROWS, 1)
        await by_state.apply([ChangePage(ROWS, "results", held)])
        some = held.take(pa.array(np.random.default_rng(3).choice(held.num_rows, 4_000, replace=False)))
        change = some.slice(0, 3_000).set_column(3, "votes", pa.array(np.zeros(3_000, np.int64)))
        pages = [ChangePage(ROWS, "results", change), ChangePage(KEYS, "results", some.slice(3_000).select(_KEY))]
        current = store.current
        assert current is not None
        plan = ApplyPlan(
            root=str(store.root),
            schemas=(("results", SCHEMA),),
            rules=(("results", BY_STATE),),
            pages=tuple((page.kind, page.table) for page in pages),
            whole=frozenset(),
            hooks=None,
        )
        slot = store.reserve(256 << 20)
        with store.handing({f"p{at}": page.rows for at, page in enumerate(pages)}) as handed, handed.mapped() as tables:
            with read_version(store.root, current.name) as version:
                context = JobContext(version=version, result=slot, tables=tables)
                peak = _python_bytes(lambda: apply_job(context, plan))
        segment = store.claim(slot)
        assert segment is not None
        segment.close()
        # every part touched, each rewritten: the bound is the apply's own, not an idle one's
        assert peak < 5 << 19, f"{peak} bytes of Python objects for {self._ROWS} held rows"


def _python_bytes(run: Callable[[], Any]) -> int:
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak
