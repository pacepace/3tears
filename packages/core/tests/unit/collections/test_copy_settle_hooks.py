"""The hooks a product's settle and retention plug into the shared copy
(:mod:`threetears.core.collections.shared_copy`): rows let go after a change, by key and inside a pack; the deleted rows as held and the applied version read in the
worker; a whole settle after a start that read L3 whole; a merge policy that sees each run's time and model; and a
table cut again as it crosses the split floor. Each fails on the copy before these hooks (reports R-1)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_hooks
from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.copy_parts import (
    WHOLE_PART,
    DeclaredTable,
    PartDeclaration,
    PartRule,
    is_pack,
    tier_part,
)
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column
from threetears.core.collections.shared_copy import (
    KEYS,
    ROWS,
    ApplyHooks,
    ChangePage,
    MergePolicy,
    RunTimes,
    SharedCopyBackend,
    held_parts,
)
from threetears.core.shared_arrow import SharedArrowStore, read_version

SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
_KEY = ("state", "office_key", "candidate_id")
BY_STATE = PartRule("results", _KEY, scope_column="state")

RUN_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("run_id", pa.int64()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
_RUN_KEY = ("state", "run_id", "candidate_id")
BY_RUN = PartRule("by_run", _RUN_KEY, scope_column="state", run_column="run_id")

HELD_RUNS_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("run_id", pa.int64()),
        pa.field("source", pa.string()),
        pa.field("data_time", pa.string()),
        pa.field("model_version", pa.string()),
    ]
)
HELD_RUNS = PartRule("held_runs", ("state", "run_id", "source"))

_STATES = ("AZ", "GA", "NC", "PA", "VA")


def _rows(state: str, count: int, *, votes: int = 1, start: int = 0) -> pa.Table:
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "office_key": pa.array(["G"] * count, pa.string()),
            "candidate_id": pa.array(np.arange(start, start + count), pa.int64()),
            "votes": pa.array(np.full(count, votes), pa.int64()),
        },
        schema=SCHEMA,
    )


def _run_rows(state: str, run: int, count: int) -> pa.Table:
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "run_id": pa.array([run] * count, pa.int64()),
            "candidate_id": pa.array(np.arange(count), pa.int64()),
            "votes": pa.array(np.full(count, run), pa.int64()),
        },
        schema=RUN_SCHEMA,
    )


def _held(store: SharedArrowStore, table: str, schema: pa.Schema, version: str | None = None) -> pa.Table:
    """a table's rows as a version (the current one by default) holds them, in key order."""
    if version is None:
        current = store.current
        assert current is not None
        version = current.name
    with read_version(store.root, version) as mapped:
        rows = mapped.tables.get(table, schema.empty_table())
        held = pa.table({name: rows[name].combine_chunks() for name in schema.names}, schema=schema)
    return held.sort_by([(column, "ascending") for column in schema.names]).combine_chunks()


def _inodes(store: SharedArrowStore, table: str) -> dict[str, int]:
    current = store.current
    assert current is not None
    return {part: os.stat(path).st_ino for part, path in held_parts(store.root, current.name, table).items()}


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


async def _backend(
    store: SharedArrowStore,
    tables: Sequence[tuple[str, pa.Schema, PartRule]],
    *,
    hooks: ApplyHooks | None = None,
    merges: MergePolicy | None = None,
    pack_below: int = 0,
    run_times: RunTimes | None = None,
    declared: PartDeclaration | None = None,
) -> SharedCopyBackend:
    return SharedCopyBackend(
        store,
        {name: schema for name, schema, _ in tables},
        {name: rule for name, _, rule in tables},
        hooks=hooks,
        merges=merges,
        preload=(copy_hooks.__name__,),
        pack_below=pack_below,
        run_times=run_times,
        declared=declared,
    )


class TestLetGo:
    async def test_rows_of_the_changes_own_table_are_let_go_by_key_in_a_publish_after_it(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if a hook cannot delete rows of a table the change carried (``derive``'s pages to such a table were
        dropped), or if they go in the change's own publish: the change's version shows them, the one after it does
        not, and only the part holding them is rewritten."""
        gone = (("VA", "G", 3), ("VA", "G", 4))
        backend = await _backend(store, [("results", SCHEMA, BY_STATE)], hooks=copy_hooks.LetsGo("results", _KEY, gone))
        # the change's own version, read as its let-go starts (a superseded version is let go once unheld)
        seen: list[tuple[str, int]] = []
        let_go_after = vars(SharedCopyBackend)["_let_go_after"]

        async def reading(self: SharedCopyBackend, version: str, *args: Any) -> Any:
            current = store.current
            assert current is not None
            seen.append((current.name, _held(store, "results", SCHEMA, current.name).num_rows))
            return await let_go_after(self, version, *args)

        monkeypatch.setattr(SharedCopyBackend, "_let_go_after", reading)
        try:
            rows = pa.concat_tables([_rows(state, 10) for state in _STATES])
            applied = await backend.apply([ChangePage(ROWS, "results", rows)])
            assert seen == [(applied.version, 50)]
            assert applied.let_go is not None and applied.let_go.version is not None
            assert store.current is not None and store.current.name == applied.let_go.version
            assert applied.let_go.version != applied.version
            assert applied.let_go.parts == {"results": {"VA": True}}
            assert applied.let_go.rows == {"results": 2}
            after = _held(store, "results", SCHEMA)
            va = after.filter(pc.equal(after["state"], "VA"))
            assert sorted(va["candidate_id"].to_pylist()) == [0, 1, 2, 5, 6, 7, 8, 9]
            assert after.num_rows == 48
        finally:
            await backend.close()

    async def test_a_whole_run_is_let_go_inside_a_pack_its_other_scopes_kept(self, store: SharedArrowStore) -> None:
        """Flips if a let-go drops a pack whole, or cannot reach a packed scope's rows: a run of one small state let go
        by its scope and run rewrites the pack holding it, every other state's and run's rows in it kept."""
        backend = await _backend(
            store,
            [("by_run", RUN_SCHEMA, BY_RUN)],
            hooks=copy_hooks.LetsGo("by_run", ("state", "run_id"), (("VA", 1),)),
            pack_below=1 << 20,
        )
        try:
            rows = pa.concat_tables([_run_rows(state, run, 20) for state in ("AZ", "GA", "VA") for run in (1, 2, 3)])
            applied = await backend.apply([ChangePage(ROWS, "by_run", rows)], whole={"by_run"})
            parts = held_parts(store.root, applied.version, "by_run")
            assert len(parts) == 1 and all(is_pack(part) for part in parts), sorted(parts)
            assert _held(store, "by_run", RUN_SCHEMA, applied.version).num_rows == 180
            assert applied.let_go is not None
            assert applied.let_go.parts == {"by_run": {next(iter(parts)): True}}
            assert applied.let_go.rows == {"by_run": 20}
            after = _held(store, "by_run", RUN_SCHEMA)
            assert after.num_rows == 160
            left = after.filter(pc.equal(after["state"], "VA"))
            assert sorted(set(left["run_id"].to_pylist())) == [2, 3]
            assert sorted(set(after["state"].to_pylist())) == ["AZ", "GA", "VA"]
        finally:
            await backend.close()


class TestSettleInputs:
    async def test_the_settle_sees_deleted_rows_as_held_and_reads_the_applied_version(
        self, store: SharedArrowStore
    ) -> None:
        """Flips if the hooks see only the deleted keys, or cannot read a table the change did not carry: the deleted
        rows come with every column as the version held them, and the touched state's rows of ``lookup`` are read from
        the version with the change applied, in the apply worker."""
        lookup = PartRule("lookup", _KEY, scope_column="state")
        gone_rule = PartRule(copy_hooks.GONE, _KEY)
        seen_rule = PartRule(copy_hooks.SEEN, ("state",))
        backend = await _backend(
            store,
            [
                ("results", SCHEMA, BY_STATE),
                ("lookup", SCHEMA, lookup),
                (copy_hooks.GONE, SCHEMA, gone_rule),
                (copy_hooks.SEEN, copy_hooks.SEEN_SCHEMA, seen_rule),
            ],
            hooks=copy_hooks.SettleInputs(),
        )
        try:
            first = [
                ChangePage(ROWS, "results", pa.concat_tables([_rows(state, 5, votes=7) for state in _STATES])),
                ChangePage(ROWS, "lookup", pa.concat_tables([_rows(state, n + 1) for n, state in enumerate(_STATES)])),
            ]
            await backend.apply(first)
            deleted = _rows("GA", 2, votes=7).select(list(_KEY))
            applied = await backend.apply([ChangePage(KEYS, "results", deleted)])
            assert applied.worker != os.getpid()
            gone = _held(store, copy_hooks.GONE, SCHEMA)
            assert gone.equals(_rows("GA", 2, votes=7).sort_by([(c, "ascending") for c in SCHEMA.names]))
            seen = _held(store, copy_hooks.SEEN, copy_hooks.SEEN_SCHEMA)
            # GA, the state the change touched, holds 2 lookup rows (``n + 1``); the first change touched every state
            assert dict(zip(seen["state"].to_pylist(), seen["rows"].to_pylist(), strict=True))["GA"] == 2
        finally:
            await backend.close()


class TestWholeSettle:
    async def test_changes_before_a_whole_settle_derive_nothing_and_it_settles_every_scope_in_one_publish(
        self, store: SharedArrowStore
    ) -> None:
        """Flips with no whole settle: deferred, a change derives nothing; the whole settle derives every state's
        totals and lets go of the rows of no votes in one version; changes after it derive as they apply."""
        totals = PartRule(copy_hooks.TOTALS, ("state",))
        backend = await _backend(
            store,
            [("results", SCHEMA, BY_STATE), (copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA, totals)],
            hooks=copy_hooks.WholeTotals(),
        )
        try:
            backend.defer_hooks_until_whole()
            assert backend.whole_settle_due
            rows = pa.concat_tables(
                [_rows(state, 4, votes=2) for state in _STATES] + [_rows("VA", 3, votes=0, start=10)]
            )
            await backend.apply([ChangePage(ROWS, "results", rows)], whole={"results"})
            assert _held(store, copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA).num_rows == 0
            before = store.current
            assert before is not None
            settled = await backend.settle_whole()
            assert settled is not None
            assert not backend.whole_settle_due
            assert settled.version == f"c-{int(before.name[2:]) + 1}"
            assert store.current is not None and store.current.name == settled.version
            held = _held(store, copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA)
            assert dict(zip(held["state"].to_pylist(), held["votes"].to_pylist(), strict=True)) == dict.fromkeys(
                _STATES, 8
            )
            results = _held(store, "results", SCHEMA)
            assert results.num_rows == 20 and pc.min(results["votes"]).as_py() == 2
            assert settled.let_go is not None and settled.let_go.rows == {"results": 3}
            # derived as it applies from here
            await backend.apply([ChangePage(ROWS, "results", _rows("AZ", 1, votes=100, start=50))])
            held = _held(store, copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA)
            assert dict(zip(held["state"].to_pylist(), held["votes"].to_pylist(), strict=True))["AZ"] == 108
        finally:
            await backend.close()


class TestMergeSeesRunTimes:
    async def test_the_policy_merges_by_each_runs_time_and_model_from_the_held_runs(
        self, store: SharedArrowStore
    ) -> None:
        """Flips if a merge policy sees no runs' times: runs of one model an hour behind the newest merge, the newest
        run and a run of another model do not; a run named twice is seen once, at its newest time."""
        held_runs = pa.table(
            {
                "state": ["VA"] * 6,
                "run_id": [1, 2, 2, 3, 4, 5],
                "source": ["a", "a", "b", "a", "a", "a"],
                "data_time": [
                    "2026-11-03T20:00:00Z",
                    "2026-11-03T20:01:00Z",
                    "2026-11-03T20:02:00+00:00",
                    "2026-11-03T20:03:00Z",
                    "2026-11-03T20:04:00Z",
                    "2026-11-03T21:30:00Z",
                ],
                "model_version": ["m1", "m1", "m1", "m1", "m2", "m1"],
            },
            schema=HELD_RUNS_SCHEMA,
        )
        backend = await _backend(
            store,
            [("by_run", RUN_SCHEMA, BY_RUN), ("held_runs", HELD_RUNS_SCHEMA, HELD_RUNS)],
            merges=copy_hooks.OlderThan("m1", 3600.0),
            run_times=RunTimes("held_runs", "state", "run_id", "data_time", "model_version"),
        )
        try:
            await backend.apply([ChangePage(ROWS, "held_runs", held_runs)])
            for run in (1, 2, 3, 4, 5):
                await backend.apply([ChangePage(ROWS, "by_run", _run_rows("VA", run, 5))])
            merged = await backend.merge()
            assert merged is not None
            assert merged.parts == tuple(tier_part("VA", 0, run, run) for run in (1, 2, 3))
            assert merged.into == tier_part("VA", 1, 1, 3)
            assert await backend.merge() is None
        finally:
            await backend.close()


def _followed(name: str) -> FollowedTable:
    columns = (
        Column(name="state", column_type=STRING_TYPE),
        Column(name="office_key", column_type=STRING_TYPE),
        Column(name="candidate_id", column_type=BIGINT_TYPE),
        Column(name="votes", column_type=BIGINT_TYPE),
    )
    return FollowedTable(name=name, key=_KEY, columns=columns, scope=("state", "state:"))


class TestRecutAsATableGrows:
    async def test_a_table_crossing_the_floor_is_cut_by_its_declared_scope_from_the_next_change(
        self, store: SharedArrowStore
    ) -> None:
        """Flips if a table's parts are fixed as made: declared with its scope and no hardcoded list, a table under the
        floor is one part; once a publish leaves it over the floor its rule splits it, and the next change to it cuts
        it a part a state."""
        declared = PartDeclaration(tables=(_followed("results"),), floor=8_000)
        backend = await _backend(store, [("results", SCHEMA, PartRule("results", _KEY))], declared=declared)
        try:
            small = await backend.apply([ChangePage(ROWS, "results", _rows("VA", 20))])
            assert small.rules_changed == frozenset()
            assert sorted(_inodes(store, "results")) == [WHOLE_PART]
            grown = await backend.apply(
                [ChangePage(ROWS, "results", pa.concat_tables([_rows(state, 200, start=100) for state in _STATES]))]
            )
            assert grown.rules_changed == frozenset({"results"})
            assert backend.rules["results"].split and backend.rules["results"].scope_column == "state"
            await backend.apply([ChangePage(ROWS, "results", _rows("GA", 1, votes=9))])
            assert sorted(_inodes(store, "results")) == sorted(_STATES)
            assert _held(store, "results", SCHEMA).num_rows == 20 + 1000 + 1
        finally:
            await backend.close()

    async def test_a_derived_table_declared_by_its_shape_alone_is_cut_by_its_scope(
        self, store: SharedArrowStore
    ) -> None:
        # the totals L3 does not hold: a name, a key and a scope, no L3 columns
        totals = DeclaredTable(copy_hooks.TOTALS, ("state",), ("state", "state:"))
        declared = PartDeclaration(tables=(_followed("results"), totals), floor=1)
        backend = await _backend(
            store,
            [
                ("results", SCHEMA, PartRule("results", _KEY)),
                (copy_hooks.TOTALS, copy_hooks.TOTALS_SCHEMA, PartRule(copy_hooks.TOTALS, ("state",))),
            ],
            hooks=copy_hooks.WholeTotals(),
            declared=declared,
        )
        try:
            first = await backend.apply([ChangePage(ROWS, "results", _rows("VA", 5))])
            assert copy_hooks.TOTALS in first.rules_changed
            assert backend.rules[copy_hooks.TOTALS].scope_column == "state"
            assert sorted(_inodes(store, copy_hooks.TOTALS)) == [WHOLE_PART]
            await backend.apply([ChangePage(ROWS, "results", _rows("GA", 3))])
            assert sorted(_inodes(store, copy_hooks.TOTALS)) == ["GA", "VA"]
        finally:
            await backend.close()
