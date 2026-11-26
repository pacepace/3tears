"""The copy's apply lane (:mod:`threetears.core.collections.copy_lane`): changes given faster than they apply are
applied in the order given and none lost; a full lane holds its caller (or refuses at once) and never grows; a merge
runs after the changes that make it due; a dead apply worker is replaced and its change given once more, a second
death failing it loudly; a failure reaches its caller, and the changes queued behind it fail too."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_hooks
from threetears.core.collections.copy_lane import (
    ApplyLane,
    ApplyLaneFull,
    ApplyWorkerDied,
    EarlierChangeFailed,
)
from threetears.core.collections.copy_parts import PACK_BELOW_BYTES, PartRule, tier_part
from threetears.core.collections.shared_copy import (
    MARK_TABLE,
    ROWS,
    ApplyHooks,
    ChangePage,
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
RUN_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("run_id", pa.int64()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
BY_STATE = PartRule("results", ("state", "office_key", "candidate_id"), scope_column="state")
BY_RUN = PartRule("runs", ("state", "run_id", "candidate_id"), scope_column="state", run_column="run_id")


def _vote(votes: int, *, candidate: int = 1, state: str = "VA") -> ChangePage:
    """one key's row: a change setting its votes."""
    return ChangePage(
        ROWS,
        "results",
        pa.table(
            {
                "state": [state],
                "office_key": ["G"],
                "candidate_id": pa.array([candidate], pa.int64()),
                "votes": pa.array([votes], pa.int64()),
            },
            schema=SCHEMA,
        ),
    )


def _run(run: int) -> ChangePage:
    """one run's rows over two states."""
    return ChangePage(
        ROWS,
        "runs",
        pa.table(
            {
                "state": pa.array(["VA"] * 20 + ["NC"] * 20, pa.string()),
                "run_id": pa.array([run] * 40, pa.int64()),
                "candidate_id": pa.array(np.tile(np.arange(20), 2), pa.int64()),
                "votes": pa.array(np.full(40, run), pa.int64()),
            }
        ),
    )


def _held(store: SharedArrowStore, table: str) -> pa.Table:
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        rows = mapped.tables[table]
        return pa.table({name: rows[name].combine_chunks() for name in rows.column_names})


def _mark(store: SharedArrowStore) -> int:
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        mark: int = mapped.tables[MARK_TABLE]["mark"][0].as_py()
        return mark


def _number(version: str) -> int:
    return int(version.removeprefix("c-"))


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


async def _backend(
    store: SharedArrowStore,
    hooks: ApplyHooks | None = None,
    *,
    runs: bool = False,
    pack_below: int = PACK_BELOW_BYTES,
) -> SharedCopyBackend:
    table, schema, rule = ("runs", RUN_SCHEMA, BY_RUN) if runs else ("results", SCHEMA, BY_STATE)
    return SharedCopyBackend(
        store, {table: schema}, {table: rule}, hooks=hooks, preload=(copy_hooks.__name__,), pack_below=pack_below
    )


@pytest.fixture
async def backend(store: SharedArrowStore) -> AsyncIterator[SharedCopyBackend]:
    made = await _backend(store)
    yield made
    await made.close()


async def _until(check: object, what: str) -> None:
    """wait (10 s at most) for ``check()`` to hold."""
    assert callable(check)
    for _ in range(1_000):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never: {what}")


class TestOrder:
    async def test_changes_given_faster_than_they_apply_are_applied_in_order_none_lost(
        self, backend: SharedCopyBackend, store: SharedArrowStore
    ) -> None:
        lane = ApplyLane(backend, max_changes=3)
        try:
            # 40 changes given at once to a lane holding 3: each sets one key, so the last given is the one held
            outcomes = await asyncio.gather(*(lane.apply([_vote(n), _vote(n, candidate=n)], mark=n) for n in range(40)))
            numbers = [_number(applied.version) for applied in outcomes]
            assert numbers == list(range(numbers[0], numbers[0] + 40))
            held = _held(store, "results")
            assert held.num_rows == 40
            votes = dict(zip(held["candidate_id"].to_pylist(), held["votes"].to_pylist(), strict=True))
            assert votes == {1: 39, **{n: n for n in range(40) if n != 1}}
            assert _mark(store) == 39
            status = lane.status()
            assert (status.applied, status.failed, status.queued) == (40, 0, 0)
            assert status.waits > 0
        finally:
            await lane.close()

    async def test_a_failure_reaches_its_caller_and_fails_those_queued_behind_it(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        gate = tmp_path / "go"
        backend = await _backend(store, copy_hooks.WaitFor(str(gate)))
        lane = ApplyLane(backend)
        try:
            first = await lane.submit([_vote(1)])
            await _until(lambda: lane.status().queued == 0, "the first change applying")
            unknown = await lane.submit([ChangePage(ROWS, "nowhere", pa.table({"a": [1]}))])
            behind = await lane.submit([_vote(3)])
            gate.touch()
            applied = await first
            with pytest.raises(ValueError, match="nowhere"):
                await unknown
            with pytest.raises(EarlierChangeFailed, match="nowhere"):
                await behind
            assert store.current is not None and store.current.name == applied.version
            # the caller gives the change again, and the lane takes it as any
            again = await lane.apply([_vote(3)])
            assert _number(again.version) == _number(applied.version) + 1
            status = lane.status()
            assert (status.applied, status.failed) == (2, 2)
            assert status.last_error is not None and "nowhere" in status.last_error
        finally:
            await lane.close()
            await backend.close()


class TestBackPressure:
    async def test_a_full_lane_holds_its_caller_and_refuses_a_change_given_now(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        gate = tmp_path / "go"
        backend = await _backend(store, copy_hooks.WaitFor(str(gate)))
        lane = ApplyLane(backend, max_changes=1)
        try:
            applying = await lane.submit([_vote(1)], mark=1)
            await _until(lambda: lane.status().queued == 0, "the first change applying")
            queued = await lane.submit([_vote(2)], mark=2)
            assert lane.status().queued == 1
            with pytest.raises(ApplyLaneFull):
                lane.submit_nowait([_vote(3)], mark=3)
            waiting = asyncio.create_task(lane.apply([_vote(4)], mark=4))
            await asyncio.sleep(0.3)
            # held, not queued: the lane never holds more than its bound
            assert not waiting.done()
            assert (lane.status().queued, lane.status().waits) == (1, 1)
            gate.touch()
            first, second, last = await applying, await queued, await waiting
            assert _number(first.version) < _number(second.version) < _number(last.version)
            assert _mark(store) == 4
            assert _held(store, "results")["votes"].to_pylist() == [4]
        finally:
            gate.touch()
            await lane.close()
            await backend.close()

    async def test_a_change_past_the_byte_bound_waits_and_one_bigger_alone_is_taken_into_an_empty_lane(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        gate = tmp_path / "go"
        backend = await _backend(store, copy_hooks.WaitFor(str(gate)))
        page = _vote(1)
        lane = ApplyLane(backend, max_changes=10, max_bytes=page.rows.nbytes)
        try:
            applying = await lane.submit([page])
            await _until(lambda: lane.status().queued == 0, "the first change applying")
            # twice the bound, into an empty queue: taken
            big = await lane.submit([_vote(2), _vote(2, candidate=2)])
            assert lane.status().queued == 1
            with pytest.raises(ApplyLaneFull):
                lane.submit_nowait([_vote(3)])
            gate.touch()
            await applying
            await big
            assert lane.status().waits == 0
        finally:
            gate.touch()
            await lane.close()
            await backend.close()


class TestMerges:
    async def test_a_merge_runs_after_the_changes_that_make_it_due(self, store: SharedArrowStore) -> None:
        # tier merges themselves, the small scopes unpacked (packing: test_copy_bounded_files)
        backend = await _backend(store, runs=True, pack_below=0)
        lane = ApplyLane(backend)
        try:
            outcomes = await asyncio.gather(*(lane.apply([_run(run)]) for run in range(1, 13)))
            await lane.drained()
            current = store.current
            assert current is not None
            # each state's runs 1-10 merged into one tier; 11 and the newest, 12, parts of their own
            for state in ("NC", "VA"):
                parts = {name for name in held_parts(store.root, current.name, "runs") if name.startswith(state)}
                assert parts == {
                    tier_part(state, 1, 1, 10),
                    tier_part(state, 0, 11, 11),
                    tier_part(state, 0, 12, 12),
                }
            status = lane.status()
            assert status.merges == 2
            # the merges are versions of their own, after the changes: none between a change and its publish
            assert _number(current.name) == _number(outcomes[-1].version) + 1
            held = _held(store, "runs")
            assert held.num_rows == 12 * 40
            assert pc.max(held["run_id"]).as_py() == 12
        finally:
            await lane.close()
            await backend.close()


class TestADeadWorker:
    async def test_a_dead_apply_worker_is_replaced_and_its_change_applied_once_more(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        dies = tmp_path / "die"
        backend = await _backend(store, copy_hooks.DiesWhile(str(dies), once=True))
        lane = ApplyLane(backend)
        try:
            before = await lane.apply([_vote(1)])
            dies.touch()
            applied = await lane.apply([_vote(2)], mark=2)
            assert not dies.exists()
            assert applied.worker not in (before.worker, os.getpid())
            assert _number(applied.version) == _number(before.version) + 1
            assert _held(store, "results")["votes"].to_pylist() == [2]
            assert lane.status().restarts == 1
        finally:
            await lane.close()
            await backend.close()

    async def test_a_second_death_fails_the_change_loudly_and_publishes_nothing(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        dies = tmp_path / "die"
        backend = await _backend(store, copy_hooks.DiesWhile(str(dies), once=False))
        lane = ApplyLane(backend)
        try:
            before = await lane.apply([_vote(1)])
            dies.touch()
            dying = await lane.submit([_vote(2)])
            behind = await lane.submit([_vote(3)])
            with pytest.raises(ApplyWorkerDied):
                await dying
            with pytest.raises(EarlierChangeFailed):
                await behind
            current = store.current
            assert current is not None and current.name == before.version
            assert lane.status().restarts == 2
            dies.unlink()
            # a fresh worker applies what comes next
            after = await lane.apply([_vote(4)])
            assert _number(after.version) == _number(before.version) + 1
        finally:
            await lane.close()
            await backend.close()
