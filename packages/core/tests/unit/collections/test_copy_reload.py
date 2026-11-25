"""The copy's reload lane (:mod:`threetears.core.collections.copy_reload`): a reload staging for a long time never
delays an apply; a change given during a reload is in the version that publishes it; the publish is ordered with the
apply lane's changes; a dead reload worker is replaced; a cancelled reload publishes nothing; a per-scope reload
replaces its scope's parts alone; the serving process holds none of a reload's rows while it stages."""

from __future__ import annotations

import asyncio
import gc
import os
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_hooks
from threetears.core.collections.copy_lane import ApplyLane, ReloadCancelled, ReloadOverrun
from threetears.core.collections.copy_parts import PartRule
from threetears.core.collections.copy_reload import ReloadLane, ReloadWorkerDied
from threetears.core.collections.shared_copy import (
    KEYS,
    MARK_TABLE,
    ROWS,
    ApplyHooks,
    ChangePage,
    ReloadScope,
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
BY_STATE = PartRule("results", ("state", "office_key", "candidate_id"), scope_column="state")
#: the same columns held as one part (a scoped table below the size floor)
WHOLE = PartRule("small", ("state", "office_key", "candidate_id"))
OTHER = PartRule("other", ("state", "office_key", "candidate_id"), scope_column="state")


def _rows(state: str, candidates: range, votes: int) -> pa.Table:
    count = len(candidates)
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "office_key": pa.array(["G"] * count, pa.string()),
            "candidate_id": pa.array(np.arange(candidates.start, candidates.stop), pa.int64()),
            "votes": pa.array(np.full(count, votes), pa.int64()),
        },
        schema=SCHEMA,
    )


def _page(state: str, candidates: range, votes: int, table: str = "results") -> ChangePage:
    return ChangePage(ROWS, table, _rows(state, candidates, votes))


def _held(store: SharedArrowStore, table: str) -> dict[tuple[str, int], int]:
    """the table's rows the current version holds: (state, candidate) -> votes."""
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        rows = mapped.tables[table]
        return {
            (state, candidate): votes
            for state, candidate, votes in zip(
                rows["state"].to_pylist(), rows["candidate_id"].to_pylist(), rows["votes"].to_pylist(), strict=True
            )
        }


def _mark(store: SharedArrowStore) -> int:
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        mark: int = mapped.tables[MARK_TABLE]["mark"][0].as_py()
        return mark


def _number(version: str) -> int:
    return int(version.removeprefix("c-"))


async def _until(check: object, what: str) -> None:
    """wait (20 s at most) for ``check()`` to hold."""
    assert callable(check)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never: {what}")


@dataclass
class Copy:
    store: SharedArrowStore
    backend: SharedCopyBackend
    lane: ApplyLane
    reloads: ReloadLane


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    # room for an apply's result piece beside a staging reload's (each up to 1 GiB, ``shared_copy``)
    return SharedArrowStore(tmp_path / "shm", max_bytes=4 << 30)


async def _copy(
    store: SharedArrowStore,
    *,
    hooks: ApplyHooks | None = None,
    check: ApplyHooks | None = None,
    max_replay_bytes: int = 1 << 20,
) -> Copy:
    backend = SharedCopyBackend(
        store,
        {"results": SCHEMA, "small": SCHEMA, "other": SCHEMA},
        {"results": BY_STATE, "small": WHOLE, "other": OTHER},
        hooks=hooks,
        preload=(copy_hooks.__name__,),
    )
    lane = ApplyLane(backend)
    reloads = ReloadLane(backend, lane, check=check, preload=(copy_hooks.__name__,), max_replay_bytes=max_replay_bytes)
    return Copy(store, backend, lane, reloads)


async def _closed(copy: Copy) -> None:
    await copy.reloads.close()
    await copy.lane.close()
    await copy.backend.close()


@pytest.fixture
async def gated(store: SharedArrowStore, tmp_path: Path) -> AsyncIterator[tuple[Copy, Path]]:
    """a copy whose reloads stage until a file exists."""
    gate = tmp_path / "reload-go"
    copy = await _copy(store, check=copy_hooks.WaitFor(str(gate)))
    await copy.lane.apply([_page("VA", range(0, 10), 1), _page("NC", range(0, 10), 1)], mark=1)
    try:
        yield copy, gate
    finally:
        gate.touch()
        await _closed(copy)


async def _timed(apply: Awaitable[object]) -> float:
    started = time.monotonic()
    await apply
    return time.monotonic() - started


class TestAReloadNeverDelaysAnApply:
    async def test_applies_keep_their_latency_while_a_reload_stages(self, gated: tuple[Copy, Path]) -> None:
        copy, gate = gated
        before = [await _timed(copy.lane.apply([_page("NC", range(0, 1), n, "other")])) for n in range(5)]
        reload = await copy.reloads.reload([_page("VA", range(0, 50), 100), _page("NC", range(0, 50), 100)])
        await _until(lambda: copy.reloads.status().staging == 1, "the reload staging")
        # the reload holds its worker until the gate opens: every apply meanwhile lands, each within its usual time
        during = [await _timed(copy.lane.apply([_page("NC", range(0, 1), n, "other")])) for n in range(5)]
        assert not reload.done.done()
        assert copy.reloads.status().staging == 1
        assert max(during) < max(1.0, 5 * max(before)), (before, during)
        gate.touch()
        applied = await reload.done
        assert len(_held(copy.store, "results")) == 100
        assert applied.rows == 100


class TestAChangeDuringAReload:
    async def test_a_change_given_during_a_reload_is_present_after_it_publishes(self, gated: tuple[Copy, Path]) -> None:
        copy, gate = gated
        # given before the reload: older than its full, replaced by it
        await copy.lane.apply([_page("VA", range(500, 501), 7)], mark=2)
        reload = await copy.reloads.reload([_page("VA", range(0, 20), 100), _page("NC", range(0, 20), 100)], mark=3)
        await _until(lambda: copy.reloads.status().staging == 1, "the reload staging")
        # given after it: a key it holds changed, a new key, a key deleted, while it stages
        await copy.lane.apply([_page("VA", range(0, 1), 7), _page("NC", range(900, 901), 9)], mark=4)
        await copy.lane.apply(
            [ChangePage(KEYS, "results", _rows("NC", range(5, 6), 0).select(["state", "office_key", "candidate_id"]))],
            mark=5,
        )
        assert _held(copy.store, "results")[("VA", 0)] == 7
        gate.touch()
        await reload.done
        held = _held(copy.store, "results")
        assert held[("VA", 0)] == 7
        assert held[("NC", 900)] == 9
        assert ("NC", 5) not in held
        assert ("VA", 500) not in held
        assert held[("VA", 1)] == 100
        assert len(held) == 40 - 1 + 1
        # a change since the reload published a newer mark: the reload's older one is not published over it
        assert _mark(copy.store) == 5

    async def test_too_many_changes_during_a_reload_fail_it_and_publish_nothing(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        gate = tmp_path / "reload-go"
        copy = await _copy(store, check=copy_hooks.WaitFor(str(gate)), max_replay_bytes=1)
        try:
            await copy.lane.apply([_page("VA", range(0, 10), 1)])
            reload = await copy.reloads.reload([_page("VA", range(0, 20), 100)])
            await _until(lambda: copy.reloads.status().staging == 1, "the reload staging")
            last = await copy.lane.apply([_page("VA", range(0, 1), 7)])
            gate.touch()
            with pytest.raises(ReloadOverrun):
                await reload.done
            assert store.current is not None and store.current.name == last.version
            assert len(_held(store, "results")) == 10
            assert copy.reloads.status().failed == 1
        finally:
            gate.touch()
            await _closed(copy)


class TestOrder:
    async def test_a_reloads_publish_is_ordered_with_the_lanes_changes(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        apply_gate = tmp_path / "apply-go"
        copy = await _copy(store, hooks=copy_hooks.WaitFor(str(apply_gate)))
        try:
            apply_gate.touch()
            await copy.lane.apply([_page("VA", range(0, 10), 1)], mark=1)
            apply_gate.unlink()
            # the apply lane held on a change given before the reload
            first = await copy.lane.submit([_page("VA", range(600, 601), 2)], mark=2)
            await _until(lambda: copy.lane.status().queued == 0, "the first change applying")
            reload = await copy.reloads.reload([_page("VA", range(0, 20), 100)], mark=3)
            # staged beside the held apply lane, its publish queued behind the change before it
            await _until(lambda: copy.reloads.status().queued == 1, "the reload staged")
            await _until(lambda: copy.lane.status().queued == 1, "the reload's publish queued")
            after = await copy.lane.submit([_page("VA", range(700, 701), 4)], mark=4)
            apply_gate.touch()
            before, published, behind = await first, await reload.done, await after
            assert _number(before.version) + 1 == _number(published.version)
            assert _number(published.version) + 1 == _number(behind.version)
            held = _held(store, "results")
            # the change before the reload is replaced by it; the one after lands on it
            assert ("VA", 600) not in held
            assert held[("VA", 700)] == 4
            assert len(held) == 21
            assert _mark(store) == 4
            assert copy.lane.status().reloads == 1
        finally:
            apply_gate.touch()
            await _closed(copy)


class TestADeadReloadWorker:
    async def test_a_dead_reload_worker_is_replaced(self, store: SharedArrowStore, tmp_path: Path) -> None:
        dies = tmp_path / "die"
        copy = await _copy(store, check=copy_hooks.DiesWhile(str(dies), once=True))
        try:
            before = await copy.lane.apply([_page("VA", range(0, 10), 1)])
            dies.touch()
            dying = await copy.reloads.reload([_page("VA", range(0, 20), 100)])
            with pytest.raises(ReloadWorkerDied):
                await dying.done
            assert not dies.exists()
            assert store.current is not None and store.current.name == before.version
            # the apply lane never noticed; the next reload stages in a fresh worker
            await copy.lane.apply([_page("NC", range(0, 1), 2, "other")])
            again = await copy.reloads.reload([_page("VA", range(0, 20), 100)])
            await again.done
            assert len(_held(store, "results")) == 20
            status = copy.reloads.status()
            assert (status.restarts, status.failed, status.reloads) == (1, 1, 1)
        finally:
            await _closed(copy)


class TestCancel:
    async def test_cancelling_a_staging_reload_publishes_nothing(self, gated: tuple[Copy, Path]) -> None:
        copy, gate = gated
        reload = await copy.reloads.reload([_page("VA", range(0, 20), 100)])
        await _until(lambda: copy.reloads.status().staging == 1, "the reload staging")
        current = copy.store.current
        assert current is not None
        assert reload.cancel()
        with pytest.raises(ReloadCancelled):
            await reload.done
        gate.touch()
        await copy.lane.drained()
        assert copy.store.current is not None and copy.store.current.name == current.name
        assert len(_held(copy.store, "results")) == 20
        assert copy.reloads.status().cancelled == 1

    async def test_cancelling_a_staged_reload_before_its_turn_publishes_nothing(
        self, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        apply_gate = tmp_path / "apply-go"
        copy = await _copy(store, hooks=copy_hooks.WaitFor(str(apply_gate)))
        try:
            apply_gate.touch()
            await copy.lane.apply([_page("VA", range(0, 10), 1)])
            apply_gate.unlink()
            held_on = await copy.lane.submit([_page("NC", range(0, 1), 2, "other")])
            await _until(lambda: copy.lane.status().queued == 0, "the change applying")
            reload = await copy.reloads.reload([_page("VA", range(0, 20), 100)])
            await _until(lambda: copy.lane.status().queued == 1, "the reload's publish queued")
            assert reload.cancel()
            apply_gate.touch()
            last = await held_on
            with pytest.raises(ReloadCancelled):
                await reload.done
            await copy.lane.drained()
            assert store.current is not None and store.current.name == last.version
            assert len(_held(store, "results")) == 10
            assert copy.lane.status().reloads == 0
        finally:
            apply_gate.touch()
            await _closed(copy)


class TestPerScope:
    async def test_a_per_scope_reload_replaces_its_scopes_parts_alone(self, store: SharedArrowStore) -> None:
        copy = await _copy(store)
        try:
            await copy.lane.apply([_page("VA", range(0, 10), 1), _page("NC", range(0, 10), 1)])
            current = store.current
            assert current is not None
            nc = held_parts(store.root, current.name, "results")["NC"]
            reload = await copy.reloads.reload(
                [_page("VA", range(5, 8), 100)], scopes={"results": ReloadScope("state", frozenset({"VA"}))}
            )
            applied = await reload.done
            assert set(applied.parts["results"]) == {"VA"}
            held = _held(store, "results")
            assert {key: votes for key, votes in held.items() if key[0] == "VA"} == {("VA", n): 100 for n in (5, 6, 7)}
            assert sum(1 for key in held if key[0] == "NC") == 10
            published = store.current
            assert published is not None
            # NC's part is the version before's own file, never rewritten
            assert held_parts(store.root, published.name, "results")["NC"].stat().st_ino == nc.stat().st_ino
        finally:
            await _closed(copy)

    async def test_a_per_scope_reload_of_a_table_held_whole_replaces_its_scopes_rows(
        self, store: SharedArrowStore
    ) -> None:
        copy = await _copy(store)
        try:
            await copy.lane.apply([_page("VA", range(0, 10), 1, "small"), _page("NC", range(0, 10), 1, "small")])
            reload = await copy.reloads.reload(
                [_page("NC", range(3, 5), 50, "small")], scopes={"small": ReloadScope("state", frozenset({"NC"}))}
            )
            await reload.done
            held = _held(store, "small")
            assert {key: votes for key, votes in held.items() if key[0] == "NC"} == {("NC", 3): 50, ("NC", 4): 50}
            assert sum(1 for key in held if key[0] == "VA") == 10
        finally:
            await _closed(copy)

    async def test_a_per_scope_reload_refuses_rows_of_other_values(self, store: SharedArrowStore) -> None:
        copy = await _copy(store)
        try:
            await copy.lane.apply([_page("VA", range(0, 10), 1)])
            reload = await copy.reloads.reload(
                [_page("NC", range(0, 3), 50)], scopes={"results": ReloadScope("state", frozenset({"VA"}))}
            )
            with pytest.raises(ValueError, match="rows of other values"):
                await reload.done
            assert len(_held(store, "results")) == 10
        finally:
            await _closed(copy)


def _big_reload(rows: int) -> list[ChangePage]:
    """a reload's pages whose buffers this process's Arrow pool allocated (a text column cast from numbers)."""
    numbers = pa.array(np.arange(rows), pa.int64())
    table = pa.table(
        {
            "state": pc.if_else(pc.less(numbers, rows // 2), "VA", "NC"),
            "office_key": pc.cast(numbers, pa.string()),
            "candidate_id": pc.add(numbers, 0),
            "votes": pc.multiply(numbers, 2),
        }
    ).cast(SCHEMA)
    return [ChangePage(ROWS, "results", table)]


class TestMemory:
    async def test_the_serving_process_holds_none_of_a_reloads_rows_while_it_stages(
        self, gated: tuple[Copy, Path]
    ) -> None:
        copy, gate = gated
        gc.collect()
        baseline = pa.total_allocated_bytes()
        pages = _big_reload(400_000)
        size = sum(page.rows.nbytes for page in pages)
        reload = await copy.reloads.reload(pages)
        del pages
        await _until(lambda: copy.reloads.status().staging == 1, "the reload staging")
        # handed over: once the worker holds them, this process's Arrow pool holds none of the reload's rows
        await _until(
            lambda: gc.collect() is not None and pa.total_allocated_bytes() - baseline < size // 10,
            "the reload's pages let go",
        )
        # still staging (the worker holds the gate): let go as handed over, not as the reload finished
        assert copy.reloads.status().staging == 1
        assert not reload.done.done()
        assert size > 10 << 20
        gate.touch()
        applied = await reload.done
        assert applied.rows == 400_000
        gc.collect()
        assert pa.total_allocated_bytes() - baseline < size // 10
        assert os.getpid() != applied.worker
