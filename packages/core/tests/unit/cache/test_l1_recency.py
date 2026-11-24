"""The L1 ledger's recency as columns (threetears.core.cache.l1_lru, Pace 2026-11-23): readers and writers hand the
ledger's thread immutable batches and touch nothing shared; the thread owns each table's recency as a packed key
batch beside array columns, folded latest-use-wins. Eviction order and byte accounting are the OrderedDict's they
replace: checked here against that model over a random workload, under threads, and for cost at 1M cached rows."""

from __future__ import annotations

import gc
import random
import threading
import time
import tracemalloc
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest

from threetears.core.cache.l1_lru import L1Ledger
from threetears.core.testing.loop_lateness import quietly

#: the ledger's evict-to share and the rows of a write it checks against the budget at a time (l1_lru's own)
_EVICT_TO = 0.9
_SLICE_ROWS = 256
#: the most the host's idle loop may be late over a timed measure, as the other cost tests take it
_HOST_MAY_BE_LATE = 0.020


class _Store:
    """a backend as the ledger sees one: it records what the ledger evicts, call by call."""

    store_name = "fake"

    def __init__(self) -> None:
        self.evicted: list[tuple[str, tuple[tuple[Any, ...], ...]]] = []
        self.guard = threading.Lock()

    def evict_keys(self, table: str, keys: Any) -> None:
        with self.guard:
            self.evicted = [*self.evicted, (table, tuple(keys))]

    def table_bytes(self) -> dict[str, int]:
        return {}

    def table_policies(self) -> dict[str, Any]:
        return {}

    def side_bytes(self) -> int:
        return 0


def _held(ledger: L1Ledger, store: _Store, names: tuple[str, ...] = ("alpha", "beta", "gamma")) -> dict[str, list[Any]]:
    """each table's held keys, least recent first; the tables holding none left out."""
    return {name: list(keys) for name in names if (keys := ledger.held(store, name))}


# --- the model: the OrderedDict recency this replaced, step for step, on a clock that ticks once an operation ---


def _model(ops: list[tuple[Any, ...]], budget: int) -> tuple[list[dict[str, list[Any]]], dict[str, list[Any]], int]:
    """the old ledger's evictions (per operation, each table's keys in order), held keys and bytes for ``ops``."""
    tables: OrderedDict[str, OrderedDict[Any, list[float]]] = OrderedDict()
    estimated = 0
    events: list[dict[str, list[Any]]] = []
    for clock, op in enumerate(ops):
        event: dict[str, list[Any]] = {}
        kind, name = op[0], op[1]
        if kind == "w":
            order = tables.setdefault(name, OrderedDict())
            rows = op[2]
            for start in range(0, len(rows), _SLICE_ROWS):
                for key, size in rows[start : start + _SLICE_ROWS]:
                    entry = order.get(key)
                    if entry is None:
                        order[key] = [size, clock]
                        estimated += size
                    else:
                        estimated += size - int(entry[0])
                        entry[0], entry[1] = size, clock
                        order.move_to_end(key)
                if estimated > budget:
                    target = budget * _EVICT_TO
                    while estimated > target:
                        oldest, oldest_tick = None, float("inf")
                        for held in tables.values():
                            if held:
                                head = next(iter(held.values()))
                                if head[1] < oldest_tick:
                                    oldest, oldest_tick = held, head[1]
                        if oldest is None:
                            break
                        victim, (size, _tick) = oldest.popitem(last=False)
                        estimated -= int(size)
                        owner = next(table for table, held in tables.items() if held is oldest)
                        event.setdefault(owner, []).append(victim)
        elif kind == "t":
            touched = tables.get(name)
            if touched is not None and op[2] in touched:
                touched[op[2]][1] = clock
                touched.move_to_end(op[2])
        elif kind == "r":
            removing = tables.get(name)
            if removing is not None:
                for key in op[2]:
                    entry = removing.pop(key, None)
                    if entry is not None:
                        estimated -= int(entry[0])
        else:
            gone = tables.pop(name, None)
            if gone is not None:
                estimated -= sum(int(entry[0]) for entry in gone.values())
        events.append(event)
    return events, {name: list(order) for name, order in tables.items()}, estimated


def _apply(ledger: L1Ledger, store: _Store, op: tuple[Any, ...]) -> None:
    kind, name = op[0], op[1]
    if kind == "w":
        ledger.written(store, name, op[2])
    elif kind == "t":
        ledger.touched(store, name, op[2])
    elif kind == "r":
        ledger.removed(store, name, op[2])
    else:
        ledger.emptied(store, name)


def _keys() -> list[tuple[Any, ...]]:
    """keys of every shape a serialized key takes: text, integers, floats, None, and text holding the separators."""
    keys: list[tuple[Any, ...]] = [(f"k{i}",) for i in range(120)]
    keys += [(i,) for i in range(60)]
    keys += [(f"a{i}", None) for i in range(20)]
    keys += [(i + 0.5, f"y{i}") for i in range(20)]
    keys += [("\x1dtagged",), ("x\x1fy",), ("nul\x00",), ("\x1elead",), ("",), ("", "")]
    return keys


def _workload(seed: int, count: int) -> list[tuple[Any, ...]]:
    draw = random.Random(seed)
    keys = _keys()
    ops: list[tuple[Any, ...]] = []
    for _ in range(count):
        name = draw.choice(("alpha", "beta", "gamma"))
        roll = draw.random()
        if roll < 0.45:
            width = draw.choice((1, 1, 3, 20, 300))
            ops.append(("w", name, [(draw.choice(keys), draw.randint(10, 400)) for _ in range(width)]))
        elif roll < 0.85:
            ops.append(("t", name, draw.choice(keys)))
        elif roll < 0.98:
            ops.append(("r", name, [draw.choice(keys) for _ in range(draw.randint(1, 5))]))
        else:
            ops.append(("e", name))
    return ops


class TestTheSameOrderAsTheOrderedDict:
    @pytest.mark.parametrize("seed", range(6))
    def test_each_operation_evicts_what_the_ordered_dict_did(self, seed: int) -> None:
        ops = _workload(seed, 1500)
        budget = 9000
        events, held, estimated = _model(ops, budget)
        store, ledger = _Store(), L1Ledger(budget, thread=False)
        for op, expected in zip(ops, events, strict=True):
            before = len(store.evicted)
            _apply(ledger, store, op)
            ledger.drain()
            got: dict[str, list[Any]] = {}
            for table, keys in store.evicted[before:]:
                got.setdefault(table, []).extend(keys)
            assert got == expected, f"seed {seed}, after {op[:2]}"
        assert sum(len(keys) for _t, keys in store.evicted) > 100, "the workload evicted too little to prove order"
        assert _held(ledger, store) == {name: keys for name, keys in held.items() if keys}
        assert ledger.status()["evictable_bytes"] == estimated

    @pytest.mark.parametrize("seed", range(3))
    def test_hand_offs_taken_in_one_pass_evict_the_same(self, seed: int) -> None:
        """the ledger's thread takes whatever is queued in one pass; the order is the queue's, so nothing moves."""
        ops = _workload(seed + 100, 1500)
        events, held, _estimated = _model(ops, 9000)
        store, ledger = _Store(), L1Ledger(9000, thread=False)
        for at, op in enumerate(ops):
            _apply(ledger, store, op)
            if at % 97 == 0:
                ledger.drain()
        ledger.drain()
        flat: dict[str, list[Any]] = {}
        for event in events:
            for table, keys in event.items():
                flat.setdefault(table, []).extend(keys)
        got: dict[str, list[Any]] = {}
        for table, evicted in store.evicted:
            got.setdefault(table, []).extend(evicted)
        assert got == flat
        assert _held(ledger, store) == {name: keys for name, keys in held.items() if keys}

    def test_a_key_comes_back_exactly_as_written(self) -> None:
        keys = [*_keys(), (1000,), ("1000",), (None,), (1000.0,)]
        store, ledger = _Store(), L1Ledger(1 << 30, thread=False)
        ledger.written(store, "t", [(key, 10) for key in keys])
        assert list(ledger.held(store, "t")) == keys, "a key came back changed, or two keys as one"
        ledger.set_budget(1)
        ledger.written(store, "t", [(("last",), 10)])
        ledger.drain()
        assert [key for _table, evicted in store.evicted for key in evicted] == [*keys, ("last",)]


class TestATouchAlwaysCounts:
    def test_a_touch_while_the_ledger_is_held_still_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the OrderedDict's touch was skipped while anything held the ledger's lock (a writer's batch, an eviction, a
        store joining); a hand-off never is. A store joining with no budget set reads its directory's file system
        under that lock, so a read made then (here, from the directory's own path) was lost."""
        monkeypatch.delenv("THREETEARS_L1_BUDGET_BYTES", raising=False)
        store, ledger = _Store(), L1Ledger(None, thread=False)
        ledger.written(store, "t", [((f"k{i:03d}",), 100) for i in range(90)])
        ledger.drain()

        class _ReadWhileJoining:
            """a directory whose path, asked for while the ledger is held, reads the oldest row."""

            def __fspath__(self) -> str:
                ledger.touched(store, "t", ("k000",))
                return str(tmp_path)

        ledger.join(store, _ReadWhileJoining())  # type: ignore[arg-type]
        ledger.set_budget(100 * 100)
        ledger.written(store, "t", [((f"n{i:03d}",), 100) for i in range(20)])
        ledger.drain()
        evicted = [key for _table, keys in store.evicted for key in keys]
        assert ("k001",) in evicted
        assert ("k000",) not in evicted, "a touch made while the ledger was held was lost"


class TestUnderThreads:
    def test_readers_and_writers_at_once_leave_the_rows_and_bytes_they_wrote(self) -> None:
        store, ledger = _Store(), L1Ledger(1 << 40)
        ledger.join(store, directory=None)  # type: ignore[arg-type]
        final: list[dict[tuple[Any, ...], int]] = [{} for _ in range(4)]
        stop = threading.Event()

        def write(lane: int) -> None:
            draw = random.Random(lane)
            mine = final[lane]  # this thread's alone, read after it is joined
            for _ in range(3000):
                key = (f"w{lane}-{draw.randrange(400)}",)
                if draw.random() < 0.8:
                    size = draw.randint(10, 500)
                    ledger.written(store, "t", [(key, size)])
                    mine[key] = size
                else:
                    ledger.removed(store, "t", [key])
                    mine.pop(key, None)

        def read(lane: int) -> None:
            draw = random.Random(100 + lane)
            while not stop.is_set():
                ledger.touched(store, "t", (f"w{draw.randrange(4)}-{draw.randrange(400)}",))

        readers = [threading.Thread(target=read, args=(lane,)) for lane in range(4)]
        writers = [threading.Thread(target=write, args=(lane,)) for lane in range(4)]
        for thread in [*readers, *writers]:
            thread.start()
        for thread in writers:
            thread.join(60)
        stop.set()
        for thread in readers:
            thread.join(60)
        ledger.drain()
        expected = {key: size for lane in final for key, size in lane.items()}
        held = ledger.held(store, "t")
        assert len(held) == len(set(held)), "a key is held twice"
        assert set(held) == set(expected)
        assert ledger.status()["evictable_bytes"] == sum(expected.values())
        assert store.evicted == []
        ledger.leave(store)

    def test_a_tight_budget_under_threads_holds_its_bytes_and_evicts_only_what_it_held(self) -> None:
        store, ledger = _Store(), L1Ledger(20_000)
        ledger.join(store, directory=None)  # type: ignore[arg-type]
        stop = threading.Event()
        written: list[set[tuple[Any, ...]]] = [set() for _ in range(4)]

        def write(lane: int) -> None:
            draw = random.Random(lane)
            mine = written[lane]  # this thread's alone, read after it is joined
            for _ in range(2000):
                rows = [
                    ((f"w{lane}-{draw.randrange(2000)}",), draw.randint(50, 300)) for _ in range(draw.randint(1, 30))
                ]
                ledger.written(store, "t", rows)
                mine.update(key for key, _size in rows)

        def read(lane: int) -> None:
            draw = random.Random(100 + lane)
            while not stop.is_set():
                ledger.touched(store, "t", (f"w{draw.randrange(4)}-{draw.randrange(2000)}",))

        readers = [threading.Thread(target=read, args=(lane,)) for lane in range(4)]
        writers = [threading.Thread(target=write, args=(lane,)) for lane in range(4)]
        for thread in [*readers, *writers]:
            thread.start()
        for thread in writers:
            thread.join(120)
        stop.set()
        for thread in readers:
            thread.join(60)
        ledger.drain()
        held = ledger.held(store, "t")
        evicted = {key for _table, keys in store.evicted for key in keys}
        every = set().union(*written)
        assert len(held) == len(set(held))
        assert set(held) <= every
        assert evicted <= every
        assert every <= set(held) | evicted, "a written row was neither held nor evicted"
        assert ledger.status()["evictable_bytes"] <= 20_000
        ledger.leave(store)


_MILLION = 1_000_000


def _a_million(ledger: L1Ledger, store: _Store) -> list[tuple[str]]:
    keys = [(f"row-{i:07d}",) for i in range(_MILLION)]
    for start in range(0, _MILLION, 1000):
        ledger.written(store, "t", [(key, 1056) for key in keys[start : start + 1000]])
        if start % 50_000 == 0:
            ledger.drain()
    ledger.drain()
    return keys


class TestAtAMillionRows:
    def test_a_cached_row_costs_tens_of_bytes_not_hundreds(self) -> None:
        """measured 2026-11-23 (laptop): the OrderedDict held 163 bytes a row beside keys it did not own (about 280
        with them); the columns hold 37, key text included."""
        store, ledger = _Store(), L1Ledger(1 << 50, thread=False)
        keys = [(f"row-{i:07d}",) for i in range(_MILLION)]
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for start in range(0, _MILLION, 1000):
                ledger.written(store, "t", [(key, 1056) for key in keys[start : start + 1000]])
                if start % 50_000 == 0:
                    ledger.drain()
            ledger.drain(fold=True)
            gc.collect()
            held = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()
        assert ledger.status()["evictable_bytes"] == _MILLION * 1056
        assert held / _MILLION < 64, f"{held / _MILLION:.0f} bytes a cached row"

    @pytest.mark.loop_timing
    def test_a_touch_costs_its_caller_a_hand_off(self) -> None:
        """the caller's whole cost of a read's recency: a key text, a tuple, a queue put. Measured 2026-11-23
        (laptop) at 1M held rows: 0.2-0.35 us, where the OrderedDict's lock and move took 0.5 us."""
        store, ledger = _Store(), L1Ledger(1 << 50, thread=False)
        keys = _a_million(ledger, store)
        sample = keys[::100]

        def per_touch() -> float:
            best = float("inf")
            for _ in range(5):
                started = time.perf_counter()
                for key in sample:
                    ledger.touched(store, "t", key)
                best = min(best, (time.perf_counter() - started) / len(sample))
                ledger.drain()
            return best

        taken = quietly(per_touch, host_may_be_late=_HOST_MAY_BE_LATE, threads=1)
        assert taken < 2e-6, f"a touch took its caller {taken * 1e6:.2f} us"
