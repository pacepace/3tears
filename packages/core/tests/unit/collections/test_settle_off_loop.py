"""A caller transaction of many rows settles them without holding the event loop.

At the commit of a 33,786-row write (one state's candidates, 179 columns) the transaction grouped its enrollments and
built a change record a key on the loop, and freed the rows its enrollments held there too: 41-71 ms with the loop
held (the WD2 bench, 2026-11-16). The grouping and the records are made on a worker thread, the rows freed there a
slice at a time, and every key is still settled, in the order written, with the advance a table's instances share
covering every instance's rows.
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.collections import WRITE_GENERATION
from threetears.core.collections.caller_transaction import CallerTransaction, shared_advance_for
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the key's columns: the 11-03 candidates table's seven
_KEY = tuple(f"k{index}" for index in range(7))
#: past the count from which a settle leaves the loop (64)
_MANY = 200
#: the 11-03 state that held the loop: 33,786 rows
_REAL = 33_786
#: what each tick of the loop's ticker asks to sleep
_TICK = 0.002


class _Candidate(BaseEntity):
    """a candidate's row: its own id is the last key column, within the contest its key leads with."""

    primary_key_field = "k6"


def _schema() -> TableSchema:
    return TableSchema(
        name="candidates",
        primary_key=_KEY,
        columns=[*(Column(name, STRING_TYPE) for name in _KEY), Column("payload", STRING_TYPE, nullable=True)],
        l1_eviction=TEST_L1_TABLE,
    )


class _Source:
    """a generation source that counts each table's advances."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def current(self, table_name: str) -> str:
        return f"inc:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc:{self.counts[table_name]}"


_Candidates = collection_for_schema(_schema(), entity_class=_Candidate)


class _AdvancingCandidates(_Candidates):  # type: ignore[misc,valid-type]
    """the table switched on: each commit of its rows advances its write generation."""

    write_generation = WRITE_GENERATION


def _registry(source: _Source | None = None) -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal")
    if source is not None:
        registry.set_generation_source(source)
    return registry


def _collection(registry: CollectionRegistry | None = None, *, advancing: bool = False) -> Any:
    cls = _AdvancingCandidates if advancing else _Candidates
    return cls(registry if registry is not None else _registry(), DefaultCoreConfig(), None)


def _key(index: int) -> tuple[str, ...]:
    return tuple(f"{name}-{index % (position + 3)}-{index}" for position, name in enumerate(_KEY))


def _row(index: int, width: int) -> dict[str, Any]:
    """a row as a write sends it: its key, and ``width`` more columns."""
    return {
        **dict(zip(_KEY, _key(index), strict=True)),
        **{f"c{column}": f"v{column}-{index}" for column in range(width)},
    }


class _Conn:
    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


def _enroll(transaction: CallerTransaction, collections: list[Any], count: int, *, width: int) -> None:
    """``count`` rows enrolled round the ``collections``, each row held by its enrollment alone."""
    for index in range(count):
        transaction.enroll(collections[index % len(collections)], _key(index), row=_row(index, width))


class TestManyRowsSettleOffTheLoop:
    @pytest.mark.parametrize("instances", [1, 2], ids=["one-instance", "two-instances"])
    async def test_rows_no_instance_advances_for_build_no_change_record(
        self, instances: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _registry()
        collections = [_collection(registry) for _ in range(instances)]
        built: list[Any] = []
        original = type(collections[0]).changed_row

        def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            built.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(collections[0]), "changed_row", recording)
        settled: list[tuple[list[Any], Any]] = []

        async def settle(keys: Any) -> None:
            settled.append((list(keys), shared_advance_for("candidates")))

        for collection in collections:
            collection.invalidate_cache_many = settle
        async with CallerTransaction(_Conn()) as transaction:
            _enroll(transaction, collections, _MANY, width=1)
        assert [key for keys, _ in settled for key in keys] == [
            _key(index) for start in range(instances) for index in range(start, _MANY, instances)
        ]
        assert [shared for _, shared in settled] == [None] * instances
        assert built == [], "a change record was built for rows no advance names"

    async def test_two_instances_share_one_advance_covering_every_row_built_on_a_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _registry(_Source())
        first, second = _collection(registry, advancing=True), _collection(registry, advancing=True)
        threads: set[int] = set()
        original = _AdvancingCandidates.changed_row

        def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            threads.add(threading.get_ident())
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_AdvancingCandidates, "changed_row", recording)
        seen: list[tuple[str, list[Any], int, list[list[str]]]] = []

        def spy(name: str) -> Any:
            async def settle(keys: Any) -> None:
                shared = shared_advance_for("candidates")
                assert shared is not None
                seen.append((name, list(keys), shared.rows, [row.ids for row in shared.changed]))

            return settle

        first.invalidate_cache_many = spy("first")
        second.invalidate_cache_many = spy("second")
        async with CallerTransaction(_Conn()) as transaction:
            _enroll(transaction, [first, second], _MANY, width=1)
        evens = [_key(index) for index in range(0, _MANY, 2)]
        odds = [_key(index) for index in range(1, _MANY, 2)]
        # each instance's rows in the order written, and the shared advance names the first instance's, then the
        # second's, as one count
        every = [list(key) for key in evens] + [list(key) for key in odds]
        assert seen == [("first", evens, _MANY, every), ("second", odds, _MANY, every)]
        assert threads and threading.get_ident() not in threads, "the change records were built on the loop"

    async def test_a_few_rows_settle_on_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = _registry(_Source())
        first, second = _collection(registry, advancing=True), _collection(registry, advancing=True)
        threads: set[int] = set()
        original = _AdvancingCandidates.changed_row

        def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            threads.add(threading.get_ident())
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_AdvancingCandidates, "changed_row", recording)

        async def settle(keys: Any) -> None:
            return None

        first.invalidate_cache_many = settle
        second.invalidate_cache_many = settle
        async with CallerTransaction(_Conn()) as transaction:
            _enroll(transaction, [first, second], 6, width=1)
        assert threads == {threading.get_ident()}

    async def test_every_key_still_reaches_every_listener_in_the_order_written(self) -> None:
        collection = _collection()
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)
        async with CallerTransaction(_Conn()) as transaction:
            _enroll(transaction, [collection], _MANY, width=1)
        assert heard == [_key(index) for index in range(_MANY)]


async def _settled(instances: int, *, advancing: bool) -> tuple[list[Tick], list[Tick]]:
    """the loop's ticks and the host's idle loop's while a transaction of :data:`_REAL` rows, round ``instances``
    instances of the table (``advancing``: switched on), ends; each row holding 40 columns more than its key, held by
    its enrollment alone.

    The rows the transaction holds are frozen out of the garbage collector's reach for the window: an advancing
    settle builds a change record a key, on its thread, and the collections that allocation sets off walk the heap
    with the GIL held, 10-25 ms here over these rows (CPython 3.14's increments of the old generation, 2026-11-16).
    That walk is the runtime's, sized by the process's heap, wherever the allocation runs; what this measures is
    the settle's own hold."""
    registry = _registry(_Source() if advancing else None)
    collections = [_collection(registry, advancing=advancing) for _ in range(instances)]
    transaction = CallerTransaction(_Conn())
    await transaction.__aenter__()
    _enroll(transaction, collections, _REAL, width=40)
    gc.freeze()
    try:
        with HostTicker() as host:
            ticks = await ticks_while(lambda: transaction.__aexit__(None, None, None), every=_TICK)
    finally:
        gc.unfreeze()
    return ticks, host.ticks


@pytest.mark.loop_timing
class TestTheLoopRunsWhileARealSizeTransactionSettles:
    @pytest.mark.parametrize(
        ("instances", "advancing"),
        [(1, False), (2, False), (1, True), (2, True)],
        ids=["one-instance", "two-instances", "one-instance-advancing", "two-instances-sharing-an-advance"],
    )
    async def test_a_ticker_is_never_held_while_33786_rows_settle(self, instances: int, advancing: bool) -> None:
        # what the per-key work alone takes here -- a change record a key -- so the bound is in proportion to this
        # machine: the loop once ran all of it (and more) in one stretch, 41-71 ms in the WD2 bench
        collection = _collection()
        rows = [_row(index, 0) for index in range(_REAL)]
        gc.disable()
        try:
            started = time.perf_counter()
            for index, row in enumerate(rows):
                collection.changed_row(_key(index), row)
            one_build = time.perf_counter() - started
        finally:
            gc.enable()
        bound = max(0.02, one_build / 2)
        ticks, host = await trusted(
            lambda: _settled(instances, advancing=advancing), lambda measured: measured, host_may_be_late=bound
        )
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.0f} ms past the host's idle loop while {_REAL:,} rows settled "
            f"({latest(ticks, every=_TICK) * 1000:.0f} ms in all; a change record a key takes "
            f"{one_build * 1000:.0f} ms)"
        )
