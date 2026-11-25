"""A bulk write of many rows on a caller's transaction holds the event loop no longer than a few milliseconds at a time.

One state's candidates (33,786 rows) written through ``save_arrow`` held the serving loop 6-7 ms with the collector held
and 50-55 ms with it running, and through ``save_rows`` 87-103 ms and 134-156 ms (the WD2 bench, 2026-11-16, 179
columns, a hub answering at once): every key read into Python, checked for a twin, enrolled and fenced on the loop, a
call a key; every row checked, stamped and keyed on the loop; each statement's text and flattened parameters built on
the loop, 7-8 ms a statement; and the statements' parameters freed on the loop as the write returned. All of it now
runs on worker threads that hand the loop a turn every half millisecond (:mod:`threetears.core.loop_turns`).

A bulk delete through ``delete_rows`` held it the same way, its keys checked, bound, grouped into statements and
enrolled on the loop, and a collection with L1 change listeners (a complete copy's, a scope snapshot's) heard every key
of a write a call a key on the loop, twice: as the write evicted it and as the commit settled it.

The window is the whole write: ``tx.begin``, the save, ``tx.commit`` and the settle, the hub a bus that answers at once.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from threetears.nats import Subject

import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the 11-03 state that held the loop: 33,786 rows
_REAL = 33_786
#: the key's columns: the 11-03 candidates table's seven
_KEY = tuple(f"k{index}" for index in range(7))
#: the columns past the key, of the candidates' kinds in turn
_WIDTH = 40
_KINDS = (STRING_TYPE, BIGINT_TYPE, NUMERIC_TYPE, BOOL_TYPE, DATETIMETZ_TYPE)
_ARROW_TYPES = {
    STRING_TYPE: pa.string(),
    BIGINT_TYPE: pa.int64(),
    NUMERIC_TYPE: pa.decimal128(18, 6),
    BOOL_TYPE: pa.bool_(),
    DATETIMETZ_TYPE: pa.timestamp("us", tz="UTC"),
}
_VALUES = tuple((f"c{index}", _KINDS[index % len(_KINDS)]) for index in range(_WIDTH))
#: what each tick of the loop's ticker asks to sleep
_TICK = 0.002
_AT = datetime(2026, 11, 3, tzinfo=UTC)

_SCHEMA = TableSchema(
    name="candidates",
    primary_key=_KEY,
    columns=[
        *(Column(name, STRING_TYPE) for name in _KEY),
        *(
            Column(
                name,
                kind,
                nullable=True,
                precision=18 if kind == NUMERIC_TYPE else None,
                scale=6 if kind == NUMERIC_TYPE else None,
            )
            for name, kind in _VALUES
        ),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    l1_eviction=TEST_L1_TABLE,
)


class _Candidate(BaseEntity):
    """a candidate's row: its own id is the last key column."""

    primary_key_field = "k6"


_Candidates = collection_for_schema(_SCHEMA, entity_class=_Candidate)


class _Kept(_Candidates):  # type: ignore[misc, valid-type]
    """candidates whose row messages carry a column of each row: a delete returns the rows it removes."""

    invalidation_columns: ClassVar[tuple[str, ...]] = ("c0", "c1")


def _value(kind: str, row: int, column: int) -> Any:
    if kind == STRING_TYPE:
        return f"v{row % 97}-{column}"
    if kind == BIGINT_TYPE:
        return row * 3 + column
    if kind == NUMERIC_TYPE:
        return Decimal(f"{row % 1000}.{(row + column) % 1000000:06d}")
    if kind == BOOL_TYPE:
        return (row + column) % 2 == 0
    return _AT + timedelta(seconds=row + column)


def _key(row: int) -> tuple[str, ...]:
    return tuple(f"{name}-{row % (position + 3)}-{row}" for position, name in enumerate(_KEY))


def _rows_of(count: int) -> list[dict[str, Any]]:
    return [
        {
            **dict(zip(_KEY, _key(row), strict=True)),
            **{name: _value(kind, row, column) for column, (name, kind) in enumerate(_VALUES)},
        }
        for row in range(count)
    ]


def _rows() -> list[dict[str, Any]]:
    return _rows_of(_REAL)


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table(
        {
            **{name: pa.array([row[name] for row in rows], pa.string()) for name in _KEY},
            **{name: pa.array([row[name] for row in rows], _ARROW_TYPES[kind]) for name, kind in _VALUES},
        }
    )


class _Hub:
    """a hub that answers every request at once, after one turn of the loop as a reply off the network would; its
    ``tx.begin`` announces the Arrow copy and, when ``door``, the Arrow write door."""

    def __init__(self, *, door: bool) -> None:
        self.begin = json.dumps(
            {"success": True, "tx_id": "019d9a00-0000-7000-8000-0000000000aa", "copy_arrow": True, "write_arrow": door}
        ).encode()
        self.reply = json.dumps({"success": True, "row_count": 1}).encode()

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del payload, timeout
        await asyncio.sleep(0)
        return self.begin if subject.path.endswith("tx.begin") else self.reply

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 16 * 1024 * 1024
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "t" * 900
        )


def _wire(value: Any) -> Any:
    """a value as the broker's JSON carries it."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


class _ReturningHub(_Hub):
    """:class:`_Hub` whose ``tx.fetch`` answers each ``DELETE ... RETURNING`` with the next batch of ``rows``, encoded
    before the write starts, as the broker would serialize them."""

    def __init__(self, rows: list[dict[str, Any]], *, batch: int = 1000) -> None:
        super().__init__(door=False)
        stamped = [{**row, "date_created": _AT, "date_updated": _AT} for row in rows]
        self.replies = [
            json.dumps(
                {
                    "success": True,
                    "rows": [
                        {name: _wire(value) for name, value in row.items()} for row in stamped[start : start + batch]
                    ],
                }
            ).encode()
            for start in range(0, len(stamped), batch)
        ]
        self.fetched = 0

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        if not subject.path.endswith("tx.fetch"):
            return await super().request_raw(subject=subject, payload=payload, timeout=timeout)
        await asyncio.sleep(0)
        reply = self.replies[self.fetched]
        self.fetched += 1
        return reply


async def _written(
    save: Callable[[Any, Any], Any],
    *,
    door: bool,
    listeners: tuple[Callable[[], None], ...] = (),
    hub: _Hub | None = None,
    kind: type[Any] = _Candidates,
) -> tuple[list[Tick], list[Tick]]:
    """the loop's ticks and the host's idle loop's while one transaction writes :data:`_REAL` rows with ``save``.

    The collector is held for the window, as a serving process holds it through a refresh (the reports pod's
    ``collector_held``): a collection the write's allocations set off walks the heap with the GIL held, wherever the
    allocation runs, 4-6 ms at a time here and 40-50 ms beside a heap of rows as Python values; that walk is the
    runtime's, sized by the process's heap. What this measures is the write's own hold."""
    backend = (hub or _Hub(door=door)).proxy()
    collection = kind(CollectionRegistry(), DefaultCoreConfig(), None)
    collection.l3_pool = backend
    for listener in listeners:
        # as a scope snapshot's ``on_change`` adds its holders' (scope_epochs)
        collection.add_l1_change_listener(_on_change(listener))

    async def write() -> None:
        async with backend.acquire() as conn, CallerTransaction(conn):
            await save(collection, conn)

    gc.collect()
    gc.disable()
    try:
        with HostTicker() as host:
            ticks = await ticks_while(write, every=_TICK)
    finally:
        gc.enable()
    return ticks, host.ticks


#: the collection a baseline's steps enroll their keys under: only its identity is read
_STAND_IN = MagicMock()


def _loop_steps(rows: list[dict[str, Any]], *, stamped: bool) -> float:
    """what the write's per-row steps alone take here, done as the loop once did them, so the bound is in proportion
    to this machine: ``save_arrow``'s keys zipped from their columns, checked for a twin and enrolled; ``save_rows``'
    rows each stamped and keyed, then the same."""
    transaction = CallerTransaction(MagicMock())
    stamps = {"date_created": _AT, "date_updated": _AT}
    columns = [[row[name] for row in rows] for name in _KEY]
    gc.disable()
    try:
        started = time.perf_counter()
        if stamped:
            data: list[dict[str, Any] | None] = []
            keys: list[tuple[Any, ...]] = []
            for row in rows:
                data.append({**row, **stamps})
                keys.append(tuple(row[name] for name in _KEY))
        else:
            keys = list(zip(*columns, strict=True))
            data = [None] * len(keys)
        assert len(set(keys)) == len(keys)
        for key, each in zip(keys, data, strict=True):
            transaction.enroll(_STAND_IN, key, row=each)
        return time.perf_counter() - started
    finally:
        gc.enable()


def _delete_steps(rows: list[dict[str, Any]]) -> float:
    """what a delete's per-key steps alone take here, done as the loop once did them: each key made a tuple, checked
    for its width, made the key it is enrolled by and enrolled."""
    transaction = CallerTransaction(MagicMock())
    keys = [tuple(row[name] for name in _KEY) for row in rows]
    gc.disable()
    try:
        started = time.perf_counter()
        named = [tuple(key) for key in keys]
        assert not [key for key in named if len(key) != len(_KEY)]
        for key in named:
            transaction.enroll(_STAND_IN, key if len(key) > 1 else key[0], row=None)
        return time.perf_counter() - started
    finally:
        gc.enable()


def _rehydrated(row: dict[str, Any]) -> dict[str, Any]:
    """a row as the hub's connection reads it off the wire, each ``date_`` text a datetime again, a value a value."""
    return {
        name: datetime.fromisoformat(value) if name.startswith("date_") and isinstance(value, str) else value
        for name, value in row.items()
    }


def _returned_steps(rows: list[dict[str, Any]]) -> float:
    """what a returning delete's per-row steps alone take here, done as the loop once did them: each row the delete
    returned read from the broker's JSON, its key bound, its values coerced as read, and its key enrolled then restated
    with it."""
    hub = _ReturningHub(rows)
    replies = [json.loads(reply)["rows"] for reply in hub.replies]
    transaction = CallerTransaction(MagicMock())
    key_columns = [_SCHEMA.column(name) for name in _KEY]
    gc.disable()
    try:
        started = time.perf_counter()
        named = [tuple(row[name] for name in _KEY) for row in rows]
        for key in named:
            transaction.enroll(_STAND_IN, key, row=None)
        found: dict[tuple[Any, ...], dict[str, Any]] = {}
        for reply in replies:
            for row in reply:
                read = _rehydrated(row)
                bound = tuple(schema_sql.normalize_write_value(column, read[column.name]) for column in key_columns)
                found[bound] = schema_sql.coerce_row(_SCHEMA, read)
        transaction.restate_many(_STAND_IN, found)
        return time.perf_counter() - started
    finally:
        gc.enable()


def _on_change(listener: Callable[[], None]) -> Callable[[Any], None]:
    """``listener`` as a scope snapshot's ``on_change`` adds it to its collection's L1 (scope_epochs)."""
    return lambda _key: listener()


class _Holder:
    """a holder listening to a collection's L1 through a scope snapshot (the reports pod's rows function and its
    loaded check, two a snapshot): each change heard marks it dirty and starts a rebuild unless one is running."""

    def __init__(self) -> None:
        self.heard = 0
        self._dirty = asyncio.Event()
        self._builder: asyncio.Task[Any] | None = None

    def changed(self) -> None:
        self.heard += 1
        self._dirty.set()
        if self._builder is None or self._builder.done():
            self._builder = asyncio.get_running_loop().create_task(self._dirty.wait())

    async def settled(self) -> None:
        if self._builder is not None:
            await self._builder


def _heard_keys(count: int) -> float:
    """what hearing ``count`` keys alone takes here, two :class:`_Holder`'s calls a key as the loop once made them;
    from a coroutine."""
    listeners = [_on_change(_Holder().changed), _on_change(_Holder().changed)]
    gc.disable()
    try:
        started = time.perf_counter()
        for row in range(count):
            for listener in tuple(listeners):
                listener(row)
        return time.perf_counter() - started
    finally:
        gc.enable()


class TestTheWritesPerRowStepsLeaveTheLoop:
    @pytest.mark.parametrize("how", ["save_arrow", "save_rows"])
    async def test_many_rows_are_enrolled_from_a_thread_every_key_in_order(
        self, how: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _rows_of(200)
        made: list[int] = []
        original = CallerTransaction.enrollments

        def recording(*args: Any, **kwargs: Any) -> Any:
            made.append(threading.get_ident())
            return original(*args, **kwargs)

        monkeypatch.setattr(CallerTransaction, "enrollments", staticmethod(recording))
        backend = _Hub(door=how == "save_arrow").proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        settled: list[Any] = []

        async def settle(keys: Any) -> None:
            settled.extend(keys)

        monkeypatch.setattr(collection, "invalidate_cache_many", settle)
        async with backend.acquire() as conn, CallerTransaction(conn):
            if how == "save_arrow":
                await collection.save_arrow(_table(rows), conn=conn)
            else:
                await collection.save_rows(rows, conn=conn)
        assert made and threading.get_ident() not in made, "the enrollments were made on the loop"
        assert settled == [_key(index) for index in range(200)]

    @pytest.mark.parametrize("how", ["save_arrow", "save_rows"])
    async def test_a_key_twice_is_still_refused_before_anything_is_enrolled(
        self, monkeypatch: pytest.MonkeyPatch, how: str
    ) -> None:
        rows = _rows_of(200)
        rows[150] = {**rows[150], **dict(zip(_KEY, _key(3), strict=True))}
        backend = _Hub(door=how == "save_arrow").proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        settled: list[Any] = []

        async def settle(keys: Any) -> None:
            settled.extend(keys)

        monkeypatch.setattr(collection, "invalidate_cache_many", settle)
        async with backend.acquire() as conn, CallerTransaction(conn):
            with pytest.raises(ValueError, match="two rows share a key"):
                if how == "save_arrow":
                    await collection.save_arrow(_table(rows), conn=conn)
                else:
                    await collection.save_rows(rows, conn=conn)
        assert settled == [], "a refused write enrolled keys"


class TestABulkDeletesKeysLeaveTheLoop:
    async def test_many_keys_are_enrolled_from_a_thread_every_key_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made: list[int] = []
        original = CallerTransaction.enrollments

        def recording(*args: Any, **kwargs: Any) -> Any:
            made.append(threading.get_ident())
            return original(*args, **kwargs)

        monkeypatch.setattr(CallerTransaction, "enrollments", staticmethod(recording))
        backend = _Hub(door=False).proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        settled: list[Any] = []

        async def settle(keys: Any) -> None:
            settled.extend(keys)

        monkeypatch.setattr(collection, "invalidate_cache_many", settle)
        async with backend.acquire() as conn, CallerTransaction(conn):
            assert await collection.delete_rows([_key(index) for index in range(200)], conn=conn) == 200
        assert made and threading.get_ident() not in made, "the enrollments were made on the loop"
        assert settled == [_key(index) for index in range(200)]

    async def test_a_key_of_the_wrong_width_is_still_refused_before_anything_is_enrolled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keys = [_key(index) for index in range(200)]
        keys[150] = keys[150][:-1]
        backend = _Hub(door=False).proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        settled: list[Any] = []

        async def settle(keys: Any) -> None:
            settled.extend(keys)

        monkeypatch.setattr(collection, "invalidate_cache_many", settle)
        async with backend.acquire() as conn, CallerTransaction(conn):
            with pytest.raises(ValueError, match=r"key \('k0-0-150'.*has 6 values where the key has 7"):
                await collection.delete_rows(keys, conn=conn)
        assert settled == [], "a refused delete enrolled keys"


class TestADeleteReturningItsRowsLeavesTheLoop:
    async def test_every_key_is_settled_with_the_row_its_delete_returned_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = await asyncio.to_thread(_rows_of, 2500)
        hub = await asyncio.to_thread(_ReturningHub, rows)
        backend = hub.proxy()
        collection = _Kept(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        coerced: list[int] = []
        original = schema_sql.coerce_row

        def recording(*args: Any, **kwargs: Any) -> Any:
            coerced.append(threading.get_ident())
            return original(*args, **kwargs)

        monkeypatch.setattr(schema_sql, "coerce_row", recording)
        settled: list[tuple[list[Any], list[dict[str, Any] | None]]] = []

        async def settle(keys: list[Any], *, rows: list[dict[str, Any] | None] | None = None) -> None:
            settled.append((list(keys), list(rows or [])))

        monkeypatch.setattr(collection, "invalidate_cache_many", settle)
        keys = [_key(index) for index in range(2500)]
        async with backend.acquire() as conn, CallerTransaction(conn):
            assert await collection.delete_rows([*keys, _key(9999)], conn=conn) == 2501
        assert hub.fetched == 3, "the delete did not return its rows a statement a batch"
        assert len(coerced) == 2500 and threading.get_ident() not in coerced, "the returned rows were read on the loop"
        assert len(settled) == 1
        assert settled[0][0] == [*keys, _key(9999)], "a key was settled out of order"
        assert settled[0][1][-1] is None, "a key no row held was settled with a row"
        for index, row in enumerate(settled[0][1][:-1]):
            assert row is not None and (row["k6"], row["c0"], row["c1"]) == (
                keys[index][6],
                rows[index]["c0"],
                rows[index]["c1"],
            ), f"key {index} was settled without the row its delete returned"
            assert row["date_updated"] == _AT, "a returned row was not read as fetch_many reads it"

    async def test_the_loop_runs_between_stretches_of_33786_rows_returned(self) -> None:
        # what the timing budget below bounds by the clock, counted: the loop's other work runs while the returned
        # rows are read and restated, not once they all are (the old delete read each statement's 1,000 rows and
        # restated all 33,786 in stretches the ticker never ran inside)
        rows = await asyncio.to_thread(_rows)
        hub = await asyncio.to_thread(_ReturningHub, rows)
        backend = hub.proxy()
        collection = _Kept(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        turns = 0
        seen: list[int] = []
        original = schema_sql.coerce_row

        def counting(*args: Any, **kwargs: Any) -> Any:
            seen.append(turns)
            return original(*args, **kwargs)

        async def ticker() -> None:
            nonlocal turns
            while True:
                turns += 1
                await asyncio.sleep(0)

        keys = await asyncio.to_thread(lambda: [_key(index) for index in range(_REAL)])
        ticking = asyncio.get_running_loop().create_task(ticker())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(schema_sql, "coerce_row", counting)
            try:
                async with backend.acquire() as conn, CallerTransaction(conn):
                    await collection.delete_rows(keys, conn=conn)
            finally:
                ticking.cancel()
        assert len(seen) == _REAL
        # a statement's rows read in one stretch is 1,000 rows a stretch: 34 stretches for 34 statements
        stretches = sum(1 for before, after in zip(seen, seen[1:]) if after != before) + 1
        assert stretches > 2 * len(hub.replies), (
            f"the {_REAL:,} returned rows were read in {stretches} stretches over {len(hub.replies)} statements, "
            f"the loop held through each statement's rows"
        )


class TestAListenerHearsEveryKeyOnTheLoop:
    @pytest.mark.parametrize("how", ["save_arrow", "save_rows", "delete_rows"])
    async def test_every_key_is_heard_in_order_on_the_loop_as_written_and_as_settled(self, how: str) -> None:
        rows = _rows_of(300)
        backend = _Hub(door=how == "save_arrow").proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        loop_thread = threading.get_ident()
        heard: list[Any] = []
        off_loop: list[int] = []

        def listener(key: Any) -> None:
            if threading.get_ident() != loop_thread:
                off_loop.append(threading.get_ident())
            heard.append(key)

        collection.add_l1_change_listener(listener)
        async with backend.acquire() as conn, CallerTransaction(conn):
            if how == "save_arrow":
                await collection.save_arrow(_table(rows), conn=conn)
            elif how == "save_rows":
                await collection.save_rows(rows, conn=conn)
            else:
                await collection.delete_rows([_key(index) for index in range(300)], conn=conn)
            assert heard == [_key(index) for index in range(300)], "the write's eviction was not heard key by key"
        assert off_loop == [], "a listener was called off the loop"
        assert heard == [_key(index) for index in range(300)] * 2, "the commit's settle was not heard key by key"

    async def test_the_loop_runs_between_stretches_of_33786_keys_heard(self) -> None:
        # what the timing budget below bounds by the clock, counted: the loop's other work runs while the keys are
        # heard, not once they all are (the old write heard every key of a pass in one stretch: two stretches)
        rows = await asyncio.to_thread(_rows)
        table = await asyncio.to_thread(_table, rows)
        backend = _Hub(door=True).proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        turns = 0
        seen: list[int] = []

        async def ticker() -> None:
            nonlocal turns
            while True:
                turns += 1
                await asyncio.sleep(0)

        holders = (_Holder(), _Holder())
        for holder in holders:
            collection.add_l1_change_listener(_on_change(holder.changed))
        collection.add_l1_change_listener(lambda _key: seen.append(turns))
        ticking = asyncio.get_running_loop().create_task(ticker())
        try:
            async with backend.acquire() as conn, CallerTransaction(conn):
                await collection.save_arrow(table, conn=conn)
        finally:
            ticking.cancel()
        for holder in holders:
            await holder.settled()
        stretches = sum(1 for before, after in zip(seen, seen[1:]) if after != before) + 1
        assert len(seen) == 2 * _REAL
        assert stretches > 2, f"the {2 * _REAL:,} keys were heard in {stretches} stretches, the loop held throughout"


@pytest.mark.loop_timing
class TestTheLoopRunsWhileARealSizeWriteIsSaved:
    @pytest.mark.parametrize("how", ["save_arrow", "save_rows"])
    async def test_a_ticker_is_never_held_while_33786_rows_are_written(self, how: str) -> None:
        rows = await asyncio.to_thread(_rows)
        table = await asyncio.to_thread(_table, rows)
        one_pass = _loop_steps(rows, stamped=how == "save_rows")
        # the old write ran these steps, and more, in one stretch on the loop (save_arrow 10-11 ms held against 5 ms
        # of them, save_rows 43-46 ms against 17-20 ms, a 10-core Apple host, 2026-11-16); a stretch of half of them is a
        # step back onto the loop. 5 ms at the least, a GIL switch interval: the new write holds it 1.6-1.9 ms
        # (save_arrow) and 3.1-4.5 ms (save_rows), each statement's JSON encoded in one C call
        bound = max(0.005, one_pass / 2)

        async def save(collection: Any, conn: Any) -> None:
            if how == "save_arrow":
                await collection.save_arrow(table, conn=conn)
            else:
                await collection.save_rows(rows, conn=conn)

        ticks, host = await trusted(
            lambda: _written(save, door=how == "save_arrow"), lambda measured: measured, host_may_be_late=bound
        )
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while {_REAL:,} rows were written "
            f"with {how} ({latest(ticks, every=_TICK) * 1000:.1f} ms in all; the per-row steps take "
            f"{one_pass * 1000:.1f} ms here)"
        )

    async def test_a_ticker_is_never_held_while_33786_keys_are_deleted(self) -> None:
        rows = await asyncio.to_thread(_rows)
        keys = [_key(row) for row in range(_REAL)]
        one_pass = _delete_steps(rows)
        # the old delete ran these steps, and more (each key's values bound, the keys grouped into statements a JSON
        # text a key column a key), in one stretch on the loop; 5 ms at the least, a GIL switch interval
        bound = max(0.005, one_pass / 2)

        async def delete(collection: Any, conn: Any) -> None:
            await collection.delete_rows(keys, conn=conn)

        ticks, host = await trusted(
            lambda: _written(delete, door=False), lambda measured: measured, host_may_be_late=bound
        )
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while {_REAL:,} keys were deleted "
            f"({latest(ticks, every=_TICK) * 1000:.1f} ms in all; the per-key steps take {one_pass * 1000:.1f} ms here)"
        )

    async def test_a_ticker_is_never_held_while_a_listened_collection_writes_33786_rows(self) -> None:
        rows = await asyncio.to_thread(_rows)
        table = await asyncio.to_thread(_table, rows)
        one_pass = _heard_keys(_REAL)
        # the old write called the listeners a key at a time on the loop, every key in one stretch, as the write
        # evicted it and again as the commit settled it; 5 ms at the least, a GIL switch interval
        bound = max(0.005, one_pass / 2)
        holders = (_Holder(), _Holder())

        async def save(collection: Any, conn: Any) -> None:
            await collection.save_arrow(table, conn=conn)

        ticks, host = await trusted(
            lambda: _written(save, door=True, listeners=tuple(holder.changed for holder in holders)),
            lambda measured: measured,
            host_may_be_late=bound,
        )
        for holder in holders:
            await holder.settled()
            assert holder.heard >= 2 * _REAL, "a holder did not hear every key as written and as settled"
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while a listened collection wrote "
            f"{_REAL:,} rows ({latest(ticks, every=_TICK) * 1000:.1f} ms in all; hearing the keys takes "
            f"{one_pass * 1000:.1f} ms here)"
        )

    async def test_a_ticker_is_never_held_while_33786_keys_are_deleted_returning_their_rows(self) -> None:
        rows = await asyncio.to_thread(_rows)
        keys = [_key(row) for row in range(_REAL)]
        one_pass = _returned_steps(rows)
        # the old delete ran these steps on the loop, each statement's 1,000 rows (its reply decoded, its rows read) in
        # one stretch and every key enrolled and restated in one more: held 35.6 ms (a 10-core Apple host, 2026-11-16).
        # Half of one statement's share of the steps is a step back; 5 ms at the least, a GIL switch interval. What is
        # left is each reply's JSON decode, one C call no turn interrupts (3.5 ms for a megabyte here)
        statements = -(-_REAL // _Kept.BULK_MAX_ROWS)
        bound = max(0.005, one_pass / statements / 2)

        async def delete(collection: Any, conn: Any) -> None:
            await collection.delete_rows(keys, conn=conn)

        ticks, host = await trusted(
            lambda: _written(delete, door=False, hub=_ReturningHub(rows), kind=_Kept),
            lambda measured: measured,
            host_may_be_late=bound,
        )
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while {_REAL:,} keys were deleted "
            f"returning their rows ({latest(ticks, every=_TICK) * 1000:.1f} ms in all; the per-row steps take "
            f"{one_pass * 1000:.1f} ms here)"
        )
