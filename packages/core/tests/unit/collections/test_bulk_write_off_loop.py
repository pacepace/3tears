"""A bulk write of many rows on a caller's transaction holds the event loop no longer than a few milliseconds at a time.

One state's candidates (33,786 rows) written through ``save_arrow`` held the serving loop 6-7 ms with the collector held
and 50-55 ms with it running, and through ``save_rows`` 87-103 ms and 134-156 ms (the WD2 bench, 2026-11-16, 179
columns, a hub answering at once): every key read into Python, checked for a twin, enrolled and fenced on the loop, a
call a key; every row checked, stamped and keyed on the loop; each statement's text and flattened parameters built on
the loop, 7-8 ms a statement; and the statements' parameters freed on the loop as the write returned. All of it now
runs on worker threads that hand the loop a turn every half millisecond (:mod:`threetears.core.loop_turns`).

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
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from threetears.nats import Subject

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
            Column(name, kind, nullable=True, **({"precision": 18, "scale": 6} if kind == NUMERIC_TYPE else {}))
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


async def _written(save: Callable[[Any, Any], Any], *, door: bool) -> tuple[list[Tick], list[Tick]]:
    """the loop's ticks and the host's idle loop's while one transaction writes :data:`_REAL` rows with ``save``.

    The collector is held for the window, as a serving process holds it through a refresh (the reports pod's
    ``collector_held``): a collection the write's allocations set off walks the heap with the GIL held, wherever the
    allocation runs, 4-6 ms at a time here and 40-50 ms beside a heap of rows as Python values; that walk is the
    runtime's, sized by the process's heap. What this measures is the write's own hold."""
    backend = _Hub(door=door).proxy()
    collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
    collection.l3_pool = backend

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
            transaction.enroll(MagicMock, key, row=each)
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

        collection.invalidate_cache_many = settle
        async with backend.acquire() as conn, CallerTransaction(conn):
            if how == "save_arrow":
                await collection.save_arrow(_table(rows), conn=conn)
            else:
                await collection.save_rows(rows, conn=conn)
        assert made and threading.get_ident() not in made, "the enrollments were made on the loop"
        assert settled == [_key(index) for index in range(200)]

    @pytest.mark.parametrize("how", ["save_arrow", "save_rows"])
    async def test_a_key_twice_is_still_refused_before_anything_is_enrolled(self, how: str) -> None:
        rows = _rows_of(200)
        rows[150] = {**rows[150], **dict(zip(_KEY, _key(3), strict=True))}
        backend = _Hub(door=how == "save_arrow").proxy()
        collection = _Candidates(CollectionRegistry(), DefaultCoreConfig(), None)
        collection.l3_pool = backend
        settled: list[Any] = []

        async def settle(keys: Any) -> None:
            settled.extend(keys)

        collection.invalidate_cache_many = settle
        async with backend.acquire() as conn, CallerTransaction(conn):
            with pytest.raises(ValueError, match="two rows share a key"):
                if how == "save_arrow":
                    await collection.save_arrow(_table(rows), conn=conn)
                else:
                    await collection.save_rows(rows, conn=conn)
        assert settled == [], "a refused write enrolled keys"


@pytest.mark.loop_timing
class TestTheLoopRunsWhileARealSizeWriteIsSaved:
    @pytest.mark.parametrize("how", ["save_arrow", "save_rows"])
    async def test_a_ticker_is_never_held_while_33786_rows_are_written(self, how: str) -> None:
        rows = _rows()
        table = _table(rows)
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
