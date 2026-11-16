"""An L2 refill's read of L3 holds the event loop a fraction of a millisecond at a time: its pages are read-coerced and
freed off the loop.

``L2FirstSupport.refill_from_l3`` reads its table from L3 a keyset page at a time (``SchemaBackedCollection``'s pages,
read-coerced as ``fetch_from_store`` reads a row) and puts each row into L2. At the production reply cap (12 MiB, the
hub's ``FOURTEENAIBOTS_DB_L3_REPLY_MAX_BYTES``) a page of a 49-column table is about 10,000 rows: each page was coerced
on the loop in one stretch (55-59 ms), its rows as read freed there as the pager read the next (3 ms), and the coerced
rows freed as the refill took the next page (6.5 ms). One state's 33,786 rows held the loop 70-75 ms a turn (the
longest turn's CPU time, a 10-core Apple host, 2026-11-16); the pages are now coerced on a worker thread taking turns
with the loop and emptied there (:mod:`threetears.core.loop_turns`), the longest turn 0.3-0.4 ms.
"""

from __future__ import annotations

import asyncio
import gc
import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from sqlalchemy import MetaData
from threetears.nats import Subject, set_default_namespace

import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.nats_proxy import NatsProxyL3Backend, deserialize_row
from threetears.core.backends.protocol import L3Page
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import CollectionRegistry
from threetears.core.collections.l2_first import L2_FIRST
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
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the 11-03 state that held the loop: 33,786 rows
_REAL = 33_786
#: the broker's reply cap in production (the hub's FOURTEENAIBOTS_DB_L3_REPLY_MAX_BYTES)
_CAP = 12 * 1024 * 1024
#: what the broker keeps of the cap for a reply's other fields
_CAP_ROOM = 64 * 1024
#: the text columns beside the key, as the candidates table's other six key columns
_TEXT = tuple(f"k{index}" for index in range(1, 7))
#: the columns past them, of the candidates' kinds in turn
_WIDTH = 40
_KINDS = (STRING_TYPE, BIGINT_TYPE, NUMERIC_TYPE, BOOL_TYPE, DATETIMETZ_TYPE)
#: the refiller's page (l2_first's refill page)
_PAGE_SIZE = 500
#: what each tick of the loop's ticker asks to sleep
_TICK = 0.002
_AT = datetime(2026, 11, 3, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("refilloffloop")


def _id(row: int) -> str:
    # zero-padded, so the keys' order is the rows' order
    return f"r{row:06d}"


def _wire_row(row: int) -> dict[str, Any]:
    """row ``row`` as the broker's JSON carries it: text, numbers, a numeric as text, a flag, a timestamp as ISO text,
    and two ``date_`` columns the reader makes datetimes again: 49 columns."""
    values: dict[str, Any] = {"id": _id(row)}
    values.update({name: f"{name}-{row % (position + 3)}" for position, name in enumerate(_TEXT)})
    for column in range(_WIDTH):
        kind = column % 5
        if kind == 0:
            values[f"c{column}"] = f"v{row % 97}-{column}"
        elif kind == 1:
            values[f"c{column}"] = row * 3 + column
        elif kind == 2:
            values[f"c{column}"] = f"{row % 1000}.{(row + column) % 1000000:06d}"
        elif kind == 3:
            values[f"c{column}"] = (row + column) % 2 == 0
        else:
            values[f"c{column}"] = (_AT + timedelta(seconds=row + column)).isoformat()
    values["date_created"] = _AT.isoformat()
    values["date_updated"] = (_AT + timedelta(seconds=row)).isoformat()
    return values


def _schema() -> TableSchema:
    columns = [Column("id", STRING_TYPE)]
    columns += [Column(name, STRING_TYPE, nullable=True) for name in _TEXT]
    columns += [
        Column(
            f"c{column}",
            _KINDS[column % 5],
            nullable=True,
            **({"precision": 12, "scale": 6} if _KINDS[column % 5] == NUMERIC_TYPE else {}),
        )
        for column in range(_WIDTH)
    ]
    columns += [Column("date_created", DATETIMETZ_TYPE, nullable=True)]
    columns += [Column("date_updated", DATETIMETZ_TYPE, nullable=True)]
    return TableSchema(
        name="candidates", primary_key="id", columns=columns, on_conflict="update", l1_eviction=TEST_L1_TABLE
    )


class _Hub:
    """a hub answering a keyed read's ``l3.query`` as the broker does at a byte cap: the rows past the request's
    cursor, at most its ``LIMIT``, cut at the cap (``truncated``), after one turn of the loop. Each reply is built once
    and kept, so a refill measured after a first one measures the reader, not the hub."""

    def __init__(self, count: int = _REAL, *, cap: int = _CAP) -> None:
        self.rows = [_wire_row(row) for row in range(count)]
        self.encoded = [json.dumps(row).encode() for row in self.rows]
        self.at = {_id(row): row for row in range(count)}
        self.cap = cap
        self.replies: dict[tuple[int, int], bytes] = {}
        self.answered: list[int] = []

    def reply(self, start: int, limit: int) -> bytes:
        kept = self.replies.get((start, limit))
        if kept is None:
            end, size = start, 0
            while (
                end < len(self.encoded)
                and end - start < limit
                and size + len(self.encoded[end]) <= self.cap - _CAP_ROOM
            ):
                size += len(self.encoded[end]) + 1
                end += 1
            cut = end < len(self.encoded) and end - start < limit
            kept = (
                b'{"success": true, "truncated": '
                + (b"true" if cut else b"false")
                + b', "max_reply_bytes": '
                + str(self.cap).encode()
                + b', "rows": ['
                + b",".join(self.encoded[start:end])
                + b"]}"
            )
            self.replies[(start, limit)] = kept
        return kept

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, timeout
        await asyncio.sleep(0)
        request = json.loads(payload)
        match = re.search(r"LIMIT (\d+)", request["query"])
        assert match is not None, "a keyed read's page names its LIMIT"
        params = request["params"]
        start = 0 if not params else self.at[params[-1]] + 1
        reply = self.reply(start, int(match.group(1)))
        self.answered.append(len(reply))
        return reply

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 16 * 1024 * 1024
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "t" * 900
        )


def _collection(l3: Any, bus: FakeNatsClient) -> Any:
    """an L2-first collection of :func:`_schema` over ``l3``, its L2 the fake bus's bucket."""
    schema = _schema()
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    declared = type("Candidates", (collection_for_schema(schema),), {"l2_first": L2_FIRST})
    l1 = SQLiteBackend(db_name=f"refill_off_loop_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=bus, l3_pool=SqlL3Backend(l3), kv_key_scope="fleet")
    return declared(registry, DefaultCoreConfig())


async def _refill(l3: Any, bus: FakeNatsClient | None = None) -> tuple[Any, int]:
    collection = _collection(l3, bus if bus is not None else FakeNatsClient())
    written: int = await collection.l2_first_support.refill_from_l3(page_size=_PAGE_SIZE)
    return collection, written


class TestARefillPutsEveryRowAsRead:
    @pytest.mark.parametrize("count", [50, 5000])
    async def test_every_row_is_put_read_coerced_in_key_order(self, count: int) -> None:
        # 50 rows are one page coerced on the loop; 5,000 at a 1 MiB cap are pages of about 800, on a thread
        hub = _Hub(count, cap=1024 * 1024)
        bus = FakeNatsClient()
        collection, written = await _refill(hub.proxy(), bus)
        assert written == count
        assert (len(hub.answered) > 2) == (count > 50), "the read was not paged as the test means"
        kv = await bus.kv_bucket(name=collection.L2_BUCKET_SUFFIX)
        assert kv.keys() == tuple(collection.l2_key(_id(row)) for row in range(count))
        schema = _schema()
        for row in (0, count // 2, count - 1):
            expected = schema_sql.coerce_row(schema, deserialize_row(hub.rows[row]))
            held = collection.deserialize(await kv.get(key=collection.l2_key(_id(row))))
            assert held == collection.deserialize(collection.serialize(expected)), f"row {row} was not put as read"
            assert held["date_updated"] == _AT + timedelta(seconds=row)


class _Freed(str):
    """a text value that says on which thread it was freed."""

    __slots__ = ()
    freed: ClassVar[dict[str, int]] = {}

    def __del__(self) -> None:
        type(self).freed[str(self)] = threading.get_ident()


class _Paged:
    """an L3 answering keyset pages of ``count`` rows from fresh dicts each time, sized to the production cap; each
    row's ``c0`` a :class:`_Freed` naming its row, so the test sees where the rows it answered were freed."""

    rows_per_statement = None

    def __init__(self, count: int) -> None:
        self.count = count
        self.answered: list[int] = []

    def _row(self, row: int) -> dict[str, Any]:
        values = deserialize_row(_wire_row(row))
        values["c0"] = _Freed(f"freed-{row}")
        return values

    async def fetch_page(self, query: str, *params: Any) -> L3Page:
        await asyncio.sleep(0)
        match = re.search(r"LIMIT (\d+)", query)
        assert match is not None, "a keyed read's page names its LIMIT"
        start = 0 if not params else int(str(params[-1])[1:]) + 1
        end = min(self.count, start + int(match.group(1)))
        self.answered.append(end - start)
        # 1,200 bytes a row of a 12 MiB cap: the next page asks for about as many rows as the cap holds
        return L3Page(
            rows=[self._row(row) for row in range(start, end)],
            truncated=False,
            reply_bytes=1200 * (end - start),
            reply_cap=_CAP,
        )

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        return (await self.fetch_page(query, *params)).rows


class TestTheLoopRunsWhileARefillReadsItsRows:
    async def test_the_loop_runs_between_stretches_of_33786_rows_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # what the timing budget below bounds by the clock, counted: the loop's other work runs while a page's rows are
        # coerced, not once they all are (the old refill coerced each page's rows in one stretch on the loop)
        hub = _Hub()
        turns = 0
        seen: list[tuple[int, int]] = []
        original = schema_sql.normalize_read_value

        def counting(column: Column, value: Any) -> Any:
            if column.name == "id":
                seen.append((turns, threading.get_ident()))
            return original(column, value)

        async def ticker() -> None:
            nonlocal turns
            while True:
                turns += 1
                await asyncio.sleep(0)

        monkeypatch.setattr(schema_sql, "normalize_read_value", counting)
        ticking = asyncio.get_running_loop().create_task(ticker())
        try:
            _, written = await _refill(hub.proxy())
        finally:
            ticking.cancel()
        assert written == _REAL and len(seen) == _REAL
        assert max(hub.answered) > 8 * 1024 * 1024, "the refill's pages were not sized to the cap"
        statements = len(hub.answered)
        stretches = sum(1 for (before, _), (after, _) in zip(seen, seen[1:]) if after != before) + 1
        assert stretches > 2 * statements, (
            f"the {_REAL:,} rows were coerced in {stretches} stretches over {statements} statements, the loop held "
            f"through each page's rows"
        )
        on_loop = threading.get_ident()
        assert sum(1 for _, thread in seen if thread == on_loop) == 0, "a page's rows were coerced on the loop"

    async def test_a_long_page_s_rows_are_freed_off_the_loop(self) -> None:
        # a page's rows as read are freed where its coerce runs, and the coerced rows where the refill empties the
        # page once they are put: on a worker thread for a long page (the old refill freed both on the loop)
        l3 = _Paged(6000)
        _Freed.freed.clear()
        gc.collect()
        _, written = await _refill(l3)
        gc.collect()
        assert written == 6000 and l3.answered[0] == _PAGE_SIZE + 1 and max(l3.answered) > 2000, l3.answered
        on_loop = threading.get_ident()
        long_pages = [f"freed-{row}" for row in range(_PAGE_SIZE, 6000)]
        assert all(name in _Freed.freed for name in long_pages), "a row's value was never freed"
        held_on_loop = sum(1 for name in long_pages if _Freed.freed[name] == on_loop)
        assert held_on_loop == 0, f"{held_on_loop:,} rows of the long pages were freed on the loop"


def _coerce_steps(hub: _Hub) -> tuple[float, int]:
    """what the old refill's per-page steps alone take here, done as the loop once did them: each reply's rows read,
    then coerced; and the statements."""
    schema = _schema()
    replies = list(hub.replies.values())
    read = [[deserialize_row(row) for row in json.loads(reply)["rows"]] for reply in replies]
    gc.disable()
    try:
        started = time.perf_counter()
        for rows in read:
            [schema_sql.coerce_row(schema, row) for row in rows]
        return time.perf_counter() - started, len(replies)
    finally:
        gc.enable()


async def _refill_ticks(hub: _Hub, kept: list[int]) -> tuple[list[Tick], list[Tick]]:
    """the loop's ticks and the host's idle loop's while a refill reads :data:`_REAL` rows; the count put kept in
    ``kept``.

    The collector is held for the window, as ``test_reads_off_loop`` holds it: what this measures is the refill's own
    hold, not a walk of the heap its allocations set off."""
    backend = hub.proxy()

    async def refill() -> None:
        kept.append((await _refill(backend))[1])

    gc.collect()
    gc.disable()
    try:
        with HostTicker() as host:
            ticks = await ticks_while(refill, every=_TICK)
    finally:
        gc.enable()
    return ticks, host.ticks


@pytest.mark.loop_timing
class TestTheLoopRunsWhileARealSizeRefillIsRead:
    async def test_a_ticker_is_never_held_while_33786_rows_are_refilled_at_the_production_cap(self) -> None:
        hub = _Hub()
        kept: list[int] = []
        # the hub's replies built once, outside any window
        await _refill(hub.proxy())
        one_pass, statements = _coerce_steps(hub)
        # the old refill coerced each page (about 10,000 rows) in one stretch on the loop, and freed its rows there:
        # held 70-75 ms a turn (a 10-core Apple host, 2026-11-16). Whatever a page's size, the refill holds the loop no
        # longer than a GIL switch interval, 5 ms
        bound = 0.005
        ticks, host = await trusted(lambda: _refill_ticks(hub, kept), lambda measured: measured, host_may_be_late=bound)
        assert kept and all(written == _REAL for written in kept)
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while {_REAL:,} rows were refilled in "
            f"{statements} statements ({latest(ticks, every=_TICK) * 1000:.1f} ms in all; the pages' coerce takes "
            f"{one_pass * 1000:.1f} ms here)"
        )
