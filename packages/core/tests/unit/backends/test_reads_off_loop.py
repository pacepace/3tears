"""A read's long reply is decoded and its rows read off the event loop: ``fetch``, ``fetchrow`` and ``fetch_page``.

A keyed read of one state's candidates (33,786 rows of 49 columns) at the production reply cap (12 MiB, the hub's
``FOURTEENAIBOTS_DB_L3_REPLY_MAX_BYTES``) is five statements, the largest replies about 10 MB: each reply's JSON
decoded and its rows read (``deserialize_row``) on the serving loop held it 180-210 ms a reply, and its decoded rows
freed there 5 ms more (a 10-core Apple host, 2026-11-16). A transaction's ``fetch`` read its reply on a thread already
(a delete returning its rows, ``test_bulk_write_off_loop``); the backend's own reads now share that reader
(``_read_rows_reply``).
"""

from __future__ import annotations

import asyncio
import gc
import json
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from threetears.nats import Subject

import threetears.core.backends.nats_proxy as nats_proxy
from threetears.core.backends.nats_proxy import READ_ON_THREAD_BYTES, NatsProxyL3Backend, deserialize_row
from threetears.core.backends.protocol import L3AnswerCutError
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.keyset import read_keyset_pages
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the 11-03 state that held the loop: 33,786 rows
_REAL = 33_786
#: the broker's reply cap in production (the hub's FOURTEENAIBOTS_DB_L3_REPLY_MAX_BYTES)
_CAP = 12 * 1024 * 1024
#: what the broker keeps of the cap for a reply's other fields
_CAP_ROOM = 64 * 1024
#: the key's columns: the 11-03 candidates table's seven
_KEY = tuple(f"k{index}" for index in range(7))
#: the columns past the key
_WIDTH = 40
#: a page's rows as the complete copy's keyed read asks for the first (``page_size + 1`` within the rail's 1,000)
_PAGE_SIZE = 999
#: what each tick of the loop's ticker asks to sleep
_TICK = 0.002
_AT = datetime(2026, 11, 3, tzinfo=UTC)


def _key(row: int) -> tuple[str, ...]:
    # zero-padded, so the keys' order is the rows' order
    return tuple(f"{name}-{row % (position + 3)}-{row:06d}" for position, name in enumerate(_KEY))


def _wire_row(row: int) -> dict[str, Any]:
    """row ``row`` as the broker's JSON carries it: text, numbers, a numeric as text, a flag, a timestamp as ISO text,
    and two ``date_`` columns the reader makes datetimes again."""
    values: dict[str, Any] = dict(zip(_KEY, _key(row), strict=True))
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


class _Hub:
    """a hub answering a keyed read's ``l3.query`` as the broker does at a byte cap: the rows past the request's
    cursor, at most its ``LIMIT``, cut at the cap (``truncated``), after one turn of the loop. Each reply is built once
    and kept, so a read measured after a first one measures the reader, not the hub."""

    def __init__(self, count: int = _REAL, *, cap: int = _CAP) -> None:
        self.rows = [_wire_row(row) for row in range(count)]
        self.encoded = [json.dumps(row).encode() for row in self.rows]
        self.at = {_key(row): row for row in range(count)}
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
        start = 0 if not params else self.at[tuple(params[-len(_KEY) :])] + 1
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


class _Answering:
    """a hub answering every request with ``reply``, after one turn of the loop."""

    def __init__(self, reply: bytes) -> None:
        self.reply = reply

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, payload, timeout
        await asyncio.sleep(0)
        return self.reply

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "t" * 900
        )


def _reply(rows: list[dict[str, Any]], **fields: Any) -> bytes:
    return json.dumps({"success": True, **fields, "rows": rows}).encode()


async def _keyed_read(backend: NatsProxyL3Backend) -> Any:
    return await read_keyset_pages(backend, "SELECT * FROM candidates", list(_KEY), page_size=_PAGE_SIZE)


def _recording_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """the thread each row is read on, in order."""
    threads: list[int] = []
    original = deserialize_row

    def recording(row: dict[str, Any]) -> dict[str, Any]:
        threads.append(threading.get_ident())
        return original(row)

    monkeypatch.setattr(nats_proxy, "deserialize_row", recording)
    return threads


class TestALongReplyIsReadOffTheLoopAsAShortOneIsOnIt:
    @pytest.mark.parametrize("count", [3, 300])
    async def test_fetch_answers_its_rows_as_read(self, count: int, monkeypatch: pytest.MonkeyPatch) -> None:
        wire = [_wire_row(row) for row in range(count)]
        reply = _reply(wire)
        threads = _recording_reads(monkeypatch)
        rows = await _Answering(reply).proxy().fetch("SELECT * FROM candidates")
        assert rows == [deserialize_row(row) for row in wire][: len(rows)] and len(rows) == count
        assert rows[0]["date_updated"] == _AT and rows[0]["c4"] == wire[0]["c4"], "a row was not read as the wire's"
        on_loop = threading.get_ident()
        if len(reply) > READ_ON_THREAD_BYTES:
            assert on_loop not in threads[:count], f"a {len(reply):,}-byte reply's rows were read on the loop"
        else:
            assert set(threads[:count]) == {on_loop}, f"a {len(reply):,}-byte reply's rows went to a thread"

    async def test_fetchrow_reads_the_first_row_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wire = [_wire_row(row) for row in range(300)]
        reply = _reply(wire, truncated=True)
        assert len(reply) > READ_ON_THREAD_BYTES
        expected = deserialize_row(wire[0])
        threads = _recording_reads(monkeypatch)
        row = await _Answering(reply).proxy().fetchrow("SELECT * FROM candidates")
        assert row == expected
        assert len(threads) == 1 and threads[0] != threading.get_ident(), (
            "fetchrow read more than its row, or on the loop"
        )
        assert await _Answering(_reply([])).proxy().fetchrow("SELECT 1") is None

    async def test_fetch_page_answers_its_rows_cut_and_sized_as_the_reply_says(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wire = [_wire_row(row) for row in range(300)]
        reply = _reply(wire, truncated=True, max_reply_bytes=_CAP)
        threads = _recording_reads(monkeypatch)
        page = await _Answering(reply).proxy().fetch_page("SELECT * FROM candidates LIMIT 301")
        assert page.rows == [deserialize_row(row) for row in wire]
        assert (page.truncated, page.reply_bytes, page.reply_cap) == (True, len(reply), _CAP)
        assert threading.get_ident() not in threads[:300], "a long page's rows were read on the loop"

    async def test_a_long_reply_the_rail_cut_is_refused_naming_its_rows(self) -> None:
        reply = _reply([_wire_row(row) for row in range(300)], truncated=True)
        with pytest.raises(L3AnswerCutError, match="at 300 rows"):
            await _Answering(reply).proxy().fetch("SELECT * FROM candidates")

    @pytest.mark.parametrize("size", [100, 200_000])
    async def test_an_unreadable_reply_is_refused(self, size: int) -> None:
        with pytest.raises(DataLayerUnavailableError):
            await _Answering(b"{" + b" " * size).proxy().fetch_page("SELECT * FROM candidates")

    async def test_a_failed_long_reply_raises_as_a_short_one(self) -> None:
        reply = json.dumps({"success": False, "error_message": "relation missing", "pad": "x" * 200_000}).encode()
        with pytest.raises(DataLayerUnavailableError, match="relation missing"):
            await _Answering(reply).proxy().fetch("SELECT * FROM candidates")

    async def test_a_keyed_read_answers_every_row_in_key_order(self) -> None:
        hub = _Hub(5000, cap=1024 * 1024)
        read = await _keyed_read(hub.proxy())
        assert [tuple(row[name] for name in _KEY) for row in read.rows] == [_key(row) for row in range(5000)]
        assert read.rows[-1] == deserialize_row(hub.rows[-1])
        assert read.statements > 2, "the read was not paged at the cap"


class TestTheLoopRunsWhileAKeyedReadReadsItsRows:
    async def test_the_loop_runs_between_stretches_of_33786_rows_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # what the timing budget below bounds by the clock, counted: the loop's other work runs while a reply's rows
        # are read, not once they all are (the old reads read each reply's rows in one stretch: five stretches)
        hub = _Hub()
        turns = 0
        seen: list[int] = []
        original = deserialize_row

        def counting(row: dict[str, Any]) -> dict[str, Any]:
            seen.append(turns)
            return original(row)

        async def ticker() -> None:
            nonlocal turns
            while True:
                turns += 1
                await asyncio.sleep(0)

        monkeypatch.setattr(nats_proxy, "deserialize_row", counting)
        ticking = asyncio.get_running_loop().create_task(ticker())
        try:
            read = await _keyed_read(hub.proxy())
        finally:
            ticking.cancel()
        assert len(read.rows) == _REAL and len(seen) >= _REAL
        assert max(hub.answered) > 8 * 1024 * 1024, "the read's pages were not sized to the cap"
        stretches = sum(1 for before, after in zip(seen, seen[1:]) if after != before) + 1
        assert stretches > 2 * read.statements, (
            f"the {_REAL:,} rows were read in {stretches} stretches over {read.statements} statements, the loop held "
            f"through each reply's rows"
        )


def _read_steps(hub: _Hub) -> tuple[float, int]:
    """what the old reads' per-reply steps alone take here, done as the loop once did them: each reply's JSON decoded
    and its rows read; and the statements."""
    replies = list(hub.replies.values())
    gc.disable()
    try:
        started = time.perf_counter()
        for reply in replies:
            [deserialize_row(row) for row in json.loads(reply)["rows"]]
        return time.perf_counter() - started, len(replies)
    finally:
        gc.enable()


async def _read_ticks(hub: _Hub, kept: list[Any]) -> tuple[list[Tick], list[Tick]]:
    """the loop's ticks and the host's idle loop's while a keyed read reads :data:`_REAL` rows; the rows are kept past
    the window in ``kept``, as the reader keeps them.

    The collector is held for the window, as a serving process holds it through a refresh (``collector_held``): what
    this measures is the read's own hold, not a walk of the heap its allocations set off."""
    backend = hub.proxy()

    async def read() -> None:
        kept.append(await _keyed_read(backend))

    gc.collect()
    gc.disable()
    try:
        with HostTicker() as host:
            ticks = await ticks_while(read, every=_TICK)
    finally:
        gc.enable()
    return ticks, host.ticks


@pytest.mark.loop_timing
class TestTheLoopRunsWhileARealSizeKeyedReadIsRead:
    async def test_a_ticker_is_never_held_while_33786_rows_are_read_at_the_production_cap(self) -> None:
        hub = _Hub()
        kept: list[Any] = []
        # the hub's replies built once, outside any window
        await _keyed_read(hub.proxy())
        one_pass, statements = _read_steps(hub)
        # the old reads ran these steps on the loop, each reply's (up to 10 MB) in one stretch: held 190-215 ms (a
        # 10-core Apple host, 2026-11-16). Whatever a reply's size, the reads hold the loop no longer than a GIL switch
        # interval, 5 ms
        bound = 0.005
        ticks, host = await trusted(lambda: _read_ticks(hub, kept), lambda measured: measured, host_may_be_late=bound)
        assert kept and all(len(read.rows) == _REAL for read in kept)
        worst = added_lateness(ticks, host, every=_TICK)
        assert worst < bound, (
            f"the loop was held {worst * 1000:.1f} ms past the host's idle loop while {_REAL:,} rows were read in "
            f"{statements} statements ({latest(ticks, every=_TICK) * 1000:.1f} ms in all; the per-reply steps take "
            f"{one_pass * 1000:.1f} ms here)"
        )
