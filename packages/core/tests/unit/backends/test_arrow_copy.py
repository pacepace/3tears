"""An ``l3.tx.copy`` request's rows as Arrow, sent only to a hub that announces it reads them.

What the pod sends is the contract the hub's broker implements (its ``test_copy_arrow`` holds the other half: the
``COPY`` bytes of an Arrow request are those of the JSON request of the same rows). Pinned here:

- to a hub whose ``tx.begin`` reply says ``copy_arrow``, the request is the magic, the envelope (the JSON request's
  fields but its rows, the rows' digest and their JSON columns) and the rows' IPC stream, whose values are those
  the JSON request carries, value for value and kind for kind;
- to a hub that does not, and for rows that as Arrow pass the request's budget, the JSON request exactly as before;
- a proof over the envelope verifies at the door and covers the rows through their digest;
- the rows are laid out on a thread, never the event loop.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pyarrow as pa
import pytest
from threetears.nats import Subject

from threetears.core.backends import arrow_copy, nats_proxy
from threetears.core.backends.arrow_copy import (
    COPY_ARROW_FIELD,
    JSON_COLUMNS_FIELD,
    ROWS_DIGEST_FIELD,
    copy_rows_arrow,
    rows_digest,
    split_copy_request,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import json_default, transport_param
from threetears.core.security import (
    PopKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.core.testing.loop_lateness import (
    MEASURE_ATTEMPTS,
    HostTicker,
    Tick,
    added_lateness,
    hold_the_loop,
    latest,
    quietly,
    ticks_while,
    trusted,
)

_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_AGENT, _AGENT_PUB = generate_signing_keypair()

_COLUMNS = ["name", "n", "big", "ratio", "flag", "id", "share", "at", "on", "blob", "doc", "tags", "mixed", "none"]


def _record(n: int) -> tuple[Any, ...]:
    """one row of the values a pod hands a copy, every kind the rail carries."""
    return (
        f'name {n} ✓ "q" \\',
        n,
        2**63 - 1 if n % 2 else 2**64,
        (-0.0, 0.0, math.nan, math.inf, n / 7)[n % 5],
        n % 2 == 0,
        UUID(int=n + 1),
        Decimal(f"{n}.50"),
        datetime(2026, 11, 4, 4, 41, 7, n, tzinfo=UTC if n % 2 else None),
        date(2026, 11, 1 + n % 8),
        bytes([n % 256, 0]),
        {"k": [1, 2.5, None], "at": datetime(2026, 1, 1, tzinfo=UTC)},
        ["a", None, n],
        (1, 2.5, "x", None)[n % 4],
        None,
    )


_RECORDS = [_record(n) for n in range(40)] + [tuple([None] * len(_COLUMNS))]


def _token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class _Hub:
    """the bus a proxy sends on: keeps each request's bytes, and answers ``tx.begin`` with ``begin``."""

    def __init__(self, begin: dict[str, Any]) -> None:
        self.raw: list[tuple[str, bytes]] = []
        self._begin = begin

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.raw.append((subject.path, payload))
        reply = self._begin if subject.path.endswith("tx.begin") else {"success": True, "row_count": 41}
        return json.dumps(reply).encode("utf-8")

    def proxy(self, *, proven: bool = False) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 64 * 1024 * 1024
        token = _token()
        return NatsProxyL3Backend(
            nats_client=nats,
            namespace_prefix="test",
            agent_id="agent-1",
            identity_token=lambda: token,
            pop_key=(lambda: PopKey(_AGENT)) if proven else None,
        )

    def copy(self) -> bytes:
        [data] = [payload for subject, payload in self.raw if subject.endswith("tx.copy")]
        return data


async def _copied(begin: dict[str, Any], records: list[tuple[Any, ...]], *, proven: bool = False) -> bytes:
    hub = _Hub({"success": True, "tx_id": _TX_ID, **begin})
    async with hub.proxy(proven=proven).acquire() as conn, conn.transaction():
        assert await conn.copy_records_to_table("rows", records=records, columns=_COLUMNS) == "COPY 41"
    return hub.copy()


def _arrow_parts(data: bytes) -> tuple[dict[str, Any], bytes]:
    split = split_copy_request(data)
    assert split is not None
    envelope, offset = split
    return json.loads(bytes(envelope)), data[offset:]


def _json_rows(records: list[tuple[Any, ...]]) -> str:
    """the rows as the JSON request carries them, read back and written again: every kind and value it holds."""
    sent = json.dumps([[transport_param(value) for value in record] for record in records], default=json_default)
    return json.dumps(json.loads(sent))


async def test_a_hub_that_reads_arrow_is_sent_the_rows_the_json_request_carries_as_arrow() -> None:
    data = await _copied({COPY_ARROW_FIELD: True}, _RECORDS)
    assert split_copy_request(data) is not None
    envelope, ipc = _arrow_parts(data)
    assert set(envelope) == {
        "correlation_id", "identity_token", "tx_id", "table", "columns", ROWS_DIGEST_FIELD, JSON_COLUMNS_FIELD,
    }  # fmt: skip
    assert (envelope["tx_id"], envelope["table"], envelope["columns"]) == (_TX_ID, "rows", _COLUMNS)
    assert envelope[ROWS_DIGEST_FIELD] == rows_digest(ipc)
    table = pa.ipc.open_stream(ipc).read_all()
    as_json = set(envelope[JSON_COLUMNS_FIELD])
    # an integer past 64 bits, a date, an object, an array and a mix of kinds go as their JSON text
    assert as_json == {_COLUMNS.index(name) for name in ("big", "on", "doc", "tags", "mixed")}
    assert table.schema.field("ratio").type == pa.float64()
    assert table.schema.field("n").type == pa.int64()
    columns = [
        [None if text is None else json.loads(text) for text in table.column(index).to_pylist()]
        if index in as_json
        else table.column(index).to_pylist()
        for index in range(len(_COLUMNS))
    ]
    # value for value and kind for kind (1 and 1.0, 0.0 and -0.0, NaN), as JSON writes them
    assert json.dumps([list(row) for row in zip(*columns, strict=True)]) == _json_rows(_RECORDS)


async def test_a_hub_that_does_not_announce_it_is_sent_json_as_before() -> None:
    for begin in ({}, {COPY_ARROW_FIELD: False}, {COPY_ARROW_FIELD: "yes"}):
        body = json.loads(await _copied(begin, _RECORDS))
        assert json.dumps(body["rows"]) == _json_rows(_RECORDS)


async def test_rows_that_as_arrow_pass_the_requests_budget_go_as_json() -> None:
    # nulls and small integers take more bytes as Arrow than as JSON
    records = [(None, 1) + (None,) * (len(_COLUMNS) - 2)] * 41
    budget = len(copy_rows_arrow(_COLUMNS, records).ipc) + 64 * 1024 - 1
    data = await _copied({COPY_ARROW_FIELD: True, "max_request_bytes": budget}, records)
    assert data.startswith(b"{")
    data = await _copied({COPY_ARROW_FIELD: True, "max_request_bytes": budget + 1}, records)
    assert split_copy_request(data) is not None


async def test_a_proof_over_the_envelope_verifies_at_the_door_and_names_the_rows() -> None:
    data = await _copied({COPY_ARROW_FIELD: True}, _RECORDS, proven=True)
    envelope, ipc = _arrow_parts(data)
    caller = verify_door_request(envelope, jwks=_JWKS, hub_issuer="hub", door="l3", op="tx.copy")
    assert caller.agent_id == "agent-1"
    assert envelope[ROWS_DIGEST_FIELD] == rows_digest(ipc)


async def test_the_rows_are_laid_out_on_a_thread_never_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[threading.Thread] = []

    def laid_out(columns: Any, records: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread())
        return copy_rows_arrow(columns, records, **kwargs)

    monkeypatch.setattr(nats_proxy, "copy_rows_arrow", laid_out)
    await _copied({COPY_ARROW_FIELD: True}, _RECORDS)
    assert threads and threads[0] is not threading.main_thread()


async def test_a_copy_starts_no_thread_on_the_callers_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a copy (a process's first, or the first on a new loop) starts a thread from the caller's loop: a
    thread's start waits for it to run (1-2 ms on a loaded host), so its build and framing run on the copy thread
    started at import."""
    started: list[str] = []
    start = threading.Thread.start

    def recorded(thread: threading.Thread) -> None:
        if threading.current_thread() is threading.main_thread():
            started.append(thread.name)
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", recorded)
    await _copied({COPY_ARROW_FIELD: True}, _RECORDS)
    await _copied({}, _RECORDS)
    assert started == []


class _HandingLoop:
    """the caller's loop as a build sees it, counting the turns handed to it and the turns it ran."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.handed = 0
        self.ran = 0
        self.waited: list[float] = []
        self.first_handed = threading.Event()

    def is_closed(self) -> bool:
        return self.loop.is_closed()

    def call_soon_threadsafe(self, callback: Any) -> Any:
        self.handed += 1
        posted = time.perf_counter()

        def run() -> None:
            self.ran += 1
            self.waited.append(time.perf_counter() - posted)
            callback()

        handle = self.loop.call_soon_threadsafe(run)
        self.first_handed.set()
        return handle


async def test_a_build_hands_its_turns_to_the_callers_loop_and_waits_for_each() -> None:
    """a turn is handed, not offered: the build schedules each turn on the caller's loop and goes on only once the loop
    ran it. A sleep-only turn let a build on a loaded host take the lock back before the loop's thread was scheduled,
    and the loop waited out the 5 ms switch interval turn after turn.

    The loop stays busy from the moment the build hands its first turn, not from the build's start: how long a build
    on a loaded host takes to reach that first turn is the host's, and a busy stretch timed from the start could end
    before the turn was handed (pool, brooks-nas: the first turn waited 7 ms of a 30 ms busy loop). The busy stretch
    stays well inside the build's bounded wait for a turn (50 ms), so a slow sleep cannot let the build go on."""
    loop = _HandingLoop(asyncio.get_running_loop())

    def build() -> Any:
        return copy_rows_arrow([f"c{n}" for n in range(42)], _wide_records(4_000), loop=loop)  # type: ignore[arg-type]

    building = loop.loop.run_in_executor(None, build)
    assert loop.first_handed.wait(5), "the build handed the loop no turn"
    time.sleep(0.02)  # the loop busy past the first turn's handing: the build waits for it
    handed_while_busy = loop.handed
    rows = await building
    assert rows.ipc
    assert loop.handed > 1 and loop.ran == loop.handed
    assert handed_while_busy == 1, "the build went on while its handed turn waited for the loop"
    assert loop.waited[0] >= 0.02, loop.waited


class _TimedLoop:
    """the caller's loop as a build sees it: each handed turn runs after the lock is let go for a moment (the loop's
    turn), and the stretch the build ran before handing it is timed on the build thread's own CPU clock, with the
    build's step that handed it."""

    def __init__(self) -> None:
        self.stretches: list[tuple[float, str]] = []
        self._since = time.thread_time()

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback: Any) -> None:
        held = time.thread_time() - self._since
        step = next(
            frame.name for frame in reversed(traceback.extract_stack(limit=6)) if frame.name not in _TURN_FRAMES
        )
        self.stretches.append((held, step))
        time.sleep(0.0002)  # the loop's turn: the lock let go, as a real loop's turn lets it go
        callback()
        self._since = time.thread_time()


#: the frames of the turn itself and of the fake loop, not the build's step that took it
_TURN_FRAMES = frozenset({"call_soon_threadsafe", "hand", "take"})


@pytest.mark.loop_timing
def test_the_build_holds_the_lock_under_1_ms_between_turns() -> None:
    """every stretch the build runs holding the interpreter lock, between two of its turns, costs its thread under
    1 ms of CPU over a 12 MB copy. On a loaded host the OS can stop the build mid-stretch, and the caller's loop is late
    by that stop only when its tick falls inside a stretch: 1 ms stretches left it 47.5 ms late at worst on a Linux
    container with every CPU busy. The build thread's own CPU time is counted, never the wall clock, so a host's load
    cannot stretch a stretch past its work. The one stretch ending in the IPC write is Arrow's, with the lock
    released (it hands a turn the moment it returns), and is not a hold."""
    records = _wide_records(16_000)

    def measure() -> _TimedLoop:
        loop = _TimedLoop()
        rows = copy_rows_arrow([f"c{n}" for n in range(42)], records, loop=loop)  # type: ignore[arg-type]
        assert len(rows.ipc) > 10_000_000
        return loop

    # thread CPU time is still stretched by a busy sibling core or a throttled one: judged only when a spin on one
    # thread ran at the host's best around it
    loop = quietly(measure, host_may_be_late=0.020, threads=1)
    written = [held for held, step in loop.stretches if step == "_ipc_stream"]
    holds = sorted((held, step) for held, step in loop.stretches if step != "_ipc_stream")
    assert len(written) == 1 and len(holds) > 20
    assert holds[-1][0] < 0.001, holds[-5:]


def test_a_request_that_is_not_one_is_not_split() -> None:
    assert split_copy_request(b'{"rows": []}') is None
    with pytest.raises(ValueError, match="past its end"):
        split_copy_request(b"L3COPYA1" + (100).to_bytes(4, "big") + b"{}")


def _wide_records(count: int) -> list[tuple[Any, ...]]:
    """rows shaped as the reports pod's candidates (42 columns: text, integers, floats, booleans, nulls), ~490 bytes."""
    return [
        tuple(
            [f"text {n % 97} of the row {column}" for column in range(26)]
            + [n * column for column in range(6)]
            + [n / 7, float(n % 11)]
            + [n % 2 == 0, None]
            + [None] * 6
        )
        for n in range(count)
    ]


#: rounds of idle, copy, idle each run: the copy judged against the idle windows either side of it, so load arriving or
#: leaving the host mid-run is seen in the windows it falls in
_ROUNDS = 4

#: how long each idle window ticks
_IDLE_SECONDS = 0.1


@dataclass
class _Round:
    """one round's measurements: the test loop's worst idle lateness either side, and its ticks during the copy."""

    before: float
    during: list[Tick]
    after: float


def _held_too_long(rounds: list[_Round], host: list[Tick]) -> str | None:
    """the first round whose copy held the caller's loop past its own idle jitter and a turn: 5 ms (CPython's switch
    interval, the most a thread waiting for the lock waits once the holder yields it) plus twice the worse of the idle
    windows either side of it -- counting only lateness the host's idle loop, ticking in its own process at the same
    moments, did not share.

    :param rounds: each round's idle lateness before and after, and its ticks during the copy
    :ptype rounds: list[_Round]
    :param host: the host loop's ticks over the whole run
    :ptype host: list[Tick]
    :return: what held too long, or None
    :rtype: str | None
    """
    for index, measured in enumerate(rounds):
        idle = max(measured.before, measured.after)
        added = added_lateness(measured.during, host)
        if added >= 0.005 + 2 * idle:
            return (
                f"round {index + 1}: a 12 MB copy's build held the caller's loop {added * 1000:.1f} ms past the "
                f"host's own idle loop ({latest(measured.during) * 1000:.1f} ms late in all; idle "
                f"{measured.before * 1000:.1f} ms before, {measured.after * 1000:.1f} ms after)"
            )
    return None


async def _copy_rounds(records: list[tuple[Any, ...]], *, during_copy: Any = None) -> tuple[list[_Round], list[Tick]]:
    """each round's idle lateness, the ticks during a copy of ``records`` (sent as Arrow), and the idle lateness
    after; with the host's idle loop ticking beside the whole run.

    :param during_copy: called on the loop as each copy starts (a red case plants its blocking here)
    """
    rounds: list[_Round] = []
    with HostTicker() as host:
        after = latest(await ticks_while(lambda: asyncio.sleep(_IDLE_SECONDS)))
        for _ in range(_ROUNDS):
            hub = _Hub({"success": True, "tx_id": _TX_ID, COPY_ARROW_FIELD: True})
            proxy = hub.proxy()

            async def copy(proxy: Any = proxy) -> None:
                if during_copy is not None:
                    asyncio.get_running_loop().call_soon(during_copy)
                async with proxy.acquire() as conn, conn.transaction():
                    await conn.copy_records_to_table("rows", records=records, columns=[f"c{n}" for n in range(42)])

            before = after
            during = await ticks_while(copy)
            after = latest(await ticks_while(lambda: asyncio.sleep(_IDLE_SECONDS)))
            assert split_copy_request(hub.copy()) is not None
            rounds.append(_Round(before, during, after))
    return rounds, host.ticks


#: the most the host's own idle loop may be late over a run of copies for the run to judge the code: the least a copy
#: may hold the loop (:func:`_held_too_long`'s turn, before any idle jitter)
_HOST_MAY_BE_LATE = 0.005


async def _trusted_rounds(
    records: list[tuple[Any, ...]], *, during_copy: Any = None
) -> tuple[list[_Round], list[Tick]]:
    """:func:`_copy_rounds`, retaken while the host's own idle loop was later over the copies than a copy may hold
    the loop (:func:`~threetears.core.testing.loop_lateness.trusted`: such a run says nothing about the code)."""

    def window(measured: tuple[list[_Round], list[Tick]]) -> tuple[list[Tick], list[Tick]]:
        rounds, host = measured
        return [rounds[0].during[0], rounds[-1].during[-1]], host

    return await trusted(
        lambda: _copy_rounds(records, during_copy=during_copy), window, host_may_be_late=_HOST_MAY_BE_LATE
    )


def test_a_load_spike_at_another_moment_hides_no_stall() -> None:
    """the host loop 30 ms late before the copy credits none of a 20 ms stall after it: the round is held too long."""
    host = [(0.0, 0.031), *((0.031 + n * 0.001, 0.032 + n * 0.001) for n in range(100))]
    stall = [(0.050, 0.071)]
    assert added_lateness(stall, host) == pytest.approx(0.020, abs=1e-9)
    assert _held_too_long([_Round(0.0005, stall, 0.0005)], host) is not None


@pytest.mark.loop_timing
class TestTheCallersLoopRunsWhileACopyIsBuilt:
    async def test_a_12_mb_copy_holds_the_callers_loop_no_longer_than_a_ticker_idles(self) -> None:
        """the build reads the caller's rows (Python objects) on a thread and takes turns with the loop: each 12 MB
        copy holds the loop no longer than its own idle jitter and a turn, the idle measured either side of it, past
        what the host's own idle loop was late at the same moments."""
        rounds, host = await _trusted_rounds(_wide_records(16_000))
        problem = _held_too_long(rounds, host)
        assert problem is None, problem

    async def test_the_host_baseline_ticks_through_every_copy(self) -> None:
        """the host loop's ticks span each copy with no gap past its own lateness: the baseline is taken at the
        copy's moments, never minutes away."""
        rounds, host = await _trusted_rounds(_wide_records(16_000))
        for measured in rounds:
            assert host[0][0] <= measured.during[0][0] and measured.during[-1][1] <= host[-1][1]
        assert all(later[0] - earlier[1] < 0.001 for earlier, later in itertools.pairwise(host))

    async def test_a_build_holding_the_lock_without_turns_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the measurement's red case: a build stretch holding the interpreter lock with no turn (one C call) fails
        the same judgement."""
        original = arrow_copy.rows_digest
        held: list[bool] = []

        def holding(ipc: Any) -> str:
            held.append(True)
            # one C call: no turn, and no switch of the lock until it returns
            sum(range(4_000_000))
            return original(ipc)

        monkeypatch.setattr(arrow_copy, "rows_digest", holding)
        rounds, host = await _trusted_rounds(_wide_records(16_000))
        # every copy of every measurement taken (a retake runs its rounds again) went through the holding build
        assert len(held) in {_ROUNDS * taken for taken in range(1, MEASURE_ATTEMPTS + 1)}, len(held)
        assert _held_too_long(rounds, host) is not None, rounds

    async def test_blocking_planted_on_the_loop_is_caught(self) -> None:
        """the second red case: 30 ms of blocking on the caller's loop itself during each copy fails the judgement;
        the host loop, in its own process, is not late for it, so it credits none of it."""
        rounds, host = await _trusted_rounds(_wide_records(16_000), during_copy=lambda: hold_the_loop(0.03))
        assert _held_too_long(rounds, host) is not None, rounds
