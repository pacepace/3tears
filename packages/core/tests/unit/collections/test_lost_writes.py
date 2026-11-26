"""a lost advance's backstop: a dead writer's window announced from its beat, a migration run's from the runner.

What these hold:

- a writer's beat carries its ledger's watermark; a dead writer's window runs from it (less the clocks' skew) to now
  (its last beat plus the grace at most), plus the skew;
- the recovery holder announces a stale writer's window from the change notes of exactly that window and deletes its
  key; a live writer's key is left; a window that fails is kept for the next pass; a replica that finds the lease
  held does nothing;
- a process stopping with its lost advances landed deletes its key; one with some left beats instead.

The migration runner's window: ``tests/unit/data/migrations/test_runner_announces.py``.

Against a real database: ``tests/integration/test_lost_writes_live.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import REGISTRY

from packages.core.tests.unit.collections.epoch_range_db import RESULTS, RUNS, STAMPED, VERSIONED, RangeDb
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.lost_writes import WriteBackstop, WriterBeats
from threetears.core.coordination.lease import KVLease
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.utils.write_ledger import CLOCK_SKEW, WRITE_GRACE, WriteLedger

_TABLE = "lost_writes_rows"
_START = 1_800_000_000.0

#: the stale time the backstops here are built with
_STALE = 30.0


class _Clock:
    def __init__(self, at: float = _START) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def _at(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


class _Pool(RangeDb):
    """the database: one stamped table in the platform schema, holding one row ``(CA, x)`` written as the dead
    writer's window began; each window read is the epoch range its bucket reads bind."""

    def __init__(self) -> None:
        super().__init__()
        self.declared = {STAMPED: RUNS}
        self.stamped("CA", "x", 3, int(_START * 1000) << 15)

    @property
    def windows(self) -> list[tuple[int, int]]:
        """each window read, as its floor and ceiling: one a pass, however many buckets it reads."""
        bounds = {
            tuple(arg for arg in args if isinstance(arg, int) and arg > 1 << 40) for _, args in self.data_reads(STAMPED)
        }
        return sorted(bounds)  # type: ignore[arg-type]


class _Source:
    """a recording source; fails while ``failing`` is set."""

    def __init__(self) -> None:
        self.records: list[tuple[str, list[list[str]] | None, bool]] = []
        self.failing = False

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise AssertionError("a recording source is advanced with its change")

    async def advance_with_change(
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        if self.failing:
            raise GenerationUnavailableError("the epoch bucket is away")
        self.records.append((table_name, None if rows is None else [row.ids for row in rows], keys_unknown))
        return f"inc:{len(self.records)}"


async def _beaten(bucket: Any, key: str) -> bool:
    """whether ``key`` holds a beat within a second: the first beat runs in the beat loop, not in start()."""
    for _ in range(100):
        if await bucket.get(key=key) is not None:
            return True
        await asyncio.sleep(0.01)
    return False


def _backstop(
    nats: FakeNatsClient,
    process: str,
    clock: _Clock,
    source: _Source,
    pool: _Pool,
    ledger: WriteLedger,
    *,
    recover: bool = True,
) -> WriteBackstop:
    return WriteBackstop(
        pool,  # type: ignore[arg-type]
        generation_source=source,
        notes_schema="platform",
        lease=KVLease(nats, pod_id=process),
        process_id=process,
        ledger=ledger,
        recover=recover,
        stale_seconds=_STALE,
        clock=clock,
    )


async def test_a_dead_writer_s_window_is_announced_from_its_rows_and_its_key_deleted() -> None:
    """Flips if a writer that died after committing, before its advance, is never announced, or is announced from
    another window than the one its beat names, or its key outlives the recovery (announced again every pass)."""
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    writer_clock, holder_clock = _Clock(), _Clock()
    writer_ledger = WriteLedger(clock=writer_clock)
    writer = _backstop(nats, "writer", writer_clock, source, pool, writer_ledger)
    holder = _backstop(nats, "holder", holder_clock, source, pool, WriteLedger(clock=holder_clock))

    writer_ledger.begin()  # a write open when the writer beat, then it died
    writer_clock.at += 10
    beat = await writer.beat()
    assert beat.watermark == _at(_START)

    holder_clock.at = writer_clock.at + _STALE - 1
    await holder.beat()
    early = await holder.recover_pass()
    assert early.live == 2 and early.recovered == () and source.records == []

    holder_clock.at = writer_clock.at + _STALE + 5
    done = await holder.recover_pass()
    assert done.recovered == ("writer",) and done.live == 1
    assert source.records == [(STAMPED, [["CA", "x"]], False)]
    (window,) = pool.windows
    since, until = _at(_START) - CLOCK_SKEW, _at(holder_clock.at) + CLOCK_SKEW
    assert window == (int(since.timestamp() * 1000) << 15, ((int(until.timestamp() * 1000) + 1) << 15) - 1)
    assert until < beat.beat_at + WRITE_GRACE + CLOCK_SKEW + timedelta(seconds=1)
    bucket = await KVLease(nats, pod_id="reader").bucket()
    assert await bucket.get(key=writer.key) is None
    assert await bucket.get(key=holder.key) is not None
    assert (await holder.recover_pass()).recovered == ()


async def test_a_window_that_cannot_be_announced_keeps_the_key_for_the_next_pass() -> None:
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    writer = _backstop(nats, "writer", _Clock(), source, pool, WriteLedger(clock=clock))
    holder = _backstop(nats, "holder", clock, source, pool, WriteLedger(clock=clock))
    await writer.beat()
    clock.at += _STALE + 1
    source.failing = True
    failed = await holder.recover_pass()
    assert failed.failed == ("writer",) and failed.recovered == ()
    source.failing = False
    assert (await holder.recover_pass()).recovered == ("writer",)


async def test_a_replica_that_finds_the_recovery_lease_held_does_nothing() -> None:
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    held = await KVLease(nats, pod_id="other").acquire("write-ledger-recovery", ttl_seconds=15, max_wait_seconds=0)
    try:
        skipped = await _backstop(nats, "holder", clock, source, pool, WriteLedger(clock=clock)).recover_pass()
    finally:
        await held.release()
    assert skipped.skipped


async def test_stopping_with_every_repair_landed_deletes_the_key_and_with_one_left_beats() -> None:
    """Flips if a clean stop leaves its key (read as a dead writer: its window announced for nothing), or a stop with a
    lost advance still held deletes it (the advance would never be announced)."""
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    backstop = _backstop(nats, "writer", clock, source, pool, ledger, recover=False)
    bucket = await KVLease(nats, pod_id="reader").bucket()
    await backstop.start()
    assert await _beaten(bucket, backstop.key)
    assert await backstop.stop(drain_seconds=0) == 0
    assert await bucket.get(key=backstop.key) is None

    async def never() -> None:
        raise GenerationUnavailableError("away")

    await backstop.start()
    ledger.lost_advance(_TABLE, never)
    assert await backstop.stop(drain_seconds=0) == 1
    raw = await bucket.get(key=backstop.key)
    assert raw is not None and datetime.fromisoformat(json.loads(raw)["watermark"]) == _at(_START)


async def test_start_never_waits_on_the_leases_bucket_and_beats_once_it_is_declared() -> None:
    """Flips if start() waits on (or fails for) a leases bucket the hub has not declared yet: a service the hub's own
    start waits on (identity-core) then never starts (2026-11-22 replay), or if no beat lands once it is declared."""
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    backstop = WriteBackstop(
        pool,  # type: ignore[arg-type]
        generation_source=source,
        notes_schema="platform",
        lease=KVLease(nats, pod_id="writer", create_if_missing=False),
        process_id="writer",
        ledger=WriteLedger(clock=clock),
        recover=False,
        beat_seconds=0.02,
        stale_seconds=_STALE,
        clock=clock,
    )
    await asyncio.wait_for(backstop.start(), timeout=5)
    try:
        # the hub declares it after
        bucket = await KVLease(nats, pod_id="hub").bucket()
        assert await _beaten(bucket, backstop.key)
    finally:
        await backstop.stop(drain_seconds=0)


async def test_a_retry_that_hangs_never_holds_a_stop_past_its_drain() -> None:
    """Flips if a lost advance whose retry never answers (an epoch bucket that hangs) holds stop() past drain_seconds:
    the retry is bounded, stays held, and the stop beats once more so recovery announces it."""
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    backstop = _backstop(nats, "writer", clock, source, pool, ledger, recover=False)
    bucket = await KVLease(nats, pod_id="reader").bucket()
    await backstop.start()

    async def hangs() -> None:
        await asyncio.Event().wait()

    ledger.lost_advance(_TABLE, hangs)
    started = asyncio.get_running_loop().time()
    left = await asyncio.wait_for(backstop.stop(drain_seconds=0.3), timeout=5)
    assert left == 1
    assert asyncio.get_running_loop().time() - started < 2
    assert await bucket.get(key=backstop.key) is not None


async def test_writer_beats_tell_a_writer_gone_only_once_its_key_is_deleted_never_by_its_beat_s_age() -> None:
    """Flips if a writer is read as gone while its beat key is kept (however old: a hub still loading, its loop slow),
    or never once the key is deleted (a clean stop, or the recovery's delete of a dead writer's key), or a key that is
    no beat's is asked after."""
    nats, pool, source = FakeNatsClient(), _Pool(), _Source()
    clock = _Clock()
    backstop = _backstop(nats, "hub-1", clock, source, pool, WriteLedger(clock=clock), recover=False)
    beats = WriterBeats(KVLease(nats, pod_id="pod", create_if_missing=False))
    await backstop.beat()
    clock.at += _STALE * 100
    assert await beats.gone(backstop.key) is False, "an old beat is still a kept beat: the recovery judges age"
    assert await backstop.stop(drain_seconds=0) == 0
    assert await beats.gone(backstop.key) is True
    with pytest.raises(ValueError, match="beat"):
        await beats.gone("leases.someone-else")


def _lost_writes_reloads(table: str) -> float:
    """``collection_whole_reloads_total{why="lost_writes"}`` of ``table``."""
    return REGISTRY.get_sample_value("collection_whole_reloads_total", {"table": table, "why": "lost_writes"}) or 0.0


class _RunDeletedPool(_Pool):
    """the database: the versioned table alone declared, one of its runs deleted (its run marker) as the dead
    writer's window began: a whole change of the window, recorded again by its prefixes."""

    def __init__(self) -> None:
        super().__init__()
        self.declared = {VERSIONED: RESULTS}
        self.run_marker("CA", "run9", int(_START * 1000) << 15)


class _WholeSource(_Source):
    """a recording source that records a whole change's tenant and prefixes too."""

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        prefixes: Sequence[Any] | None = None,
        reason: str | None = None,
    ) -> str:
        return await super().advance_with_change(table_name, rows, keys_unknown=keys_unknown)


async def test_a_dead_writer_s_whole_change_recorded_again_is_counted_as_a_lost_writes_reload_and_said_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flips if the backstop records a dead writer's whole change again (every copy then reloads the table whole)
    without counting it under ``collection_whole_reloads_total{why="lost_writes"}`` with its table, or without saying
    so at WARNING (Pace, 2026-11-27: a recovery reload is never silent, and a replay fails on one); or if a migration
    run's own whole change, announced from its window, is counted as one."""
    nats, pool, source = FakeNatsClient(), _RunDeletedPool(), _WholeSource()
    clock = _Clock()
    writer = _backstop(nats, "writer", _Clock(), source, pool, WriteLedger(clock=clock))
    holder = _backstop(nats, "holder", clock, source, pool, WriteLedger(clock=clock))

    migrated = await holder.announce_window(_at(_START - 1), _at(_START + 1))
    assert migrated.advanced == (VERSIONED,)
    before = _lost_writes_reloads(VERSIONED)
    assert before == 0.0, "a migration's own window is no lost write"

    await writer.beat()
    clock.at += _STALE + 1
    with caplog.at_level("WARNING"):
        assert (await holder.recover_pass()).recovered == ("writer",)

    assert _lost_writes_reloads(VERSIONED) == before + 1
    said = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and getattr(record, "extra_data", {}).get("why") == "lost_writes"
    ]
    assert said and said[0].extra_data["table"] == VERSIONED  # type: ignore[attr-defined]
