"""An emptying of a ring slot bigger than one batch runs bounded statements and resumes after a timeout.

Replay 7 (2026-11-25, local YugabyteDB): the keeper's first band of a slot holding far more notes than the count that
sized it ran past the connection's 5 s statement timeout, was rolled back, and the keeper sent the SAME statement every
pass (about every 6 s, 80+ timeouts) -- never finishing. These hold the fix (``_delete_band``, ``_delete_past``,
``_resized``, ``_timed_out``): a statement the database cancels keeps the walk where it was and the next one is smaller,
never the same; a finished band is never deleted again; and a slow statement shrinks the next.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, cast

from asyncpg.exceptions import QueryCanceledError

import threetears.core.collections.change_note_keeper as change_note_keeper

#: the hash space a YugabyteDB key's hash spans
_CODES = 65536

#: notes a hash code holds in the slot: 6.5 M notes, far past the count that sizes a walk's first band (64,000)
_PER_CODE = 100

#: the most notes a statement deletes before the stand-in database cancels it (its statement timeout)
_TIMEOUT_NOTES = 5000

#: a run of hash codes holding more notes each (keys are never perfectly even): a band sized from the codes before
#: it runs into the timeout there, so the walk meets a cancelled statement mid-way as well as at its start
_HOT = range(30_000, 30_200)
_HOT_PER_CODE = 3000

_YUGABYTE = "PostgreSQL 11.2-YB-2026.1.0.0-b0"
_POSTGRES = "PostgreSQL 16.4"


class _YugabyteSlot:
    """a YugabyteDB pool over one idle slot: its notes by hash code, a band over ``_TIMEOUT_NOTES`` cancelled; the first
    band from ``slow_from`` on takes ``slow_seconds`` (a slow tablet)."""

    def __init__(self, *, slow_from: int | None = None, slow_seconds: float = 0.0) -> None:
        self.slow_from = slow_from
        self.slow_seconds = slow_seconds
        #: where in ``bands`` the slow one is
        self.slowed: int | None = None
        self.held = [_HOT_PER_CODE if code in _HOT else _PER_CODE for code in range(_CODES)]
        self.notes = sum(self.held)
        #: each band delete sent: (start, end, notes deleted or None when cancelled)
        self.bands: list[tuple[int, int, int | None]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if query.startswith("SELECT version()"):
            return [{"v": _YUGABYTE}]
        if query.startswith("SELECT count(*) AS notes"):
            return [{"notes": min(sum(self.held), int(args[0]))}]
        if query.startswith("SELECT 1 FROM"):
            return [{"?column?": 1}] if any(self.held) else []
        raise AssertionError(query)

    async def execute(self, query: str, *args: Any) -> str:
        assert query.startswith("DELETE FROM") and "yb_hash_code" in query, query
        start, end = int(args[0]), int(args[1])
        notes = sum(self.held[start:end])
        if notes > _TIMEOUT_NOTES:
            self.bands.append((start, end, None))
            raise QueryCanceledError("canceling statement due to statement timeout")
        if self.slow_from is not None and self.slowed is None and start >= self.slow_from:
            self.slowed = len(self.bands)
            await asyncio.sleep(self.slow_seconds)
        self.held[start:end] = [0] * (end - start)
        self.bands.append((start, end, notes))
        return f"DELETE {notes}"

    def acquire(self) -> Any:
        raise AssertionError("an emptying shares no session")


class _PostgresSlot:
    """a Postgres pool over one idle slot: its notes' ids in order, a batch over ``_TIMEOUT_NOTES`` cancelled."""

    def __init__(self, notes: int) -> None:
        self.ids = list(range(1, notes + 1))
        #: each batch delete sent: (its LIMIT, the id it started past, notes deleted or None when cancelled)
        self.batches: list[tuple[int, int | None, int | None]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if query.startswith("SELECT version()"):
            return [{"v": _POSTGRES}]
        if query.startswith("SELECT 1 FROM"):
            return [{"?column?": 1}] if self.ids else []
        assert query.startswith("DELETE FROM") and "RETURNING id" in query, query
        limit, past = int(args[0]), args[1]
        if limit > _TIMEOUT_NOTES:
            self.batches.append((limit, past, None))
            raise QueryCanceledError("canceling statement due to statement timeout")
        taken = [note for note in self.ids if past is None or note > past][:limit]
        gone = set(taken)
        self.ids = [note for note in self.ids if note not in gone]
        self.batches.append((limit, past, len(taken)))
        return [{"id": note} for note in taken]

    async def execute(self, query: str, *args: Any) -> str:
        assert query.startswith("VACUUM"), query
        return "VACUUM"

    def acquire(self) -> Any:
        raise AssertionError("an emptying shares no session")


def _keeper(pool: Any, batch: int) -> change_note_keeper.ChangeNoteKeeper:
    return change_note_keeper.ChangeNoteKeeper(
        pool,
        notes_schema="ring",
        log=cast(Any, object()),
        empty_batch=batch,
        empty_pause=timedelta(0),
    )


async def _passes(keeper: change_note_keeper.ChangeNoteKeeper, most: int) -> bool:
    """run the keeper's emptying as its passes do, a failed pass contained and the next run; whether it finished."""
    for _ in range(most):
        try:
            done, _deleted = await keeper.empty_slot(1)
        except QueryCanceledError:
            continue  # run_change_note_keeper contains a failed pass and runs the next
        if done:
            return True
    return False


async def test_a_yugabyte_emptying_past_a_batch_resumes_after_a_timeout_with_a_smaller_band() -> None:
    """Flips if a band the database cancelled is sent again unchanged (replay 7's loop), if the walk goes back over a
    band it finished, or if the slot is not emptied: every band after a cancelled one starts where it did and is
    smaller, the finished bands tile the hash space once, and every note is gone."""
    slot = _YugabyteSlot()
    assert await _passes(_keeper(slot, 1000), most=2000), slot.bands[-5:]
    assert not any(slot.held)
    for (start, end, notes), (after_start, after_end, _) in zip(slot.bands, slot.bands[1:], strict=False):
        if notes is None:
            assert after_start == start and after_end - after_start < end - start, (start, end, after_start, after_end)
    finished = [(start, end) for start, end, notes in slot.bands if notes is not None]
    assert finished[0][0] == 0 and finished[-1][1] == _CODES
    assert all(end == next_start for (_, end), (next_start, _) in zip(finished, finished[1:], strict=False)), finished
    assert sum(notes for _, _, notes in slot.bands if notes is not None) == slot.notes
    cancelled = [band for band in slot.bands if band[2] is None]
    assert cancelled and any(start >= _HOT.start for start, _, _ in cancelled), cancelled


async def test_a_postgres_emptying_resumes_after_a_timeout_with_a_smaller_batch() -> None:
    """Flips if a Postgres batch the database cancelled is sent again unchanged, or the walk restarts from the table's
    start after it: the next batch starts past the same id with a smaller LIMIT, and every note is deleted once."""
    slot = _PostgresSlot(notes=40_000)
    keeper = _keeper(slot, 8000)
    assert await _passes(keeper, most=200), slot.batches[-5:]
    assert slot.ids == []
    for (limit, past, notes), (after_limit, after_past, _) in zip(slot.batches, slot.batches[1:], strict=False):
        if notes is None:
            assert after_past == past and after_limit < limit, (limit, past, after_limit, after_past)
    assert sum(notes for _, _, notes in slot.batches if notes is not None) == 40_000
    assert any(notes is None for _, _, notes in slot.batches)


async def test_a_slow_yugabyte_band_makes_the_next_one_smaller() -> None:
    """Flips if the next band ignores how long the last took (``_resized``): a band that took 1 s (twice
    ``_DELETE_SECONDS``) with a batch's notes is followed by a narrower one, where its notes alone would keep the band
    as it was, so a slow tablet's bands never grow into the statement timeout."""
    slot = _YugabyteSlot(slow_from=_CODES // 2, slow_seconds=1.0)
    assert await _passes(_keeper(slot, 1000), most=2000), slot.bands[-5:]
    assert slot.slowed is not None
    (start, end, _), (_, after_end, _) = slot.bands[slot.slowed : slot.slowed + 2]
    before = [end - start for start, end, _ in slot.bands[slot.slowed - 3 : slot.slowed]]
    assert all(width == end - start for width in before), (before, end - start)
    assert after_end - end < end - start, slot.bands[slot.slowed - 3 : slot.slowed + 2]
