"""the writer ledger: the watermark of a process's writes, held by an open write, a write in its grace, or a lost advance.

What these hold:

- an empty ledger's watermark is now; an open write holds it at the write's start until it ends, and an ended write
  for :data:`WRITE_GRACE` after;
- a lost advance holds it from the watermark when it failed until a retry lands; a retry that fails keeps it held;
- ``advance_recording`` hands an advance that failed after its commit to the process's ledger, as it was, and never one
  whose commit did not land or that was refused before anything was advanced.

The connection class is proven against a real database: ``tests/integration/test_lost_writes_live.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from threetears.core.collections.epoch_change import ChangedRow, advance_recording
from threetears.core.exceptions import GenerationNotCommittedError, GenerationUnavailableError
from threetears.core.utils.write_ledger import WRITE_GRACE, WriteLedger, process_write_ledger


class _Clock:
    """a wall clock the test moves."""

    def __init__(self, at: float = 1_800_000_000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def _when(clock: _Clock, seconds: float = 0.0) -> datetime:
    return datetime.fromtimestamp(clock.at + seconds, tz=UTC)


def test_an_empty_ledger_s_watermark_is_now() -> None:
    clock = _Clock()
    assert WriteLedger(clock=clock).watermark() == _when(clock)


def test_an_open_write_holds_the_watermark_at_its_start_until_it_ends_and_its_grace_passes() -> None:
    """Flips if a write's start stops holding the watermark while it is open, or before its grace after it ends: a
    dead writer's recovery would start after that write's notes and miss them."""
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    started = _when(clock)
    token = ledger.begin()
    clock.at += 300
    assert ledger.watermark() == started
    ledger.end(token)
    clock.at += WRITE_GRACE.total_seconds() - 1
    assert ledger.watermark() == started
    clock.at += 2
    assert ledger.watermark() == _when(clock)


def test_the_oldest_of_several_writes_holds_it() -> None:
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    first = ledger.begin()
    oldest = _when(clock)
    clock.at += 5
    second = ledger.begin()
    ledger.end(first)
    clock.at += 5
    assert ledger.watermark() == oldest
    ledger.end(second)
    assert ledger.open_writes == 0
    assert ledger.watermark() == oldest


def test_ending_a_write_twice_or_one_never_begun_changes_nothing() -> None:
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    token = ledger.begin()
    ledger.end(token)
    ledger.end(token)
    ledger.end(12345)
    clock.at += WRITE_GRACE.total_seconds() + 2
    assert ledger.watermark() == _when(clock)


async def test_a_lost_advance_holds_the_watermark_from_its_failure_until_a_retry_lands() -> None:
    """Flips if a lost advance stops holding the watermark before its retry lands (a writer dying meanwhile would be
    recovered from too late a window), or a failed retry releases it."""
    clock = _Clock()
    ledger = WriteLedger(clock=clock)
    token = ledger.begin()
    write_start = _when(clock)
    clock.at += 1
    ledger.end(token)
    attempts: list[int] = []

    async def retry() -> None:
        attempts.append(1)
        if len(attempts) < 2:
            raise GenerationUnavailableError("the bucket is still away")

    ledger.lost_advance("rows", retry, error="first failure")
    clock.at += WRITE_GRACE.total_seconds() * 3
    assert ledger.watermark() == write_start
    assert await ledger.repair_pending() == 1
    assert ledger.watermark() == write_start
    (held,) = ledger.lost
    assert held.attempts == 1 and "still away" in held.error
    assert await ledger.repair_pending() == 0
    assert ledger.watermark() == _when(clock)
    assert len(attempts) == 2


def test_a_grace_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError):
        WriteLedger(grace=timedelta(0))


class _Failing:
    """a recording source whose advances fail ``failures`` times, then land."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.landed: list[tuple[str, list[str] | None]] = []

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise AssertionError("a recording source is advanced with its change")

    async def advance_with_change(
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str:
        if self.failures:
            self.failures -= 1
            raise self.error
        self.landed.append((table_name, None if rows is None else [row.ids[0] for row in rows]))
        return f"inc:{len(self.landed)}"


@pytest.fixture
async def ledger() -> AsyncIterator[WriteLedger]:
    """the process's ledger, emptied of the lost advances a test left (each made to land)."""
    kept = process_write_ledger()
    yield kept
    for lost in kept.lost:
        lost.retry = _landed
    await kept.repair_pending()


async def _landed() -> None:
    return None


async def test_an_advance_that_fails_after_its_commit_is_handed_to_the_ledger_as_it_was(ledger: WriteLedger) -> None:
    """Flips if a failed advance is dropped (its rows reach no follower until the table moves again), or its retry
    names other rows than the write's."""
    source = _Failing(1, GenerationUnavailableError("the epoch bucket timed out"))
    before = len(ledger.lost)
    with pytest.raises(GenerationUnavailableError):
        await advance_recording(source, "lost_rows", [ChangedRow(ids=["a"]), ChangedRow(ids=["b"])])
    assert len(ledger.lost) == before + 1
    held = ledger.lost[-1]
    assert held.table == "lost_rows" and "timed out" in held.error
    await held.retry()
    assert source.landed == [("lost_rows", ["a", "b"])]
    await ledger.repair_pending()
    assert held not in ledger.lost


async def test_a_commit_that_did_not_land_or_a_refused_change_is_not_handed_over(ledger: WriteLedger) -> None:
    before = len(ledger.lost)
    with pytest.raises(GenerationNotCommittedError):
        await advance_recording(_Failing(1, GenerationNotCommittedError("rolled back")), "lost_rows", None)
    with pytest.raises(ValueError):
        await advance_recording(_Failing(0, RuntimeError()), "lost_rows", [ChangedRow(ids=["a"])], keys_unknown=True)
    assert len(ledger.lost) == before


async def test_the_retry_itself_is_never_handed_over_again(ledger: WriteLedger) -> None:
    before = len(ledger.lost)
    with pytest.raises(GenerationUnavailableError):
        await advance_recording(
            _Failing(1, GenerationUnavailableError("away")), "lost_rows", None, keys_unknown=True, repair=False
        )
    assert len(ledger.lost) == before
