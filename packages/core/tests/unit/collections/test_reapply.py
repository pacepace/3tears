"""
reapply_on_lost_race: the bounded re-read-and-reapply loop around a fenced save.

a fenced save (``TableSchema(cas_column=...)``) refuses a writer whose read another save has
overtaken with ``ConcurrentModificationError``. a change that is a pure function of the row it
lands on is right to re-read the winner's row and apply itself again; this is the one loop that
does it, bounded, with full-jitter backoff between attempts. the loop is driven here with a
recorded pause in place of a sleep, so every assertion is about which attempts ran, never about
timing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from threetears.core.collections import REAPPLY_BACKOFF_SECONDS, REAPPLY_MAX_ATTEMPTS, reapply_on_lost_race
from threetears.core.exceptions import ConcurrentModificationError


class RecordedPauses:
    """
    records every pause the loop asks for instead of sleeping.

    :ivar seconds: each pause requested, in order
    """

    def __init__(self) -> None:
        """
        start with no pauses recorded.

        :return: none
        :rtype: None
        """
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """
        record one pause.

        :param seconds: pause requested
        :ptype seconds: float
        :return: none
        :rtype: None
        """
        self.seconds.append(seconds)


def refusal() -> ConcurrentModificationError:
    """
    build the refusal a fenced save raises when its read was overtaken.

    :return: refusal for one lost race
    :rtype: ConcurrentModificationError
    """
    return ConcurrentModificationError("rows", uuid.uuid4(), datetime(2026, 1, 1, tzinfo=UTC))


async def test_an_attempt_that_lands_runs_once_and_never_pauses() -> None:
    """the first attempt's save lands: its value is returned and nothing is retried."""
    pauses = RecordedPauses()
    runs: list[int] = []

    async def attempt() -> str:
        runs.append(1)
        return "landed"

    result = await reapply_on_lost_race(attempt, what="row update", pause=pauses)

    assert result == "landed"
    assert runs == [1]
    assert pauses.seconds == []


async def test_a_lost_race_reruns_the_whole_attempt_until_it_lands() -> None:
    """each refusal re-runs the attempt from its read, with one bounded pause between runs."""
    pauses = RecordedPauses()
    runs: list[int] = []

    async def attempt() -> int:
        runs.append(len(runs))
        if len(runs) <= 2:
            raise refusal()
        return len(runs)

    result = await reapply_on_lost_race(attempt, what="row update", pause=pauses)

    assert result == 3
    assert runs == [0, 1, 2]
    assert len(pauses.seconds) == 2
    assert all(0 <= seconds <= REAPPLY_BACKOFF_SECONDS for seconds in pauses.seconds)


async def test_an_attempt_that_returns_none_is_a_landed_none_not_a_lost_race() -> None:
    """a save that lands with nothing to return is still landed: it runs once and returns none."""
    pauses = RecordedPauses()
    runs: list[int] = []

    async def attempt() -> None:
        runs.append(1)

    result = await reapply_on_lost_race(attempt, what="row update", pause=pauses)

    assert result is None
    assert runs == [1]
    assert pauses.seconds == []


async def test_a_row_that_keeps_losing_raises_the_last_refusal_after_the_budget() -> None:
    """bounded: after the budget the last refusal propagates, never a silent give-up."""
    pauses = RecordedPauses()
    refusals: list[ConcurrentModificationError] = []

    async def attempt() -> None:
        lost = refusal()
        refusals.append(lost)
        raise lost

    with pytest.raises(ConcurrentModificationError) as raised:
        await reapply_on_lost_race(attempt, what="row update", pause=pauses)

    assert len(refusals) == REAPPLY_MAX_ATTEMPTS
    assert raised.value is refusals[-1]
    # no pause after the last attempt: there is nothing left to wait for
    assert len(pauses.seconds) == REAPPLY_MAX_ATTEMPTS - 1


async def test_a_smaller_budget_is_honoured() -> None:
    """a caller's own budget bounds the attempts, first included."""
    pauses = RecordedPauses()
    runs: list[int] = []

    async def attempt() -> None:
        runs.append(1)
        raise refusal()

    with pytest.raises(ConcurrentModificationError):
        await reapply_on_lost_race(attempt, what="row update", max_attempts=3, pause=pauses)

    assert len(runs) == 3
    assert len(pauses.seconds) == 2


async def test_any_other_failure_propagates_at_once() -> None:
    """only a lost race is retried; anything else is the caller's to answer."""
    pauses = RecordedPauses()
    runs: list[int] = []

    async def attempt() -> None:
        runs.append(1)
        raise KeyError("row not found")

    with pytest.raises(KeyError, match="row not found"):
        await reapply_on_lost_race(attempt, what="row update", pause=pauses)

    assert runs == [1]
    assert pauses.seconds == []


async def test_a_budget_below_one_is_refused_before_anything_runs() -> None:
    """a budget that would never run the attempt is a caller error, not a silent no-op."""
    runs: list[int] = []

    async def attempt() -> None:
        runs.append(1)

    with pytest.raises(ValueError, match="max_attempts"):
        await reapply_on_lost_race(attempt, what="row update", max_attempts=0)

    assert runs == []
