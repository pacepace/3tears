"""``retry_until_done``: the one "keep going until it is done" engine, with capped doubling backoff.

Three owners grew their own copy of this loop -- the NATS client's restore after a reconnect, the
registry's catalog restore, and their startup paths -- each with the same 0.5s first pause and 30s
cap. ``retry_with_backoff`` could not serve them: it gives up and never raises. This is the shared
form; each caller's attempt logs its own failure, because only the caller can name what is missing.
"""

from __future__ import annotations

import pytest

from threetears.observe.resilience import retry_until_done


class _Attempts:
    """an attempt that reports not-done a scripted number of times, then done."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.failures


@pytest.mark.asyncio
async def test_it_runs_until_an_attempt_reports_done_and_returns_how_many_ran() -> None:
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    attempts = _Attempts(failures=3)
    ran = await retry_until_done(attempts, first_delay=0.5, max_delay=30.0, sleep=_sleep)

    assert ran == 4
    assert attempts.calls == 4
    assert slept == [0.5, 1.0, 2.0], "the pause doubles after each attempt that was not done"


@pytest.mark.asyncio
async def test_the_pause_is_capped() -> None:
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    await retry_until_done(_Attempts(failures=6), first_delay=1.0, max_delay=4.0, sleep=_sleep)

    assert slept == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]


@pytest.mark.asyncio
async def test_an_attempt_done_at_once_never_sleeps() -> None:
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    assert await retry_until_done(_Attempts(failures=0), first_delay=0.5, max_delay=30.0, sleep=_sleep) == 1
    assert slept == []


@pytest.mark.asyncio
async def test_an_attempt_that_raises_propagates() -> None:
    """the attempt owns its failure handling; an exception it lets out is not swallowed here."""

    async def _boom() -> bool:
        raise RuntimeError("attempt failed in a way its owner did not handle")

    with pytest.raises(RuntimeError, match="did not handle"):
        await retry_until_done(_boom, first_delay=0.5, max_delay=30.0)


@pytest.mark.parametrize(("first_delay", "max_delay"), [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0)])
@pytest.mark.asyncio
async def test_a_schedule_that_cannot_back_off_is_refused(first_delay: float, max_delay: float) -> None:
    with pytest.raises(ValueError):
        await retry_until_done(_Attempts(failures=0), first_delay=first_delay, max_delay=max_delay)
