"""``retry_bounded``: the one owner of a BOUNDED retry that classifies each failure.

Three loops hand-rolled it -- the KV bind waiting for a bucket's declarer (deadline-bounded,
retrying only an absence), the collections-bucket bind (attempt-bounded, an absence retried at
once because the bind already waited, anything else after a pause), and a tool pod's NATS connect
(deadline-bounded). Each carried its own ``min(delay * 2, cap)``. This is their shared form: it
retries what ``retry_on`` accepts, raises everything else at once, and when the bound is spent
raises the LAST failure unchanged, so the caller turns it into its own error.
"""

from __future__ import annotations

import pytest

from threetears.observe.resilience import retry_bounded


class _Transient(Exception):
    """a failure worth retrying."""


class _Paced(Exception):
    """a failure whose attempt already waited, so it is retried without a pause."""


class _Fatal(Exception):
    """a failure no retry can clear."""


class _Script:
    """an attempt that raises each scripted failure in turn, then returns ``"done"``."""

    def __init__(self, failures: list[Exception]) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "done"


class _Clock:
    """a monotonic clock that only moves when a pause is taken."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _transient(exc: Exception) -> bool:
    return isinstance(exc, _Transient | _Paced)


@pytest.mark.asyncio
async def test_it_retries_what_it_is_told_to_and_returns_the_value() -> None:
    clock = _Clock()
    script = _Script([_Transient(), _Transient()])

    result = await retry_bounded(
        script, retry_on=_transient, first_delay=1.0, max_delay=30.0, max_attempts=5, sleep=clock.sleep
    )

    assert result == "done"
    assert script.calls == 3
    assert clock.slept == [1.0, 2.0]


@pytest.mark.asyncio
async def test_a_failure_it_is_not_told_to_retry_is_raised_at_once() -> None:
    clock = _Clock()
    script = _Script([_Fatal()])

    with pytest.raises(_Fatal):
        await retry_bounded(
            script, retry_on=_transient, first_delay=1.0, max_delay=30.0, max_attempts=5, sleep=clock.sleep
        )

    assert script.calls == 1
    assert clock.slept == []


@pytest.mark.asyncio
async def test_a_spent_attempt_budget_raises_the_last_failure_unchanged() -> None:
    clock = _Clock()
    last = _Transient("the third")
    script = _Script([_Transient("the first"), _Transient("the second"), last])

    with pytest.raises(_Transient) as raised:
        await retry_bounded(
            script, retry_on=_transient, first_delay=1.0, max_delay=30.0, max_attempts=3, sleep=clock.sleep
        )

    assert raised.value is last
    assert script.calls == 3
    assert clock.slept == [1.0, 2.0], "no pause after the last attempt"


@pytest.mark.asyncio
async def test_a_deadline_bounds_the_retries_and_the_last_pause() -> None:
    clock = _Clock()
    script = _Script([_Transient() for _ in range(10)])

    with pytest.raises(_Transient):
        await retry_bounded(
            script,
            retry_on=_transient,
            first_delay=1.0,
            max_delay=30.0,
            deadline_seconds=5.0,
            sleep=clock.sleep,
            clock=clock.time,
        )

    assert clock.slept == [1.0, 2.0, 2.0], "the last pause is cut to what the deadline leaves"
    assert script.calls == 4


@pytest.mark.asyncio
async def test_the_pause_is_capped() -> None:
    clock = _Clock()
    script = _Script([_Transient() for _ in range(5)])

    await retry_bounded(script, retry_on=_transient, first_delay=1.0, max_delay=3.0, max_attempts=6, sleep=clock.sleep)

    assert clock.slept == [1.0, 2.0, 3.0, 3.0, 3.0]


@pytest.mark.asyncio
async def test_a_failure_that_does_not_back_off_is_retried_at_once_without_advancing_the_pause() -> None:
    clock = _Clock()
    script = _Script([_Paced(), _Transient(), _Paced(), _Transient()])

    await retry_bounded(
        script,
        retry_on=_transient,
        backs_off=lambda exc: not isinstance(exc, _Paced),
        first_delay=1.0,
        max_delay=30.0,
        max_attempts=10,
        sleep=clock.sleep,
    )

    assert clock.slept == [1.0, 2.0]


@pytest.mark.asyncio
async def test_each_retry_is_reported_with_its_attempt_and_pause() -> None:
    clock = _Clock()
    reported: list[tuple[str, int, float]] = []
    script = _Script([_Transient("a"), _Paced("b")])

    await retry_bounded(
        script,
        retry_on=_transient,
        backs_off=lambda exc: not isinstance(exc, _Paced),
        first_delay=1.0,
        max_delay=30.0,
        max_attempts=5,
        on_retry=lambda exc, attempt, pause: reported.append((str(exc), attempt, pause)),
        sleep=clock.sleep,
    )

    assert reported == [("a", 1, 1.0), ("b", 2, 0.0)]


@pytest.mark.asyncio
async def test_a_retry_with_no_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="bound"):
        await retry_bounded(_Script([]), retry_on=_transient, first_delay=1.0, max_delay=30.0)


@pytest.mark.parametrize(("first_delay", "max_delay"), [(0.0, 1.0), (2.0, 1.0)])
@pytest.mark.asyncio
async def test_a_schedule_that_cannot_back_off_is_refused(first_delay: float, max_delay: float) -> None:
    with pytest.raises(ValueError):
        await retry_bounded(
            _Script([]), retry_on=_transient, first_delay=first_delay, max_delay=max_delay, max_attempts=1
        )
