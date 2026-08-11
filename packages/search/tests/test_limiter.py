"""What the in-process limiter promises: D8's pair keying, and pacing that holds unturned.

The behaviour worth pinning is not "a token bucket works" -- that shape is
well understood -- but the rulings this particular bucket exists to carry:

- the key is the ``(provider instance, egress)`` pair, and both halves
  separate (D8, D20, SR-N4);
- refill is computed from a monotonic clock at acquire time, so there is no
  background task and a limiter is inert between calls (SR-L5);
- ``max_wait_seconds`` blocks toward the computed availability moment and
  never past the caller's deadline (SR-G2);
- ``retry_after_seconds`` is honest: waiting exactly that long is enough;
- concurrent acquisitions never grant more than the bucket held;
- the defaults are on, safe, and not read from anywhere (SR-L6, SR-K1).

Time is driven rather than waited on. :class:`_ManualClock` owns both halves
of the seam -- the clock the limiter reads and the sleeper it awaits -- so a
blocking acquisition advances simulated time by exactly what it slept, and the
suite has no timing tolerance to tune. Two tests deliberately use the real
defaults instead, because a seam that is only ever exercised by its test
double proves nothing about the shipped path.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from threetears.search.contracts import EGRESS_DIRECT, RateLimiterPort
from threetears.search.limiter import (
    DEFAULT_BURST_TOKENS,
    DEFAULT_RATE_PER_SECOND,
    InProcessRateLimiter,
    RatePolicy,
)


class _ManualClock:
    """A monotonic source the test advances, and the sleeper that advances it.

    Not a fake of any production protocol -- it is the two callables the
    limiter's constructor takes, which is why it declares none.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        """Start at an arbitrary non-zero reading.

        :param start: initial monotonic value; non-zero so a test cannot pass
            by accidentally comparing against an uninitialised 0.0
        :ptype start: float
        """
        self.now = start
        self.sleeps: list[float] = []

    def read(self) -> float:
        """Report the current simulated monotonic reading.

        :return: seconds since this clock's arbitrary origin
        :rtype: float
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """Move simulated time forward.

        :param seconds: how far forward
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        """Record a sleep, advance the clock by it, and yield to the loop.

        :param seconds: what the limiter asked to wait
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def _limiter(clock: _ManualClock, **kwargs: object) -> InProcessRateLimiter:
    """Build a limiter driven by ``clock``.

    :param clock: the manual clock to read and sleep against
    :ptype clock: _ManualClock
    :param kwargs: pacing overrides forwarded to the constructor
    :ptype kwargs: object
    :return: the limiter under test
    :rtype: InProcessRateLimiter
    """
    return InProcessRateLimiter(clock=clock.read, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


async def test_the_in_process_limiter_satisfies_the_port() -> None:
    """P9/D8: the shipped implementation is interchangeable with an injected one."""
    limiter = InProcessRateLimiter()

    assert isinstance(limiter, RateLimiterPort)
    port: RateLimiterPort = limiter
    decision = await port.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    assert decision.acquired
    assert decision.retry_after_seconds == 0.0


async def test_defaults_are_on_and_pace_at_the_documented_rate() -> None:
    """SR-L6: a limiter nobody tuned still paces, at the rate the docstring argues for.

    The numbers are asserted rather than trusted, because "on by default with
    safe rates" is only a promise if a change to the constants is a visible
    event.
    """
    clock = _ManualClock()
    limiter = _limiter(clock)

    assert (DEFAULT_RATE_PER_SECOND, DEFAULT_BURST_TOKENS) == (1.0, 3.0)
    assert limiter.policy_for(provider_instance="searxng-main", egress=EGRESS_DIRECT) == RatePolicy()

    granted = 0
    for _ in range(5):
        if (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired:
            granted += 1

    assert granted == 3, "the burst is three calls, and the fourth waits"
    denied = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)
    assert denied.retry_after_seconds == pytest.approx(1.0), "one call a second is the sustained pace"


async def test_pacing_is_keyed_on_the_pair_not_the_provider() -> None:
    """D8/D20: two exits out of one instance are two allowances, not one."""
    clock = _ManualClock()
    limiter = _limiter(clock, burst_tokens=1.0)

    assert (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired
    assert (await limiter.acquire(provider_instance="searxng-main", egress="warp")).acquired
    assert not (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired
    assert not (await limiter.acquire(provider_instance="searxng-main", egress="warp")).acquired


async def test_two_instances_of_one_product_are_paced_separately() -> None:
    """SR-N4: two SearXNG deployments are two subjects behind one exit."""
    clock = _ManualClock()
    limiter = _limiter(clock, burst_tokens=1.0)

    assert (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired
    assert (await limiter.acquire(provider_instance="searxng-spare", egress=EGRESS_DIRECT)).acquired
    assert not (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired


async def test_tokens_refill_continuously_from_the_clock() -> None:
    """Refill is elapsed time times rate, computed on acquire -- no task, no ticks."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=2.0, burst_tokens=2.0)

    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=2.0)).acquired
    assert not (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired

    clock.advance(0.25)  # 2/sec * 0.25s == half a token: still short
    assert not (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired

    clock.advance(0.25)  # now a full token has accrued
    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired
    assert clock.sleeps == [], "a fail-fast acquisition never sleeps"


async def test_an_idle_key_banks_no_more_than_its_burst() -> None:
    """Idling for an hour does not buy an hour's worth of calls."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=2.0)

    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, tokens=2.0)
    clock.advance(3600.0)

    granted = 0
    for _ in range(5):
        if (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired:
            granted += 1

    assert granted == 2, "the ceiling is the burst, however long the key idled"


async def test_a_fail_fast_denial_is_a_value_and_costs_no_time() -> None:
    """A denial is returned, not raised, and ``0.0`` means a single answer."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=0.5, burst_tokens=1.0)
    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    denied = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    assert not denied.acquired
    assert denied.retry_after_seconds == pytest.approx(2.0), "one token short at half a token a second"
    assert clock.sleeps == []
    assert clock.now == 1_000.0


async def test_retry_after_is_honest_about_when_to_come_back() -> None:
    """Waiting exactly ``retry_after_seconds`` is enough -- no more, and no less."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=4.0, burst_tokens=1.0)
    await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)

    denied = await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)
    clock.advance(denied.retry_after_seconds * 0.5)
    assert not (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired, (
        "half the advertised wait is not enough -- the estimate is not padded"
    )

    clock.advance(denied.retry_after_seconds * 0.5)
    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired


async def test_blocking_sleeps_toward_availability_rather_than_polling() -> None:
    """``max_wait_seconds`` above zero waits for the computed moment, once."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=2.0, burst_tokens=1.0)
    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    granted = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, max_wait_seconds=5.0)

    assert granted.acquired
    assert granted.retry_after_seconds == 0.0
    assert clock.sleeps == [pytest.approx(0.5)], "one computed sleep, not a poll loop"


async def test_a_blocking_acquisition_respects_the_deadline() -> None:
    """SR-G2: a caller's remaining deadline bounds the wait, and a denial comes back."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=0.5, burst_tokens=1.0)
    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    denied = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, max_wait_seconds=1.0)

    assert not denied.acquired, "two seconds of refill were needed and one was offered"
    assert sum(clock.sleeps) <= 1.0
    assert clock.now <= 1_001.0, "the limiter never slept past the deadline it was given"
    assert denied.retry_after_seconds > 0.0


async def test_concurrent_acquisitions_never_over_grant() -> None:
    """N concurrent fail-fast acquires against N tokens grant exactly N."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=5.0)

    decisions = await asyncio.gather(
        *(limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT) for _ in range(12))
    )

    assert sum(1 for decision in decisions if decision.acquired) == 5


async def test_concurrent_waiters_are_paced_by_the_clock_not_the_burst() -> None:
    """Blocking waiters interleave without minting tokens: six calls cost three seconds.

    Three come out of the burst; the other three can only be granted as time
    passes, and simulated time only passes because a waiter slept for it.
    """
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=3.0)

    decisions = await asyncio.gather(
        *(
            limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, max_wait_seconds=60.0)
            for _ in range(6)
        )
    )

    assert all(decision.acquired for decision in decisions)
    assert clock.now - 1_000.0 >= 3.0, "three unbanked calls at one a second cannot cost less than three seconds"


async def test_a_weighted_call_costs_more_than_one_token() -> None:
    """SR-E4's weighted units have a pacing analogue: an advanced call costs two."""
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=3.0)

    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=2.0)).acquired
    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired

    denied = await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=2.0)
    assert not denied.acquired
    assert denied.retry_after_seconds == pytest.approx(2.0)


async def test_an_unsatisfiable_ask_is_a_programming_error_not_a_denial() -> None:
    """A call weighing more than the burst can never be granted by waiting."""
    clock = _ManualClock()
    limiter = _limiter(clock, burst_tokens=2.0)

    with pytest.raises(ValueError, match="exceeds the key's burst"):
        await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=3.0)
    with pytest.raises(ValueError, match="must not be negative"):
        await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=-1.0)


def test_a_policy_that_could_never_grant_is_refused_at_construction() -> None:
    """A zero rate never refills and a zero burst never holds a token."""
    with pytest.raises(ValueError, match="rate_per_second"):
        InProcessRateLimiter(rate_per_second=0.0)
    with pytest.raises(ValueError, match="burst_tokens"):
        InProcessRateLimiter(burst_tokens=0.0)
    with pytest.raises(ValueError, match="max_tracked_keys"):
        InProcessRateLimiter(max_tracked_keys=0)


async def test_rates_are_configured_by_the_host_per_pair_and_per_instance() -> None:
    """SR-K1: pacing comes from the constructor, resolved most specific first."""
    clock = _ManualClock()
    limiter = _limiter(
        clock,
        burst_tokens=1.0,
        policies={
            ("searxng-main", "warp"): RatePolicy(rate_per_second=10.0, burst_tokens=2.0),
            "tavily": RatePolicy(rate_per_second=5.0, burst_tokens=4.0),
        },
    )

    assert limiter.policy_for(provider_instance="searxng-main", egress="warp").burst_tokens == 2.0
    assert limiter.policy_for(provider_instance="searxng-main", egress=EGRESS_DIRECT).burst_tokens == 1.0
    assert limiter.policy_for(provider_instance="tavily", egress="warp").burst_tokens == 4.0, (
        "a bare instance name binds every exit out of that instance"
    )

    granted = 0
    for _ in range(4):
        if (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired:
            granted += 1
    assert granted == 4, "the per-instance override, not the default burst of one"


async def test_tracked_keys_stay_bounded_by_evicting_full_buckets() -> None:
    """SR-L6's resting half: the map cannot grow with traffic without bound.

    A full bucket is indistinguishable from an untracked key -- both start the
    next acquisition at the burst -- so evicting one is unobservable. A key
    still in debt is kept whatever the ceiling says, which is why the bound is
    soft: forgiving debt to save two floats would be the wrong trade.
    """
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=1.0, max_tracked_keys=4)

    for index in range(40):
        assert (await limiter.acquire(provider_instance=f"searxng-{index}", egress=EGRESS_DIRECT)).acquired
        assert limiter.tracked_key_count <= 4, "full buckets are evicted rather than accumulated"
        clock.advance(60.0)  # the keys behind this one refill to full

    assert (await limiter.acquire(provider_instance="searxng-39", egress="warp")).acquired, (
        "eviction changes no pacing: a pair with no bucket starts at the burst, like a full one"
    )
    assert not (await limiter.acquire(provider_instance="searxng-39", egress="warp")).acquired


def test_one_limiter_serves_two_one_shot_event_loops() -> None:
    """SR-L5: no loop-bound handle, so a module-scope limiter survives ``asyncio.run``.

    This is what an :class:`asyncio.Lock` in the limiter's state would break,
    and it is why there is not one.
    """
    clock = _ManualClock()
    limiter = _limiter(clock, rate_per_second=1.0, burst_tokens=1.0)

    first = asyncio.run(limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT))
    second = asyncio.run(limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT))

    assert first.acquired
    assert not second.acquired, "the second run sees the state the first left, not a fresh bucket"


async def test_the_shipped_sleeper_is_asyncio_sleep() -> None:
    """The default path blocks on the loop rather than in a thread, and comes back.

    Real time, deliberately: every other blocking test drives an injected
    sleeper, and a seam only ever exercised through its double would not prove
    the default one works.
    """
    limiter = InProcessRateLimiter(rate_per_second=100.0, burst_tokens=1.0)
    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    started = time.monotonic()
    granted = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, max_wait_seconds=2.0)
    elapsed = time.monotonic() - started

    assert granted.acquired
    assert elapsed < 1.0, "a hundred tokens a second means about ten milliseconds, not a poll interval"


async def test_the_limiter_yields_while_it_waits() -> None:
    """A blocking acquisition must not hold the loop: other work runs during it."""
    limiter = InProcessRateLimiter(rate_per_second=50.0, burst_tokens=1.0)
    await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)
    progressed = 0

    async def _other_work() -> None:
        nonlocal progressed
        for _ in range(5):
            progressed += 1
            await asyncio.sleep(0)

    granted, _ = await asyncio.gather(
        limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, max_wait_seconds=2.0),
        _other_work(),
    )

    assert granted.acquired
    assert progressed == 5
