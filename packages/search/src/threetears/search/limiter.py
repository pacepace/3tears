"""In-process token-bucket pacing, keyed ``(provider instance, egress)`` -- D8's other half.

D8 rules two mechanisms for one shape. Where a host has a bus it injects an
adapter over ``core``'s NATS ``TokenBucket`` and the bound is shared across
every pod. Where it has no bus -- §5.4's embedded mode, a Pi, samsung -- there
is nothing to share state through, and the answer is this module: a token
bucket per ``(provider instance, egress)`` pair held in the process that is
making the calls. The leaf takes no ``threetears.core`` dependency (SR-L7), so
this is written rather than reused; the port both satisfy lives in
:mod:`threetears.search.contracts.limiter`.

**On by default, with rates chosen to be uninteresting rather than optimal**
(SR-L6). A no-argument :class:`InProcessRateLimiter` paces one call per second
per pair with a burst of three:

- SR-H4's context is a *shared* self-hosted SearXNG, where no single consumer
  sees the aggregate load and the instance's own limiter is the backstop for
  non-cooperating clients rather than our budget to spend. A default should
  therefore sit far below whatever a deployment allows, not probe it. One call
  a second is slower than a person clicking through result pages.
- The asymmetry decides the direction: too slow is latency the caller can see
  and tune away at construction, too fast is a ban on an instance shared by
  everybody, which nobody can tune away afterwards.
- The burst of three is what lets the shapes this package actually issues pass
  unpaced -- a handful of queries fired back to back at one instance -- while
  still pacing a loop. Transport retries sit *below* this seam and cost no
  tokens, so the burst is not sized for them.
- The resting-footprint half of SR-L6: a key costs two floats, the map is
  soft-capped at :data:`DEFAULT_MAX_TRACKED_KEYS`, and the limiter holds no
  task, timer, thread or connection. Nothing here grows with traffic.

Rates are host configuration, passed at construction -- per pair, per provider
instance, or as the default. Nothing in this module reads the environment or
any ambient config (SR-K1).

**No background refill task.** Tokens are recomputed from a monotonic clock on
each :meth:`InProcessRateLimiter.acquire`, so a limiter is inert between calls
and a single search works from a one-shot ``asyncio.run()`` with nothing to
start and nothing to close (SR-L5).

**No lock, deliberately.** The refill-compare-consume sequence in
``_attempt`` is synchronous: it contains no ``await``, so the event loop cannot
interleave two acquisitions inside it, and N concurrent acquires against a
bucket holding N tokens grant exactly N. A lock would buy nothing there and
would cost SR-L5: an :class:`asyncio.Lock` binds to the loop it is first
awaited on, so a limiter constructed once and used from two separate
``asyncio.run()`` calls would raise on the second. Holding no loop-bound handle
is what keeps one instance reusable across one-shot calls. *Anything added to
``_attempt`` must stay synchronous* -- the first ``await`` inside it reopens
the race a lock would have closed. Waiting happens outside it, between
attempts.

Fairness is not promised, and neither does the distributed implementation
promise it: a caller arriving fail-fast at the instant a token refills may take
it ahead of a waiter that has been sleeping for exactly that token. What the
bucket paces is the aggregate rate, which is what the upstream actually sees,
and each waiter is bounded by its own deadline.

**The state here is the argued SR-O2 allowlist entry** (search-spec.md §3.9),
argued rather than assumed:

1. It is *meaningless* outside this process. A bucket's timestamp is a
   monotonic-clock reading, which is not comparable across processes, let
   alone across hosts -- there is no serialisation of this state that another
   reader could use.
2. The cross-instance version already exists and is a different object:
   ``core``'s NATS ``TokenBucket``, host-injected (D8). This is not a
   backend-shaped thing built badly; it is the half of the ruling that must
   hold where there is no backend to reach.
3. "A restart forgets" is correct behaviour here, not a defect. Every key
   resets to a full bucket, which is exactly the state an upstream would infer
   from a process that had been making no calls -- the forgetting is bounded
   by one burst.
4. It is bounded by construction: keys are the host's configured
   ``(instance, egress)`` pairs, two floats each, soft-capped with full
   buckets evicted first -- and evicting a full bucket is unobservable, since
   a fresh key starts full.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final

from threetears.search.contracts import RateLimitDecision

__all__ = [
    "DEFAULT_BURST_TOKENS",
    "DEFAULT_MAX_TRACKED_KEYS",
    "DEFAULT_RATE_PER_SECOND",
    "InProcessRateLimiter",
    "RatePolicy",
]

#: sustained pace per ``(provider instance, egress)``, in calls per second.
#: Deliberately below any rate a shared instance would notice (SR-L6, SR-H4);
#: a host that knows its instance can take more says so at construction.
DEFAULT_RATE_PER_SECOND: Final[float] = 1.0

#: how much allowance one pair may bank while idle, in calls. Three lets a
#: short back-to-back batch through unpaced without letting a loop run.
DEFAULT_BURST_TOKENS: Final[float] = 3.0

#: soft ceiling on tracked pairs. Reached only by a host calling more distinct
#: pairs than it has configured providers; full buckets are evicted first, and
#: a pair genuinely being paced is never evicted, because dropping it would
#: forgive the debt this module exists to collect.
DEFAULT_MAX_TRACKED_KEYS: Final[int] = 256


@dataclass(frozen=True, slots=True)
class RatePolicy:
    """The pace and the burst allowed to one paced key.

    A configuration value, not a wire type: hosts build these at construction
    and this package never puts one on a payload (SR-L4).
    """

    #: tokens added per second, continuously rather than in ticks.
    rate_per_second: float = DEFAULT_RATE_PER_SECOND
    #: ceiling on banked tokens -- the largest unpaced run a key may make.
    burst_tokens: float = DEFAULT_BURST_TOKENS

    def __post_init__(self) -> None:
        """Reject a policy that could never grant.

        :return: nothing
        :rtype: None
        :raises ValueError: when either field is not positive -- a zero rate
            never refills and a zero burst never holds a token, so both are
            configuration mistakes rather than very strict pacing
        """
        if self.rate_per_second <= 0:
            raise ValueError(f"rate_per_second must be positive, got {self.rate_per_second}")
        if self.burst_tokens <= 0:
            raise ValueError(f"burst_tokens must be positive, got {self.burst_tokens}")


@dataclass(slots=True)
class _Bucket:
    """One key's tokens, as of one monotonic reading.

    :ivar tokens: tokens available at ``updated_at``
    :ivar updated_at: monotonic reading the count is as of
    """

    tokens: float
    updated_at: float


class InProcessRateLimiter:
    """A token bucket per ``(provider instance, egress)``, held in this process.

    Satisfies :class:`~threetears.search.contracts.limiter.RateLimiterPort`
    structurally, the same way
    :class:`~threetears.search.standalone.StandaloneTransport` satisfies
    ``SearchTransport``: a consumer may inject this, ``core``'s distributed
    bucket behind a thin adapter, or its own, and nothing in the package knows
    which it got.

    Construct one per process and share it -- two limiters do not share
    buckets, so two of them pace at twice the configured rate.
    """

    def __init__(
        self,
        *,
        rate_per_second: float = DEFAULT_RATE_PER_SECOND,
        burst_tokens: float = DEFAULT_BURST_TOKENS,
        policies: Mapping[tuple[str, str] | str, RatePolicy] | None = None,
        max_tracked_keys: int = DEFAULT_MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Configure the pacing every key gets, and the exceptions to it.

        :param rate_per_second: sustained calls per second for any key no
            entry in ``policies`` names
        :ptype rate_per_second: float
        :param burst_tokens: banked allowance for any key no entry in
            ``policies`` names
        :ptype burst_tokens: float
        :param policies: per-key overrides, resolved most specific first: a
            ``(provider_instance, egress)`` tuple binds that exact pair, a
            bare ``provider_instance`` string binds every exit out of that
            instance. Host configuration, passed here rather than read from
            anywhere (SR-K1)
        :ptype policies: Mapping[tuple[str, str] | str, RatePolicy] | None
        :param max_tracked_keys: soft ceiling on tracked pairs; full buckets
            are evicted once it is reached
        :ptype max_tracked_keys: int
        :param clock: monotonic seconds source. Injectable so a test can drive
            time rather than wait for it; a wall clock does not belong here,
            because a clock step backwards would mint tokens
        :ptype clock: Callable[[], float]
        :param sleep: how a blocking acquisition waits. ``asyncio.sleep`` by
            default, and any replacement must yield to the loop rather than
            block it -- a limiter that blocks the loop paces the whole
            process. Injected alongside ``clock`` on purpose: a test that
            advances a clock must also own the sleeping, or a blocking
            acquisition would sleep in real time against a clock that never
            moves
        :ptype sleep: Callable[[float], Awaitable[None]]
        :return: nothing
        :rtype: None
        :raises ValueError: when the default rate or burst is not positive,
            or ``max_tracked_keys`` is below 1
        """
        if max_tracked_keys < 1:
            raise ValueError(f"max_tracked_keys must be at least 1, got {max_tracked_keys}")
        self._default_policy = RatePolicy(rate_per_second=rate_per_second, burst_tokens=burst_tokens)
        self._policies: Mapping[tuple[str, str] | str, RatePolicy] = dict(policies or {})
        self._max_tracked_keys = max_tracked_keys
        self._clock = clock
        self._sleep = sleep
        # The SR-O2 allowlist entry this module's docstring argues for, and the
        # only mutable state here: pair -> live bucket, per-process by nature.
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    @property
    def tracked_key_count(self) -> int:
        """How many pairs currently hold state.

        Public so a host (or SR-L6's resting-footprint check) can see the
        bound hold without reaching into private state. Not a count of pairs
        ever seen: a key that has refilled to full may have been evicted, and
        that is unobservable in pacing terms.

        :return: number of live buckets
        :rtype: int
        """
        return len(self._buckets)

    def policy_for(self, *, provider_instance: str, egress: str) -> RatePolicy:
        """Report the policy this limiter would apply to one key.

        Public because a host that configured an override deserves to be able
        to assert it took, without reaching into private state.

        :param provider_instance: the configured deployment
        :ptype provider_instance: str
        :param egress: the exit calls to it leave by
        :ptype egress: str
        :return: the exact-pair policy, else the per-instance one, else the
            default
        :rtype: RatePolicy
        """
        return self._policy_for((provider_instance, egress))

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Ask for permission to make one call against one paced key.

        ``max_wait_seconds=0.0`` is a single fail-fast answer. Above zero, the
        call sleeps toward the moment the shortfall will have refilled --
        computed, not polled -- and re-attempts, until it is granted or the
        deadline passes. A denial is always a returned decision, never an
        exception.

        :param provider_instance: the deployment about to be called; two
            deployments of one product are two subjects (SR-N4)
        :ptype provider_instance: str
        :param egress: the exit the call will leave by (D20)
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: seconds the caller will block for; a caller
            under a deadline passes what remains of it (SR-G2). Anything at
            or below zero is fail-fast
        :ptype max_wait_seconds: float
        :return: the decision, with an honest ``retry_after_seconds`` on a
            denial -- the shortfall over the key's refill rate
        :rtype: RateLimitDecision
        :raises ValueError: when ``tokens`` is negative (the port declares no
            release, so a negative acquisition is not a refund), or exceeds
            the key's burst, which no amount of waiting could satisfy
        """
        key = (provider_instance, egress)
        policy = self._policy_for(key)
        if tokens < 0:
            raise ValueError(f"tokens must not be negative, got {tokens}")
        if tokens > policy.burst_tokens:
            raise ValueError(
                f"cannot acquire {tokens} tokens for {key!r}: exceeds the key's burst of {policy.burst_tokens}"
            )
        deadline = self._clock() + max_wait_seconds if max_wait_seconds > 0 else None
        while True:
            decision = self._attempt(key, policy, tokens)
            if decision.acquired or deadline is None:
                return decision
            remaining = deadline - self._clock()
            if remaining <= 0:
                return decision
            await self._sleep(max(0.0, min(decision.retry_after_seconds, remaining)))

    def _policy_for(self, key: tuple[str, str]) -> RatePolicy:
        """Resolve one key's policy, most specific first.

        :param key: the ``(provider instance, egress)`` pair
        :ptype key: tuple[str, str]
        :return: the policy to pace this key by
        :rtype: RatePolicy
        """
        exact = self._policies.get(key)
        if exact is not None:
            return exact
        by_instance = self._policies.get(key[0])
        if by_instance is not None:
            return by_instance
        return self._default_policy

    def _attempt(self, key: tuple[str, str], policy: RatePolicy, tokens: float) -> RateLimitDecision:
        """Refill from the clock, then consume if the key can afford it.

        Synchronous on purpose -- see this module's docstring. Adding an
        ``await`` here would let two acquisitions interleave between the
        affordability test and the debit, which is the one race a token bucket
        must not have.

        :param key: the pair being paced
        :ptype key: tuple[str, str]
        :param policy: the rate and burst for this key
        :ptype policy: RatePolicy
        :param tokens: how much this call consumes
        :ptype tokens: float
        :return: the decision for this instant
        :rtype: RateLimitDecision
        """
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            self._evict_full(now)
            bucket = _Bucket(tokens=policy.burst_tokens, updated_at=now)
            self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(policy.burst_tokens, bucket.tokens + elapsed * policy.rate_per_second)
            bucket.updated_at = now
        if bucket.tokens < tokens:
            shortfall = tokens - bucket.tokens
            return RateLimitDecision(acquired=False, retry_after_seconds=shortfall / policy.rate_per_second)
        bucket.tokens -= tokens
        return RateLimitDecision(acquired=True)

    def _evict_full(self, now: float) -> None:
        """Drop tracked keys that have refilled to full, once at the ceiling.

        A full bucket carries no information: a fresh key starts full, so a
        later acquisition against an evicted key behaves identically. A key
        still in debt is never dropped, which is why the ceiling is soft --
        forgiving debt to save two floats would be the wrong trade.

        :param now: the monotonic reading to refill against
        :ptype now: float
        :return: nothing
        :rtype: None
        """
        if len(self._buckets) < self._max_tracked_keys:
            return
        for tracked_key, bucket in list(self._buckets.items()):
            policy = self._policy_for(tracked_key)
            refilled = bucket.tokens + max(0.0, now - bucket.updated_at) * policy.rate_per_second
            if refilled >= policy.burst_tokens:
                del self._buckets[tracked_key]
