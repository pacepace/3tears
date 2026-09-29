"""The :class:`~threetears.iam.stores.base.AttemptLimiter` implementation, over durable counters.

This module exists because its contents stopped being a KV store. Until 0.43.0 the limiter kept
its counts in a JetStream KV bucket and was named and filed accordingly, beside the two stores
that really are KV. Attempt counts are durable security state -- a lockout that a broker restart
releases is a lockout an attacker can release -- so they moved to the coordination tables, reached
through a :class:`~threetears.core.collections.registry.CollectionRegistry` with L2 as the shared
fence and L3 as the record behind it.

The name moved with the state. A wave-2 adopter reading ``NatsKv`` in the old name would reason
about ephemerality, a bucket TTL and a broker restart, and every one of those conclusions is now
wrong in the direction that matters: it would under-trust a counter that is in fact durable.
"""

from __future__ import annotations

from datetime import timedelta

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import WindowedCounter, WindowState

from threetears.iam.stores.base import AttemptWindow

__all__ = ["CollectionAttemptLimiter"]


class CollectionAttemptLimiter:
    """:class:`~threetears.iam.stores.base.AttemptLimiter` over a
    :class:`~threetears.core.coordination.WindowedCounter`.

    The counting, the window, the CAS loop and the fail-open decision are all the counter's;
    this class only supplies the threshold and shapes the answer into an
    :class:`~threetears.iam.stores.base.AttemptWindow`. A second counting implementation
    would be a second set of window semantics to get wrong, which is exactly how a lockout
    ends up lasting a hundred milliseconds.

    **The window is anchored at the first failure, not at a wall-clock boundary.** Five
    failures buy a full window of lockout measured from the fifth attempt. An epoch-aligned
    window -- ``floor(now / window)`` -- looks equivalent and is not: every key's window
    rolls at the same instant, so an attacker who straddles a boundary gets ``2 x
    max_attempts`` back to back and a victim's lockout can expire almost immediately.

    **Fail-open is the caller's choice and defaults to closed.** It is defensible for a cheap
    edge throttle sitting in front of an authoritative check. It is not defensible for
    credential lockout, which has nothing behind it: there, a storage outage that reports "not
    limited" is an unlimited password-guessing window.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        *,
        purpose: str,
        max_attempts: int = 5,
        window: timedelta = timedelta(minutes=15),
        fail_open: bool = False,
    ) -> None:
        """
        :param registry: the collection registry the counter reads and writes through. L2 is
            the fence every replica shares and L3 the durable record behind it; both are
            optional, and a deployment with neither counts per process.
        :ptype registry: CollectionRegistry
        :param purpose: what this limiter protects, carried in the row key. Give each
            protected surface its own, so unrelated counters never share a budget. This is
            what the KV bucket name used to be.
        :ptype purpose: str
        :param max_attempts: failures within one window before :attr:`AttemptWindow.limited`.
        :ptype max_attempts: int
        :param window: the window length, measured from the first failure in it.
        :ptype window: timedelta
        :param fail_open: whether a storage failure, or exhausted compare-and-swap contention,
            reports "not limited" instead of raising. Defaults to ``False`` -- pass ``True``
            only with an authoritative check behind this one.
        :ptype fail_open: bool
        """
        self._max_attempts = max_attempts
        self._window = window
        self._counter = WindowedCounter(
            registry,
            purpose=purpose,
            window_seconds=int(window.total_seconds()),
            fail_open=fail_open,
        )

    def _verdict(self, state: WindowState | None) -> AttemptWindow:
        if state is None:
            return AttemptWindow(count=0, limited=False)
        limited = state.count >= self._max_attempts
        if not limited:
            return AttemptWindow(count=state.count, limited=False)
        # Time actually remaining, not the window length: a caller surfacing `Retry-After`
        # should not tell a user to wait fifteen minutes when three are left.
        # Read through the counter's own clock, not `time.time()` -- two clocks in one
        # verdict is two answers, and the retry_after is the one a user is shown.
        elapsed = self._counter.clock() - state.window_start
        remaining = max(self._window.total_seconds() - elapsed, 0.0)
        return AttemptWindow(count=state.count, limited=True, retry_after=timedelta(seconds=remaining))

    async def record_failure(self, key: str) -> AttemptWindow:
        count = await self._counter.record_attempt(key)
        if count == 0:
            # fail-open: the counter swallowed a storage failure and recorded nothing.
            return AttemptWindow(count=0, limited=False)
        return self._verdict(await self._counter.state(key))

    async def check(self, key: str) -> AttemptWindow:
        return self._verdict(await self._counter.state(key))

    async def clear(self, key: str) -> None:
        await self._counter.clear(key)
