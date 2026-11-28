"""the rules for a tool call's updates (progress and log entries) while it runs: who may publish where, and how much.

A call's updates travel in two hops, as a durable result does (:mod:`threetears.nats.result_delivery`): the pod
publishes to ``{ns}.tools.progress.{pod_id}.{call_id}`` under its own id, and the registry relays each to
``{ns}.tools.updates.{caller_id}.{call_id}`` under the caller's verified id. This module owns the pure decisions both
ends share -- whether a subject is a party's own, how fast updates may flow and how large one may be -- so the pod,
the registry and the caller cannot disagree about them. It imports nothing from the NATS client.

Updates are best effort. They ride core NATS, a lost one loses only itself, and the tool that sends them never waits
on the link: the bounds here are what a sender drops or coalesces to, never what it blocks on.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from uuid import UUID

from threetears.nats.subjects import Subjects

__all__ = [
    "MAX_UPDATE_BYTES",
    "POD_UPDATE_BURST",
    "POD_UPDATES_PER_SECOND",
    "RELAY_UPDATE_BURST",
    "RELAY_UPDATES_PER_SECOND",
    "UPDATE_FLUSH_SECONDS",
    "UpdateBudget",
    "UpdateDrain",
    "progress_subject_is_owned_by_pod",
    "updates_subject_is_owned_by_caller",
]

#: the most updates a pod sends a second for one call, and the burst it may spend at once. Progress coalesces to the
#: latest value while the budget refills, so a tool reporting in a tight loop costs the link this and no more.
POD_UPDATES_PER_SECOND = 10.0
POD_UPDATE_BURST = 10

#: the most updates the registry relays a second for one call, and its burst: twice the pod's, so an honest pod never
#: meets it. A tool pod runs a tenant's code, so the registry does not take the pod's bound on trust; what is over it
#: is dropped, not queued.
RELAY_UPDATES_PER_SECOND = 20.0
RELAY_UPDATE_BURST = 40

#: the largest encoded update any hop sends or relays. A log entry over it is sent with its data replaced by a note of
#: its size; a relay drops one over it.
MAX_UPDATE_BYTES = 16 * 1024

#: how long a pod spends sending what is still pending before it answers the call. An MCP client ignores progress
#: after the result, so the last progress leaves first; the bound keeps a dead link from delaying the answer.
UPDATE_FLUSH_SECONDS = 1.0

#: once a call has answered, how long a receiving hop (the registry's relay, the caller) keeps listening for updates
#: still on their way: until none has arrived for :data:`_DRAIN_QUIET_SECONDS`, and no longer than
#: :data:`_DRAIN_SECONDS` in all. A subscription's messages are dispatched from a task of their own, apart from
#: the reply's, so an update the pod sent before its answer can still be queued when the answer is read; ending the
#: subscription then would drop exactly the last progress.
_DRAIN_QUIET_SECONDS = 0.05
_DRAIN_SECONDS = 0.5


class UpdateBudget:
    """a token bucket: ``rate`` updates a second, at most ``burst`` at once.

    one per call, on each hop that bounds it. :meth:`take` spends one when it can (the relay drops what it cannot);
    :meth:`wait` says how long until one can be spent (the pod sleeps that long, coalescing meanwhile).

    :param rate: updates a second the bucket refills by
    :ptype rate: float
    :param burst: the most the bucket holds, and starts with
    :ptype burst: int
    :param clock: monotonic seconds; ``time.monotonic`` when not given
    :ptype clock: Callable[[], float] | None
    """

    def __init__(self, *, rate: float, burst: int, clock: Callable[[], float] | None = None) -> None:
        """start the bucket full.

        :param rate: updates a second
        :ptype rate: float
        :param burst: the bucket's size
        :ptype burst: int
        :param clock: monotonic seconds
        :ptype clock: Callable[[], float] | None
        :return: nothing
        :rtype: None
        :raises ValueError: ``rate`` is not positive or ``burst`` is below one
        """
        if rate <= 0 or burst < 1:
            raise ValueError(f"an update budget needs a positive rate and a burst of at least one; got {rate}, {burst}")
        self._rate = rate
        self._burst = float(burst)
        self._clock = clock if clock is not None else time.monotonic
        self._tokens = float(burst)
        self._at = self._clock()

    def _refill(self) -> None:
        """add what the bucket earned since it was last read.

        :return: nothing
        :rtype: None
        """
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._at) * self._rate)
        self._at = now

    def take(self) -> bool:
        """spend one update when the bucket holds one.

        :return: ``True`` when one was spent
        :rtype: bool
        """
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait(self) -> float:
        """seconds until :meth:`take` can spend one; ``0.0`` when it can now.

        :return: the wait
        :rtype: float
        """
        self._refill()
        return 0.0 if self._tokens >= 1.0 else (1.0 - self._tokens) / self._rate


class UpdateDrain:
    """when a receiving hop may stop listening for a call's updates: after it has gone quiet, within a bound.

    the hop's handler handles each update inside ``with drain:``; once the call has answered the hop awaits
    :meth:`settled` and then ends its subscription.

    :param clock: monotonic seconds; ``time.monotonic`` when not given
    :ptype clock: Callable[[], float] | None
    :param sleep: awaits a number of seconds; ``asyncio.sleep`` when not given
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """start with nothing received.

        :param clock: monotonic seconds
        :ptype clock: Callable[[], float] | None
        :param sleep: awaits a number of seconds
        :ptype sleep: Callable[[float], Awaitable[None]] | None
        :return: nothing
        :rtype: None
        """
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._busy = 0
        self._last = self._clock()

    def __enter__(self) -> None:
        """one update is being handled.

        :return: nothing
        :rtype: None
        """
        self._busy += 1
        self._last = self._clock()

    def __exit__(self, *exc: object) -> None:
        """the update has been handled.

        :param exc: the exception the block raised, if any
        :ptype exc: object
        :return: nothing
        :rtype: None
        """
        self._busy -= 1
        self._last = self._clock()

    async def settled(self, *, quiet: float = _DRAIN_QUIET_SECONDS, limit: float = _DRAIN_SECONDS) -> bool:
        """wait until no update is being handled and none has arrived for ``quiet``, or ``limit`` has passed.

        :param quiet: the quiet that means nothing more is on its way
        :ptype quiet: float
        :param limit: the most this waits
        :ptype limit: float
        :return: ``True`` when it settled, ``False`` when the limit ended the wait
        :rtype: bool
        """
        deadline = self._clock() + limit
        while True:
            now = self._clock()
            idle_for = now - self._last
            if self._busy == 0 and idle_for >= quiet:
                return True
            if now >= deadline:
                return False
            await self._sleep(min(max(quiet - idle_for, quiet / 5), max(deadline - now, 0.0)))


def progress_subject_is_owned_by_pod(subject: str, *, pod_id: str | UUID, namespace: str | None = None) -> bool:
    """whether ``subject`` is one call's updates subject inside pod ``pod_id``'s own grant.

    the pod checks the subject it was handed before it publishes there: the broker would refuse a publish outside its
    grant without a word to anyone, and a pod's updates would then vanish with nothing in its log to say why.

    :param subject: the updates subject the registry asked the pod to publish to
    :ptype subject: str
    :param pod_id: the pod's own routing identifier
    :ptype pod_id: str | UUID
    :param namespace: the pod's subject namespace; the process's default when None
    :ptype namespace: str | None
    :return: ``True`` when the subject is the pod's own
    :rtype: bool
    """
    prefix = str(Subjects.tools_progress_pod_wildcard(pod_id, namespace=namespace)).removesuffix(">")
    return _is_single_token_under(subject, prefix)


def updates_subject_is_owned_by_caller(subject: str, *, caller_id: str | UUID) -> bool:
    """whether ``subject`` is one call's updates subject of caller ``caller_id``.

    the registry relays to every caller's updates, so it relays only to a subject naming the call's VERIFIED caller:
    this check is what keeps that wildcard grant from being a way to push updates into a peer's call.

    :param subject: the updates subject the caller asked for
    :ptype subject: str
    :param caller_id: the verified identity of the caller
    :ptype caller_id: str | UUID
    :return: ``True`` when the subject belongs to that caller
    :rtype: bool
    """
    prefix = str(Subjects.tools_updates_caller_subtree(caller_id)).removesuffix("*")
    return _is_single_token_under(subject, prefix)


def _is_single_token_under(subject: str, prefix: str) -> bool:
    """whether ``subject`` is ``prefix`` plus exactly one further non-empty token, with no wildcard.

    :param subject: candidate subject
    :ptype subject: str
    :param prefix: the ``.``-terminated prefix
    :ptype prefix: str
    :return: ``True`` when the subject is one token below the prefix
    :rtype: bool
    """
    if not subject.startswith(prefix):
        return False
    tail = subject[len(prefix) :]
    return bool(tail) and "." not in tail and "*" not in tail and ">" not in tail
