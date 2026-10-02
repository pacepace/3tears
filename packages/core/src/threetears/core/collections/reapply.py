"""re-run a read-apply-save whose fenced save lost its race, bounded.

A table declaring ``TableSchema(cas_column=...)`` fences every save of a loaded row on the fence
value its writer READ: when another save has landed since, :meth:`BaseCollection.save_entity`
refuses the stale one with :class:`~threetears.core.exceptions.ConcurrentModificationError`
rather than letting it overwrite the winner blind. What a caller does with that refusal depends
on what its change is:

- a change that is a **pure function of the current row** -- a set of named fields, a counter,
  a flag, a merge, a get-or-create on a derived id -- is right to re-read the winner's row and
  apply itself again over it. That is this module.
- a change computed from **what a person saw** answers a conflict instead, because re-applying
  it over a row they never saw is an overwrite under a different name.

**The attempt is the whole read-apply-save, not the save alone.** The losing save's working copy
is withdrawn from L1 when the fence refuses it, so re-running the read reaches the winner's row;
re-saving the stale handle would only lose again. The caller passes the attempt as a closure, so
the ``.save()`` stays in the method that owns the table.

**Bounded, then the refusal itself propagates.** A row that keeps losing is sustained contention,
and the caller answers it (a conflict to a person, a logged failure for a background job). It is
never swallowed here and never turned into a success. Any other exception propagates at once:
only a lost race is retried.

**The pause between attempts is the caller's schedule.** The default, :func:`full_jitter_backoff`,
is a short uniform pause sized for many writers racing one row in-process. A caller racing a
slower writer -- a background net that rewrites the row a beat after it lands -- passes
:class:`ExponentialBackoff` with a smaller budget, so its attempts spread out over seconds rather
than spending thirty of them inside one.

This is not :meth:`BaseCollection.l2_cas_mutate`, which is the L2-revision-ordered swap with its
own retry; this loop rides the L3 ``cas_column`` fence that an ordinary entity save carries.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from threetears.core.exceptions import ConcurrentModificationError
from threetears.observe import get_logger

__all__ = [
    "REAPPLY_BACKOFF_SECONDS",
    "REAPPLY_MAX_ATTEMPTS",
    "ExponentialBackoff",
    "full_jitter_backoff",
    "reapply_on_lost_race",
]

log = get_logger(__name__)

#: attempts, first included, before the refusal propagates. Sized for a 20-way simultaneous race
#: on one row: measured against a live 20-writer test, one writer exhausted exactly 20 rounds, so
#: the margin is kept.
REAPPLY_MAX_ATTEMPTS = 30

#: ceiling of the full-jitter pause between attempts, in seconds.
REAPPLY_BACKOFF_SECONDS = 0.02

T = TypeVar("T")


def full_jitter_backoff(lost_attempt: int) -> float:
    """the default pause after a lost race: uniform between zero and :data:`REAPPLY_BACKOFF_SECONDS`.

    Full jitter, not security: it spreads the losers of one race so they do not collide again in
    step. The same ceiling whatever the attempt, because the writers it separates are in-process
    and fast.

    :param lost_attempt: the number of the attempt that just lost, first is one; unused here
    :ptype lost_attempt: int
    :return: seconds to pause before the next attempt
    :rtype: float
    """
    del lost_attempt  # every attempt waits on the same ceiling
    return random.uniform(0, REAPPLY_BACKOFF_SECONDS)


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """a pause that doubles after each lost race, up to a cap, jittered to between half and one and a half of it.

    After attempt ``n`` loses, the pause is ``min(base_seconds * 2 ** (n - 1), cap_seconds)``
    times a uniform jitter in ``[0.5, 1.5)``. With an 8-attempt budget on a 50ms base capped at
    2s, the seven pauses sum to 5.15s of steps, so a create that keeps losing gives up after
    roughly 2.5-8s.

    :ivar base_seconds: the first pause before jitter; positive
    :ivar cap_seconds: the largest pause before jitter; at least ``base_seconds``
    """

    base_seconds: float
    cap_seconds: float

    def __post_init__(self) -> None:
        """refuse a schedule that cannot back off.

        :return: None
        :rtype: None
        :raises ValueError: when ``base_seconds`` is not positive, or ``cap_seconds`` is below it
        """
        if self.base_seconds <= 0:
            raise ValueError(f"base_seconds must be positive, got {self.base_seconds}")
        if self.cap_seconds < self.base_seconds:
            raise ValueError(
                f"cap_seconds must be at least base_seconds, got cap {self.cap_seconds} below base {self.base_seconds}"
            )

    def __call__(self, lost_attempt: int) -> float:
        """the pause after attempt ``lost_attempt`` lost its race.

        :param lost_attempt: the number of the attempt that just lost, first is one
        :ptype lost_attempt: int
        :return: seconds to pause before the next attempt
        :rtype: float
        :raises ValueError: when ``lost_attempt`` is below one
        """
        if lost_attempt < 1:
            raise ValueError(f"lost_attempt must be at least 1, got {lost_attempt}")
        step: float = min(self.base_seconds * 2.0 ** (lost_attempt - 1), self.cap_seconds)
        # jitter, not security: spreads the losers so they do not collide again in step
        return step * random.uniform(0.5, 1.5)


async def reapply_on_lost_race(
    attempt: Callable[[], Awaitable[T]],
    *,
    what: str,
    max_attempts: int = REAPPLY_MAX_ATTEMPTS,
    backoff: Callable[[int], float] = full_jitter_backoff,
    pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """run a read-apply-save attempt until its fenced save lands, re-running it on a lost race.

    :param attempt: one whole read, apply and fenced save; re-run from its read after every
        refusal
    :ptype attempt: Callable[[], Awaitable[T]]
    :param what: what is being written, for logs
    :ptype what: str
    :param max_attempts: attempts, first included, before the refusal propagates
    :ptype max_attempts: int
    :param backoff: the pause schedule: given the number of the attempt that just lost (first is
        one), the seconds to wait before the next; :func:`full_jitter_backoff` by default, or an
        :class:`ExponentialBackoff`
    :ptype backoff: Callable[[int], float]
    :param pause: awaitable sleep between attempts; the production default is ``asyncio.sleep``
    :ptype pause: Callable[[float], Awaitable[None]]
    :return: the value of the attempt whose save landed
    :rtype: T
    :raises ConcurrentModificationError: the last refusal, once every attempt lost its race
    :raises ValueError: when ``max_attempts`` is below one
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    last_refusal: ConcurrentModificationError | None = None
    landed: tuple[T] | None = None
    for attempt_number in range(1, max_attempts + 1):
        try:
            landed = (await attempt(),)
        except ConcurrentModificationError as refusal:
            last_refusal = refusal
            log.info(
                f"{what} lost a compare-and-swap race (attempt {attempt_number}/{max_attempts}); re-reading the row"
            )
            if attempt_number < max_attempts:
                await pause(backoff(attempt_number))
            continue
        if attempt_number > 1:
            log.info(f"{what} landed on attempt {attempt_number} after re-reading the winner's row")
        break
    if landed is None:
        assert last_refusal is not None
        log.error(
            f"{what} lost {max_attempts} consecutive compare-and-swap races and was not applied: another writer "
            f"is updating the same row continuously; find that writer, or answer the caller with a conflict and "
            f"let it retry"
        )
        raise last_refusal
    return landed[0]
