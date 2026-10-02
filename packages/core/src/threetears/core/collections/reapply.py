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

This is not :meth:`BaseCollection.l2_cas_mutate`, which is the L2-revision-ordered swap with its
own retry; this loop rides the L3 ``cas_column`` fence that an ordinary entity save carries.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from threetears.core.exceptions import ConcurrentModificationError
from threetears.observe import get_logger

__all__ = ["REAPPLY_BACKOFF_SECONDS", "REAPPLY_MAX_ATTEMPTS", "reapply_on_lost_race"]

log = get_logger(__name__)

#: attempts, first included, before the refusal propagates. Sized for a 20-way simultaneous race
#: on one row: measured against a live 20-writer test, one writer exhausted exactly 20 rounds, so
#: the margin is kept.
REAPPLY_MAX_ATTEMPTS = 30

#: ceiling of the full-jitter pause between attempts, in seconds.
REAPPLY_BACKOFF_SECONDS = 0.02

T = TypeVar("T")


async def reapply_on_lost_race(
    attempt: Callable[[], Awaitable[T]],
    *,
    what: str,
    max_attempts: int = REAPPLY_MAX_ATTEMPTS,
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
                # full jitter, not security: spreads the losers so they do not collide again in step
                await pause(random.uniform(0, REAPPLY_BACKOFF_SECONDS))
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
