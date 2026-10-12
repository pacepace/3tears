"""one catch-up pass over a listener's registered subjects.

:func:`catchup_tick` is the safety net for everything the broadcast path can
lose: the documented prime/subscribe race window, a message dropped on the
wire, a subscriber blip -- and, since the counter moved to a memory-backed KV
bucket, a broker restart that replaces the counter outright while every
operation against it keeps succeeding.

**Pure-async, one pass per call. No internal polling; the consumer's scheduler
drives cadence.** That shape is not a preference, it is what the package
dependency arrow allows. ``3tears-epoch`` depends on ``3tears`` and
``3tears-nats``; neither may depend on it. A framework-owned loop would have to
live where the collections registry lives, in ``3tears``, and importing the
listener there is a circular dependency -- which, under this family's lockstep
version bounds, makes the whole family unresolvable rather than merely wrong.
It also matches :mod:`threetears.scheduled_jobs.tick`, which states the same
position for the same reason.

The consumer keeps its own loop, its own interval and its own shutdown. What it
stops keeping is a per-consumer opinion about what a pass DOES -- which subjects
to poll, whether one subject's failure should abandon the others, and whether a
raise should kill the loop that schedules it.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.nats.subjects import Subject
from threetears.observe import get_logger

from threetears.epoch.listener import BumpCallback, EpochListener

__all__ = [
    "catchup_tick",
]

log = get_logger(__name__)


async def catchup_tick(
    listener: EpochListener,
    subjects: Sequence[tuple[Subject, BumpCallback]],
) -> int:
    """run one catch-up pass over ``subjects``, returning how many advanced.

    **A failing subject does not abandon the others.** Each is attempted; the
    first exception is re-raised after the pass so a consumer bug still
    surfaces, and the consumer's own loop decides whether that ends its
    scheduling. Letting the first failure return would mean one broken domain
    silently stopped the catch-up for every other domain sharing the pass --
    the failure mode being fixed here, one level up.

    Ordering is the caller's: subjects are polled in the sequence given.

    :param listener: the listener whose last-seen state this pass advances
    :ptype listener: EpochListener
    :param subjects: the ``(subject, on_bump)`` pairs to poll, one per domain
        this consumer tracks
    :ptype subjects: Sequence[tuple[Subject, BumpCallback]]
    :return: how many subjects advanced (fired their callback) in this pass
    :rtype: int
    :raises Exception: the first exception raised by any subject's catch-up,
        after every other subject has been attempted
    """
    advanced = 0
    first_error: BaseException | None = None
    for subject, on_bump in subjects:
        before = listener.last_seen(subject)
        try:
            after = await listener.catch_up(subject, on_bump)
        # prawduct:allow prawduct/broad-except -- one domain's failure must not
        # abandon the remaining domains in the same pass. re-raised below.
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "epoch catch-up failed for a subject; continuing with the rest of the pass",
                exc_info=True,
                extra={"extra_data": {"subject": subject.path}},
            )
            if first_error is None:
                first_error = exc
            continue
        if after != before:
            advanced += 1
    if first_error is not None:
        raise first_error
    return advanced
