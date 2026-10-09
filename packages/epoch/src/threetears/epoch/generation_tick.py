"""following collection write generations: one catch-up pass, and a watcher for a few keys.

A registry that follows a table (``CollectionRegistry.follow_generation``) keeps a mark: the last
write generation whose writes it has accounted for, moved on by the row broadcasts it hears. The
broadcasts are at most once, so something has to compare the mark with the table's generation as
the epoch bucket holds it, and drop the table in this pod when they disagree. That is all either
function here does. What "disagree" means is the mark's to say
(``threetears.core.collections.generation.GenerationMarks``):

- the generation belongs to another incarnation than the mark's -- the bucket was replaced, by a
  broker restart -- so the table is dropped;
- same incarnation, and an advance this pod did not hear every row of -- a broadcast was missed --
  so the table is dropped;
- everything up to it heard: nothing happens.

A generation that cannot be read is none of those. Nothing is dropped and the mark does not move;
the next pass tries again.

**:func:`generation_catchup_tick` is pure-async, one pass per call, with no internal polling: the
consumer's scheduler drives cadence**, for the reason :mod:`threetears.epoch.tick` gives -- the
package dependency arrow allows nothing else -- and with the same contract: one table's failure
does not abandon the rest. Whoever wires a registry to follow tables also schedules this pass, or
runs :func:`follow_generation_key` for each; following alone drops nothing.

**:func:`follow_generation_key` is the alternative for a pod following a few keys.** The broker
pushes each new generation through one named consumer per key, so a missed broadcast is noticed
when the generation moves rather than at the next pass. It costs a consumer per table per pod,
which is right for the four access tables and wrong for a registry following every table.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from datetime import timedelta
from typing import Final

from threetears.core.collections.generation import GenerationVerdict
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.exceptions import GenerationUnavailableError
from threetears.observe import get_logger

from threetears.epoch.generation import EpochGenerationReader

__all__ = [
    "DEFAULT_BROADCAST_GRACE",
    "follow_generation_key",
    "generation_catchup_tick",
]

log = get_logger(__name__)

#: how long :func:`follow_generation_key` waits for the row broadcasts of an advance it has just
#: been pushed before judging them missed. A writer advances the generation and THEN publishes its
#: rows, so the pushed generation routinely arrives first; judged at once, every heard write would
#: read as a missed one and drop the table.
DEFAULT_BROADCAST_GRACE: Final = timedelta(seconds=2)

#: how often that wait looks at the mark again.
_GRACE_POLL_SECONDS: Final = 0.02


async def generation_catchup_tick(registry: CollectionRegistry, reader: EpochGenerationReader) -> int:
    """run one catch-up pass over every table ``registry`` follows, returning how many it dropped.

    Each followed table's generation is read once and judged against the registry's mark
    (``CollectionRegistry.settle_generation``), which drops the table in this pod when a broadcast
    was missed or the bucket was replaced.

    **A generation that cannot be read is not a reset.** ``GenerationUnavailableError`` for a
    table is logged and that table is left exactly as it was: nothing dropped, the mark unmoved.
    The pod goes on serving what it has, as it did before the pass, and the next pass reads again.

    **A failing table does not abandon the others.** Every table is attempted; any other exception
    is re-raised after the pass, the first one if there were several, so a consumer bug still
    surfaces and the consumer's own loop decides whether that ends its scheduling.

    A pass that reads a generation while that advance's broadcasts are still in flight sees them
    unheard and drops the table needlessly. That is the safe direction.

    :param registry: the registry whose followed tables this pass judges
    :ptype registry: CollectionRegistry
    :param reader: reads each table's generation from the epoch bucket
    :ptype reader: EpochGenerationReader
    :return: how many tables were dropped in this pass
    :rtype: int
    :raises Exception: the first exception other than ``GenerationUnavailableError`` raised for
        any table, after every other table has been attempted
    """
    dropped = 0
    first_error: BaseException | None = None
    for table_name in registry.generation_marks.followed:
        try:
            token = await reader.read(table_name)
            verdict = registry.settle_generation(table_name, token)
        except GenerationUnavailableError as exc:
            log.warning(
                "write generation unavailable; the table is left as it is and the next pass reads again",
                extra={"extra_data": {"table": table_name, "error": str(exc)}},
            )
            continue
        # prawduct:allow prawduct/broad-except -- one table's failure must not abandon the
        # remaining tables in the same pass. re-raised below.
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "write generation catch-up failed for a table; continuing with the rest of the pass",
                exc_info=True,
                extra={"extra_data": {"table": table_name}},
            )
            if first_error is None:
                first_error = exc
            continue
        if verdict.drops:
            dropped += 1
    if first_error is not None:
        raise first_error
    return dropped


async def follow_generation_key(
    registry: CollectionRegistry,
    reader: EpochGenerationReader,
    table_name: str,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
) -> None:
    """follow one table's write generation by watching its key, until cancelled.

    Starts following the table on ``registry`` and then judges every generation the broker pushes,
    as a catch-up pass would judge one it read. The first value pushed is the key's latest, so a
    watcher that starts, or whose consumer the watch replaced after a broker restart, needs no
    read of its own.

    An advance is pushed as soon as it is written, ahead of the row broadcasts the writer sends
    next. So a generation whose advances are not yet all accounted for is given ``grace`` for
    their rows to arrive before it is judged; only then is an advance still unheard a missed
    broadcast, and the table dropped. A generation from another incarnation is never waited on.

    Runs until the consumer cancels it. Run it as a task per table, beside whatever else the pod
    schedules.

    What a watch cannot see: the broker pushes nothing when the bucket is emptied, so a replaced
    bucket is noticed at the table's next advance, which arrives under a new incarnation. An
    advance whose push and whose broadcasts were all lost in the same restart is noticed then too,
    and not before. A pod for which that window matters also runs
    :func:`generation_catchup_tick`.

    :param registry: the registry that follows the table
    :ptype registry: CollectionRegistry
    :param reader: watches the table's generation key in the epoch bucket
    :ptype reader: EpochGenerationReader
    :param table_name: the table to follow
    :ptype table_name: str
    :param grace: how long to wait for an advance's row broadcasts before judging them missed
    :ptype grace: timedelta
    :return: nothing; it returns only if the watch ends
    :rtype: None
    :raises GenerationUnavailableError: when the bucket cannot be opened or watched
    :raises KvError: when the connection closes, so nothing can ever be pushed again
    """
    registry.follow_generation(table_name)
    marks = registry.generation_marks
    loop = asyncio.get_running_loop()
    async with aclosing(reader.watch(table_name)) as generations:
        async for token in generations:
            deadline = loop.time() + grace.total_seconds()
            while marks.judge(table_name, token) is GenerationVerdict.MISSED and loop.time() < deadline:
                await asyncio.sleep(_GRACE_POLL_SECONDS)
            registry.settle_generation(table_name, token)
