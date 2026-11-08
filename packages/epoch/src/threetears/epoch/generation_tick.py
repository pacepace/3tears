"""following collection write generations: one catch-up pass, and a watcher for a few keys.

A registry that follows a table (``CollectionRegistry.follow_generation``) keeps a mark: the last
write generation whose writes it has accounted for, moved on by the row broadcasts it hears. The
broadcasts are at most once, so something has to compare the mark with the table's generation as
the epoch bucket holds it, and catch the pod up when they disagree. That is all either function
here does: ``CollectionRegistry.catch_up_generation`` fetches the change record of each epoch the
pod did not hear in full and applies exactly its rows. A gap never drops a table:

- same incarnation, and an advance this pod did not hear every row of -- a broadcast was missed --
  so that epoch's record is read and its rows evicted;
- a record the epoch change log does not hold, or a generation from another incarnation (the
  bucket was replaced): the pod is behind on the table, serves what it holds, and is counted and
  logged until the record can be read;
- everything up to it heard: nothing happens.

A generation that cannot be read is none of those. Nothing is applied and the mark does not move;
the next pass tries again.

**:func:`generation_catchup_tick` is pure-async, one pass per call, with no internal polling: the
consumer's scheduler drives cadence**, for the reason :mod:`threetears.epoch.tick` gives -- the
package dependency arrow allows nothing else -- and with the same contract: one table's failure
does not abandon the rest. Whoever wires a registry to follow tables also schedules this pass, or
runs :func:`follow_generation_key` for each; following alone applies nothing.

**:func:`follow_generation_key` is the alternative for a pod following a few keys.** The broker
pushes each new generation through one named consumer per key, so a missed broadcast is noticed
when the generation moves rather than at the next pass. It costs a consumer per table per pod,
which is right for the four access tables and wrong for a registry following every table.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, runtime_checkable

from threetears.core.collections.epoch_change import (
    EPOCH_CHANGE_MAX_AGE,
    EpochChange,
    EpochChangeReader,
    EpochLogExtent,
    EpochQuiet,
    EpochWalk,
)
from threetears.core.collections.generation import GenerationVerdict, split_generation_token
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.exceptions import GenerationUnavailableError
from threetears.nats.change_log import ChangeLogState
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter

from threetears.epoch.generation import EpochGenerationReader

__all__ = [
    "DEFAULT_BROADCAST_GRACE",
    "GenerationWatcher",
    "QuietPass",
    "QuietReader",
    "follow_generation_key",
    "generation_catchup_tick",
    "quiet_marks_tick",
    "quiet_pass_interval",
]

log = get_logger(__name__)

#: how long :func:`follow_generation_key` waits for the row broadcasts of an advance it has just
#: been pushed before judging them missed. A writer advances the generation and THEN publishes its
#: rows, so the pushed generation routinely arrives first; judged at once, every heard write would
#: read as a missed one and fetch a change record the broadcasts were about to deliver.
DEFAULT_BROADCAST_GRACE: Final = timedelta(seconds=2)


class GenerationWatcher(Protocol):
    """what :func:`follow_generation_key` watches a table's generation through.

    :class:`~threetears.epoch.generation.EpochGenerationReader` is the platform's; a follower that
    wants to see each push (to keep a health state) wraps one.
    """

    def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        """the table's write generation as it stands, then each value it takes.

        :param table_name: the table
        :ptype table_name: str
        :return: the generations, ``None`` while the table has none
        :rtype: AsyncGenerator[str | None]
        """
        ...


#: how often that wait looks at the mark again.
_GRACE_POLL_SECONDS: Final = 0.02


async def generation_catchup_tick(registry: CollectionRegistry, reader: EpochGenerationReader) -> int:
    """run one catch-up pass over every table ``registry`` follows, returning how many it caught up.

    Each followed table's generation is read once and caught up
    (``CollectionRegistry.catch_up_generation``): the change record of every epoch the pod did not
    hear in full is read through ``reader`` and its rows evicted. A table whose record is not in the
    log is left behind, serving what it holds, and tried again at the next pass.

    **A generation that cannot be read is not a reset.** ``GenerationUnavailableError`` for a
    table is logged and that table is left exactly as it was: nothing applied, the mark unmoved.
    The pod goes on serving what it has, as it did before the pass, and the next pass reads again.

    **A failing table does not abandon the others.** Every table is attempted; any other exception
    is re-raised after the pass, the first one if there were several, so a consumer bug still
    surfaces and the consumer's own loop decides whether that ends its scheduling.

    A pass that reads a generation while that advance's broadcasts are still in flight sees them
    unheard and applies its record, evicting rows the broadcasts are about to evict anyway.

    **A table whose generation is kept in L2** (``L2_WRITE_GENERATION``) has none in the epoch bucket:
    it is judged by ``CollectionRegistry.catch_up_l2_generations``, which this pass runs after the
    others, reading each missed epoch's change record from beside the generation in L2.

    :param registry: the registry whose followed tables this pass judges
    :ptype registry: CollectionRegistry
    :param reader: reads each table's generation from the epoch bucket
    :ptype reader: EpochGenerationReader
    :return: how many tables this pass applied missed epochs to
    :rtype: int
    :raises Exception: the first exception other than ``GenerationUnavailableError`` raised for
        any table, after every other table has been attempted
    """
    caught_up = 0
    first_error: BaseException | None = None
    for table_name in registry.generation_marks.followed:
        if registry.generation_kept_in_l2(table_name):
            continue
        try:
            token = await reader.read(table_name)
            verdict = await registry.catch_up_generation(table_name, token, changes=reader)
        except GenerationUnavailableError as exc:
            log.warning(
                "write generation unavailable; the table is left as it is and the next pass reads again",
                extra={"extra_data": {"table": table_name, "error": str(exc)}},
            )
            continue
        # prawduct:allow prawduct/broad-except -- one table's failure must not abandon the
        # remaining tables in the same pass. re-raised below.
        except Exception as exc:  # noqa: BLE001
            contain(
                exc,
                log=log,
                what="write generation catch-up failed for a table; continuing with the rest of the pass",
                extra={"table": table_name},
            )
            if first_error is None:
                first_error = exc
            continue
        if verdict is GenerationVerdict.MISSED and not registry.behind_on(table_name):
            caught_up += 1
    try:
        caught_up += await registry.catch_up_l2_generations()
    # prawduct:allow prawduct/broad-except -- the L2-kept tables are one more table's worth of the pass;
    # re-raised below with the first error, like any table's.
    except Exception as exc:  # noqa: BLE001
        contain(exc, log=log, what="write generation catch-up failed for the tables kept in L2")
        if first_error is None:
            first_error = exc
    if first_error is not None:
        raise first_error
    return caught_up


async def follow_generation_key(
    registry: CollectionRegistry,
    reader: GenerationWatcher,
    table_name: str,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    changes: EpochChangeReader | None = None,
) -> None:
    """follow one table's write generation by watching its key, until cancelled.

    Starts following the table on ``registry`` and then judges every generation the broker pushes,
    as a catch-up pass would judge one it read. The first value pushed is the key's latest, so a
    watcher that starts, or whose consumer the watch replaced after a broker restart, needs no
    read of its own.

    An advance is pushed as soon as it is written, ahead of the row broadcasts the writer sends
    next. So a generation whose advances are not yet all accounted for is given ``grace`` for
    their rows to arrive before it is judged; only then is an advance still unheard a missed
    broadcast, and its change record is read and applied. A generation from another incarnation is
    never waited on. A record that cannot be read leaves the pod behind on the table, logged, and
    the watch goes on: the next generation pushed is caught up from the same mark.

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
    :ptype reader: GenerationWatcher
    :param table_name: the table to follow
    :ptype table_name: str
    :param grace: how long to wait for an advance's row broadcasts before judging them missed
    :ptype grace: timedelta
    :param changes: reads each missed epoch's change record; ``None`` reads through ``reader`` when it
        can (:class:`~threetears.epoch.generation.EpochGenerationReader` does), else every missed
        epoch leaves the pod behind
    :ptype changes: EpochChangeReader | None
    :return: nothing; it returns only if the watch ends
    :rtype: None
    :raises GenerationUnavailableError: when the bucket cannot be opened or watched
    :raises KvError: when the connection closes, so nothing can ever be pushed again
    """
    registry.follow_generation(table_name)
    marks = registry.generation_marks
    loop = asyncio.get_running_loop()
    records = changes if changes is not None else reader if isinstance(reader, EpochChangeReader) else None
    async with aclosing(reader.watch(table_name)) as generations:
        async for token in generations:
            deadline = loop.time() + grace.total_seconds()
            # the pushed epoch's broadcasts are given the grace to arrive, so the walk that follows
            # finds it heard and applies nothing for it
            while (
                token is not None
                and not marks.heard_in_full(table_name, token)
                and marks.judge(table_name, token) is GenerationVerdict.MISSED
                and loop.time() < deadline
            ):
                await asyncio.sleep(_GRACE_POLL_SECONDS)
            try:
                await registry.catch_up_generation(table_name, token, changes=records)
            except GenerationUnavailableError as exc:
                # NOSILENT: the registry recorded itself behind on the table and logged it; the next push catches up
                log.warning(
                    "an epoch's change record could not be read; behind on the table until the next generation",
                    extra={"extra_data": {"table": table_name, "error": str(exc)}},
                )


#: how many quiet passes a follower runs per retention window of the change log: a quiet table's mark is
#: moved several times before retention could take the epoch after it, so one late or failed pass is
#: never the one that lets it age out
_QUIET_PASSES_PER_RETENTION: Final = 4

_QUIET_MOVED = counter(
    "collection_epoch_quiet_marks_moved_total",
    description=(
        "times a follower's quiet pass moved a table's mark to the change log's head because the log held "
        "nothing of the table after it, so a quiet table never ages out of retention"
    ),
    label_names=("table",),
)


@runtime_checkable
class QuietReader(EpochChangeReader, Protocol):
    """what the quiet pass reads: the log's state and every table's newest epoch, in one pass.

    :class:`~threetears.epoch.generation.EpochGenerationReader` is one.
    """

    async def newest_epochs(self, tables: Sequence[str]) -> tuple[ChangeLogState | None, dict[str, int]]:
        """the log's state, then each table's newest epoch's sequence (``0`` for none).

        :param tables: the tables
        :ptype tables: Sequence[str]
        :return: the state (``None`` when the broker holds no log) and each table's newest
        :rtype: tuple[ChangeLogState | None, dict[str, int]]
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...


@dataclass(frozen=True, slots=True)
class QuietPass:
    """what one quiet pass did.

    :ivar moved: tables whose mark it moved to the log's head, the log holding nothing of them after it
    :ivar walked: tables the log held an epoch of after the mark, caught up through the ordinary walk
    :ivar behind: tables the registry is behind on after the pass
    :ivar interval: how long until the next pass: the log's retention over :data:`_QUIET_PASSES_PER_RETENTION`
    """

    moved: tuple[str, ...]
    walked: tuple[str, ...]
    behind: tuple[str, ...]
    interval: timedelta


def quiet_pass_interval(state: ChangeLogState | None) -> timedelta:
    """how long a follower waits between quiet passes: well inside the change log's retention.

    The log's own declared retention (``max_age``) over :data:`_QUIET_PASSES_PER_RETENTION`; where the log
    does not say, the retention a source declares by default (:data:`EPOCH_CHANGE_MAX_AGE`).

    :param state: the log's state, or ``None``
    :ptype state: ChangeLogState | None
    :return: the interval
    :rtype: timedelta
    """
    retention = state.max_age if state is not None and state.max_age else EPOCH_CHANGE_MAX_AGE
    return retention / _QUIET_PASSES_PER_RETENTION


class _QuietSnapshot:
    """the change log as one quiet pass read it: a vouch for each quiet table from one batch read.

    The judgement a catch-up runs asks the log what it can vouch for after a mark
    (:meth:`~threetears.core.collections.epoch_change.EpochLogExtent.quiet_after`); here that is answered
    from the pass's one read of the state and each table's newest epoch, instead of two reads per table.
    Every other read goes to the reader.
    """

    def __init__(self, reader: QuietReader, state: ChangeLogState, newest: dict[str, int]) -> None:
        """capture the pass's read; no I/O.

        :param reader: the reader the pass read through
        :ptype reader: QuietReader
        :param state: the log's state, read first
        :ptype state: ChangeLogState
        :param newest: each table's newest epoch, read after
        :ptype newest: dict[str, int]
        :return: nothing
        :rtype: None
        """
        self._reader = reader
        self._state = state
        self._newest = newest

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the record of one epoch, through the reader."""
        return await self._reader.change(table_name, token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """the table's epochs after a mark, through the reader."""
        return await self._reader.changes_after(table_name, after, upto)

    async def continues(self, incarnation: str) -> tuple[str, int | None] | None:
        """the lost log a restored one continues, through the reader when it can say."""
        if isinstance(self._reader, EpochLogExtent):
            return await self._reader.continues(incarnation)
        return None

    async def quiet_after(self, table_name: str, mark: str) -> EpochQuiet:
        """what the pass's read vouches for of ``table_name`` after ``mark``.

        The state was read before the table's newest epoch, so an epoch appended between the two is seen
        as the table's newest and the table is not vouched quiet; one appended after both lies past the
        state's head, and so past where the mark moves.

        :param table_name: the table
        :ptype table_name: str
        :param mark: the follower's mark
        :ptype mark: str
        :return: how far the mark may move, and whether the log has aged past it
        :rtype: EpochQuiet
        """
        parts = split_generation_token(mark)
        state = self._state
        if parts is None or parts[0] != state.incarnation:
            return EpochQuiet(through=None)
        aged_out = state.first_seq > parts[1] + 1 and state.last_seq > parts[1]
        quiet = self._newest.get(table_name, 0) <= parts[1] and state.last_seq > parts[1]
        return EpochQuiet(through=f"{state.incarnation}:{state.last_seq}" if quiet else None, aged_out=aged_out)


async def quiet_marks_tick(
    registry: CollectionRegistry, reader: QuietReader, tables: Sequence[str] | None = None
) -> QuietPass:
    """move every quiet table's mark to the change log's head, and catch up every table that is not quiet.

    One pass per registry, never one per table: the log's state and every followed table's newest epoch
    are read in one batch from NATS -- never from L3 -- and each table is judged against them. A table
    the log holds nothing of after its mark has its mark moved to the log's head, so it stays inside the
    log's retention however long it stays quiet, and is never later judged ``aged_out`` for a stretch in
    which nothing happened to it. A table the log holds an epoch of after its mark goes through the
    ordinary walk and apply (``CollectionRegistry.catch_up_generation``), so a mark never moves past a
    change it has not applied. A table kept in L2 (``L2_WRITE_GENERATION``) is not in the log, and is left
    to its own catch-up.

    Run it every :func:`quiet_pass_interval` (``QuietPass.interval``); :class:`~threetears.epoch.table_follow
    .TableFollower` does, for the tables it watches.

    :param registry: the registry whose followed tables the pass judges
    :ptype registry: CollectionRegistry
    :param reader: reads the change log
    :ptype reader: QuietReader
    :param tables: the followed tables to judge; ``None`` judges every table the registry follows
    :ptype tables: Sequence[str] | None
    :return: what the pass did
    :rtype: QuietPass
    :raises GenerationUnavailableError: when the log cannot be read; nothing is moved
    :raises Exception: the first failure other than an unreadable record, after every table was tried
    """
    followed = registry.generation_marks.followed
    judged = [
        table
        for table in (followed if tables is None else [each for each in tables if each in followed])
        if not registry.generation_kept_in_l2(table)
    ]
    state, newest = await reader.newest_epochs(judged)
    if state is None:
        return QuietPass(moved=(), walked=(), behind=(), interval=quiet_pass_interval(None))
    snapshot = _QuietSnapshot(reader, state, newest)
    marks = registry.generation_marks
    moved: list[str] = []
    walked: list[str] = []
    first_error: BaseException | None = None
    for table in judged:
        before = marks.recorded(table)
        token = f"{state.incarnation}:{newest.get(table, 0)}"
        try:
            verdict = await registry.catch_up_generation(table, token, changes=snapshot)
        except GenerationUnavailableError as exc:
            # NOSILENT: the registry recorded itself behind on the table and logged it; the next pass reads again
            log.warning(
                "a quiet pass could not read an epoch's change record; the table is left behind until the next pass",
                extra={"extra_data": {"table": table, "error": str(exc)}},
            )
            continue
        # prawduct:allow prawduct/broad-except -- one table's failure must not abandon the rest of the pass; re-raised below
        except Exception as exc:  # noqa: BLE001
            contain(
                exc, log=log, what="a quiet pass failed for a table; continuing with the rest", extra={"table": table}
            )
            first_error = first_error or exc
            continue
        after = marks.recorded(table)
        if verdict is GenerationVerdict.MISSED:
            walked.append(table)
        elif after != before and before is not None:
            moved.append(table)
            _QUIET_MOVED.labels(table=table).inc()
    behind = tuple(table for table in judged if registry.behind_on(table))
    result = QuietPass(moved=tuple(moved), walked=tuple(walked), behind=behind, interval=quiet_pass_interval(state))
    log.info(
        "quiet pass over the followed tables",
        extra={
            "extra_data": {
                "tables": len(judged),
                "moved": len(result.moved),
                "walked": list(result.walked),
                "behind": list(result.behind),
                "next_in_seconds": result.interval.total_seconds(),
            }
        },
    )
    if first_error is not None:
        raise first_error
    return result
