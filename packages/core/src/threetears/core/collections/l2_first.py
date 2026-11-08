"""an L2-first collection commits its writes in NATS, and a saver makes L3 match.

**The commit.** A collection that declares :attr:`~threetears.core.collections.base.BaseCollection.l2_first`
first appends one epoch record carrying the row and the revision of its L2 key the writer read
(:meth:`L2FirstEpochs.advance_continuing`): that record landing in the epoch change log is the commit. Only
then is the row applied to L2, at that revision; a writer that dies between the two leaves the key where it
was, and the saver applies the row from the record. A follower that hears the broadcast, or fetches the
record after missing it, reads the row from L2. Every L2 bucket is memory storage, replicated three ways;
nothing here is on disk.

**The save.** L3 is made to match L2 key by key: a row L2 holds is upserted, a key L2 no longer holds is
deleted, and the table's change-note trigger writes each row's note in the same L3 transaction
(:mod:`threetears.core.collections.change_notes`), as it does for every noted table. Doing it twice is
harmless, so whoever does it -- the writer, or the saver after the writer died -- needs no coordination
beyond an epoch it has finished with being marked.

- **Waiting** (``L2First()``, the default): the writer saves to L3 before it returns, so a row that
  exists nowhere else survives a full NATS restart, and L3's constraints decide whether it is saved. A
  refusal is the caller's error, and the L2 write is withdrawn (its prior value put back at the revision
  the write won) and announced by one more epoch naming the key.
- **Not waiting** (``L2First(wait_for_l3=False, refetchable_from=...)``): for data its source can send
  again. The writer returns at the commit and the saver saves it. An L3 refusal is an ERROR, counted,
  and the saver moves on: one refused row never holds up the rest.

The saver (:class:`L3Saver`) walks each table's epochs after its saved position, which it keeps in L2.
An epoch the writer already saved is marked "settled" and skipped; any other has its pending rows made to
match in L3.

**L2 lost.** A pod never reads L3 for an L2-first collection. While L2 holds the table complete (the
refiller's mark), a key L2 does not hold is absent; while it does not, a lookup raises
:class:`L2LostError` and the pod serves what it holds, reporting itself behind through the epoch catch-up.
When NATS loses the epoch change log, the change-note keeper
(:class:`~threetears.core.collections.change_notes.ChangeNoteKeeper`) puts it back from the L3 change notes
under a new incarnation that continues the lost one, as for every other table. The refiller
(:class:`L2Refiller`) then puts back, from L3, every row (only where L2 holds nothing for the key, so a write
since the loss is never overwritten), and last the mark that L2 is complete. A pod catches up through the
ordinary epoch path, walking the restored log. Nothing drops a table and nothing is rebuilt.

**One claimed worker.** The saver and the refiller are run by one process at a time, under a
:class:`~threetears.core.coordination.lease.KVLease` (:func:`run_l2_first_worker`). Two at once is safe,
only wasteful.

**Without L2** the same collection writes straight to L3, as any collection does.
"""

from __future__ import annotations

import asyncio
import time
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk
from threetears.core.collections.generation import split_generation_token
from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from threetears.core.collections.base import BaseCollection
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.coordination.lease import HeldLease

__all__ = [
    "L2_FIRST",
    "L2First",
    "L2FirstEpochs",
    "L2FirstTableStatus",
    "L2Refiller",
    "L3Saver",
    "SaverPass",
    "complete_mark",
    "l2_first_status",
    "run_l2_first_worker",
]

log = get_logger(__name__)


def _worker_last_pass() -> Any:
    return gauge(
        "threetears_l2_first_worker_last_pass_timestamp_seconds",
        description="unix time of the L2-first worker's last whole pass (refill and save) in this process; alert when "
        "no process has moved it for several passes",
    )


#: rows the refiller reads from L3 per statement, under the L3 rail's row cap.
_REFILL_PAGE_ROWS: Final = 500

#: the SQLSTATE class of an integrity constraint violation (unique, foreign key, not null, check).
_INTEGRITY_SQLSTATE_CLASS: Final = "23"

#: the counters, fetched where they count rather than registered at import (a module reloaded under a
#: test would otherwise register them twice): name, help, labels
_SAVED: Final = (
    "threetears_l2_first_saved_total",
    "rows the L3 saver made L3 match L2 for, after their writer did not",
)
_REFUSED: Final = (
    "threetears_l2_first_l3_refused_total",
    "committed L2-first writes L3 refused (a constraint); alert on any increase",
)
_MISSING: Final = (
    "threetears_l2_first_records_missing_total",
    "epochs the L3 saver found no change record for; a write that did not wait may be unsaved",
)
_REFILLED: Final = (
    "threetears_l2_first_refills_total",
    "tables the refiller put back into L2 from L3 because L2 did not hold them complete",
)


def _count(metric: tuple[str, str], table: str) -> None:
    """add one to a per-table L2-first counter.

    :param metric: the counter's name and help
    :ptype metric: tuple[str, str]
    :param table: the table label
    :ptype table: str
    :return: nothing
    :rtype: None
    """
    counter(metric[0], description=metric[1], label_names=("table",)).labels(table=table).inc()


@dataclass(frozen=True, slots=True)
class L2First:
    """a collection that commits its writes in NATS (L2) and saves them to L3 from there.

    :ivar wait_for_l3: whether a write returns only once L3 has saved it. ``True`` (the default) for
        data that exists nowhere else, and so that L3's constraints decide "saved"
    :ivar refetchable_from: where the data can be fetched again from, which is what makes not waiting
        safe: a write lost with NATS before it reached L3 is sent again by its source. Required when
        ``wait_for_l3`` is ``False``
    """

    wait_for_l3: bool = True
    refetchable_from: str | None = None

    def __post_init__(self) -> None:
        """refuse a collection that does not wait without saying where its data comes back from.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``wait_for_l3`` is ``False`` and ``refetchable_from`` is empty
        """
        if not self.wait_for_l3 and not (self.refetchable_from or "").strip():
            raise ValueError(
                "L2First(wait_for_l3=False) needs refetchable_from: a write that does not wait for L3 is lost "
                "with NATS until its source sends it again, so the declaration names that source"
            )


#: commit in NATS and wait for L3: the declaration for data that exists nowhere else.
L2_FIRST: Final = L2First()


@runtime_checkable
class L2FirstEpochs(Protocol):
    """the epoch log an L2-first collection commits through, and the saver and refiller read."""

    async def read(self, table_name: str) -> str | None:
        """the table's current generation, or ``None`` when the store holds none; never mints one.

        :param table_name: the table
        :ptype table_name: str
        :return: the generation token, ``{incarnation}:{count}``, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the store cannot be read
        """
        ...

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the change record of the advance that wrote ``token``, or ``None`` when the log does not hold it.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record, or ``None``
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def advance_continuing(self, table_name: str, rows: Sequence[ChangedRow]) -> str:
        """advance the table's generation once and store ``rows`` as the advance's change: the commit.

        Unlike an ordinary advance this never mints a new incarnation: a table with no generation is
        one whose L2 was lost, and an epoch started there is one no pod could follow from its mark.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the write changed
        :ptype rows: Sequence[ChangedRow]
        :return: the generation token the advance wrote
        :rtype: str
        :raises L2LostError: when the table has no generation
        :raises GenerationUnavailableError: when the advance or its record failed
        """
        ...

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """every epoch of the table after ``after`` and up to ``upto``, oldest first, as the log holds them.

        :param table_name: the table
        :ptype table_name: str
        :param after: the epoch to read after
        :ptype after: str
        :param upto: the newest epoch to read
        :ptype upto: str
        :return: the walk, saying whether the log may be missing one of them
        :rtype: EpochWalk
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def observe_change_log_fill(self) -> float | None:
        """measure how full the epoch change log is, export it, and warn when it nears its bound.

        :return: the fraction of its byte bound the log holds, or ``None`` when it cannot be measured
        :rtype: float | None
        """
        ...


def _is_l3_refusal(exc: BaseException) -> bool:
    """whether an L3 write failed because L3 refused the row (a constraint), rather than being unreachable.

    A refusal is final: the same row will be refused again. Anything else -- a connection lost, a
    timeout -- is retried. The SQLSTATE class ``23`` (integrity constraint violation) is the test, as
    asyncpg and every DB-API driver report it.

    :param exc: the failure
    :ptype exc: BaseException
    :return: ``True`` for a refusal
    :rtype: bool
    """
    state = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
    return isinstance(state, str) and state.startswith(_INTEGRITY_SQLSTATE_CLASS)


def _parts(table_name: str, token: str) -> tuple[str, int]:
    """a token's incarnation and count.

    :param table_name: the table, for the error
    :ptype table_name: str
    :param token: ``{incarnation}:{count}``
    :ptype token: str
    :return: ``(incarnation, count)``
    :rtype: tuple[str, int]
    :raises GenerationUnavailableError: when the token is not followable
    """
    parts = split_generation_token(token)
    if parts is None:
        raise GenerationUnavailableError(f"generation {token!r} of {table_name!r} is not {{incarnation}}:{{count}}")
    return parts


def l2_first_collections(registry: CollectionRegistry) -> list[BaseCollection[Any]]:
    """every collection of ``registry`` that declares itself L2-first and has L2 and L3 to work between.

    :param registry: the registry
    :ptype registry: CollectionRegistry
    :return: the collections, in table order
    :rtype: list[BaseCollection[Any]]
    """
    found = []
    for table in sorted(registry.registered_tables()):
        collection = registry.get_collection(table)
        if collection is not None and collection.writes_l2_first:
            found.append(collection)
    return found


@dataclass(slots=True)
class SaverPass:
    """what one pass of the L3 saver did.

    :ivar saved: rows it made L3 match L2 for
    :ivar settled: epochs it found already saved by their writer
    :ivar rejected: rows L3 refused, each counted and logged at ERROR
    :ivar missing: epochs whose change record the log no longer held
    :ivar change_log_fill: how full the epoch change log was, or ``None`` when not measured
    """

    saved: int = 0
    settled: int = 0
    rejected: int = 0
    missing: int = 0
    change_log_fill: float | None = None


@dataclass(slots=True)
class _Deferred:
    """the epochs a pass left once because their L3 refusal may be their own writer's, still settling."""

    tokens: set[str] = field(default_factory=set)


class L3Saver:
    """makes L3 match L2 for every epoch an L2-first collection committed."""

    def __init__(self, registry: CollectionRegistry, *, epochs: L2FirstEpochs) -> None:
        """capture what the saver works between; no I/O.

        :param registry: holds the L2-first collections, wired to L2 and L3
        :ptype registry: CollectionRegistry
        :param epochs: the epoch log the collections commit through
        :ptype epochs: L2FirstEpochs
        :return: nothing
        :rtype: None
        """
        self._registry = registry
        self._epochs = epochs
        self._deferred = _Deferred()

    async def run_pass(self) -> SaverPass:
        """save every epoch after each table's saved position, then measure the change log.

        Every table is attempted. A table whose L3 write failed for any reason but a refusal stops at
        that epoch, keeping its position, and the first such failure is raised after the other tables,
        so a dead L3 is seen and nothing is skipped.

        :return: what the pass did
        :rtype: SaverPass
        :raises Exception: the first L3 or L2 failure that was not a refusal, after every table was tried
        """
        result = SaverPass()
        failure: BaseException | None = None
        for collection in l2_first_collections(self._registry):
            try:
                await self._save_table(collection, result)
            # prawduct:allow prawduct/broad-except -- one table's failure keeps its position; the rest are saved and it is raised after
            except Exception as exc:
                failure = failure or exc
                contain(
                    exc,
                    log=log,
                    what="the L3 saver stopped on a table; its position is kept and the next pass resumes there",
                    extra={"table": collection.table_name, "error": f"{type(exc).__name__}: {exc}"},
                    level="error",
                )
        result.change_log_fill = await self._epochs.observe_change_log_fill()
        if failure is not None:
            raise failure
        return result

    async def _save_table(self, collection: BaseCollection[Any], result: SaverPass) -> None:
        """save one table's epochs after its saved position.

        :param collection: the L2-first collection
        :ptype collection: BaseCollection[Any]
        :param result: the pass's tally
        :ptype result: SaverPass
        :return: nothing
        :rtype: None
        """
        table = collection.table_name
        head = await self._epochs.read(table)
        if head is None:
            # L2 lost, or never written: the refiller puts it back first
            return
        incarnation, _ = _parts(table, head)
        start = await self._start(collection, incarnation)
        # a table's epochs are walked, never counted: other tables' epochs sit between them in the log
        walk = await self._epochs.changes_after(table, f"{incarnation}:{start}", head)
        if not walk.complete:
            result.missing += 1
            _count(_MISSING, table)
            log.error(
                "the change log may have lost an epoch of an L2-first table before the saver reached it; a write "
                "that did not wait for L3 may never reach it, until its source sends it again",
                extra={"extra_data": {"table": table, "first_missing": walk.first_missing, "head": head}},
            )
        for change in walk.changes:
            if not await self._save_epoch(collection, change, result):
                return
            await collection.write_l2_first_state(change.token.encode("utf-8"), "saved")  # convert at border: KV value
        await collection.write_l2_first_state(head.encode("utf-8"), "saved")  # convert at border: KV value

    async def _start(self, collection: BaseCollection[Any], incarnation: str) -> int:
        """the count after which this table's epochs are unsaved.

        The saved position, under the head's incarnation. The refiller sets it whenever it completes a
        table, so a position missing, or under another incarnation, means none was ever kept for this
        one: every epoch of it is walked. Never the newest epoch a writer saved: a waiting write saves its
        own epoch while a write before it that did not wait may still be unsaved.

        :param collection: the collection
        :ptype collection: BaseCollection[Any]
        :param incarnation: the head's incarnation
        :ptype incarnation: str
        :return: the count
        :rtype: int
        """
        held = await collection.read_l2_first_state("saved")
        parts = None if held is None else split_generation_token(held.decode("utf-8"))
        return parts[1] if parts is not None and parts[0] == incarnation else 0

    async def _save_epoch(self, collection: BaseCollection[Any], change: EpochChange, result: SaverPass) -> bool:
        """save one epoch, or say the table's pass stops here.

        :param collection: the collection
        :ptype collection: BaseCollection[Any]
        :param change: the epoch's record, as the walk read it
        :ptype change: EpochChange
        :param result: the pass's tally
        :ptype result: SaverPass
        :return: whether the epoch is done with and the position may move past it
        :rtype: bool
        """
        table = collection.table_name
        token = change.token
        incarnation, count = _parts(table, token)
        if await collection.read_l2_first_state("settled", incarnation, str(count)) is not None:
            result.settled += 1
            await collection.drop_l2_first_state("settled", incarnation, str(count))
            return True
        for row in change.rows:
            if not row.l3_pending:
                continue
            try:
                await collection.carry_to_l3(row)
            # prawduct:allow prawduct/broad-except -- a refusal is counted and passed; anything else stops the table here and is raised by run_pass
            except Exception as exc:
                if not _is_l3_refusal(exc):
                    raise
                if await self._refused_by_its_writer(collection, incarnation, count, token):
                    result.settled += 1
                    return True
                if collection.waits_for_l3 and token not in self._deferred.tokens:
                    # a waiting writer may be withdrawing this very write: look again next pass
                    self._deferred.tokens.add(token)
                    return False
                self._deferred.tokens.discard(token)
                result.rejected += 1
                _count(_REFUSED, table)
                log.error(
                    "L3 refused a committed L2-first write; L2 keeps it and L3 does not. Its source must send a "
                    "row L3 accepts",
                    extra={
                        "extra_data": {
                            "table": table,
                            "epoch": token,
                            "ids": list(row.ids),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
                continue
            result.saved += 1
            _count(_SAVED, table)
        return True

    async def _refused_by_its_writer(
        self, collection: BaseCollection[Any], incarnation: str, count: int, token: str
    ) -> bool:
        """whether the epoch L3 just refused was settled by its own writer meanwhile.

        :param collection: the collection
        :ptype collection: BaseCollection[Any]
        :param incarnation: the epoch's incarnation
        :ptype incarnation: str
        :param count: the epoch's count
        :ptype count: int
        :param token: the epoch
        :ptype token: str
        :return: ``True`` when the writer marked it settled
        :rtype: bool
        """
        if await collection.read_l2_first_state("settled", incarnation, str(count)) is None:
            return False
        await collection.drop_l2_first_state("settled", incarnation, str(count))
        self._deferred.tokens.discard(token)
        return True


class L2Refiller:
    """puts an L2-first table back into L2 from L3 when L2 does not hold it complete."""

    def __init__(self, registry: CollectionRegistry, *, epochs: L2FirstEpochs) -> None:
        """capture what the refiller works between; no I/O.

        :param registry: holds the L2-first collections, wired to L2 and L3
        :ptype registry: CollectionRegistry
        :param epochs: the epoch log the collections commit through
        :ptype epochs: L2FirstEpochs
        :return: nothing
        :rtype: None
        """
        self._registry = registry
        self._epochs = epochs

    async def run_pass(self) -> int:
        """refill every L2-first table L2 does not hold complete, once the change log is there.

        A lost change log is put back by the change-note keeper, not here: until it is, no table has a
        generation to refill under, and this pass refills nothing.

        :return: how many tables were refilled
        :rtype: int
        """
        refilled = 0
        for collection in l2_first_collections(self._registry):
            if await self.refill(collection):
                refilled += 1
        return refilled

    async def refill(self, collection: BaseCollection[Any]) -> bool:
        """put one table back into L2 from L3, when L2 does not hold it complete.

        Every row L2 holds nothing for; then the saver's position, at the table's generation (every epoch up
        to it is in L3, or was lost with NATS before it got there); then the mark that L2 holds the table
        complete, last, so nothing reads "complete" before it is.

        :param collection: the L2-first collection
        :ptype collection: BaseCollection[Any]
        :return: whether it was refilled; ``False`` when L2 already held it complete, or the change log is
            still absent
        :rtype: bool
        """
        table = collection.table_name
        head = await self._epochs.read(table)
        if head is None or await collection.l2_complete():
            return False
        rows = await collection.refill_l2_from_l3(page_size=_REFILL_PAGE_ROWS)
        if await collection.read_l2_first_state("saved") is None:
            await collection.write_l2_first_state(head.encode("utf-8"), "saved")  # convert at border: KV value
        await collection.mark_l2_complete(head)
        _count(_REFILLED, table)
        log.warning(
            "an L2-first table was put back into L2 from L3; a pod behind on it catches up from here",
            extra={"extra_data": {"table": table, "rows": rows, "generation": head}},
        )
        return True


@dataclass(frozen=True, slots=True)
class L2FirstTableStatus:
    """where one L2-first table stands, as an operator reads it.

    :ivar table: the table
    :ivar generation: its epoch head in NATS; ``None`` while the change log is lost (the change-note keeper
        has not put it back)
    :ivar complete: whether L2 holds the table complete; ``False`` while lost or before the first refill
    :ivar saved_through: the epoch the saver has finished with; ``None`` before its first pass
    :ivar unsaved_epochs: epochs after it up to the head, which the saver has yet to pass (a waiting
        write's own epoch counts until the saver passes its mark); ``None`` when either is unknown
    """

    table: str
    generation: str | None
    complete: bool
    saved_through: str | None
    unsaved_epochs: int | None


async def l2_first_status(registry: CollectionRegistry, *, epochs: L2FirstEpochs) -> list[L2FirstTableStatus]:
    """where every L2-first table of ``registry`` stands: lost or complete, and how far the saver is behind.

    Reads only: one generation and two L2 keys per table. The operator's script
    prints it; a table with no generation is waiting for the change-note keeper, and one whose unsaved epochs keep
    growing has no saver running.

    :param registry: holds the L2-first collections, wired to L2 and L3
    :ptype registry: CollectionRegistry
    :param epochs: the epoch log the collections commit through
    :ptype epochs: L2FirstEpochs
    :return: one status per table, in table order
    :rtype: list[L2FirstTableStatus]
    """
    found = []
    for collection in l2_first_collections(registry):
        table = collection.table_name
        head = await epochs.read(table)
        held = await collection.read_l2_first_state("saved")
        saved = None if held is None else held.decode("utf-8")
        head_parts = None if head is None else split_generation_token(head)
        saved_parts = None if saved is None else split_generation_token(saved)
        unsaved = (
            head_parts[1] - saved_parts[1]
            if head_parts is not None and saved_parts is not None and head_parts[0] == saved_parts[0]
            else None
        )
        found.append(
            L2FirstTableStatus(
                table=table,
                generation=head,
                complete=await collection.l2_complete(),
                saved_through=saved,
                unsaved_epochs=unsaved,
            )
        )
    return found


async def run_l2_first_worker(
    held: HeldLease, *, saver: L3Saver, refiller: L2Refiller, every: timedelta = timedelta(seconds=1)
) -> None:
    """run the refiller, then the saver, every ``every``, for as long as ``held`` is held.

    The one claimed worker: take the lease (``KVLease.hold``) and hand it here. A pass that fails is
    logged and the next pass resumes where it stopped.

    :param held: the lease that makes this process the worker
    :ptype held: HeldLease
    :param saver: the L3 saver
    :ptype saver: L3Saver
    :param refiller: the L2 refiller
    :ptype refiller: L2Refiller
    :param every: the pause between passes
    :ptype every: timedelta
    :return: nothing, once the lease is lost
    :rtype: None
    """
    while held.lost_reason is None:
        try:
            await refiller.run_pass()
            await saver.run_pass()
            _worker_last_pass().set(time.time())
        # prawduct:allow prawduct/broad-except -- the worker outlives one failed pass; the next resumes where it stopped
        except Exception as exc:
            contain(exc, log=log, what="an L2-first worker pass failed; the next pass resumes", level="error")
        await asyncio.sleep(every.total_seconds())


def complete_mark(token: str | None) -> bytes:
    """the value of the mark that L2 holds a table complete.

    :param token: the generation the refill completed under
    :ptype token: str | None
    :return: the stored mark
    :rtype: bytes
    """
    return json.dumps({"generation": token}).encode("utf-8")
