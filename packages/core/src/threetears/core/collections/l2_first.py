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
from threetears.core.collections.l2_degrade import degrade_l2
from threetears.core.declared import whole_read
from threetears.core.exceptions import GenerationUnavailableError, L2LostError, L2ScopeNotConfiguredError
from threetears.nats.errors import KvError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from threetears.core.collections.base import BaseCollection
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.coordination.lease import HeldLease

__all__ = [
    "L2_FIRST",
    "L2First",
    "L2FirstEpochs",
    "L2FirstSupport",
    "L2FirstTableStatus",
    "L2Refiller",
    "L3Saver",
    "SaverPass",
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


#: how many of a table's epochs the saver saves at once: each a read of L2 and a write to L3, so a backlog of a
#: thousand epochs is not two thousand round trips one after another
_SAVE_CONCURRENCY: Final = 1000

#: how many L2 reads a many-row carry makes at once
_CARRY_CONCURRENCY: Final = 32


@dataclass(slots=True)
class _Deferred:
    """the epochs a pass left once because their L3 refusal may be their own writer's, still settling."""

    tokens: set[str] = field(default_factory=set)


class L3Saver:
    """makes L3 match L2 for every epoch an L2-first collection committed."""

    def __init__(
        self, registry: CollectionRegistry, *, epochs: L2FirstEpochs, batched_reads: bool | None = None
    ) -> None:
        """capture what the saver works between; no I/O.

        :param registry: holds the L2-first collections, wired to L2 and L3
        :ptype registry: CollectionRegistry
        :param epochs: the epoch log the collections commit through
        :ptype epochs: L2FirstEpochs
        :param batched_reads: read a window's L2 rows in batched direct gets. **Needs the body-carried direct get
            granted to this process's NATS connection**, which an infrastructure identity holds and a pod never
            does (:meth:`~threetears.nats.kv.NatsKvBucket.get_latest_many`); a refused batch is probed briefly and
            the bucket read a key at a time after. ``None`` (the default) is off, except under
            :func:`run_l2_first_worker`, whose runner is the infrastructure identity, and which turns it on
        :ptype batched_reads: bool | None
        :return: nothing
        :rtype: None
        """
        self._registry = registry
        self._epochs = epochs
        self._deferred = _Deferred()
        self.batched_reads = batched_reads

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
        # each epoch's carry is its own (carry_to_l3 writes L2's value as it is now, safe repeated and out of
        # order), so a few epochs are saved at once, not one read of L2 and one write to L3 after another; the
        # position still moves past an epoch only once it and every epoch before it are done
        changes = list(walk.changes)
        for start in range(0, len(changes), _SAVE_CONCURRENCY):
            window = changes[start : start + _SAVE_CONCURRENCY]
            if await self._save_window_at_once(collection, window, result):
                await collection.l2_first_support.write_state(
                    window[-1].token.encode("utf-8"), "saved"
                )  # convert at border: KV value
                continue
            outcomes = await asyncio.gather(
                *(self._save_epoch(collection, change, result) for change in window), return_exceptions=True
            )
            for change, outcome in zip(window, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    raise outcome
                if not outcome:
                    return
                await collection.l2_first_support.write_state(
                    change.token.encode("utf-8"), "saved"
                )  # convert at border: KV value
        await collection.l2_first_support.write_state(head.encode("utf-8"), "saved")  # convert at border: KV value

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
        held = await collection.l2_first_support.read_state("saved")
        parts = None if held is None else split_generation_token(held.decode("utf-8"))
        return parts[1] if parts is not None and parts[0] == incarnation else 0

    async def _save_window_at_once(
        self, collection: BaseCollection[Any], window: Sequence[EpochChange], result: SaverPass
    ) -> bool:
        """save a window of epochs in the collection's many-row save, when every row of them can go that way.

        The epochs a writer settled are let go as :meth:`_save_epoch` lets them go; the rest's rows are carried
        at once (:meth:`L2FirstSupport.carry_many_to_l3`). Anything that path does not take (a key L2 holds no
        row for, a dead writer's row to repair, a collection with no many-row save) or that L3 refuses or fails
        answers ``False`` with no epoch counted, and the caller saves the window an epoch at a time, as it
        always did: a carry is safe repeated.

        :param collection: the collection
        :ptype collection: BaseCollection[Any]
        :param window: the epochs, in order
        :ptype window: Sequence[EpochChange]
        :param result: the pass's tally
        :ptype result: SaverPass
        :return: whether every epoch of the window is done with
        :rtype: bool
        """
        support = collection.l2_first_support
        # the window's settled epochs from one listing of the marks, not a read an epoch
        marks = await support.settled_marks()
        settled: list[tuple[str, str]] = []
        rows: list[ChangedRow] = []
        for change in window:
            incarnation, number = _parts(collection.table_name, change.token)
            if (incarnation, str(number)) in marks:
                settled.append((incarnation, str(number)))
            else:
                rows.extend(row for row in change.rows if row.l3_pending)
        try:
            carried = await support.carry_many_to_l3(rows, batched=bool(self.batched_reads)) if rows else True
        # prawduct:allow prawduct/broad-except -- the window is saved again an epoch at a time, which meets each failure as it always did
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="an L2-first window could not be saved at once; saving it an epoch at a time",
                extra={"table": collection.table_name, "error": f"{type(exc).__name__}: {exc}"},
            )
            return False
        if not carried:
            return False
        for incarnation, settled_number in settled:
            result.settled += 1
            await support.drop_state("settled", incarnation, settled_number)
        result.saved += len(rows)
        for _ in rows:
            _count(_SAVED, collection.table_name)
        return True

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
        if await collection.l2_first_support.read_state("settled", incarnation, str(count)) is not None:
            result.settled += 1
            await collection.l2_first_support.drop_state("settled", incarnation, str(count))
            return True
        for row in change.rows:
            if not row.l3_pending:
                continue
            try:
                await collection.l2_first_support.carry_to_l3(row)
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
        if await collection.l2_first_support.read_state("settled", incarnation, str(count)) is None:
            return False
        await collection.l2_first_support.drop_state("settled", incarnation, str(count))
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
            if await whole_read(
                self.refill(collection),
                kind="cold_start",
                reason="L2 holds nothing for this table (its first fill, or NATS lost the memory bucket and the complete mark with it)",
                ruling="Pace 2026-11-12: the one full load is a true no-data cold start",
                proof="packages/epoch/tests/unit/test_l2_first.py::test_the_refill_reads_l3_only_while_l2_holds_no_data",
                speed="packages/epoch/tests/integration/test_l2_refill_speed_live.py::test_refill_from_l3_speed_at_real_size",
            ):
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
        if head is None or await collection.l2_first_support.complete():
            return False
        rows = await whole_read(
            collection.l2_first_support.refill_from_l3(page_size=_REFILL_PAGE_ROWS),
            kind="cold_start",
            reason="L2 holds nothing for this table (its first fill, or NATS lost the memory bucket and the complete mark with it)",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start",
            proof="packages/epoch/tests/unit/test_l2_first.py::test_the_refill_reads_l3_only_while_l2_holds_no_data",
            speed="packages/epoch/tests/integration/test_l2_refill_speed_live.py::test_refill_from_l3_speed_at_real_size",
        )
        if await collection.l2_first_support.read_state("saved") is None:
            await collection.l2_first_support.write_state(head.encode("utf-8"), "saved")  # convert at border: KV value
        await collection.l2_first_support.mark_complete(head)
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
        held = await collection.l2_first_support.read_state("saved")
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
                complete=await collection.l2_first_support.complete(),
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

    The worker is run by an infrastructure identity, never a pod, so its saver reads in batched direct gets
    unless it was built saying not to (:class:`L3Saver` ``batched_reads``): a connection without the grant
    is probed once and read a key at a time.

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
    if getattr(saver, "batched_reads", False) is None:
        saver.batched_reads = True
    while held.lost_reason is None:
        try:
            await refiller.run_pass()
            await saver.run_pass()
            _worker_last_pass().set(time.time())
        # prawduct:allow prawduct/broad-except -- the worker outlives one failed pass; the next resumes where it stopped
        except Exception as exc:
            contain(exc, log=log, what="an L2-first worker pass failed; the next pass resumes", level="error")
        await asyncio.sleep(every.total_seconds())


#: how long the marker a dropped L2-first state key leaves lives: one settled epoch is one key, so a
#: marker kept for ever would fill the bucket with one per epoch.
_L2_FIRST_STATE_MARKER_TTL: Final = timedelta(seconds=5)


class L2FirstSupport:
    """what the saver and the refiller do to one L2-first collection: its bookkeeping, a save from L2, a refill.

    Owned by the collection (``BaseCollection.l2_first_support``), which hands it the operations it needs, so
    none of them is public on every collection. Only this module's workers use it.
    """

    def __init__(
        self,
        *,
        table_name: str,
        scope_of: Callable[[], str | None],
        ensure_kv: Callable[[], Awaitable[Any]],
        l2_key: Callable[[Any], str],
        apply: Callable[[Any, Any, bytes | None, int], Awaitable[int | None]],
        decode: Callable[[bytes], Any],
        is_expired: Callable[[dict[str, Any]], bool],
        save: Callable[..., Awaitable[int]],
        delete: Callable[[Any], Awaitable[None]],
        pages: Callable[..., AsyncIterator[list[dict[str, Any]]]],
        id_of_changed: Callable[[ChangedRow], Any],
        id_of_row: Callable[[dict[str, Any]], Any],
        entry_lifetime: Callable[[dict[str, Any]], timedelta | None],
        encode: Callable[[dict[str, Any]], bytes],
        save_many: Callable[[list[dict[str, Any]]], Awaitable[int]] | None = None,
    ) -> None:
        """capture the collection's operations; no I/O.

        :param table_name: the collection's table
        :ptype table_name: str
        :param scope_of: the registry's key scope, read when a key is built
        :ptype scope_of: Callable[[], str | None]
        :param ensure_kv: the collection's L2 bucket, bound on first use
        :ptype ensure_kv: Callable[[], Awaitable[Any]]
        :param l2_key: a row's L2 key
        :ptype l2_key: Callable[[Any], str]
        :param apply: applies a committed value to its key at a revision
        :ptype apply: Callable[[Any, Any, bytes | None, int], Awaitable[int | None]]
        :param decode: decodes an L2 value
        :ptype decode: Callable[[bytes], Any]
        :param is_expired: whether a row is past its expiry
        :ptype is_expired: Callable[[dict[str, Any]], bool]
        :param save: the collection's L3 save
        :ptype save: Callable[..., Awaitable[int]]
        :param delete: the collection's L3 delete
        :ptype delete: Callable[[Any], Awaitable[None]]
        :param pages: the collection's L3 rows, a page at a time
        :ptype pages: Callable[..., AsyncIterator[list[dict[str, Any]]]]
        :param id_of_changed: a changed row's key
        :ptype id_of_changed: Callable[[ChangedRow], Any]
        :param id_of_row: a whole row's key
        :ptype id_of_row: Callable[[dict[str, Any]], Any]
        :param entry_lifetime: an L2 entry's lifetime for a row
        :ptype entry_lifetime: Callable[[dict[str, Any]], timedelta | None]
        :param encode: a row as its L2 value
        :ptype encode: Callable[[dict[str, Any]], bytes]
        :param save_many: the collection's L3 save of many rows in a few statements, as :paramref:`save` writes
            each; ``None`` for a collection that saves a row at a time
        :ptype save_many: Callable[[list[dict[str, Any]]], Awaitable[int]] | None
        """
        self._table = table_name
        self._scope = scope_of
        self._ensure_kv = ensure_kv
        self._l2_key = l2_key
        self._apply = apply
        self._decode = decode
        self._is_expired = is_expired
        self._save = save
        self._delete = delete
        self._l3_pages = pages
        self._id_of_changed = id_of_changed
        self._id_of_row = id_of_row
        self._entry_lifetime = entry_lifetime
        self._encode = encode
        self._save_many = save_many

    def state_key(self, *parts: str) -> str:
        """the L2 key of one piece of this table's L2-first bookkeeping, under the principal's scope.

        ``{scope}._l2first.{table}.{parts...}``: in the same bucket and under the same grant as the rows,
        and never a row's key, since ``_l2first`` cannot be a table name.

        :param parts: the piece's name, as key tokens
        :ptype parts: str
        :return: the key
        :rtype: str
        :raises L2ScopeNotConfiguredError: when the registry carries no ``kv_key_scope``
        """
        scope = self._scope()
        if scope is None:
            raise L2ScopeNotConfiguredError(f"{self._table}: no kv_key_scope, so no L2-first state key")
        return ".".join((scope, "_l2first", self._table, *parts))

    async def read_state(self, *parts: str) -> bytes | None:
        """one piece of this table's L2-first bookkeeping, or ``None`` when L2 holds none.

        :param parts: the piece's name
        :ptype parts: str
        :return: its value
        :rtype: bytes | None
        :raises KvError: when L2 cannot be read
        """
        kv = await self._ensure_kv()
        if kv is None:
            return None
        value: bytes | None = await kv.get(key=self.state_key(*parts))
        return value

    async def settled_marks(self) -> set[tuple[str, str]]:
        """every epoch of this table its own writer marked settled, as ``(incarnation, count)``: one listing, not a
        read an epoch.

        :return: the marked epochs
        :rtype: set[tuple[str, str]]
        :raises KvError: when L2 cannot be read
        """
        kv = await self._ensure_kv()
        if kv is None:
            return set()
        prefix = self.state_key("settled") + "."
        marks: set[tuple[str, str]] = set()
        for key in await kv.list_keys(prefix=prefix):
            incarnation, _, count = key[len(prefix) :].rpartition(".")
            if incarnation:
                marks.add((incarnation, count))
        return marks

    async def write_state(self, value: bytes, *parts: str) -> None:
        """set one piece of this table's L2-first bookkeeping.

        :param value: its value
        :ptype value: bytes
        :param parts: the piece's name
        :ptype parts: str
        :raises KvError: when L2 refuses it
        """
        kv = await self._ensure_kv()
        if kv is not None:
            await kv.put(key=self.state_key(*parts), value=value)

    async def drop_state(self, *parts: str) -> None:
        """remove one piece of this table's L2-first bookkeeping, leaving a marker that lapses.

        :param parts: the piece's name
        :ptype parts: str
        :raises KvError: when L2 refuses it
        """
        kv = await self._ensure_kv()
        if kv is not None:
            await kv.delete(key=self.state_key(*parts), marker_ttl=_L2_FIRST_STATE_MARKER_TTL)

    async def complete(self) -> bool:
        """whether L2 holds this table complete: the refiller filled it, and NATS has not lost it since.

        :return: ``True`` when the refiller's mark is in L2
        :rtype: bool
        :raises L2LostError: when L2 cannot be read, so whether it is complete cannot be told
        """
        try:
            return await self.read_state("complete") is not None
        except KvError as exc:
            degrade_l2(
                exc,
                table=self._table,
                op="get",
                what="L2 unreadable while asking whether it holds an L2-first table complete",
                l2_is_the_store=True,
                refused_as=L2LostError(f"{self._table}: L2 cannot be read ({type(exc).__name__}: {exc})"),
            )
            raise

    async def mark_complete(self, generation: str | None) -> None:
        """record that L2 holds this table complete, as of ``generation``: the refiller's last step.

        :param generation: the table's generation when the refill finished
        :ptype generation: str | None
        :raises KvError: when L2 refuses it
        """
        await self.write_state(_complete_mark(generation), "complete")

    async def carry_to_l3(self, row: ChangedRow) -> None:
        """make L3 hold what L2 holds for one committed key: its row, or no row.

        Safe to repeat and safe out of order: it writes L2's value as it is now, whichever epoch asked.
        A key L2 holds no row for is deleted from L3 only while L2 holds the table complete.

        :param row: the changed row an epoch named
        :ptype row: ChangedRow
        :raises L2LostError: when L2 holds no row for the key and does not hold the table complete
        :raises KvError: when L2 cannot be read
        :raises Exception: whatever L3 raised: a refusal, or a failure to reach it
        """
        entity_id = self._id_of_changed(row)
        kv = await self._ensure_kv()
        if kv is None:
            raise L2LostError(f"{self._table}: no L2 to save from")
        raw, latest = await kv.get_latest(key=self._l2_key(entity_id))
        if row.l2_revision is not None and latest == row.l2_revision and raw != row.value:
            # the key is where the committing writer read it: it died before applying its row, so the
            # saver applies it now, at that revision; a write since would have moved the key on
            await self._apply(kv, entity_id, row.value, latest)
            raw, latest = await kv.get_latest(key=self._l2_key(entity_id))
        held: dict[str, Any] | None = None
        if raw is not None:
            decoded = self._decode(raw)
            if isinstance(decoded, dict) and not self._is_expired(decoded):
                held = decoded
        if held is not None:
            await self._save(held, None)
            return
        if not await self.complete():
            raise L2LostError(f"{self._table}: L2 holds no row for {entity_id!s} and is not complete")
        await self._delete(entity_id)

    async def carry_many_to_l3(self, rows: Sequence[ChangedRow], *, batched: bool = False) -> bool:
        """make L3 hold what L2 holds for many committed keys at once, where every one of them L2 holds a row for.

        Each key's L2 value is read (a few at a time), and the rows go to L3 in the collection's many-row save:
        a few statements, not one a key. Only when the collection has one, and every key's row is held and
        current: a key L2 holds no row for, or one a dead writer left at its read (:meth:`carry_to_l3`'s
        repair), is not this path's, and it answers ``False`` having written nothing, for the caller to carry
        the keys one at a time.

        :param rows: the changed rows epochs named
        :ptype rows: Sequence[ChangedRow]
        :param batched: read the rows' L2 values in batched direct gets (:meth:`L3Saver` ``batched_reads``)
        :ptype batched: bool
        :return: whether every row was carried
        :rtype: bool
        :raises KvError: when L2 cannot be read
        :raises Exception: whatever L3 raised: a refusal, or a failure to reach it (nothing is known saved)
        """
        if self._save_many is None or not rows:
            return False
        kv = await self._ensure_kv()
        if kv is None:
            return False
        keys = [self._l2_key(self._id_of_changed(row)) for row in rows]
        # every key's latest message in a few batched reads, where the bucket answers them; else a few at a time
        if callable(getattr(type(kv), "get_latest_many", None)):
            latest_of: dict[str, tuple[bytes | None, int]] = await kv.get_latest_many(keys=keys, batched=batched)
        else:
            limit = asyncio.Semaphore(_CARRY_CONCURRENCY)

            async def latest(key: str) -> tuple[str, tuple[bytes | None, int]]:
                async with limit:
                    return key, await kv.get_latest(key=key)

            latest_of = dict(await asyncio.gather(*(latest(key) for key in dict.fromkeys(keys))))

        def held(row: ChangedRow, key: str) -> dict[str, Any] | None:
            raw, latest = latest_of[key]
            if row.l2_revision is not None and latest == row.l2_revision and raw != row.value:
                return None
            decoded = self._decode(raw) if raw is not None else None
            return decoded if isinstance(decoded, dict) and not self._is_expired(decoded) else None

        found = [held(row, key) for row, key in zip(rows, keys, strict=True)]
        if any(row is None for row in found):
            return False
        # a key named twice is written once, as it stands now
        latest_by_key = {repr(self._id_of_row(row)): row for row in found if row is not None}
        await self._save_many(list(latest_by_key.values()))
        return True

    async def refill_from_l3(self, *, page_size: int) -> int:
        """put every row L3 holds into L2, where L2 holds nothing at all for its key.

        A key L2 holds anything for -- a write made since L2 was lost, or its delete -- is left alone:
        each row is written at revision zero, which lands only on a key with no message.

        :param page_size: rows read from L3 per statement
        :ptype page_size: int
        :return: how many rows were put back
        :rtype: int
        :raises KvError: when L2 refuses a row
        """
        kv = await self._ensure_kv()
        if kv is None:
            return 0
        written = 0
        async for page in whole_read(
            self._l3_pages(page_size=page_size),
            kind="cold_start",
            reason="L2 holds nothing for this table (its first fill, or NATS lost the memory bucket and the complete mark with it)",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start",
            proof="packages/epoch/tests/unit/test_l2_first.py::test_the_refill_reads_l3_only_while_l2_holds_no_data",
            speed="packages/epoch/tests/integration/test_l2_refill_speed_live.py::test_refill_from_l3_speed_at_real_size",
        ):
            for row in page:
                if self._is_expired(row):
                    continue
                entity_id = self._id_of_row(row)
                lifetime = self._entry_lifetime(row)
                timed = {} if lifetime is None else {"ttl": lifetime}
                try:
                    landed = await kv.update(key=self._l2_key(entity_id), value=self._encode(row), revision=0, **timed)
                except KvError as exc:
                    degrade_l2(
                        exc,
                        table=self._table,
                        op="refill",
                        what="L2 refused a row the refiller put back from L3",
                        l2_is_the_store=True,
                        entity_id=entity_id,
                    )
                    raise
                if landed is not None:
                    written += 1
        return written


def _complete_mark(token: str | None) -> bytes:
    """the value of the mark that L2 holds a table complete.

    :param token: the generation the refill completed under
    :ptype token: str | None
    :return: the stored mark
    :rtype: bytes
    """
    return json.dumps({"generation": token}).encode("utf-8")
