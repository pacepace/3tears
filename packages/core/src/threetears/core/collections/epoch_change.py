"""an epoch is its change set: the rows one advance of a table's write generation covered.

A table's write generation (:mod:`threetears.core.collections.generation`) tells a pod THAT it
missed a write. The epoch's change record tells it WHICH rows: the writer that advances the
generation to ``{incarnation}:{N}`` stores, beside it, the rows that commit changed, addressable
by table and token. A pod that finds an advance it did not hear in full fetches that record and
evicts exactly those rows, through the same path a heard row broadcast takes. Catching up after a
missed broadcast, after a pod restart and after a partition is the same operation: fetch epochs
``mine+1 .. head`` and apply them.

A gap never drops a table. A record that cannot be read leaves the pod serving what it holds,
counted and logged as behind, until the record can be read. A record that names the whole table
(:attr:`EpochChange.whole_table`) is the change itself -- a truncate, a file load, a migration
whose rows nobody listed -- and invalidates the table because that is what the write did. Only a
write whose caller changed the whole table records one: a commit of many rows is recorded as
several consecutive keyed records (:func:`changed_row_runs`), each within the record bound, and its
epoch is the last of them.

The records live in L2 (a memory bucket, replicated); the writer that advances writes them, and
core holds only the format and the two protocols, as it does for the generation itself.
"""

from __future__ import annotations

import inspect

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from threetears.core.cache.base import stored_instant
from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = [
    "EPOCH_CHANGE_MAX_AGE",
    "RESET_REASON",
    "WholeReach",
    "key_text",
    "ChangedRow",
    "EpochChange",
    "EpochChangeReader",
    "EpochLogExtent",
    "EpochQuiet",
    "EpochWalk",
    "epoch_change_body",
    "read_epoch_change_body",
    "read_logged_epoch_change",
    "walk_by_count",
    "advance_recording",
    "records_changes",
    "changed_row_runs",
    "check_change_form",
    "check_whole_reach",
    "epoch_change_record",
    "read_epoch_change",
]

#: the one reason a whole-table record names: an operator's reset (``SchemaBackedCollection.reset``), which a follower
#: reads as that table's cold start. A whole-table record naming no reason and no key prefix is an older writer's
RESET_REASON: Final = "reset"


@dataclass(frozen=True, slots=True)
class WholeReach:
    """what a whole-table change reached, as its record names it: the key prefixes of a load or a clear, or an
    operator's reset (:data:`RESET_REASON`). A follower reads each prefix as a key range, and a reset as that table's
    cold start.

    :ivar prefixes: each prefix, the leading key columns and their values as text
    :ivar reason: :data:`RESET_REASON`, or None
    """

    prefixes: tuple[Mapping[str, str], ...] = ()
    reason: str | None = None


#: how long a change record is kept, wherever it is kept: long enough for a partitioned pod to catch
#: up from it. A record gone is a pod behind, never a drop.
EPOCH_CHANGE_MAX_AGE: Final = timedelta(hours=24)

#: the largest record of a commit's rows. A commit whose rows would not fit in one is recorded as
#: several consecutive records, each naming its run of the rows (:func:`changed_row_runs`); never as
#: the whole table. Well under a broker's default message limit.
_EPOCH_CHANGE_MAX_RECORD_BYTES: Final = 256 * 1024

#: what a record holds beside its rows (the table, the token, the field names), kept free under the bound.
_RECORD_OVERHEAD_BYTES: Final = 1024


def key_text(value: Any) -> str:
    """a key's or an invalidation column's value as a change record, a row message and a change note name it.

    A time zone-aware ``datetime`` (a ``timestamptz``) is named in UTC, to the microsecond, in ISO form
    (``2026-11-09T10:22:10.082658+00:00``): the form the note trigger writes it in
    (:mod:`threetears.core.ddl.change_notes`), whatever zone the value came in, so a note and a record of one row
    name it in one spelling, and a key read back from the notes binds as the same instant. Anything else is its
    ``str``. Every key derived from a row's key values (its L2 key, its fence key) is built from these forms too, so a
    message's text values land on the key the typed values made.

    :param value: the value
    :ptype value: Any
    :return: its text
    :rtype: str
    """
    if isinstance(value, datetime) and value.utcoffset() is not None:
        # the L1 keeps a timestamp column in this text too, so an eviction naming the key finds its row
        return stored_instant(value)
    return str(value)


class ChangedRow(BaseModel):
    """one row an epoch changed, as a row broadcast names it.

    :ivar ids: the primary key's values as strings, in declared column order
    :ivar l2_current_scope: the L2 key scope whose entry the writer left holding the newest value,
        as :attr:`CacheInvalidationMessage.l2_current_scope`; ``None`` for an unfenced write
    :ivar columns: the declared invalidation columns' values from the row the write saw, as
        :attr:`CacheInvalidationMessage.columns`; ``None`` when none are declared or none was seen
    :ivar l3_pending: the write committed in L2 first (:mod:`threetears.core.collections.l2_first`), so
        L3 is made to match L2 for this row after the commit; ``False`` for a write L3 took first. With
        :attr:`value` ``None`` it is a delete
    :ivar value: the row's stored bytes, as the write put them in L2, for an epoch that is the write's
        commit (an L2-first collection): the row is applied to L2 from here, and a saver puts back any
        row L2 is missing before it saves to L3. ``None`` where the record only names the row
    :ivar l2_revision: the revision of the row's L2 key the committing writer read before its commit. The
        saver applies :attr:`value` only while the key is still at it: a writer that died after its commit
        left it there, and any write applied since moved it on
    :ivar tenant: the tenant the row belongs to (:mod:`threetears.core.tenancy`), whose key the
        record is appended under; ``None`` for ``_all``. Not stored in the record: its key carries it
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    ids: list[str]
    l2_current_scope: str | None = None
    columns: dict[str, str | None] | None = None
    l3_pending: bool = False
    value: bytes | None = None
    l2_revision: int | None = None
    tenant: str | None = Field(default=None, exclude=True)


class EpochChange(BaseModel):
    """the change one advance of a table's write generation made.

    :ivar table: the table
    :ivar token: the generation the advance wrote, ``{incarnation}:{count}``
    :ivar whole_table: the advance covered every row of the table (a truncate, a load, an
        announced migration); :attr:`rows` is then empty
    :ivar keys_unknown: the advance's writer named no rows and did not change the whole table (a
        broker advancing for a write that sent no ``changed_rows``): a follower that missed it cannot
        apply it, and is behind on the table for that epoch -- serving, logged and counted as
        ``keys_unknown``, never a drop. :attr:`rows` is then empty
    :ivar rows: the rows the advance covered, one per row broadcast of its commit
    :ivar partition: for a whole-table record of a load from files, the columns every row it loaded holds and their
        values as text (the load's constants): the rows whose columns equal these are the load's reach, and a follower
        that knows the field reads those rows again rather than the table. A follower built before it reads the
        record as the whole table, which the record also says, so it is never misled. A record of exactly one key
        prefix names it here too, for a follower that knows this field but not :attr:`prefixes`
    :ivar prefixes: for a whole-table record of a load or a clear, the SET of primary-key prefixes it reached, each the
        leading key columns and their values as text (an AP load of a state names each office's key): a follower reads
        each prefix's rows again as a key range, never the table. A follower built before it reads :attr:`partition`,
        or the whole table
    :ivar reason: for a whole-table record, why the whole table: only :data:`RESET_REASON` (an operator's reset). A
        follower built before it reads the record as the whole table
    :ivar noted_between: for a keys-unknown record of a commit whose rows were too many to name, the database clock's
        bracket of the commit's change notes, ``(transaction start, just before COMMIT)`` on the committing
        connection: every note the commit wrote (:mod:`~threetears.core.collections.change_notes`) is in it, so a
        follower resolves the keys from the notes (perhaps with other commits' keys, which it reads again harmlessly).
        A follower built before it reads the record as keys unknown and is behind, never misled
    """

    table: str
    token: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []
    partition: dict[str, str] | None = None
    prefixes: list[dict[str, str]] | None = None
    reason: str | None = None
    noted_between: tuple[datetime, datetime] | None = None

    @model_validator(mode="after")
    def _names_its_reach(self) -> EpochChange:
        """refuse a record that is not exactly one of: its rows, the whole table, or keys unknown.

        :return: the record
        :rtype: EpochChange
        :raises ValueError: when the reach is empty or contradictory
        """
        if [bool(self.rows), self.whole_table, self.keys_unknown].count(True) != 1:
            raise ValueError(
                f"an epoch change of {self.table!r} at {self.token!r} names exactly one of its rows, the whole "
                f"table, or keys unknown"
            )
        if self.partition is not None and (not self.whole_table or not self.partition):
            raise ValueError(
                f"an epoch change of {self.table!r} at {self.token!r} names a partition only as a whole-table record, "
                f"and names at least one column"
            )
        check_whole_reach(self.prefixes, self.reason, whole_table=self.whole_table)
        _check_noted_between(self.noted_between, keys_unknown=self.keys_unknown)
        return self


def check_whole_reach(prefixes: Sequence[Mapping[str, str]] | None, reason: str | None, *, whole_table: bool) -> None:
    """refuse key prefixes or a reason on a record that is not whole-table, an empty prefix, or a reason not known.

    :param prefixes: the record's key prefixes, or None
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: the record's reason, or None
    :ptype reason: str | None
    :param whole_table: whether the record is whole-table
    :ptype whole_table: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if (prefixes is not None or reason is not None) and not whole_table:
        raise ValueError("a change names key prefixes or a reason only as a whole-table record")
    if prefixes is not None and (not prefixes or not all(prefixes)):
        raise ValueError("a change's key prefixes are at least one, each naming at least one column")
    if reason is not None and reason != RESET_REASON:
        raise ValueError(f"a whole-table change's reason is {RESET_REASON!r} or none, not {reason!r}")
    if reason is not None and prefixes is not None:
        raise ValueError("a reset names the whole table, never key prefixes")


def _check_noted_between(noted_between: tuple[datetime, datetime] | None, *, keys_unknown: bool) -> None:
    """refuse a notes bracket on a record that is not keys unknown, or one that ends before it starts.

    :param noted_between: the bracket, or None
    :ptype noted_between: tuple[datetime, datetime] | None
    :param keys_unknown: whether the record is keys unknown
    :ptype keys_unknown: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if noted_between is None:
        return
    if not keys_unknown:
        raise ValueError("a change's notes bracket is named only on a keys-unknown record")
    if noted_between[1] < noted_between[0]:
        raise ValueError(f"a change's notes bracket ends before it starts: {noted_between}")


@runtime_checkable
class EpochChangeReader(Protocol):
    """reads the change record of one advance of a table's write generation."""

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the record of the advance that wrote ``token``.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record, or ``None`` when L2 does not hold it (never written, lost with the
            store, or past its retention): the follower is then behind
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the store cannot be read
        """
        ...

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """every epoch of ``table_name`` after ``after`` and up to ``upto``, in order: what a follower applies.

        :param table_name: the table
        :ptype table_name: str
        :param after: the follower's mark, ``{incarnation}:{count}``
        :ptype after: str
        :param upto: the generation read, under the same incarnation
        :ptype upto: str
        :return: the epochs the log holds in that range, and whether it may be missing any
        :rtype: EpochWalk
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...


@dataclass(frozen=True, slots=True)
class EpochWalk:
    """the epochs of one table a follower reads to catch up, in order.

    :ivar changes: each epoch's record the log holds in the range, oldest first
    :ivar complete: ``False`` when the log may be missing an epoch in the range -- a record lost or
        aged out -- so the follower is behind on the table even after applying :attr:`changes`
    :ivar first_missing: the oldest epoch known or suspected missing, for the log line; ``None`` when
        :attr:`complete`
    :ivar reason: why the walk is not complete: the reason the registry falls behind with, one of
        :data:`threetears.core.collections.registry.BEHIND_REASONS`
    """

    changes: tuple[EpochChange, ...]
    complete: bool
    first_missing: str | None = None
    reason: str = "record_missing"


async def walk_by_count(reader: EpochChangeReader, table_name: str, after: str, upto: str) -> EpochWalk:
    """the walk of a log that numbers one table's epochs one apart: each count in turn.

    For a log that keeps a table's epochs at consecutive counts (a table kept in L2, a test's log). A count
    the log does not hold is an epoch missing; the counts after it are still read.

    :param reader: reads one epoch's record
    :ptype reader: EpochChangeReader
    :param table_name: the table
    :ptype table_name: str
    :param after: the mark
    :ptype after: str
    :param upto: the generation read
    :ptype upto: str
    :return: the walk
    :rtype: EpochWalk
    :raises GenerationUnavailableError: when a record cannot be read
    """
    incarnation, _, low = after.rpartition(":")
    _, _, high = upto.rpartition(":")
    changes: list[EpochChange] = []
    missing: str | None = None
    for count in range(int(low) + 1, int(high) + 1):
        token = f"{incarnation}:{count}"
        change = await reader.change(table_name, token)
        if change is None:
            missing = missing or token
            continue
        changes.append(change)
    return EpochWalk(changes=tuple(changes), complete=missing is None, first_missing=missing)


@dataclass(frozen=True, slots=True)
class EpochQuiet:
    """what a log can vouch for after a follower's mark, when the table's generation reads at or below it.

    :ivar through: the newest epoch through which the log holds nothing of the table after the mark, so
        the mark may move to it; ``None`` when the log holds an epoch of the table (or one it cannot
        name) after the mark, or cannot say
    :ivar aged_out: the log no longer holds the epoch after the mark (retention took it), so it cannot say
        whether the table was written since: the follower is behind (``aged_out``), never current
    """

    through: str | None
    aged_out: bool = False


@runtime_checkable
class EpochLogExtent(Protocol):
    """what a change log can say about its own reach: the lost log a restore continues, and quiet stretches."""

    async def continues(self, incarnation: str) -> tuple[str, int | None] | None:
        """the lost log's incarnation the log ``incarnation`` continues, and the sequence its notes cover it from.

        A follower whose mark under the lost incarnation is at or past that sequence missed nothing the
        restored log does not hold: it walks every record of the restored log. One whose mark is older is
        behind.

        :param incarnation: the log's incarnation, as a generation read names it
        :ptype incarnation: str
        :return: ``(lost incarnation, covered from)``, the second ``None`` when the notes cover none of the
            lost log; ``None`` when the log continues none, or is not ``incarnation`` (any more)
        :rtype: tuple[str, int | None] | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def quiet_after(self, table_name: str, mark: str) -> EpochQuiet:
        """what the log can vouch for of ``table_name`` after ``mark``.

        :param table_name: the table
        :ptype table_name: str
        :param mark: the follower's mark
        :ptype mark: str
        :return: how far the mark may move, and whether the log has aged past it
        :rtype: EpochQuiet
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...


@runtime_checkable
class _EpochChangeWriter(Protocol):
    """a generation source that records each advance's change as it makes it."""

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
    ) -> str | None:
        """advance the table's generation once and store what the advance covered.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the writer named no rows and did not change the whole table; the record
            says so and a follower that missed it is behind for that epoch. Requires ``rows=None``
        :ptype keys_unknown: bool
        :param tenant: the write's tenant, from a writer that knows it: a rowless record's key, and that of
            any row naming none; ``None`` for ``_all``
        :ptype tenant: str | None
        :return: the generation token the advance wrote, as :meth:`GenerationSource.advance`
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation could not be advanced, or its
            change could not be stored, after a committed write
        :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
        """
        ...


def check_change_form(rows: Sequence[ChangedRow] | None, *, keys_unknown: bool) -> None:
    """refuse ``keys_unknown`` together with rows: a change is its rows, the whole table, or keys unknown.

    :param rows: the rows named, or ``None``
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: whether the writer's keys are unknown
    :ptype keys_unknown: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when both are given
    """
    if keys_unknown and rows is not None:
        raise ValueError("keys_unknown=True names no rows: pass rows=None")


def records_changes(source: object) -> bool:
    """whether ``source`` stores each advance's change itself (an epoch source that keeps the change log).

    A pod's source does not: its broker advances for it and names the change.

    :param source: a generation source
    :ptype source: object
    :return: whether it records changes
    :rtype: bool
    """
    return isinstance(source, _EpochChangeWriter)


async def advance_recording(
    source: GenerationSource,
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    tenant: str | None = None,
    partition: Mapping[str, str] | None = None,
    noted_between: tuple[datetime, datetime] | None = None,
    prefixes: Sequence[Mapping[str, str]] | None = None,
    reason: str | None = None,
) -> str | None:
    """advance ``table_name``'s generation through ``source``, storing the change when the source records changes.

    Every advance goes through here, so a source that keeps the epoch change log is never handed an
    advance without its change. A source that keeps none (a pod's, whose broker advances for it)
    advances as it always has.

    :param source: the registry's generation source
    :ptype source: GenerationSource
    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows the committed write changed; ``None`` when its reach is the whole table
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows and did not change the whole table; requires ``rows=None``
    :ptype keys_unknown: bool
    :param tenant: the write's tenant, from a writer that knows it (the schema it wrote in): a rowless
        record's key, and that of any row naming none; ``None`` for ``_all``
    :ptype tenant: str | None
    :param partition: a whole-table change's partition (:attr:`EpochChange.partition`), for a source that records it
        (``advance_with_change(partition=)``); one that does not records the whole table, which the partition narrows
    :ptype partition: Mapping[str, str] | None
    :param noted_between: a keys-unknown change's notes bracket (:attr:`EpochChange.noted_between`), for a source that
        records it; one that does not records keys unknown, which the bracket would resolve
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: a whole-table change's key prefixes (:attr:`EpochChange.prefixes`), for a source that records
        them; exactly one is also named as :attr:`EpochChange.partition`, for a follower that knows only that
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: a whole-table change's reason (:data:`RESET_REASON`), for a source that records it
    :ptype reason: str | None
    :return: the token the advance wrote, as :meth:`GenerationSource.advance`
    :rtype: str | None
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    if isinstance(source, _EpochChangeWriter):
        extra: dict[str, Any] = {}
        if tenant is not None:
            extra["tenant"] = tenant
        whole = rows is None and not keys_unknown
        check_whole_reach(prefixes, reason, whole_table=whole)
        if prefixes and len(prefixes) == 1 and not partition:
            partition = prefixes[0]
        if partition and whole and _takes(source, "partition"):
            extra["partition"] = dict(partition)
        if prefixes and _takes(source, "prefixes"):
            extra["prefixes"] = [dict(prefix) for prefix in prefixes]
        if reason is not None and _takes(source, "reason"):
            extra["reason"] = reason
        if noted_between is not None and keys_unknown and _takes(source, "noted_between"):
            extra["noted_between"] = noted_between
        return await source.advance_with_change(table_name, rows, keys_unknown=keys_unknown, **extra)
    return await source.advance(table_name)


def _takes(source: object, field: str) -> bool:
    """whether ``source``'s ``advance_with_change`` takes ``field``: a source built before it is handed none.

    :param source: a source that records changes
    :ptype source: object
    :param field: the keyword (``partition``, ``noted_between``)
    :ptype field: str
    :return: whether it takes it
    :rtype: bool
    """
    method = getattr(source, "advance_with_change", None)
    return method is not None and field in inspect.signature(method).parameters


def changed_row_runs(table_name: str, rows: Sequence[ChangedRow]) -> list[list[ChangedRow]]:
    """split a commit's rows into consecutive runs whose records each fit the record bound.

    A commit of more rows than one record holds is recorded as one epoch per run, in order, so a
    follower that missed it applies every row by key; nothing becomes a whole-table change for its size.

    :param table_name: the table, for the error
    :ptype table_name: str
    :param rows: the rows the commit changed, at least one
    :ptype rows: Sequence[ChangedRow]
    :return: the runs, in order; one run when every row fits in one record
    :rtype: list[list[ChangedRow]]
    :raises GenerationUnavailableError: when one row alone is past the bound, so no record can name it
    """
    budget = _EPOCH_CHANGE_MAX_RECORD_BYTES - _RECORD_OVERHEAD_BYTES
    runs: list[list[ChangedRow]] = []
    run: list[ChangedRow] = []
    used = 0
    for row in rows:
        # measured as the fullest form any record stores it in, so every record of the run fits
        size = len(row.model_dump_json()) + 1
        if size > budget:
            raise GenerationUnavailableError(
                f"a changed row of {table_name!r} is {size} bytes, past the {budget}-byte record bound: no epoch "
                f"record can name it"
            )
        if run and used + size > budget:
            runs.append(run)
            run, used = [], 0
        run.append(row)
        used += size
    if run:
        runs.append(run)
    return runs


def _within_bound(stored: bytes, table_name: str) -> bytes:
    """``stored``, refused when it is past the record bound.

    :param stored: a record as stored
    :ptype stored: bytes
    :param table_name: the table, for the error
    :ptype table_name: str
    :return: ``stored``
    :rtype: bytes
    :raises GenerationUnavailableError: when it is past the bound; the caller splits with :func:`changed_row_runs`
    """
    if len(stored) > _EPOCH_CHANGE_MAX_RECORD_BYTES:
        raise GenerationUnavailableError(
            f"an epoch record of {table_name!r} is {len(stored)} bytes, past the {_EPOCH_CHANGE_MAX_RECORD_BYTES}-byte "
            f"bound: split its rows with changed_row_runs"
        )
    return stored


def epoch_change_record(
    table_name: str,
    token: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    reach: WholeReach | None = None,
) -> bytes:
    """the stored form of the change one advance made.

    A record past :data:`_EPOCH_CHANGE_MAX_RECORD_BYTES` is refused: the caller records a large
    commit as one advance per run of :func:`changed_row_runs`.

    :param table_name: the table
    :ptype table_name: str
    :param token: the generation the advance wrote
    :ptype token: str
    :param rows: the rows it covered; ``None`` when its reach is the whole table, or its keys unknown
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows and did not change the whole table; requires ``rows=None``
    :ptype keys_unknown: bool
    :param reach: a whole-table change's key prefixes or reason (:attr:`EpochChange.prefixes`,
        :attr:`EpochChange.reason`)
    :ptype reach: WholeReach | None
    :return: the record, as stored
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows, or a reach on a change that is not whole-table
    :raises GenerationUnavailableError: when the record is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    prefixes = [dict(prefix) for prefix in reach.prefixes] if reach is not None and reach.prefixes else None
    record = EpochChange(
        table=table_name,
        token=token,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
        partition=prefixes[0] if prefixes is not None and len(prefixes) == 1 else None,
        prefixes=prefixes,
        reason=reach.reason if reach is not None else None,
    )
    return _within_bound(record.model_dump_json().encode("utf-8"), table_name)


def read_epoch_change(raw: bytes, table_name: str, token: str) -> EpochChange:
    """the change record stored as ``raw``, checked to be the one of ``table_name`` at ``token``.

    :param raw: the stored record
    :ptype raw: bytes
    :param table_name: the table it must be of
    :ptype table_name: str
    :param token: the generation it must be of
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed, or is another epoch's
    """
    try:
        record = EpochChange.model_validate_json(raw)
    except ValidationError as exc:
        raise GenerationUnavailableError(
            f"the change record of {table_name!r} at {token!r} is malformed: {exc}"
        ) from exc
    if record.table != table_name or record.token != token:
        raise GenerationUnavailableError(
            f"the change record stored for {table_name!r} at {token!r} is of {record.table!r} at {record.token!r}"
        )
    return record


class _EpochBody(BaseModel):
    """an epoch's change as a change log stores it: the stream's sequence numbers it, so it names no token."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    table: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []
    partition: dict[str, str] | None = None
    prefixes: list[dict[str, str]] | None = None
    reason: str | None = None
    noted_between: tuple[datetime, datetime] | None = None


def epoch_change_body(
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    partition: Mapping[str, str] | None = None,
    noted_between: tuple[datetime, datetime] | None = None,
    prefixes: Sequence[Mapping[str, str]] | None = None,
    reason: str | None = None,
) -> bytes:
    """the stored form of a change for a log whose sequence is the epoch, so the record carries no token.

    A record past the record bound is refused, as :func:`epoch_change_record` refuses it: the caller
    splits a large commit's rows with :func:`changed_row_runs`. A record whose rows carry their values is
    the commit itself, so it is stored whole, and the log refuses it if it is past the log's own limit.

    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows changed; ``None`` for the whole table, or with ``keys_unknown``
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows; requires ``rows=None``
    :ptype keys_unknown: bool
    :param partition: for a whole-table change that is a load from files, the columns its rows hold and their values
        (:attr:`EpochChange.partition`); requires ``rows=None`` and not ``keys_unknown``
    :ptype partition: Mapping[str, str] | None
    :param noted_between: for a keys-unknown change, its notes bracket (:attr:`EpochChange.noted_between`)
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: for a whole-table change of a load or a clear, its key prefixes (:attr:`EpochChange.prefixes`)
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: for a whole-table change, its reason (:attr:`EpochChange.reason`)
    :ptype reason: str | None
    :return: the stored body
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows, or a partition with either
    :raises GenerationUnavailableError: when a record naming rows without their values is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    if partition is not None and (rows is not None or keys_unknown):
        raise ValueError("a partition narrows a whole-table change: pass rows=None without keys_unknown")
    _check_noted_between(noted_between, keys_unknown=keys_unknown)
    check_whole_reach(prefixes, reason, whole_table=rows is None and not keys_unknown)
    body = _EpochBody(
        table=table_name,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
        partition=dict(partition) if partition else None,
        prefixes=[dict(prefix) for prefix in prefixes] if prefixes else None,
        reason=reason,
        noted_between=noted_between,
    )
    stored = body.model_dump_json(exclude_none=True).encode("utf-8")
    carries_values = rows is not None and any(row.value is not None for row in rows)
    return stored if carries_values else _within_bound(stored, table_name)


def read_epoch_change_body(raw: bytes, table_name: str, token: str) -> EpochChange:
    """the change a log stored as ``raw`` at the epoch ``token``, checked to be ``table_name``'s.

    :param raw: the stored body
    :ptype raw: bytes
    :param table_name: the table it must be of
    :ptype table_name: str
    :param token: the epoch the log numbered it
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed, or another table's
    """
    try:
        body = _EpochBody.model_validate_json(raw)
        record = EpochChange(
            table=body.table,
            token=token,
            whole_table=body.whole_table,
            keys_unknown=body.keys_unknown,
            rows=body.rows,
            partition=body.partition,
            prefixes=body.prefixes,
            reason=body.reason,
            noted_between=body.noted_between,
        )
    except ValidationError as exc:
        raise GenerationUnavailableError(
            f"the change record of {table_name!r} at {token!r} is malformed: {exc}"
        ) from exc
    if record.table != table_name:
        raise GenerationUnavailableError(f"the change record at {token!r} is of {record.table!r}, not {table_name!r}")
    return record


def read_logged_epoch_change(raw: bytes, token: str) -> EpochChange:
    """the change a log stored as ``raw`` at the epoch ``token``, of whichever table it names.

    For a reader that walks the log by sequence rather than by table: the change-note scrub.

    :param raw: the stored body
    :ptype raw: bytes
    :param token: the epoch the log numbered it
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed
    """
    try:
        body = _EpochBody.model_validate_json(raw)
    except ValidationError as exc:
        raise GenerationUnavailableError(f"the change record at {token!r} is malformed: {exc}") from exc
    return read_epoch_change_body(raw, body.table, token)
