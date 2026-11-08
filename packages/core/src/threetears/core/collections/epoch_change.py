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

from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = [
    "EPOCH_CHANGE_MAX_AGE",
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
    "changed_row_runs",
    "check_change_form",
    "epoch_change_record",
    "read_epoch_change",
]

#: how long a change record is kept, wherever it is kept: long enough for a partitioned pod to catch
#: up from it. A record gone is a pod behind, never a drop.
EPOCH_CHANGE_MAX_AGE: Final = timedelta(hours=24)

#: the largest record of a commit's rows. A commit whose rows would not fit in one is recorded as
#: several consecutive records, each naming its run of the rows (:func:`changed_row_runs`); never as
#: the whole table. Well under a broker's default message limit.
_EPOCH_CHANGE_MAX_RECORD_BYTES: Final = 256 * 1024

#: what a record holds beside its rows (the table, the token, the field names), kept free under the bound.
_RECORD_OVERHEAD_BYTES: Final = 1024


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
    """

    table: str
    token: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []

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
        return self


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
        self, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
    ) -> str | None:
        """advance the table's generation once and store what the advance covered.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the writer named no rows and did not change the whole table; the record
            says so and a follower that missed it is behind for that epoch. Requires ``rows=None``
        :ptype keys_unknown: bool
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


async def advance_recording(
    source: GenerationSource, table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
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
    :return: the token the advance wrote, as :meth:`GenerationSource.advance`
    :rtype: str | None
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    if isinstance(source, _EpochChangeWriter):
        return await source.advance_with_change(table_name, rows, keys_unknown=keys_unknown)
    return await source.advance(table_name)


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
    table_name: str, token: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
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
    :return: the record, as stored
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows
    :raises GenerationUnavailableError: when the record is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    record = EpochChange(
        table=table_name,
        token=token,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
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


def epoch_change_body(table_name: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False) -> bytes:
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
    :return: the stored body
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows
    :raises GenerationUnavailableError: when a record naming rows without their values is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    body = _EpochBody(
        table=table_name,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
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
