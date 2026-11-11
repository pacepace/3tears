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
whose rows nobody listed -- and invalidates the table because that is what the write did.

The records live in L2 (a memory bucket, replicated); the writer that advances writes them, and
core holds only the format and the two protocols, as it does for the generation itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = [
    "EPOCH_CHANGE_MAX_AGE",
    "ChangedRow",
    "EpochChange",
    "EpochChangeReader",
    "EpochWalk",
    "epoch_change_body",
    "read_epoch_change_body",
    "walk_by_count",
    "advance_recording",
    "check_change_form",
    "epoch_change_record",
    "read_epoch_change",
]

#: how long a change record is kept, wherever it is kept: long enough for a partitioned pod to catch
#: up from it. A record gone is a pod behind, never a drop.
EPOCH_CHANGE_MAX_AGE: Final = timedelta(hours=24)

#: the largest record of a commit's rows that is stored as rows. A commit whose rows would not fit is
#: recorded as a change to the whole table: one record that always fits, where a refused one would
#: leave every follower that missed it behind for good. Well under a broker's default message limit.
_EPOCH_CHANGE_MAX_RECORD_BYTES: Final = 256 * 1024


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
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    ids: list[str]
    l2_current_scope: str | None = None
    columns: dict[str, str | None] | None = None
    l3_pending: bool = False
    value: bytes | None = None
    l2_revision: int | None = None


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
    """

    changes: tuple[EpochChange, ...]
    complete: bool
    first_missing: str | None = None


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


def epoch_change_record(
    table_name: str, token: str, rows: Sequence[ChangedRow] | None, *, keys_unknown: bool = False
) -> bytes:
    """the stored form of the change one advance made.

    Rows that would make a record past :data:`_EPOCH_CHANGE_MAX_RECORD_BYTES` are recorded as the
    whole table instead.

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
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    record = EpochChange(
        table=table_name,
        token=token,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
    )
    stored = record.model_dump_json().encode("utf-8")
    if len(stored) > _EPOCH_CHANGE_MAX_RECORD_BYTES:
        stored = EpochChange(table=table_name, token=token, whole_table=True).model_dump_json().encode("utf-8")
    return stored


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

    Rows that would make a record past the record bound are stored as the whole table, as
    :func:`epoch_change_record` stores them -- unless they carry their values: that record is the commit,
    so it is stored whole or refused.

    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows changed; ``None`` for the whole table, or with ``keys_unknown``
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows; requires ``rows=None``
    :ptype keys_unknown: bool
    :return: the stored body
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows
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
    if len(stored) > _EPOCH_CHANGE_MAX_RECORD_BYTES and not carries_values:
        stored = _EpochBody(table=table_name, whole_table=True).model_dump_json().encode("utf-8")
    return stored


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
