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

from pydantic import BaseModel, ValidationError, model_validator

from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = [
    "EPOCH_CHANGE_MAX_AGE",
    "EPOCH_CHANGE_MAX_RECORD_BYTES",
    "ChangedRow",
    "EpochChange",
    "EpochChangeReader",
    "EpochChangeWriter",
    "advance_recording",
    "epoch_change_record",
    "read_epoch_change",
]

#: how long a change record is kept, wherever it is kept: long enough for a partitioned pod to catch
#: up from it. A record gone is a pod behind, never a drop.
EPOCH_CHANGE_MAX_AGE: Final = timedelta(hours=24)

#: the largest record of a commit's rows that is stored as rows. A commit whose rows would not fit is
#: recorded as a change to the whole table: one record that always fits, where a refused one would
#: leave every follower that missed it behind for good. Well under a broker's default message limit.
EPOCH_CHANGE_MAX_RECORD_BYTES: Final = 256 * 1024


class ChangedRow(BaseModel):
    """one row an epoch changed, as a row broadcast names it.

    :ivar ids: the primary key's values as strings, in declared column order
    :ivar l2_current_scope: the L2 key scope whose entry the writer left holding the newest value,
        as :attr:`CacheInvalidationMessage.l2_current_scope`; ``None`` for an unfenced write
    :ivar columns: the declared invalidation columns' values from the row the write saw, as
        :attr:`CacheInvalidationMessage.columns`; ``None`` when none are declared or none was seen
    """

    ids: list[str]
    l2_current_scope: str | None = None
    columns: dict[str, str | None] | None = None


class EpochChange(BaseModel):
    """the change one advance of a table's write generation made.

    :ivar table: the table
    :ivar token: the generation the advance wrote, ``{incarnation}:{count}``
    :ivar whole_table: the advance covered every row of the table (a truncate, a load, an
        announced migration); :attr:`rows` is then empty
    :ivar rows: the rows the advance covered, one per row broadcast of its commit
    """

    table: str
    token: str
    whole_table: bool = False
    rows: list[ChangedRow] = []

    @model_validator(mode="after")
    def _names_its_reach(self) -> EpochChange:
        """refuse a record that names no row and is not the whole table, or names rows and is.

        :return: the record
        :rtype: EpochChange
        :raises ValueError: when the reach is empty or contradictory
        """
        if self.whole_table == bool(self.rows):
            raise ValueError(
                f"an epoch change of {self.table!r} at {self.token!r} names either the whole table or its rows, "
                f"never both and never neither"
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


@runtime_checkable
class EpochChangeWriter(Protocol):
    """a generation source that records each advance's change as it makes it."""

    async def advance_with_change(self, table_name: str, rows: Sequence[ChangedRow] | None) -> str | None:
        """advance the table's generation once and store what the advance covered.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :return: the generation token the advance wrote, as :meth:`GenerationSource.advance`
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation could not be advanced, or its
            change could not be stored, after a committed write
        """
        ...


async def advance_recording(source: GenerationSource, table_name: str, rows: Sequence[ChangedRow] | None) -> str | None:
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
    :return: the token the advance wrote, as :meth:`GenerationSource.advance`
    :rtype: str | None
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    """
    if isinstance(source, EpochChangeWriter):
        return await source.advance_with_change(table_name, rows)
    return await source.advance(table_name)


def epoch_change_record(table_name: str, token: str, rows: Sequence[ChangedRow] | None) -> bytes:
    """the stored form of the change one advance made.

    Rows that would make a record past :data:`EPOCH_CHANGE_MAX_RECORD_BYTES` are recorded as the
    whole table instead.

    :param table_name: the table
    :ptype table_name: str
    :param token: the generation the advance wrote
    :ptype token: str
    :param rows: the rows it covered; ``None`` when its reach is the whole table
    :ptype rows: Sequence[ChangedRow] | None
    :return: the record, as stored
    :rtype: bytes
    """
    record = EpochChange(
        table=table_name, token=token, whole_table=rows is None, rows=list(rows) if rows is not None else []
    )
    stored = record.model_dump_json().encode("utf-8")
    if len(stored) > EPOCH_CHANGE_MAX_RECORD_BYTES:
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
