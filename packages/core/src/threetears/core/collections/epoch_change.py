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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = ["ChangedRow", "EpochChange", "EpochChangeReader", "EpochChangeWriter", "advance_recording"]


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
