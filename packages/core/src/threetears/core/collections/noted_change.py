"""the change a restore (or a lost write's announcement) appends, folded per table: its rows, its records, or the whole
table. Rebuilt from the tables' own versions (:mod:`~threetears.core.collections.table_restore`) since the change-note
ring retired (append-only L3, chunk 7); its name stays, the epoch log's restore takes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from threetears.core.collections.epoch_change import ChangedRow, check_whole_reach

__all__ = ["NotedChange"]


@dataclass(frozen=True, slots=True)
class NotedChange:
    """what the notes of one table say changed, folded for a restore: its distinct rows, or the whole table.

    :ivar table: the table
    :ivar rows: each distinct row its notes name, with the invalidation columns they carry
    :ivar whole_table: a note of the whole table (a truncate, a file load) is among them; :attr:`rows`
        is then empty, since the whole-table record covers every row
    :ivar tenant: the tenant the rows belong to (:mod:`threetears.core.tenancy`), whose key the
        record is restored under; ``None`` for ``_all``
    :ivar prefixes: for a whole-table change, the key prefixes it names (a run deleted, from its run marker): the
        restored record names them, so a follower reads each as a key range. Rows outside them come back as their own changes
    :ivar reason: for a whole-table change, the reason its notes named: only an operator's reset
        (:data:`~threetears.core.collections.epoch_change.RESET_REASON`), restored as a reset
    :ivar bodies: for a keyed change folded in a worker process (:mod:`threetears.core.change_note_fold`), its rows'
        records as the log stores them, each within the record bound; :attr:`rows` is then empty
    :ivar s_min: the lowest row epoch its records name (:attr:`~threetears.core.collections.epoch_change.EpochChange.
        s_min`): a change rebuilt from the tables' versions names the restore's floor
        (:mod:`~threetears.core.collections.table_restore`); ``None`` for one from the change notes
    """

    table: str
    rows: tuple[ChangedRow, ...] = ()
    whole_table: bool = False
    tenant: str | None = None
    prefixes: tuple[Mapping[str, str], ...] = ()
    reason: str | None = None
    bodies: tuple[bytes, ...] = ()
    s_min: int | None = None

    def __post_init__(self) -> None:
        check_whole_reach(self.prefixes or None, self.reason, whole_table=self.whole_table)
        if self.bodies and (self.rows or self.whole_table):
            raise ValueError("a change carrying its rows' records names no rows itself and is not whole-table")
