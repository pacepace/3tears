"""Computed columns the shared copy's tests hand its apply worker (:class:`~threetears.core.collections.shared_copy.
ColumnHooks`): module-level, so the worker imports them by name. ``NewestUpload`` is a double of reports' snapshots'
``most_recent_ind``: each row's upload ranked newest first among its group's (dense, ``1`` the newest)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.arrow_keys import key_codes, rows_named
from threetears.core.collections.shared_copy import AppliedTables, ComputedRows

#: the table whose column is computed
SNAPSHOTS = "snapshots"
SNAPSHOTS_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        pa.field("upload", pa.int64()),
        pa.field("votes", pa.int64()),
        pa.field("most_recent_ind", pa.int64()),
    ]
)
SNAPSHOTS_KEY = ("state", "office_key", "candidate_id", "upload")
#: the rows one value is ranked among: a candidate's uploads in a state's race
GROUP = ("state", "office_key", "candidate_id")
COLUMN = "most_recent_ind"


def _dense_rank_newest(rows: pa.Table) -> pa.Array:
    """each row's upload's dense rank among its group's, newest ``1``: one sort of the group codes, in NumPy."""
    if not rows.num_rows:
        return pa.array([], pa.int64())
    (groups,) = key_codes([rows], GROUP)
    uploads = rows["upload"].to_numpy()
    order = np.lexsort((-uploads, groups))
    by_group, by_upload = groups[order], uploads[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = by_group[1:] != by_group[:-1]
    fresh = first.copy()
    fresh[1:] |= by_upload[1:] != by_upload[:-1]
    counted = np.cumsum(fresh)
    base = np.maximum.accumulate(np.where(first, counted, 0))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = counted - base + 1
    return pa.array(ranks, pa.int64())


@dataclass(frozen=True, slots=True)
class NewestUpload:
    """``most_recent_ind`` of the snapshots, over the change's rows and the held rows of the groups it touches (its
    rows' and its deleted rows'), read from the version before the change by its touched states."""

    @property
    def computed(self) -> Mapping[str, Collection[str]]:
        """the snapshots' ``most_recent_ind``."""
        return {SNAPSHOTS: (COLUMN,)}

    def compute(self, table: str, change: AppliedTables) -> ComputedRows:
        """the change's rows ranked, and the held rows whose rank moved.

        :param table: the table (the snapshots)
        :ptype table: str
        :param change: the change, and the version before it
        :ptype change: AppliedTables
        :return: the rows computed
        :rtype: ComputedRows
        """
        rows = change.rows[table]
        deleted_rows = change.deleted_rows.get(table, SNAPSHOTS_SCHEMA.empty_table())
        if table in change.whole:
            held = SNAPSHOTS_SCHEMA.empty_table()
        else:
            held = change.table(table, scopes=change.touched)
            touched = pa.concat_tables([rows.select(list(GROUP)), deleted_rows.select(list(GROUP))])
            held = held.take(pa.array(rows_named(held, touched, GROUP)))
            named = pa.concat_tables([rows.select(list(SNAPSHOTS_KEY)), change.deleted[table]])
            kept = np.ones(held.num_rows, dtype=bool)
            kept[rows_named(held, named, SNAPSHOTS_KEY)] = False
            held = held.filter(pa.array(kept))
        both = pa.concat_tables([rows, held])
        ranks = _dense_rank_newest(both)
        both = both.set_column(both.schema.get_field_index(COLUMN), COLUMN, ranks)
        mine = both.slice(0, rows.num_rows)
        again = both.slice(rows.num_rows)
        moved = pc.invert(pc.fill_null(pc.equal(again[COLUMN], held[COLUMN]), False))
        return ComputedRows(rows=mine, held=again.filter(moved))


@dataclass(frozen=True, slots=True)
class Reordered:
    """gives the change's rows back reversed: refused, the change's rows are given back one for one in their order."""

    @property
    def computed(self) -> Mapping[str, Collection[str]]:
        """the snapshots' ``most_recent_ind``."""
        return {SNAPSHOTS: (COLUMN,)}

    def compute(self, table: str, change: AppliedTables) -> ComputedRows:
        """the change's rows reversed.

        :param table: the table
        :ptype table: str
        :param change: the change
        :ptype change: AppliedTables
        :return: the rows reversed
        :rtype: ComputedRows
        """
        rows = change.rows[table]
        return ComputedRows(rows=rows.take(pa.array(np.arange(rows.num_rows)[::-1])))
