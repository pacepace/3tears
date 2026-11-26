"""a stamped table's row that moves between tenants records the tenant it left on itself, in the write that moves it.

Append-only L3 retires the change-note ring (``design-append-only-l3.md``, chunk 7). A pod caches a table's rows by
tenant (its tenancy column, an invalidation column), so a write that changes a row's tenant column in place must be
heard by the tenant it LEFT as well as the one it joined: the old tenant's followers still hold the row. The ring's
old-row note was the only thing that told them. Without it they would keep a row that is no longer theirs, live and
after a restore. Pace chose (2026-11-25, option A) to record the old tenant on the row itself.

**The declaration.** A stamped table (``ROW_STAMPED``) whose tenancy column can change names it in
``TableSchema(moveable=(column,))``. Each such column gets two L3-only columns beside it (:func:`moved_columns`), as the
restore bucket is L3-only: no cache, entity or ``INSERT`` names them, so a new row holds NULL in both.

- ``<column>_moved_from``, the column's type, nullable: the value the row's last move left.
- ``<column>_moved_epoch``, ``BIGINT``, nullable: the epoch of the write that moved it, NULL for a row that never
  moved. WHY a second column, not the row's epoch alone: a tenancy column may be NULL (the hub's platform datasets,
  ``customer_id IS NULL``, a tenant of their own, ``_all``), so ``moved_from IS NULL`` cannot say "never moved"; and a
  later write that does not move the row raises its epoch, so the row's epoch cannot say which write moved it.

**The write** (:func:`moved_assignments`): a statement that sets a moveable column also sets both, IN THE SAME
STATEMENT, from the row's old values (an ``UPDATE``'s right-hand sides read the row as it was), no extra read::

    c_moved_from  = CASE WHEN c IS DISTINCT FROM <new> THEN c ELSE c_moved_from END
    c_moved_epoch = CASE WHEN c IS DISTINCT FROM <new> THEN <the row's new epoch> ELSE c_moved_epoch END

A write that does not change the column leaves both as they are. WHY never cleared: clearing them "once older than the
restore window" would need the window in the statement, and the window is the change log's own (a keeper's mark may be
older than the margin M: ``row_epoch.cleanup_floor``), which no writer knows; a stale move costs nothing, since the
restore reads it only while the row's ``moved_epoch`` is at or above X (below).

**Live.** The statement returns the two columns and the row's epoch (``RETURNING``, the same statement); the row moved
in it exactly when ``moved_epoch = epoch`` (:func:`moves_returned`). The writer notes the move (:func:`note_moves`), and
the collection that made the write takes it (:func:`taken_moves`) and names the key twice in the write's epoch record:
once with its new tenant, once with the tenant it left, so each tenant's followers evict it.

**Restore.** A restore reads each bucket of a stamped table a second time for each moveable column of its tenancy,
``epoch >= X AND moved_epoch >= X`` within the same ``(bucket, epoch)`` range, the tenant named from ``moved_from``
(``table_restore``): every row that moved since X comes back for the tenant it left, as well as for its own.

**A known limit.** One row moved twice inside a restore's window (A to B, then B to C, both after X) keeps only its
last former tenant (B): A's followers that lost the log hear no eviction from a restore. Live, each move's record named
its own old tenant, so only a follower that missed both live records and is restored misses it.

A leaf module: the collections and backends both import it, and it imports nothing of theirs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Final

__all__ = [
    "MOVED_EPOCH_SUFFIX",
    "MOVED_FROM_SUFFIX",
    "moved_assignments",
    "moved_columns",
    "moved_epoch_column",
    "moved_from_column",
    "moved_returning",
    "moves_returned",
    "note_moves",
    "taken_moves",
]

#: what a moveable column's former value is kept in, beside it: ``<column>_moved_from``
MOVED_FROM_SUFFIX: Final[str] = "_moved_from"

#: what the epoch of the write that last moved it is kept in: ``<column>_moved_epoch``
MOVED_EPOCH_SUFFIX: Final[str] = "_moved_epoch"

#: the column the row's epoch is in (``row_epoch.EPOCH_COLUMN``, named here so this module stays a leaf)
_EPOCH: Final[str] = "epoch"


def moved_from_column(column: str) -> str:
    """the column a moveable ``column``'s former value is kept in.

    :param column: the moveable column
    :ptype column: str
    :return: ``<column>_moved_from``
    :rtype: str
    """
    return f"{column}{MOVED_FROM_SUFFIX}"


def moved_epoch_column(column: str) -> str:
    """the column the epoch of the write that last moved ``column`` is kept in.

    :param column: the moveable column
    :ptype column: str
    :return: ``<column>_moved_epoch``
    :rtype: str
    """
    return f"{column}{MOVED_EPOCH_SUFFIX}"


def moved_columns(column: str) -> tuple[str, str]:
    """both L3-only columns a moveable ``column`` carries: its former value's, and its move's epoch's.

    :param column: the moveable column
    :ptype column: str
    :return: ``(<column>_moved_from, <column>_moved_epoch)``
    :rtype: tuple[str, str]
    """
    return moved_from_column(column), moved_epoch_column(column)


def moved_assignments(column: str, new: str, new_epoch: str, *, target: str = "", quote: bool = False) -> str:
    """the two ``SET`` terms a write that sets a moveable ``column`` adds, in the same statement (the module's "The
    write"): its former value and its move's epoch, set only when the column's value changes.

    :param column: the moveable column
    :ptype column: str
    :param new: the expression the statement sets the column to (``EXCLUDED.c``, ``$3``)
    :ptype new: str
    :param new_epoch: the expression the statement sets the row's epoch to (its rising stamp)
    :ptype new_epoch: str
    :param target: how the statement names the stored row (an upsert's target relation), or empty for an ``UPDATE``'s
        own row
    :ptype target: str
    :param quote: whether the statement quotes its identifiers
    :ptype quote: bool
    :return: ``c_moved_from = CASE ... END, c_moved_epoch = CASE ... END``
    :rtype: str
    """
    moved_from, moved_epoch = moved_columns(column)

    def named(name: str) -> str:
        word = f'"{name}"' if quote else name
        return f"{target}.{word}" if target else word

    moving = f"{named(column)} IS DISTINCT FROM {new}"
    left = f'"{moved_from}"' if quote else moved_from
    stamped = f'"{moved_epoch}"' if quote else moved_epoch
    return (
        f"{left} = CASE WHEN {moving} THEN {named(column)} ELSE {named(moved_from)} END, "
        f"{stamped} = CASE WHEN {moving} THEN {new_epoch} ELSE {named(moved_epoch)} END"
    )


def moved_returning(columns: Sequence[str], *, quote: bool = False) -> str:
    """the ``RETURNING`` clause a write that sets the moveable ``columns`` ends with: each one's two columns and the
    row's epoch, so the writer learns from the statement itself whether, and from where, the row moved.

    :param columns: the moveable columns the statement sets
    :ptype columns: Sequence[str]
    :param quote: whether to quote the identifiers
    :ptype quote: bool
    :return: `` RETURNING epoch, c_moved_from, c_moved_epoch, ...``
    :rtype: str
    """
    names = [_EPOCH, *(name for column in columns for name in moved_columns(column))]
    return " RETURNING " + ", ".join(f'"{name}"' if quote else name for name in names)


def moves_returned(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[tuple[str, Any], ...]:
    """the moves the write that returned ``row`` (:func:`moved_returning`) made: each moveable column it moved, with the
    value it left. A column moved in that statement exactly when its ``moved_epoch`` is the row's new epoch.

    :param row: the row the statement returned
    :ptype row: Mapping[str, Any]
    :param columns: the moveable columns the statement set
    :ptype columns: Sequence[str]
    :return: ``(column, former value)`` per column the statement moved, in ``columns``' order
    :rtype: tuple[tuple[str, Any], ...]
    """
    epoch = row.get(_EPOCH)
    if epoch is None:
        return ()
    return tuple(
        (column, row.get(moved_from_column(column)))
        for column in columns
        if row.get(moved_epoch_column(column)) == epoch
    )


#: the moves the stamped writes made in this context, a table's latest only: ``((table, moves), ...)``, replaced whole
#: (:func:`note_moves`, :func:`taken_moves`). A context variable, as a write's statement stamp is
#: (``broker_generation.note_statement_stamp``): the store that ran the statement notes it, the collection that asked
#: for the write takes it, in the same task
_MOVES: ContextVar[tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]] = ContextVar(
    "threetears_tenant_moves", default=()
)


def note_moves(table: str, moves: tuple[tuple[str, Any], ...]) -> None:
    """note that the write of ``table`` just made in this context moved the row: each column it moved and the value it
    left (:func:`moves_returned`). Replaces any earlier note of the table: one write's moves at a time.

    :param table: the table written
    :ptype table: str
    :param moves: ``(column, former value)`` per column moved; empty notes that the write moved nothing
    :ptype moves: tuple[tuple[str, Any], ...]
    :return: nothing
    :rtype: None
    """
    held = tuple(pair for pair in _MOVES.get() if pair[0] != table)
    _MOVES.set((*held, (table, moves)) if moves else held)


def taken_moves(table: str) -> tuple[tuple[str, Any], ...]:
    """the moves the last write of ``table`` in this context made (:func:`note_moves`), taken: the next write starts
    afresh.

    :param table: the table
    :ptype table: str
    :return: ``(column, former value)`` per column moved; empty when the write moved nothing
    :rtype: tuple[tuple[str, Any], ...]
    """
    held = _MOVES.get()
    found = next((moves for name, moves in held if name == table), ())
    if found:
        _MOVES.set(tuple(pair for pair in held if pair[0] != table))
    return found
