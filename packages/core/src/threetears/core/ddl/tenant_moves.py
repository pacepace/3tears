"""a stamped table's row that moves between tenants: the write that moves it appends one row to the moves table.

Append-only L3 retires the change-note ring (``design-append-only-l3.md``, chunk 7). A pod caches a table's rows by
tenant (its tenancy column, an invalidation column), so a write that changes a row's tenant column in place must be
heard by the tenant it LEFT as well as the one it joined: the old tenant's followers still hold the row. The ring's
old-row note was the only thing that told them. Pace chose (2026-11-25) to record each move as its own row of a small
append-only table, so a restore reads every move since X, not only a row's last.

**The declaration.** A stamped table (``ROW_STAMPED``) whose tenancy column can change names it in
``TableSchema(moveable=(column,))``. Each such column gets two L3-only columns beside it (:func:`moved_columns`): no
cache, entity or ``INSERT`` names them, so a new row holds NULL in both.

- ``<column>_moved_from``, the column's type, nullable: the value the row's last move left.
- ``<column>_moved_epoch``, ``BIGINT``, nullable: the epoch of the write that last moved it.

WHY they stay on the row: they are how ONE statement learns, exactly, that it moved the row and from what. An
``UPDATE``'s right-hand sides read the row as it was, so the ``SET`` terms (:func:`moved_assignments`) copy the old
value into ``_moved_from`` only when the value changes, and stamp ``_moved_epoch`` with the row's new epoch. The
statement's ``RETURNING`` then carries the old value out (PostgreSQL before 18, and YugabyteDB, have no ``OLD`` in
``RETURNING``), and the row moved in that statement exactly when ``moved_epoch = epoch``: a row's epoch rises on every
write, so an earlier move's epoch is always below it. A read of the old row in a CTE beside the write would see the
statement's snapshot, not the row the write locked, and miss a move a concurrent writer made.

**The moves table** (:data:`TENANT_MOVES_TABLE`, declared by
:data:`threetears.core.collections.tenant_moves.TENANT_MOVES_SCHEMA`, in the moved table's own schema): one row a move,
append-only, ``(table_name, scope, epoch, keys, tenant, columns)``, its key ``(table_name, scope, epoch, keys)``:

- ``table_name`` the moved table; ``scope`` the row's restore bucket (``epoch_bucket``) as text, so a restore reads the
  moves of a bucket beside the bucket's rows, and a table's moves spread over its buckets' ranges;
- ``epoch`` the moving write's epoch (the row's new one);
- ``keys`` the row's key as a record spells it, a JSON array of text (a ``timestamptz`` as its UTC instant);
- ``tenant`` the tenant token the row left (:func:`~threetears.core.tenancy.column_tenant_sql`), NULL for ``_all`` or
  for a table whose tenancy is not a column;
- ``columns`` every declared invalidation column as the row left it, a JSON object of text: a column the write moved
  holds the value it left, any other the row's value.

**The write** (:func:`with_moves_appended`): a statement that sets a moveable column becomes one statement,
no extra round trip::

    WITH moved AS (<the write> RETURNING <key, bucket, epoch, columns, moved_from, moved_epoch>),
         noted AS (INSERT INTO threetears_tenant_moves ... SELECT ... FROM moved AS t WHERE t.c_moved_epoch = t.epoch)
    SELECT <what the caller asked> FROM moved

A row that did not move appends nothing. Every write that may move a row goes through it: the single-row upsert and
fenced update (``SqlL3Backend``), the bulk upsert (``schema_sql.build_bulk_insert_sql``) and a keyed update whose values
:func:`~threetears.core.collections.schema_backed.stamped_values` made (its ``_moved_epoch`` value a
:class:`MoveStamp`, which ``update_keyed`` reads the moves table's append off).

**Live.** The statement returns each moved column's two columns and the row's epoch; the writer notes the move
(:func:`note_moves`), and the collection that made the write takes it (:func:`taken_moves`) and names the key twice in
the write's epoch record: once with its new tenant, once with the tenant it left.

**Restore.** A restore reads, beside each bucket of a stamped table, the moves table's rows of the table and bucket
with ``epoch >= X`` (one range of its key) and folds each into a record for the tenant and values the row left
(``table_restore``); the epoch-range readers read the same range with a ceiling (``epoch_reads``). A row moved A to B,
then B to C, both after X, is two moves rows: A's followers and B's both evict it.

**Cleanup.** A move below the cleanup floor F is read by no restore: quiet-time cleanup deletes the moves rows of each
table and bucket stamped below F (:func:`~threetears.core.collections.tenant_moves.clear_moves`).

A leaf module: the collections and backends both import it, and it imports nothing of theirs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final

from threetears.core.tenancy import column_tenant_sql
from threetears.interfaces.keyed.filters import Fragment, quote_identifier

__all__ = [
    "MOVED_EPOCH_SUFFIX",
    "MOVED_FROM_SUFFIX",
    "TENANT_MOVES_KEY",
    "TENANT_MOVES_TABLE",
    "TENANT_MOVE_COLUMNS",
    "MoveRecord",
    "MoveStamp",
    "key_text_sql",
    "moved_assignments",
    "moved_columns",
    "moved_epoch_column",
    "moved_from_column",
    "moved_returning",
    "move_record_of_values",
    "moves_returned",
    "note_moves",
    "taken_moves",
    "with_moves_appended",
]

#: the moves table, in the same schema as the stamped tables whose moves it holds
TENANT_MOVES_TABLE: Final[str] = "threetears_tenant_moves"

#: its columns, in order (the module's "The moves table")
TENANT_MOVE_COLUMNS: Final[tuple[str, ...]] = ("table_name", "scope", "epoch", "keys", "tenant", "columns")

#: its key: a restore reads one table's bucket's moves since X as one range of it
TENANT_MOVES_KEY: Final[tuple[str, ...]] = ("table_name", "scope", "epoch", "keys")

#: the row's restore bucket column (``row_epoch.BUCKET_COLUMN``, named here so this module stays a leaf)
_BUCKET: Final[str] = "epoch_bucket"

#: a ``timestamptz`` key's text in a record: its UTC instant to the microsecond, as a change note spells it
_UTC_MICROS: Final[str] = '\'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"\''

#: what the wrapped statement names the write's returned rows, and the moves' append
_MOVED_ROWS: Final[str] = "threetears_moved"
_NOTED: Final[str] = "threetears_noted_moves"

#: what a moveable column's former value is kept in, beside it: ``<column>_moved_from``
MOVED_FROM_SUFFIX: Final[str] = "_moved_from"

#: what the epoch of the write that last moved it is kept in: ``<column>_moved_epoch``
MOVED_EPOCH_SUFFIX: Final[str] = "_moved_epoch"

#: the column the row's epoch is in (``row_epoch.EPOCH_COLUMN``, named here so this module stays a leaf)
_EPOCH: Final[str] = "epoch"


def key_text_sql(column: str, *, instant: bool, row: str = "t") -> str:
    """a key column of a row as a record spells it: its text, a ``timestamptz`` as its UTC instant. The one spelling the
    moves table's ``keys`` and a restore's read of a table's own rows share (``table_restore.key_term``).

    :param column: the key column, a TRUSTED identifier
    :ptype column: str
    :param instant: whether it is a ``timestamptz``
    :ptype instant: bool
    :param row: the row's alias in the statement
    :ptype row: str
    :return: the expression
    :rtype: str
    """
    value = f"{row}.{quote_identifier(column)}"
    if instant:
        return f"to_char({value} AT TIME ZONE 'UTC', {_UTC_MICROS})"
    return f"{value}::text"


def _literal(text: str) -> str:
    """``text`` as a SQL string literal.

    :param text: a TRUSTED name (a table's, a column's)
    :ptype text: str
    :return: the literal
    :rtype: str
    """
    return "'" + text.replace("'", "''") + "'"


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """what a stamped table's moves rows name, from its schema (``TableSchema.move_record``): the table, its key, its
    tenancy and the columns a restore's record carries, of which those a write may move.

    :ivar table: the table's name
    :ivar keys: its primary key's columns, in order
    :ivar instants: those of them that are ``timestamptz``
    :ivar tenant_column: its tenant column, for a column tenancy; None otherwise
    :ivar columns: its declared invalidation columns as its restore declaration names them (the tenant column among
        them), in order
    :ivar moved: those of them a write may move (``TableSchema.moveable``), in order; at least one
    """

    table: str
    keys: tuple[str, ...]
    instants: tuple[str, ...]
    tenant_column: str | None
    columns: tuple[str, ...]
    moved: tuple[str, ...]

    @property
    def returned(self) -> tuple[str, ...]:
        """the columns the write must return for its moves to be appended: the key, the bucket, the epoch, every
        recorded column and each moved column's two.

        :return: them, each once
        :rtype: tuple[str, ...]
        """
        named = (*self.keys, _BUCKET, _EPOCH, *self.columns, *(name for c in self.moved for name in moved_columns(c)))
        return tuple(dict.fromkeys(named))

    def _moved_in_write(self, column: str, row: str) -> str:
        """whether the write moved ``column``: its move's epoch is the row's new one.

        :param column: a moved column
        :ptype column: str
        :param row: the returned row's alias
        :ptype row: str
        :return: the condition
        :rtype: str
        """
        return f"{row}.{quote_identifier(moved_epoch_column(column))} = {row}.{quote_identifier(_EPOCH)}"

    def _left_value(self, column: str, row: str) -> str:
        """a recorded column as the row left it, its text: the value the write moved it off, else the row's own.

        :param column: a recorded column
        :ptype column: str
        :param row: the returned row's alias
        :ptype row: str
        :return: the expression
        :rtype: str
        """
        own = f"{row}.{quote_identifier(column)}::text"
        if column not in self.moved:
            return own
        left = f"{row}.{quote_identifier(moved_from_column(column))}::text"
        return f"(CASE WHEN {self._moved_in_write(column, row)} THEN {left} ELSE {own} END)"

    def _left_tenant(self, row: str) -> str:
        """the tenant token the row left: its tenant column's as the write moved it off, else its own; NULL for a
        table whose tenancy is not a column (a restore names its tenant from the table).

        :param row: the returned row's alias
        :ptype row: str
        :return: the expression
        :rtype: str
        """
        column = self.tenant_column
        if column is None:
            return "NULL::text"
        own = column_tenant_sql(column, row)
        if column not in self.moved:
            return own
        left = column_tenant_sql(moved_from_column(column), row)
        return f"(CASE WHEN {self._moved_in_write(column, row)} THEN {left} ELSE {own} END)"

    def append_sql(self, source: str) -> str:
        """the ``INSERT`` that appends one moves row a row of ``source`` (the write's returned rows) the write moved.

        :param source: the relation of the write's returned rows (:attr:`returned`), a TRUSTED name
        :ptype source: str
        :return: the statement
        :rtype: str
        """
        row = "t"
        keys = ", ".join(key_text_sql(key, instant=key in self.instants, row=row) for key in self.keys)
        pairs = ", ".join(f"{_literal(column)}, {self._left_value(column, row)}" for column in self.columns)
        moved = " OR ".join(f"({self._moved_in_write(column, row)})" for column in self.moved)
        selected = (
            _literal(self.table),
            f"{row}.{quote_identifier(_BUCKET)}::text",
            f"{row}.{quote_identifier(_EPOCH)}",
            f"array_to_json(ARRAY[{keys}])::text",
            self._left_tenant(row),
            f"json_build_object({pairs})::text",
        )
        return (
            f"INSERT INTO {TENANT_MOVES_TABLE} ({', '.join(TENANT_MOVE_COLUMNS)}) "
            f"SELECT {', '.join(selected)} FROM {source} AS {row} WHERE {moved}"
        )


def with_moves_appended(statement: str, record: MoveRecord, selected: Sequence[str]) -> str:
    """``statement`` (a write of ``record``'s table ending in a ``RETURNING`` that names :attr:`MoveRecord.returned`)
    as one statement that also appends a moves row for each row it moved (the module's "The write"), answering
    ``selected`` of each row it wrote.

    :param statement: the write, with its ``RETURNING``
    :ptype statement: str
    :param record: the table's moves
    :ptype record: MoveRecord
    :param selected: what the statement answers of the write's returned rows: select-list items, TRUSTED (column
        names, ``count(*) AS count``)
    :ptype selected: Sequence[str]
    :return: the statement
    :rtype: str
    :raises ValueError: when nothing is selected
    """
    if not selected:
        raise ValueError("a write that appends its moves answers something of its rows")
    return (
        f"WITH {_MOVED_ROWS} AS ({statement}), {_NOTED} AS ({record.append_sql(_MOVED_ROWS)}) "
        f"SELECT {', '.join(selected)} FROM {_MOVED_ROWS}"
    )


@dataclass(frozen=True)
class MoveStamp(Fragment):
    """a keyed ``UPDATE``'s value for a moveable column's ``_moved_epoch`` (``schema_backed.stamped_values``): a
    :class:`~threetears.interfaces.keyed.Fragment` the update renders as any other, and the table's moves
    (:attr:`record`), so ``update_keyed`` appends the moves rows in the same statement (:func:`with_moves_appended`).

    WHY a kind of its own, as ``row_epoch.RisingStamp`` is: the keyed layer knows a table by its name alone; the value
    is how the update learns that it may move a row, and what its moves rows name.

    :param record: the table's moves
    :ptype record: MoveRecord | None
    """

    record: MoveRecord | None = None


def move_record_of_values(values: Mapping[str, Any]) -> MoveRecord | None:
    """the moves a keyed ``UPDATE``'s ``values`` may make (a :class:`MoveStamp` among them).

    :param values: column -> its new value, as ``update_keyed`` takes them
    :ptype values: Mapping[str, Any]
    :return: the table's moves; None when the update moves nothing
    :rtype: MoveRecord | None
    """
    return next((value.record for value in values.values() if isinstance(value, MoveStamp) and value.record), None)


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
