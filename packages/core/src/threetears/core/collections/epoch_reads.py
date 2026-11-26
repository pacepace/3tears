"""what a table changed in an epoch range, read from the table itself: the restore's own read, narrowed to the range.

Append-only L3 (``design-append-only-l3.md``, decision 3, Pace 2026-11-25): every version of a versioned table and
every row of a stamped table carries its write's epoch, a hybrid logical clock stamp ``wall_ms << 15 | logical``. A
restore of a lost change log reads each table's versions with ``epoch >= X``
(:mod:`~threetears.core.collections.table_restore`). The readers that read the change-note ring until 0.95 (chunk 7)
read the same thing with a ceiling too, through the same indexes: no new mechanism.

- **Versioned tables, a scope at a time.** The scopes a stamped write touched since the range's floor, from the scope
  epochs table's ``last_epoch`` index (:func:`~threetears.core.collections.table_restore.changed_scopes`), each read
  as one range of the ``<table>_scope_epoch`` index ``(scope, epoch, key...)``: the scope bound, ``epoch`` between
  the floor and the ceiling, every version and gone marker included (``with_gone``). A run marker in the range
  (:mod:`~threetears.core.ddl.run_markers`) is read by the marker table's key
  (:func:`~threetears.core.collections.table_restore.run_marker_changes`).
- **Stamped tables, a bucket at a time.** Each of the :data:`~threetears.core.ddl.row_epoch.RESTORE_BUCKETS` ranges of
  the ``(epoch_bucket, epoch)`` index, every bucket whatever changed, the gone rows included; then, for a table whose
  rows a write moves (``TableSchema.moveable``), each bucket's range of the moves table's key
  (:mod:`~threetears.core.ddl.tenant_moves`), as further ranges of the walk.

**A bracket on the database's clock becomes an epoch range** (:func:`epoch_range`): every row a write in the bracket
stamped lies in ``[since << 15, ((until + 1 ms) << 15) - 1]``. A writer's clock is never behind the bracket by more
than the skew its makers already widen it by (:data:`~threetears.core.utils.write_ledger.CLOCK_SKEW`).

**A page at a time** (:func:`epoch_keys_page`): the walk goes a relation (``schema.table``) at a time in schema
order, a scope (in name order) or a bucket at a time within it, each a keyset page in the index's own order
(``epoch``, then the key) continued by the page's cursor: no offset, no sort past one epoch's rows. The cursor
(:class:`~threetears.core.collections.noted_keys.NotedKeysCursor`) names the relation's schema, the scope or bucket,
and the keyset place in it. A page ends at ``limit`` rows or ``max_bytes`` of keys, whichever first.

**Whose rows.** A tenant's own (:mod:`threetears.core.tenancy`): a schema-tenancy table's in the tenant's schema; a
column-tenancy table's rows whose tenant column names it; an undeclared table's in its pod's schema (``_all`` in the
platform schema). Or every tenant's (an operator's check, a platform process that caches every tenant's rows).

**What a row carries.** Its key's values as text (a ``timestamptz`` as its UTC instant), and every declared
invalidation column its restore declaration names (a column tenancy's tenant column among them): what the restore's
records carry (:func:`~threetears.core.collections.table_restore.TableChanges`), as the change notes did. Each move of a
stamped row in the range (``TableSchema.moveable``) is a row of the moves table naming the key with the values and the
tenant the row left: the followers of each old tenant and old value hear it, as the ring's old-row note had them, a row
moved twice in the range once for each tenant it left.

**A reset in the range is the whole table** (:func:`whole_changes`): a ``TRUNCATE`` leaves no row, so its reset marker
(:func:`~threetears.core.collections.table_restore.table_resets`) says so, under every tenant of the table.

Every read is keyed: a catalog statement, a keyed read of the scope epochs table, and keyset pages of one index range
each; never a table whole.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from typing import Any, Final

from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage
from threetears.core.collections.table_restore import (
    RestoredTable,
    StampedTable,
    VersionedTable,
    changed_scopes,
    column_value_sql,
    key_term,
    moved_tenant_sql,
    moves_relation,
    restored_tables,
    run_marker_changes,
    table_resets,
    tenant_value_sql,
)
from threetears.core.ddl.row_epoch import EPOCH_COLUMN
from threetears.core.keyed import Fragment, KeyPage, Selected, Sort, keyset_cursor, keyset_page
from threetears.core.sql_fragments import quote_identifier
from threetears.core.tenancy import column_tenant_sql, schema_tenant

__all__ = [
    "RangeRow",
    "epoch_keys_page",
    "epoch_range",
    "range_rows",
    "relations_named",
    "tenant_of_relation",
    "whole_changes",
]

#: the low bits of an epoch (:mod:`threetears.core.ddl.row_epoch`): the logical counter within one wall millisecond
_LOGICAL_BITS: Final = 15

#: the text column a row's key value ``i`` is read under
_KEY_TEXT: Final = "_key_text_{}"

#: the column a row's tenant token is read under
_TENANT: Final = "_tenant"

#: the text column a row's invalidation column ``i`` (its restore declaration's ``columns[i]``) is read under
_COLUMN_TEXT: Final = "_column_text_{}"

#: a moves page's order: the moves table's key after its bound table and bucket (tenant_moves)
_MOVES_ORDER: Final = (Sort("epoch", cast="bigint"), Sort("keys"))

#: a row's key column types, from the catalog: what a keyset cursor's text is cast back to
_KEY_TYPES_SQL: Final = """
SELECT a.attname::text AS column_name, format_type(a.atttypid, a.atttypmod) AS column_type
FROM pg_attribute a
WHERE a.attrelid = to_regclass($1) AND a.attname::text = ANY($2::text[]) AND a.attnum > 0 AND NOT a.attisdropped
"""

#: the bytes of framing a page counts for each key value and each column, as the notes' pages counted
_FRAMING: Final = 8


def epoch_range(bracket: tuple[datetime, datetime]) -> tuple[int, int]:
    """the epochs a write inside ``bracket`` stamped: from the first epoch of its start's millisecond to the last
    epoch of its end's.

    :param bracket: ``(since, until)``, timezone-aware
    :ptype bracket: tuple[datetime, datetime]
    :return: ``(floor, ceiling)``, both included
    :rtype: tuple[int, int]
    :raises ValueError: when the bracket ends before it starts
    """
    since, until = bracket
    if until < since:
        raise ValueError(f"a bracket ends after it starts: {since.isoformat()} .. {until.isoformat()}")
    low = max(0, int(since.timestamp() * 1000))
    high = max(0, int(until.timestamp() * 1000))
    return low << _LOGICAL_BITS, ((high + 1) << _LOGICAL_BITS) - 1


async def relations_named(pool: Any, table: str) -> list[RestoredTable]:
    """every relation of the database named ``table`` that carries an epoch, versioned or stamped, by their restore
    declarations (:func:`~threetears.core.collections.table_restore.restored_tables`): in schema order.

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the table's name
    :ptype table: str
    :return: the relations; empty for a table that carries no epoch
    :rtype: list[RestoredTable]
    """
    return sorted((each for each in await restored_tables(pool) if each.table == table), key=lambda each: each.schema)


def tenant_of_relation(table: RestoredTable, platform_schema: str) -> str | None:
    """the tenant every row of a schema-tenancy or undeclared ``table`` belongs to, ``None`` for ``_all``. A
    column-tenancy table's rows each name their own (:func:`range_rows`' ``tenant``); this answers ``None`` for it.

    :param table: the relation
    :ptype table: RestoredTable
    :param platform_schema: the platform schema: an undeclared table there is ``_all``'s
    :ptype platform_schema: str
    :return: the tenant
    :rtype: str | None
    """
    kind = table.tenancy.kind
    if kind == "schema":
        return schema_tenant(table.schema)
    if kind == "column" or table.schema == platform_schema:
        return None
    return schema_tenant(table.schema)


def _asked(
    table: RestoredTable, platform_schema: str, tenant: str | None, every: bool
) -> tuple[bool, tuple[Fragment, ...], tuple[Fragment, ...]]:
    """whether a read of ``tenant``'s rows reads ``table``, the filter on its rows that keeps them to the tenant's, and
    the filter on its moves that keeps them to those that left the tenant.

    :param table: the relation
    :ptype table: RestoredTable
    :param platform_schema: the platform schema
    :ptype platform_schema: str
    :param tenant: the tenant, ``None`` for ``_all``
    :ptype tenant: str | None
    :param every: every tenant's rows
    :ptype every: bool
    :return: ``(read, filters, move_filters)``
    :rtype: tuple[bool, tuple[Fragment, ...], tuple[Fragment, ...]]
    """
    if every:
        return True, (), ()
    if table.tenancy.kind == "column":
        named = column_tenant_sql(str(table.tenancy.column))
        if tenant is None:
            return True, (Fragment(f"{named} IS NULL", ()),), (Fragment('t."tenant" IS NULL', ()),)
        return True, (Fragment(f"{named} = $1", (tenant,)),), (Fragment('t."tenant" = $1', (tenant,)),)
    return tenant_of_relation(table, platform_schema) == tenant, (), ()


@dataclass(frozen=True, slots=True)
class RangeRow:
    """one row (a version, a gone marker, a stamped row) a read of an epoch range found.

    :ivar keys: its key's values as text, in key order
    :ivar tenant: its tenant token, ``None`` for ``_all``
    :ivar columns: its declared invalidation columns (a column-tenancy table's tenant column among them), ``None`` for
        none
    """

    keys: tuple[str, ...]
    tenant: str | None
    columns: dict[str, str | None] | None


def _columns(table: RestoredTable) -> list[str | Selected]:
    """what a range page reads: the sort columns as stored (the cursor's), each key's text, the row's tenant and every
    declared invalidation column's text.

    :param table: the relation
    :ptype table: RestoredTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    read: list[str | Selected] = [EPOCH_COLUMN, *table.keys]
    read.extend(Selected(key_term(table, column), _KEY_TEXT.format(index)) for index, column in enumerate(table.keys))
    read.append(Selected(tenant_value_sql(table), _TENANT))
    read.extend(
        Selected(column_value_sql(table, column), _COLUMN_TEXT.format(index))
        for index, column in enumerate(table.columns)
    )
    return read


def _move_columns(table: StampedTable) -> list[str | Selected]:
    """what a moves page reads (:mod:`~threetears.core.ddl.tenant_moves`): the sort columns as stored (the cursor's),
    the tenant the move left, and the key and the invalidation columns as it left them, each its JSON text.

    :param table: the moved table
    :ptype table: StampedTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    return [EPOCH_COLUMN, "keys", Selected(moved_tenant_sql(table), _TENANT), "columns"]


async def _key_types(pool: Any, table: RestoredTable) -> dict[str, str]:
    """each key column's SQL type, from the catalog: what a cursor's text is cast back to.

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the relation
    :ptype table: RestoredTable
    :return: column -> type
    :rtype: dict[str, str]
    """
    named = f"{quote_identifier(table.schema)}.{quote_identifier(table.table)}"
    rows = await pool.fetch(_KEY_TYPES_SQL, named, list(table.keys))
    return {str(row["column_name"]): str(row["column_type"]) for row in rows}


def _order(table: RestoredTable, types: dict[str, str]) -> list[Sort]:
    """a range page's order: the index's own after its bound scope or bucket, the epoch then the key.

    :param table: the relation
    :ptype table: RestoredTable
    :param types: each key column's type
    :ptype types: dict[str, str]
    :return: the order
    :rtype: list[Sort]
    """
    return [Sort(EPOCH_COLUMN, cast="bigint"), *(Sort(column, cast=types.get(column, "text")) for column in table.keys)]


async def _stations(pool: Any, table: RestoredTable, floor: int) -> list[tuple[int, str | None]]:
    """the index ranges a read of ``table`` from ``floor`` walks, in order: a versioned table's changed scopes by name
    (``(0, value)``), a stamped table's every bucket (``(bucket, None)``).

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the relation
    :ptype table: RestoredTable
    :param floor: the range's floor
    :ptype floor: int
    :return: the ranges
    :rtype: list[tuple[int, str | None]]
    """
    if isinstance(table, StampedTable):
        # each bucket's rows, then (a table a write moves) each bucket's moves: slot ``buckets + b`` is bucket b's
        moved = range(table.buckets, 2 * table.buckets) if table.moved_columns else range(0)
        return [(slot, None) for slot in (*range(table.buckets), *moved)]
    prefix = table.scope_prefix
    scopes = await changed_scopes(pool, table.schema, table.epochs, floor)
    return [(0, value) for value in sorted({scope[len(prefix) :] for scope in scopes if scope.startswith(prefix)})]


def _moves_slot(table: RestoredTable, slot: int) -> int | None:
    """the bucket whose moves a walk's ``slot`` reads (:func:`_stations`); None for a slot of a table's own rows.

    :param table: the relation
    :ptype table: RestoredTable
    :param slot: the slot
    :ptype slot: int
    :return: the bucket, or None
    :rtype: int | None
    """
    if isinstance(table, StampedTable) and slot >= table.buckets:
        return slot - table.buckets
    return None


def _rows(table: RestoredTable, raw: dict[str, Any], *, moved: bool = False) -> RangeRow:
    """a range page's row as the reader hands it on: a row of the table with its own tenant and values, or a move with
    the tenant and values the row left.

    :param table: the relation
    :ptype table: RestoredTable
    :param raw: the row as read
    :ptype raw: dict[str, Any]
    :param moved: the row is a moves row (:func:`_move_columns`)
    :ptype moved: bool
    :return: the row
    :rtype: RangeRow
    """
    tenant = raw[_TENANT]
    named = None if tenant is None else str(tenant)
    if moved:
        # function-local and read-only: one moves row's own JSON, a key and a few columns
        left = json.loads(raw["columns"]) if raw["columns"] is not None else None
        columns = {column: left.get(column) for column in table.columns} if table.columns and left else None
        return RangeRow(keys=tuple(str(key) for key in json.loads(raw["keys"])), tenant=named, columns=columns)
    keys = tuple(str(raw[_KEY_TEXT.format(index)]) for index in range(len(table.keys)))
    columns = (
        {column: raw[_COLUMN_TEXT.format(index)] for index, column in enumerate(table.columns)}
        if table.columns
        else None
    )
    return RangeRow(keys=keys, tenant=named, columns=columns)


async def _range_page(
    pool: Any,
    table: RestoredTable,
    station: tuple[int, str | None],
    floor: int,
    ceiling: int,
    *,
    filters: tuple[Fragment, ...],
    order: list[Sort],
    after: str | None,
    limit: int,
    move_filters: tuple[Fragment, ...] = (),
) -> KeyPage:
    """one keyset page of the rows of one index range of ``table`` stamped in ``[floor, ceiling]``, every version and
    gone marker (or gone row) included; for a moves slot (:func:`_stations`), of the bucket's moves in the range, one
    range of the moves table's key.

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the relation
    :ptype table: RestoredTable
    :param station: the scope (versioned) or bucket (stamped)
    :ptype station: tuple[int, str | None]
    :param floor: the range's floor
    :ptype floor: int
    :param ceiling: the range's ceiling
    :ptype ceiling: int
    :param filters: a tenant's filter on the rows
    :ptype filters: tuple[Fragment, ...]
    :param order: the page's order (:func:`_order`)
    :ptype order: list[Sort]
    :param after: the cursor the last page of this range answered; ``None`` for its first
    :ptype after: str | None
    :param limit: the most rows
    :ptype limit: int
    :param move_filters: a tenant's filter on the moves (a moves slot's)
    :ptype move_filters: tuple[Fragment, ...]
    :return: the page
    :rtype: KeyPage
    """
    bucket, scope = station
    bounds = [Fragment(f't."{EPOCH_COLUMN}" >= $1', (floor,)), Fragment(f't."{EPOCH_COLUMN}" <= $1', (ceiling,))]
    moved = _moves_slot(table, bucket)
    if moved is not None and isinstance(table, StampedTable):
        # the moves table's key prefix, the table and the bucket, then its own order (its own columns, as
        # TENANT_MOVES_SCHEMA spells them)
        return await keyset_page(
            pool,
            moves_relation(table),
            columns=_move_columns(table),
            by={"table_name": table.table, "scope": str(moved)},
            order=_MOVES_ORDER,
            limit=limit,
            cursor=after,
            where=[*bounds, *move_filters],
        )
    by: dict[str, Any] = {table.bucket: bucket} if isinstance(table, StampedTable) else {table.scope: scope}
    return await keyset_page(
        pool,
        table.relation,
        columns=_columns(table),
        by=by,
        order=order,
        limit=limit,
        cursor=after,
        where=[*bounds, *filters],
        # every version and gone marker is a change the range holds: read as stored, never the newest relation
        with_gone=True,
    )


async def range_rows(
    pool: Any,
    table: RestoredTable,
    floor: int,
    ceiling: int,
    *,
    page: int,
) -> AsyncIterator[list[RangeRow]]:
    """every row of ``table`` stamped in ``[floor, ceiling]``, every tenant's, a page at a time: a scope or a bucket
    at a time, each one index range read a keyset page a statement; then each move in the range, with the tenant and
    values the row left. For a reader that stops early (the writers' backstop, once it holds more rows than
    one record names).

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the relation
    :ptype table: RestoredTable
    :param floor: the range's floor
    :ptype floor: int
    :param ceiling: the range's ceiling
    :ptype ceiling: int
    :param page: the most rows a statement reads
    :ptype page: int
    :return: each page's rows
    :rtype: AsyncIterator[list[RangeRow]]
    """
    order = _order(table, await _key_types(pool, table))
    for station in await _stations(pool, table, floor):
        after: str | None = None
        while True:
            found = await _range_page(
                pool, table, station, floor, ceiling, filters=(), order=order, after=after, limit=page
            )
            if found.rows:
                moved = _moves_slot(table, station[0]) is not None
                yield [_rows(table, raw, moved=moved) for raw in found.rows]
            if found.next_cursor is None:
                break
            after = found.next_cursor


def _bytes(row: RangeRow) -> int:
    """the bytes a page counts for one row: its keys and its invalidation columns, each with its framing.

    :param row: the row
    :ptype row: RangeRow
    :return: the bytes
    :rtype: int
    """
    counted = sum(len(value) + _FRAMING for value in row.keys)
    if row.columns:
        counted += sum(len(name) + len(value or "") + _FRAMING for name, value in row.columns.items())
    return counted


async def whole_changes(pool: Any, table: RestoredTable, floor: int, ceiling: int) -> list[NotedChange]:
    """what of ``table`` the range changed that no key names: its resets (:func:`~threetears.core.collections.
    table_restore.table_resets`, the whole table) and a versioned table's runs deleted (their run markers' prefixes),
    each read by the run-marker table's key.

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param table: the relation
    :ptype table: RestoredTable
    :param floor: the range's floor
    :ptype floor: int
    :param ceiling: the range's ceiling
    :ptype ceiling: int
    :return: the whole-table changes, resets first
    :rtype: list[NotedChange]
    """
    if not table.run_markers:
        return []
    found = await table_resets(pool, table, floor, ceiling)
    if isinstance(table, VersionedTable):
        values = [value for _, value in await _stations(pool, table, floor) if value is not None]
        found.extend(await run_marker_changes(pool, table, values, floor, ceiling))
    return found


async def _whole_in_range(
    pool: Any,
    tables: Sequence[RestoredTable],
    floor: int,
    ceiling: int,
    *,
    platform_schema: str,
    tenant: str | None,
    every: bool,
) -> bool:
    """whether any of ``tables`` was reset, or a run of it deleted, in the range under the asked tenant
    (:func:`whole_changes`): a reach no key list names (the notes' whole-table note, until 0.95). A column-tenancy
    table's change of no one tenant (a reset; a run whose prefix names no tenant) is every tenant's.

    :param pool: the L3 pool, or one connection
    :ptype pool: Any
    :param tables: the relations read
    :ptype tables: Sequence[RestoredTable]
    :param floor: the range's floor
    :ptype floor: int
    :param ceiling: the range's ceiling
    :ptype ceiling: int
    :param platform_schema: the platform schema
    :ptype platform_schema: str
    :param tenant: the asked tenant
    :ptype tenant: str | None
    :param every: every tenant's
    :ptype every: bool
    :return: whether one was
    :rtype: bool
    """
    for table in tables:
        for change in await whole_changes(pool, table, floor, ceiling):
            if table.tenancy.kind == "column":
                if every or change.tenant is None or change.tenant == tenant:
                    return True
            elif every or tenant_of_relation(table, platform_schema) == tenant:
                return True
    return False


async def epoch_keys_page(
    pool: Any,
    *,
    table: str,
    platform_schema: str,
    tenant_token: str | None,
    noted_between: tuple[datetime, datetime],
    cursor: NotedKeysCursor | None,
    limit: int,
    max_bytes: int,
    every_tenant: bool = False,
) -> NotedKeysPage | str:
    """a page of the keys ``table`` changed in a commit's bracket, of ``tenant_token``'s rows (every tenant's with
    ``every_tenant``), read from the table's own versions or stamped rows (the module's walk).

    The bracket holds every key its commits changed (and perhaps keys other commits changed then, which a follower
    reads again harmlessly), each as often as it has versions there: a set's worth of reads, never a count. A run
    deleted in the bracket answers ``whole_table`` on the first page: its prefix is a reach no key list names.

    :param pool: the L3 pool (the broker's, on the platform schema), or one connection
    :ptype pool: Any
    :param table: the table's name
    :ptype table: str
    :param platform_schema: the platform schema: an undeclared table there is ``_all``'s
    :ptype platform_schema: str
    :param tenant_token: the tenant whose rows are read (the asker's), ``None`` for ``_all``: never another's
    :ptype tenant_token: str | None
    :param noted_between: the commit's bracket, ``(from, until)``
    :ptype noted_between: tuple[datetime, datetime]
    :param cursor: where the last page ended; ``None`` for the first
    :ptype cursor: NotedKeysCursor | None
    :param limit: the most rows a page reads
    :ptype limit: int
    :param max_bytes: the most bytes of keys a page holds
    :ptype max_bytes: int
    :param every_tenant: every tenant's rows, ``tenant_token`` unused: an operator's check, a platform process's reader
    :ptype every_tenant: bool
    :return: the page; or ``"notes_unavailable"`` when the table carries no epoch (no read can name its keys), or
        ``"not_covered"`` when the cursor names a place the walk no longer holds
    :rtype: NotedKeysPage | str
    """
    if limit < 1 or max_bytes < 1:
        raise ValueError("a page reads at least one row and one byte")
    floor, ceiling = epoch_range(noted_between)
    relations = await relations_named(pool, table)
    if not relations:
        return "notes_unavailable"
    asked = [(each, _asked(each, platform_schema, tenant_token, every_tenant)) for each in relations]
    tables = [(each, filters, moving) for each, (read, filters, moving) in asked if read]
    whole = cursor is None and await _whole_in_range(
        pool,
        [each for each, _, _ in tables],
        floor,
        ceiling,
        platform_schema=platform_schema,
        tenant=tenant_token,
        every=every_tenant,
    )
    keys: list[tuple[str, ...]] = []
    columns: list[dict[str, str | None] | None] = []
    used = 0

    def page(next_at: NotedKeysCursor | None) -> NotedKeysPage:
        return NotedKeysPage(keys=tuple(keys), columns=tuple(columns), whole_table=whole, next=next_at)

    schemas = [each.schema for each, _, _ in tables]
    start = 0
    if cursor is not None:
        if cursor.schema not in schemas:
            return "not_covered"
        start = schemas.index(cursor.schema)
    for each, filters, moving in tables[start:]:
        rows_order = _order(each, await _key_types(pool, each))
        stations = [(slot, scope or "") for slot, scope in await _stations(pool, each, floor)]
        at: tuple[int, str] | None = None
        if cursor is not None and cursor.schema == each.schema:
            # a scope or bucket at or past the cursor's: a scope first touched since sorts where it falls
            at = (cursor.slot, cursor.scope or "")
            stations = [station for station in stations if station >= at]
        for slot, scope in stations:
            after = cursor.after if cursor is not None and (slot, scope) == at else None
            if after == _PAST:
                # the last page read this range to its end
                continue
            station = (slot, None if isinstance(each, StampedTable) else scope)
            place = partial(NotedKeysCursor, slot, each.schema, station[1])
            moved = _moves_slot(each, slot) is not None
            order = list(_MOVES_ORDER) if moved else rows_order
            while True:
                found = await _range_page(
                    pool,
                    each,
                    station,
                    floor,
                    ceiling,
                    filters=filters,
                    order=order,
                    after=after,
                    limit=limit - len(keys),
                    move_filters=moving,
                )
                for index, raw in enumerate(found.rows):
                    # the row (or the move, as the row left it): the asked tenant's, as the filter kept it
                    row = _rows(each, raw, moved=moved)
                    size = _bytes(row)
                    if keys and used + size > max_bytes:
                        # the page's bytes are spent: the next page starts at this row
                        return page(place(after if index == 0 else keyset_cursor(order, found.rows[index - 1])))
                    keys.append(row.keys)
                    columns.append(row.columns)
                    used += size
                if found.next_cursor is None:
                    break
                if len(keys) >= limit:
                    return page(place(found.next_cursor))
                after = found.next_cursor
            if len(keys) >= limit:
                # this range is read to its end: the next page starts past it
                return page(place(_PAST))
    return page(None)


#: a cursor's place naming a range read to its end: the next page starts at the range after it
_PAST: Final = "past"
