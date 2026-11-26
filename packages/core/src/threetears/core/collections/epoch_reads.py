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
  the ``(epoch_bucket, epoch)`` index, every bucket whatever changed, the gone rows included.

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

**What a row carries.** Its key's values as text (a ``timestamptz`` as its UTC instant), and, for a column-tenancy
table, its tenant column as its one invalidation column: what the restore's records carry
(:func:`~threetears.core.collections.table_restore.TableChanges`). A table's other declared invalidation columns
are not stored with its versions' restore declaration, so no reader of the range names them (the restore's own
limit).

Every read is keyed: a catalog statement, a keyed read of the scope epochs table, and keyset pages of one index range
each; never a table whole.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from typing import Any, Final

from threetears.core.collections.noted_keys import NotedKeysCursor, NotedKeysPage
from threetears.core.collections.table_restore import (
    RestoredTable,
    StampedTable,
    VersionedTable,
    changed_scopes,
    key_term,
    restored_tables,
    run_marker_changes,
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
]

#: the low bits of an epoch (:mod:`threetears.core.ddl.row_epoch`): the logical counter within one wall millisecond
_LOGICAL_BITS: Final = 15

#: the text column a row's key value ``i`` is read under
_KEY_TEXT: Final = "_key_text_{}"

#: the column a row's tenant token is read under
_TENANT: Final = "_tenant"

#: the column a column-tenancy row's tenant column value is read under
_TENANT_VALUE: Final = "_tenant_value"

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
) -> tuple[bool, tuple[Fragment, ...]]:
    """whether a read of ``tenant``'s rows reads ``table``, and the filter on its rows that keeps them to the tenant's.

    :param table: the relation
    :ptype table: RestoredTable
    :param platform_schema: the platform schema
    :ptype platform_schema: str
    :param tenant: the tenant, ``None`` for ``_all``
    :ptype tenant: str | None
    :param every: every tenant's rows
    :ptype every: bool
    :return: ``(read, filters)``
    :rtype: tuple[bool, tuple[Fragment, ...]]
    """
    if every:
        return True, ()
    if table.tenancy.kind == "column":
        named = column_tenant_sql(str(table.tenancy.column))
        if tenant is None:
            return True, (Fragment(f"{named} IS NULL", ()),)
        return True, (Fragment(f"{named} = $1", (tenant,)),)
    return tenant_of_relation(table, platform_schema) == tenant, ()


@dataclass(frozen=True, slots=True)
class RangeRow:
    """one row (a version, a gone marker, a stamped row) a read of an epoch range found.

    :ivar keys: its key's values as text, in key order
    :ivar tenant: its tenant token, ``None`` for ``_all``
    :ivar columns: its invalidation columns (a column-tenancy table's tenant column), ``None`` for none
    """

    keys: tuple[str, ...]
    tenant: str | None
    columns: dict[str, str | None] | None


def _columns(table: RestoredTable) -> list[str | Selected]:
    """what a range page reads: the sort columns as stored (the cursor's), each key's text, and the row's tenant.

    :param table: the relation
    :ptype table: RestoredTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    tenancy = table.tenancy
    if tenancy.kind == "column":
        tenant = column_tenant_sql(str(tenancy.column))
    else:
        tenant = "NULL::text"
    read: list[str | Selected] = [EPOCH_COLUMN, *table.keys]
    read.extend(Selected(key_term(table, column), _KEY_TEXT.format(index)) for index, column in enumerate(table.keys))
    read.append(Selected(tenant, _TENANT))
    if tenancy.kind == "column":
        read.append(Selected(f"t.{quote_identifier(str(tenancy.column))}::text", _TENANT_VALUE))
    return read


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
        return [(bucket, None) for bucket in range(table.buckets)]
    prefix = table.scope_prefix
    scopes = await changed_scopes(pool, table.schema, table.epochs, floor)
    return [(0, value) for value in sorted({scope[len(prefix) :] for scope in scopes if scope.startswith(prefix)})]


def _row(table: RestoredTable, raw: dict[str, Any]) -> RangeRow:
    """a range page's row as the reader hands it on.

    :param table: the relation
    :ptype table: RestoredTable
    :param raw: the row as read
    :ptype raw: dict[str, Any]
    :return: the row
    :rtype: RangeRow
    """
    keys = tuple(str(raw[_KEY_TEXT.format(index)]) for index in range(len(table.keys)))
    tenant = raw[_TENANT]
    columns = None
    if table.tenancy.kind == "column":
        columns = {str(table.tenancy.column): raw[_TENANT_VALUE]}
    return RangeRow(keys=keys, tenant=None if tenant is None else str(tenant), columns=columns)


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
) -> KeyPage:
    """one keyset page of the rows of one index range of ``table`` stamped in ``[floor, ceiling]``, every version and
    gone marker (or gone row) included.

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
    :return: the page
    :rtype: KeyPage
    """
    bucket, scope = station
    by: dict[str, Any] = {table.bucket: bucket} if isinstance(table, StampedTable) else {table.scope: scope}
    return await keyset_page(
        pool,
        table.relation,
        columns=_columns(table),
        by=by,
        order=order,
        limit=limit,
        cursor=after,
        where=[
            Fragment(f't."{EPOCH_COLUMN}" >= $1', (floor,)),
            Fragment(f't."{EPOCH_COLUMN}" <= $1', (ceiling,)),
            *filters,
        ],
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
    at a time, each one index range read a keyset page a statement. For a reader that stops early (the writers'
    backstop, once it holds more rows than one record names).

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
                yield [_row(table, raw) for raw in found.rows]
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
    """whether a run of any of ``tables`` was deleted in the range under the asked tenant: a run marker's prefix is
    a reach no key list names (the notes' whole-table note, until 0.95).

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
        if not isinstance(table, VersionedTable) or not table.run_markers:
            continue
        values = [value for _, value in await _stations(pool, table, floor) if value is not None]
        for change in await run_marker_changes(pool, table, values, floor, ceiling):
            owner = change.tenant if table.tenancy.kind == "column" else tenant_of_relation(table, platform_schema)
            if every or owner == tenant:
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
    tables = [(each, filters) for each, (read, filters) in asked if read]
    whole = cursor is None and await _whole_in_range(
        pool,
        [each for each, _ in tables],
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

    schemas = [each.schema for each, _ in tables]
    start = 0
    if cursor is not None:
        if cursor.schema not in schemas:
            return "not_covered"
        start = schemas.index(cursor.schema)
    for each, filters in tables[start:]:
        order = _order(each, await _key_types(pool, each))
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
                )
                for index, raw in enumerate(found.rows):
                    row = _row(each, raw)
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
