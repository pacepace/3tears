"""a write by key: the rows a key names updated or deleted, never a table's rows by a scan; a stage applied by its key.

A collection's targeted write (a status flip, a stamp, a fence-checked save, a delete of one row, of a parent's rows
or of a batch of keys) names its rows by a leading part of a key, as a keyed read does::

    UPDATE <table> SET "c1" = $1, "c2" = now() WHERE "k1" = $2 AND "k2" = ANY($3::uuid[]) AND <where> RETURNING ...
    DELETE FROM <table> WHERE ("k1", "k2") IN (SELECT unnest($1::text[]), unnest($2::text[]::bigint[])) AND ...

``by`` pins columns (a value, ``None`` for ``IS NULL``, or an :class:`~threetears.core.keyed.AnyOf`) and ``keys``
names a batch of keys (a :class:`~threetears.core.keyed.KeyBatch`, ``batch`` keys a statement); ``where`` adds the
write's own guards (a status it must still hold, a fence value), each a fragment written as ``"col"`` from ``$1``; a
value set may be an expression (a :class:`~threetears.core.keyed.Fragment`: ``now()``) or an :class:`Increment`. A
write names a key: one with neither ``by`` nor ``keys`` is refused, never a write of every row.

:func:`update_from` sets a table's columns from a keyed derived select joined on the key; :func:`apply_stage` writes a
stage relation (a change set loaded beside the table) into it by the stage's key: replace, upsert or only the rows
that differ, on PostgreSQL, DuckDB and SQLite alike.

IO only: one statement a batch on the caller's connection, awaited in turn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.core.backends.protocol import parse_rowcount
from threetears.core.keyed._filters import (
    Derived,
    Fragment,
    binds_native_arrays,
    by_conditions,
    quoted_table,
    narrowing,
)
from threetears.core.keyed.counts import rows_affected
from threetears.core.keyed.rows import DEFAULT_KEY_BATCH, KeyBatch
from threetears.core.keyed.steps import Step, change_relation
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.writes import (
    Increment,
    KeyedWrite,
    StageMode,
    StageWrite,
    delete_keyed_statements,
    update_keyed_statements,
)

__all__ = [
    "Increment",
    "KeyedWrite",
    "StageMode",
    "StageWrite",
    "apply_stage",
    "delete_keyed",
    "update_from",
    "update_keyed",
]


async def _run(conn: Any, sql: str, params: Sequence[Any], returning: Sequence[str]) -> KeyedWrite:
    if returning:
        rows = [dict(row) for row in await conn.fetch(sql, *params)]
        return KeyedWrite(count=len(rows), rows=rows)
    status = await conn.execute(sql, *params)
    # the framework's reading of a status tag, as every hand-written write read it: a counting verb whose count
    # does not parse (a broker reply with no row_count) is refused, never read as "no rows"
    return KeyedWrite(count=parse_rowcount(status), rows=[])


async def _ran(conn: Any, statements: Sequence[tuple[str, list[Any]]], returning: Sequence[str]) -> KeyedWrite:
    """each statement run in turn on ``conn``, its counts and returned rows summed."""
    count = 0
    rows: list[dict[str, Any]] = []
    for sql, params in statements:
        written = await _run(conn, sql, params, returning)
        count += written.count
        rows += written.rows
    return KeyedWrite(count=count, rows=rows)


async def update_keyed(
    conn: Any,
    table: str,
    *,
    values: Mapping[str, Any],
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
    prefix: Mapping[str, str] | None = None,
) -> KeyedWrite:
    """set ``values`` on the rows of ``table`` that ``keys`` and ``by`` (and ``where``) name.

    :param conn: anything with an asyncpg-shaped ``execute`` (and ``fetch`` when ``returning``)
    :ptype conn: Any
    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param values: column -> its new value, an :class:`Increment`, or a :class:`~threetears.core.keyed.Fragment`
        expression (``now()``, its placeholders from ``$1``); columns are TRUSTED identifiers
    :ptype values: Mapping[str, Any]
    :param by: columns pinned (a leading part of the table's key, or of an index declared for the write), each a
        value, ``None`` (``IS NULL``) or an :class:`~threetears.core.keyed.AnyOf`
    :ptype by: Mapping[str, Any] | None
    :param keys: a batch of keys the rows are named by, ``batch`` keys a statement; none when None
    :ptype keys: KeyBatch | None
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1`` (a status the row must
        hold, a fence)
    :ptype where: Sequence[Fragment]
    :param returning: the columns the written rows return; none when empty
    :ptype returning: Sequence[str]
    :param batch: the most keys a statement names
    :ptype batch: int
    :param prefix: text key columns -> the prefix the rows written start with, a range in ``COLLATE "C"`` order
        (:func:`~threetears.core.keyed._filters.prefix_conditions`: it reads by a C-ordered index on the column, an
        ASC one on YugabyteDB); none when None
    :ptype prefix: Mapping[str, str] | None
    :return: the count written, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``values`` is empty, neither ``by``, ``keys`` nor ``prefix`` names a column, or a prefix
        is empty or not text
    """
    statements = update_keyed_statements(
        table,
        values=values,
        by=by,
        keys=keys,
        where=where,
        returning=returning,
        batch=batch,
        prefix=prefix,
        natively=binds_native_arrays(conn),
    )
    return await _ran(conn, statements, returning)


async def delete_keyed(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
    prefix: Mapping[str, str] | None = None,
) -> KeyedWrite:
    """delete the rows of ``table`` that ``keys`` and ``by`` (and ``where``) name.

    :param conn: anything with an asyncpg-shaped ``execute`` (and ``fetch`` when ``returning``)
    :ptype conn: Any
    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param by: columns pinned, as :func:`update_keyed` takes them
    :ptype by: Mapping[str, Any] | None
    :param keys: a batch of keys the rows are named by, ``batch`` keys a statement; none when None
    :ptype keys: KeyBatch | None
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param returning: the columns the deleted rows return; none when empty
    :ptype returning: Sequence[str]
    :param batch: the most keys a statement names
    :ptype batch: int
    :param prefix: text key columns -> the prefix the rows deleted start with, as :func:`update_keyed` takes it
    :ptype prefix: Mapping[str, str] | None
    :return: the count deleted, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when neither ``by``, ``keys`` nor ``prefix`` names a column, or a prefix is empty or not text
    """
    statements = delete_keyed_statements(
        table,
        by=by,
        keys=keys,
        where=where,
        returning=returning,
        batch=batch,
        prefix=prefix,
        natively=binds_native_arrays(conn),
    )
    return await _ran(conn, statements, returning)


async def update_from(
    conn: Any,
    table: str,
    *,
    key: Sequence[str],
    source: Derived | Step,
    columns: Sequence[str],
    by: Mapping[str, Any] | None = None,
    returning: Sequence[str] = (),
) -> KeyedWrite:
    """set ``columns`` of ``table``'s rows from a keyed derived select, joined on ``key``:
    ``UPDATE t SET "c" = s."c" FROM (<source>) AS s WHERE t."k" = s."k" AND <by>``.

    The source names the rows written (its own text keyed where it is written: a ``SELECT`` by a key batch, a
    stage's rows), so the update reads the table only by its key.

    :param conn: anything with an asyncpg-shaped ``execute`` (and ``fetch`` when ``returning``)
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: the columns the source joins on, the table's key (or a leading part of an index declared for the
        write); TRUSTED identifiers, the same names in the source. ``["rowid"]`` on DuckDB joins on the row itself
        (the source selects ``t.rowid AS rowid`` of the same table: a window computed over a state's rows)
    :ptype key: Sequence[str]
    :param source: the derived select (no parameters of its own), or a relational step with its parameters
        (:func:`~threetears.core.keyed.scan`, :func:`~threetears.core.keyed.windowed`)
    :ptype source: Derived | Step
    :param columns: the columns set from the source's columns of the same name
    :ptype columns: Sequence[str]
    :param by: columns of the table pinned beside the join, as :func:`update_keyed` takes them
    :ptype by: Mapping[str, Any] | None
    :param returning: the table's columns the written rows return; none when empty
    :ptype returning: Sequence[str]
    :return: the count written, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``key`` or ``columns`` is empty
    """
    if not key or not columns:
        raise ValueError(f"an update of {table} from a source joins on a key and sets at least one column")
    target = quoted_table(table)
    assignments = ", ".join(f"{quote_identifier(column)} = s.{quote_identifier(column)}" for column in columns)
    joined = [f"{target}.{quote_identifier(column)} = s.{quote_identifier(column)}" for column in key]
    stepped = list(source.params) if isinstance(source, Step) else []
    pinned, params = by_conditions(by or {}, first=len(stepped) + 1, alias=target)
    clause = " AND ".join([*joined, *pinned])
    selected = ", ".join(f"{target}.{quote_identifier(column)}" for column in returning)
    tail = f" RETURNING {selected}" if returning else ""
    sql = f"UPDATE {target} SET {assignments} FROM ({source.sql}) AS s WHERE {clause}{tail}"
    params = [*stepped, *params]
    return await _run(conn, sql, params, returning)


async def apply_stage(
    conn: Any,
    target: str,
    stage: str,
    *,
    key: Sequence[str],
    columns: Sequence[str],
    mode: StageMode | str = StageMode.REPLACE,
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    distinct: bool = False,
) -> StageWrite:
    """write ``stage``'s rows into ``target`` by the stage's ``key``, never reading ``target`` but by that key.

    - ``replace``: ``DELETE FROM target WHERE EXISTS (SELECT 1 FROM stage s WHERE s.k = target.k)``, then ``INSERT
      INTO target (cols) SELECT cols FROM stage``;
    - ``upsert``: ``INSERT ... SELECT ... FROM stage ON CONFLICT (key) DO UPDATE SET c = excluded.c`` (the key a
      unique constraint of ``target``);
    - ``append``: ``INSERT INTO target (cols) SELECT cols FROM stage``, nothing deleted and duplicates kept (``key``
      may be empty: no row is named by it);
    - ``difference``: as ``replace``, but only the stage rows whose values differ (``IS DISTINCT FROM``) from the
      row of their key, or whose key is new. With a scope (``by``, ``where``: one state, one partition), the
      table's rows in that scope whose key the stage does not hold are deleted too (``DELETE FROM target WHERE
      <scope> AND NOT EXISTS (SELECT 1 FROM stage s WHERE s.k = target.k)``), so the scope holds exactly the
      stage's rows, and rows the stage holds as they are stay untouched. With no scope no absent key is deleted:
      the stage names the rows it writes, never the table's others.

    One spelling for PostgreSQL, YugabyteDB, DuckDB (:class:`~threetears.core.keyed.ThreadedDuckDB`) and SQLite
    (:class:`~threetears.core.keyed.ThreadedSQLite`); run it in the caller's transaction so a reader sees the stage
    applied whole.

    :param conn: anything with an asyncpg-shaped ``execute``
    :ptype conn: Any
    :param target: the table written, a TRUSTED identifier
    :ptype target: str
    :param stage: the stage relation, a TRUSTED identifier, with ``columns``
    :ptype stage: str
    :param key: the key the stage names rows by, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param columns: every column written, the key's among them
    :ptype columns: Sequence[str]
    :param mode: ``replace``, ``upsert``, ``difference`` or ``append``
    :ptype mode: StageMode | str
    :param by: ``difference`` only: the scope the stage replaces whole, its columns pinned (each a value or an
        ``AnyOf``); its rows of keys the stage does not hold are deleted
    :ptype by: Mapping[str, Any] | None
    :param where: ``difference`` only: filters of that scope beside ``by``, each a fragment from ``$1`` on the
        target's columns, unqualified (``"state" = $1``)
    :ptype where: Sequence[Fragment]
    :param distinct: ``difference`` only: the stage's rows written once each however often staged (``SELECT
        DISTINCT``), as a set difference (``stage EXCEPT held``) writes them; each staged row as often as staged when
        False
    :ptype distinct: bool
    :return: the rows deleted and written
    :rtype: StageWrite
    :raises ValueError: when the key is empty or not among the columns, the mode is unknown, or a scope or
        ``distinct`` is given to a mode other than ``difference``
    """
    chosen = StageMode(mode)
    if (not key and chosen is not StageMode.APPEND) or any(column not in columns for column in key):
        raise ValueError(f"a stage applied to {target} names its key among the columns written, got {list(key)}")
    scoped = bool(by) or bool(where)
    if scoped and chosen is not StageMode.DIFFERENCE:
        raise ValueError(f"a stage's scope (by=, where=) deletes the absent keys of a difference; {chosen} has none")
    if distinct and chosen is not StageMode.DIFFERENCE:
        raise ValueError(f"a stage's rows are de-duplicated (distinct=) by a difference's insert; {chosen} has none")
    table = quoted_table(target)
    # the stage is a change set (the rows being applied), read whole by each statement below; the rule judges what
    # a caller hands as the stage where it calls, a data table there being the whole read it is
    staged = quoted_table(change_relation(stage))
    cols = ", ".join(quote_identifier(column) for column in columns)
    on_key = " AND ".join(f"s.{quote_identifier(column)} = {table}.{quote_identifier(column)}" for column in key)
    if chosen is StageMode.UPSERT:
        updated = [column for column in columns if column not in key]
        conflict = ", ".join(quote_identifier(column) for column in key)
        action = (
            "DO UPDATE SET "
            + ", ".join(f"{quote_identifier(column)} = excluded.{quote_identifier(column)}" for column in updated)
            if updated
            else "DO NOTHING"
        )
        # WHERE true: SQLite reads ON CONFLICT after a bare SELECT ... FROM as a join's ON
        status = await conn.execute(
            f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {staged} WHERE true ON CONFLICT ({conflict}) {action}"
        )
        return StageWrite(deleted=0, written=rows_affected(str(status or "")))
    if chosen is StageMode.APPEND:
        appended = await conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {staged}")
        return StageWrite(deleted=0, written=rows_affected(str(appended or "")))
    if chosen is StageMode.REPLACE:
        deleted = await conn.execute(f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM {staged} AS s WHERE {on_key})")
        written = await conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {staged}")
        return StageWrite(deleted=rows_affected(str(deleted or "")), written=rows_affected(str(written or "")))
    differs = " OR ".join(
        f"s.{quote_identifier(column)} IS DISTINCT FROM {table}.{quote_identifier(column)}"
        for column in columns
        if column not in key
    )
    changed = f"{on_key} AND ({differs})" if differs else "1 = 0"
    removed = 0
    if scoped:
        conditions, params = by_conditions(by or {})
        for fragment in where:
            conditions.append(narrowing(fragment, len(params)))
            params += list(fragment.params)
        absent_key = await conn.execute(
            f"DELETE FROM {table} WHERE {' AND '.join(conditions)} "
            f"AND NOT EXISTS (SELECT 1 FROM {staged} AS s WHERE {on_key})",
            *params,
        )
        removed = rows_affected(str(absent_key or ""))
    deleted = await conn.execute(f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM {staged} AS s WHERE {changed})")
    absent = " AND ".join(f"{table}.{quote_identifier(column)} = s.{quote_identifier(column)}" for column in key)
    selected = ", ".join(f"s.{quote_identifier(column)}" for column in columns)
    written = await conn.execute(
        f"INSERT INTO {table} ({cols}) SELECT {'DISTINCT ' if distinct else ''}{selected} FROM {staged} AS s "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE {absent})"
    )
    return StageWrite(deleted=removed + rows_affected(str(deleted or "")), written=rows_affected(str(written or "")))
