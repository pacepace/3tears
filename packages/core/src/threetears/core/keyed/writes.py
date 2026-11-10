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
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from threetears.core.backends.protocol import parse_rowcount
from threetears.core.keyed._filters import (
    Derived,
    Fragment,
    binds_native_arrays,
    by_conditions,
    quoted_table,
    renumbered,
)
from threetears.core.keyed.counts import rows_affected
from threetears.core.keyed.rows import _TEXT_TYPES, DEFAULT_KEY_BATCH, KeyBatch, _text
from threetears.core.sql_fragments import quote_identifier

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


@dataclass(frozen=True)
class KeyedWrite:
    """what a keyed write did.

    :param count: the rows it wrote (the statements' own status, or their returned rows)
    :ptype count: int
    :param rows: the rows it returned (``returning``), in no promised order; empty when it returned none
    :ptype rows: list[dict[str, Any]]
    """

    count: int
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Increment:
    """a SET value adding to the column's own: ``"col" = "col" + $n``.

    :param amount: what is added (negative to subtract)
    :ptype amount: Any
    """

    amount: Any


def _named(keys: KeyBatch, chunk: Sequence[tuple[Any, ...]], first: int, natively: bool) -> tuple[str, list[Any]]:
    """the condition naming ``chunk``'s keys in a write, placeholders from ``first``: one array a column (a text
    array cast to the column's type where the connection binds no typed arrays), a list of placeholders for a
    database with none, ``IS NOT DISTINCT FROM`` each key for a null-safe batch.

    :param keys: the batch's shape
    :ptype keys: KeyBatch
    :param chunk: the keys of one statement
    :ptype chunk: Sequence[tuple[Any, ...]]
    :param first: the first placeholder's number
    :ptype first: int
    :param natively: whether the connection binds typed arrays natively
    :ptype natively: bool
    :return: the condition and its parameters
    :rtype: tuple[str, list[Any]]
    """
    columns = [quote_identifier(column) for column in keys.columns]
    width = len(columns)
    if keys.as_list:
        if width == 1:
            marks = ", ".join(f"${first + place}" for place in range(len(chunk)))
            return f"{columns[0]} IN ({marks})", [key[0] for key in chunk]
        rows = ", ".join(
            "(" + ", ".join(f"${first + row * width + index}" for index in range(width)) + ")"
            for row in range(len(chunk))
        )
        return f"({', '.join(columns)}) IN ({rows})", [value for key in chunk for value in key]
    arrays: list[Any] = []
    typed: list[str] = []
    for index in range(width):
        cast = keys.cast(index)
        values = [key[index] for key in chunk]
        textual = cast.lower() in _TEXT_TYPES
        native = natively and not textual and not any(isinstance(value, str) for value in values)
        arrays.append(values if native else [_text(value) for value in values])
        place = first + index
        if textual:
            typed.append(f"${place}::text[]")
        else:
            typed.append(f"${place}::{cast}[]" if native else f"${place}::text[]::{cast}[]")
    if keys.null_safe:
        unnested = ", ".join(f"unnest({array}) AS k{index}" for index, array in enumerate(typed))
        joined = " AND ".join(f"{column} IS NOT DISTINCT FROM k.k{index}" for index, column in enumerate(columns))
        return f"EXISTS (SELECT 1 FROM (SELECT {unnested}) AS k WHERE {joined})", arrays
    if width == 1:
        return f"{columns[0]} = ANY({typed[0]})", arrays
    return f"({', '.join(columns)}) IN (SELECT {', '.join(f'unnest({array})' for array in typed)})", arrays


def _guarded(
    by: Mapping[str, Any],
    where: Sequence[Fragment],
    table: str,
    first: int,
    named: tuple[str, list[Any]] | None,
    alias: str | None = None,
) -> tuple[str, list[Any]]:
    """the ``WHERE`` naming a write's rows: its keys, ``by``, then ``where``, placeholders from ``first``.

    :raises ValueError: when neither keys nor ``by`` name a column
    """
    if not by and named is None:
        raise ValueError(f"a keyed write of {table} names the key of the rows it writes (by= or keys=)")
    conditions: list[str] = []
    params: list[Any] = []
    if named is not None:
        conditions.append(named[0])
        params += named[1]
    pinned, values = by_conditions(by, first=first + len(params), alias=alias)
    conditions += pinned
    params += values
    for fragment in where:
        conditions.append(renumbered(fragment, first - 1 + len(params)))
        params += list(fragment.params)
    return " WHERE " + " AND ".join(conditions), params


def _returning(columns: Sequence[str]) -> str:
    return f" RETURNING {', '.join(quote_identifier(column) for column in columns)}" if columns else ""


async def _run(conn: Any, sql: str, params: Sequence[Any], returning: Sequence[str]) -> KeyedWrite:
    if returning:
        rows = [dict(row) for row in await conn.fetch(sql, *params)]
        return KeyedWrite(count=len(rows), rows=rows)
    status = await conn.execute(sql, *params)
    # the framework's reading of a status tag, as every hand-written write read it: a counting verb whose count
    # does not parse (a broker reply with no row_count) is refused, never read as "no rows"
    return KeyedWrite(count=parse_rowcount(status), rows=[])


def _chunks(keys: KeyBatch | None, batch: int) -> list[list[tuple[Any, ...]]] | list[None]:
    if keys is None:
        return [None]
    if batch < 1:
        raise ValueError(f"a keyed write names at least one key a statement, got {batch}")
    unique = list(dict.fromkeys(tuple(key) for key in keys.values))
    return [unique[start : start + batch] for start in range(0, len(unique), batch)]


async def _batched(
    conn: Any,
    table: str,
    head: str,
    head_params: Sequence[Any],
    *,
    keys: KeyBatch | None,
    by: Mapping[str, Any],
    where: Sequence[Fragment],
    returning: Sequence[str],
    batch: int,
) -> KeyedWrite:
    """``head`` (``UPDATE ... SET ...`` or ``DELETE FROM ...``) once a batch of ``keys`` (once when None)."""
    natively = binds_native_arrays(conn)
    count = 0
    rows: list[dict[str, Any]] = []
    first = len(head_params) + 1
    for chunk in _chunks(keys, batch):
        if keys is not None and not chunk:
            continue
        named = None if chunk is None or keys is None else _named(keys, chunk, first, natively)
        clause, params = _guarded(by, where, table, first, named)
        written = await _run(conn, f"{head}{clause}{_returning(returning)}", [*head_params, *params], returning)
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
    :return: the count written, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when ``values`` is empty, or neither ``by`` nor ``keys`` names a column
    """
    if not values:
        raise ValueError(f"an update of {table} sets at least one column")
    assignments: list[str] = []
    params: list[Any] = []
    for column, value in values.items():
        quoted = quote_identifier(column)
        if isinstance(value, Fragment):
            assignments.append(f"{quoted} = {renumbered(value, len(params))}")
            params += list(value.params)
        elif isinstance(value, Increment):
            params.append(value.amount)
            assignments.append(f"{quoted} = {quoted} + ${len(params)}")
        else:
            params.append(value)
            assignments.append(f"{quoted} = ${len(params)}")
    head = f"UPDATE {quoted_table(table)} SET {', '.join(assignments)}"
    return await _batched(
        conn, table, head, params, keys=keys, by=by or {}, where=where, returning=returning, batch=batch
    )


async def delete_keyed(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
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
    :return: the count deleted, and the rows returned
    :rtype: KeyedWrite
    :raises ValueError: when neither ``by`` nor ``keys`` names a column
    """
    head = f"DELETE FROM {quoted_table(table)}"
    return await _batched(conn, table, head, [], keys=keys, by=by or {}, where=where, returning=returning, batch=batch)


async def update_from(
    conn: Any,
    table: str,
    *,
    key: Sequence[str],
    source: Derived,
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
        write); TRUSTED identifiers, the same names in the source
    :ptype key: Sequence[str]
    :param source: the derived select, with no parameters of its own
    :ptype source: Derived
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
    pinned, params = by_conditions(by or {}, alias=target)
    clause = " AND ".join([*joined, *pinned])
    selected = ", ".join(f"{target}.{quote_identifier(column)}" for column in returning)
    tail = f" RETURNING {selected}" if returning else ""
    sql = f"UPDATE {target} SET {assignments} FROM ({source.sql}) AS s WHERE {clause}{tail}"
    return await _run(conn, sql, params, returning)


class StageMode(StrEnum):
    """how :func:`apply_stage` writes a stage into its table."""

    #: the table's rows the stage's keys name are replaced by the stage's rows
    REPLACE = "replace"
    #: each stage row inserted, or the row of its key updated to it
    UPSERT = "upsert"
    #: only what differs: the table's rows the stage holds other values for are replaced, its new keys inserted,
    #: rows the stage holds as they are left untouched
    DIFFERENCE = "difference"


@dataclass(frozen=True)
class StageWrite:
    """what :func:`apply_stage` wrote.

    :param deleted: the table's rows deleted (replaced)
    :ptype deleted: int
    :param written: the stage's rows inserted or upserted
    :ptype written: int
    """

    deleted: int
    written: int


async def apply_stage(
    conn: Any,
    target: str,
    stage: str,
    *,
    key: Sequence[str],
    columns: Sequence[str],
    mode: StageMode | str = StageMode.REPLACE,
) -> StageWrite:
    """write ``stage``'s rows into ``target`` by the stage's ``key``, never reading ``target`` but by that key.

    - ``replace``: ``DELETE FROM target WHERE EXISTS (SELECT 1 FROM stage s WHERE s.k = target.k)``, then ``INSERT
      INTO target (cols) SELECT cols FROM stage``;
    - ``upsert``: ``INSERT ... SELECT ... FROM stage ON CONFLICT (key) DO UPDATE SET c = excluded.c`` (the key a
      unique constraint of ``target``);
    - ``difference``: as ``replace``, but only the stage rows whose values differ (``IS DISTINCT FROM``) from the
      row of their key, or whose key is new.

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
    :param mode: ``replace``, ``upsert`` or ``difference``
    :ptype mode: StageMode | str
    :return: the rows deleted and written
    :rtype: StageWrite
    :raises ValueError: when the key is empty or not among the columns, or the mode is unknown
    """
    chosen = StageMode(mode)
    if not key or any(column not in columns for column in key):
        raise ValueError(f"a stage applied to {target} names its key among the columns written, got {list(key)}")
    table = quoted_table(target)
    staged = quoted_table(stage)
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
    deleted = await conn.execute(f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM {staged} AS s WHERE {changed})")
    absent = " AND ".join(f"{table}.{quote_identifier(column)} = s.{quote_identifier(column)}" for column in key)
    selected = ", ".join(f"s.{quote_identifier(column)}" for column in columns)
    written = await conn.execute(
        f"INSERT INTO {table} ({cols}) SELECT {selected} FROM {staged} AS s "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE {absent})"
    )
    return StageWrite(deleted=rows_affected(str(deleted or "")), written=rows_affected(str(written or "")))
