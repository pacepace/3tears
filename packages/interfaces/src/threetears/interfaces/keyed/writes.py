"""keyed writes' statements: the rows a key names updated or deleted, never a table's rows by a scan.

A collection's targeted write (a status flip, a stamp, a fence-checked save, a delete of one row, of a parent's rows
or of a batch of keys) names its rows by a leading part of a key, as a keyed read does::

    UPDATE <table> SET "c1" = $1, "c2" = now() WHERE "k1" = $2 AND "k2" = ANY($3::uuid[]) AND <where> RETURNING ...
    DELETE FROM <table> WHERE ("k1", "k2") IN (SELECT unnest($1::text[]), unnest($2::text[]::bigint[])) AND ...

``by`` pins columns (a value, ``None`` for ``IS NULL``, or an :class:`~threetears.interfaces.keyed.AnyOf`) and
``keys`` names a batch of keys (a :class:`~threetears.interfaces.keyed.KeyBatch`, ``batch`` keys a statement);
``where`` adds the write's own guards, each a fragment written as ``"col"`` from ``$1``; a value set may be an
expression (a :class:`~threetears.interfaces.keyed.Fragment`: ``now()``) or an :class:`Increment`. A write names a
key: one with neither ``by``, ``keys`` nor ``prefix`` is refused, never a write of every row.

Pure: :func:`update_keyed_statements` and :func:`delete_keyed_statements` write each statement and its binds;
``threetears.core.keyed.update_keyed`` / ``delete_keyed`` run them, and a writer with no 3tears core runs them on its
own connection (``execute``; ``fetch`` for its ``RETURNING`` rows).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from threetears.interfaces.keyed.filters import (
    Fragment,
    by_conditions,
    narrowing,
    prefix_conditions,
    quote_identifier,
    quoted_table,
    renumbered,
)
from threetears.interfaces.keyed.rows import _TEXT_TYPES, DEFAULT_KEY_BATCH, KeyBatch, _text

__all__ = [
    "Increment",
    "KeyedWrite",
    "StageMode",
    "StageWrite",
    "delete_keyed_statements",
    "update_keyed_statements",
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
    prefix: Mapping[str, str] | None = None,
) -> tuple[str, list[Any]]:
    """the conditions naming a write's rows (its ``WHERE``): its keys, ``by``, ``prefix``, then ``where``,
    placeholders from ``first``.

    :raises ValueError: when neither keys, ``by`` nor ``prefix`` name a column
    """
    if not by and named is None and not prefix:
        raise ValueError(f"a keyed write of {table} names the key of the rows it writes (by=, keys= or prefix=)")
    conditions: list[str] = []
    params: list[Any] = []
    if named is not None:
        conditions.append(named[0])
        params += named[1]
    pinned, values = by_conditions(by, first=first + len(params), alias=alias)
    conditions += pinned
    params += values
    if prefix:
        ranged, bounds = prefix_conditions(prefix, first=first + len(params), alias=alias)
        conditions += ranged
        params += bounds
    for fragment in where:
        conditions.append(narrowing(fragment, first - 1 + len(params)))
        params += list(fragment.params)
    return " AND ".join(conditions), params


def _returning(columns: Sequence[str]) -> str:
    return f" RETURNING {', '.join(quote_identifier(column) for column in columns)}" if columns else ""


def _chunks(keys: KeyBatch | None, batch: int) -> list[list[tuple[Any, ...]]] | list[None]:
    if keys is None:
        return [None]
    if batch < 1:
        raise ValueError(f"a keyed write names at least one key a statement, got {batch}")
    unique = list(dict.fromkeys(tuple(key) for key in keys.values))
    return [unique[start : start + batch] for start in range(0, len(unique), batch)]


def _statements(
    table: str,
    assignments: Sequence[str] | None,
    head_params: Sequence[Any],
    *,
    keys: KeyBatch | None,
    by: Mapping[str, Any],
    where: Sequence[Fragment],
    returning: Sequence[str],
    batch: int,
    prefix: Mapping[str, str] | None,
    natively: bool,
) -> list[tuple[str, list[Any]]]:
    """an ``UPDATE ... SET <assignments>`` (a ``DELETE`` when None) once a batch of ``keys`` (once when None)."""
    statements: list[tuple[str, list[Any]]] = []
    first = len(head_params) + 1
    for chunk in _chunks(keys, batch):
        if keys is not None and not chunk:
            continue
        named = None if chunk is None or keys is None else _named(keys, chunk, first, natively)
        named_rows, params = _guarded(by, where, table, first, named, prefix=prefix)
        tail = _returning(returning)
        if assignments is None:
            sql = f"DELETE FROM {quoted_table(table)} WHERE {named_rows}{tail}"
        else:
            sql = f"UPDATE {quoted_table(table)} SET {', '.join(assignments)} WHERE {named_rows}{tail}"
        statements.append((sql, [*head_params, *params]))
    return statements


def update_keyed_statements(
    table: str,
    *,
    values: Mapping[str, Any],
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
    prefix: Mapping[str, str] | None = None,
    natively: bool = False,
) -> list[tuple[str, list[Any]]]:
    """every statement setting ``values`` on the rows of ``table`` that ``keys`` and ``by`` (and ``where``) name, and
    its binds: what ``threetears.core.keyed.update_keyed`` sends.

    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param values: column -> its new value, an :class:`Increment`, or a :class:`Fragment` expression (``now()``, its
        placeholders from ``$1``); columns are TRUSTED identifiers
    :ptype values: Mapping[str, Any]
    :param by: columns pinned (a leading part of the table's key, or of an index declared for the write), each a
        value, ``None`` (``IS NULL``) or an ``AnyOf``
    :ptype by: Mapping[str, Any] | None
    :param keys: a batch of keys the rows are named by, ``batch`` keys a statement; none when None
    :ptype keys: KeyBatch | None
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param returning: the columns the written rows return; none when empty
    :ptype returning: Sequence[str]
    :param batch: the most keys a statement names
    :ptype batch: int
    :param prefix: text key columns -> the prefix the rows written start with, a range in ``COLLATE "C"`` order
    :ptype prefix: Mapping[str, str] | None
    :param natively: whether the connection binds typed arrays natively (asyncpg's own)
    :ptype natively: bool
    :return: each statement and its binds, one a batch of keys (one for ``by`` alone); none for an empty batch
    :rtype: list[tuple[str, list[Any]]]
    :raises ValueError: when ``values`` is empty, neither ``by``, ``keys`` nor ``prefix`` names a column, or a prefix
        is empty or not text
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
    return _statements(
        table,
        assignments,
        params,
        keys=keys,
        by=by or {},
        where=where,
        returning=returning,
        batch=batch,
        prefix=prefix,
        natively=natively,
    )


def delete_keyed_statements(
    table: str,
    *,
    by: Mapping[str, Any] | None = None,
    keys: KeyBatch | None = None,
    where: Sequence[Fragment] = (),
    returning: Sequence[str] = (),
    batch: int = DEFAULT_KEY_BATCH,
    prefix: Mapping[str, str] | None = None,
    natively: bool = False,
) -> list[tuple[str, list[Any]]]:
    """every statement deleting the rows of ``table`` that ``keys`` and ``by`` (and ``where``) name, and its binds:
    what ``threetears.core.keyed.delete_keyed`` sends.

    :param table: the table (``schema.table``), a TRUSTED identifier
    :ptype table: str
    :param by: columns pinned, as :func:`update_keyed_statements` takes them
    :ptype by: Mapping[str, Any] | None
    :param keys: a batch of keys the rows are named by, ``batch`` keys a statement; none when None
    :ptype keys: KeyBatch | None
    :param where: guards beside the key, each a fragment written as ``"col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param returning: the columns the deleted rows return; none when empty
    :ptype returning: Sequence[str]
    :param batch: the most keys a statement names
    :ptype batch: int
    :param prefix: text key columns -> the prefix the rows deleted start with
    :ptype prefix: Mapping[str, str] | None
    :param natively: whether the connection binds typed arrays natively (asyncpg's own)
    :ptype natively: bool
    :return: each statement and its binds
    :rtype: list[tuple[str, list[Any]]]
    :raises ValueError: when neither ``by``, ``keys`` nor ``prefix`` names a column, or a prefix is empty or not text
    """
    return _statements(
        table,
        None,
        [],
        keys=keys,
        by=by or {},
        where=where,
        returning=returning,
        batch=batch,
        prefix=prefix,
        natively=natively,
    )


class StageMode(StrEnum):
    """how :func:`apply_stage` writes a stage into its table."""

    #: the table's rows the stage's keys name are replaced by the stage's rows
    REPLACE = "replace"
    #: each stage row inserted, or the row of its key updated to it
    UPSERT = "upsert"
    #: only what differs: the table's rows the stage holds other values for are replaced, its new keys inserted,
    #: rows the stage holds as they are left untouched
    DIFFERENCE = "difference"
    #: each stage row inserted as it is: nothing deleted, a key staged again (or already held) kept twice; for a
    #: table whose key allows it (a log of keys, each pass's rows)
    APPEND = "append"


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
