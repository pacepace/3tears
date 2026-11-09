"""the rows a delete's foreign-key actions reach, read before the delete and announced after it.

A foreign key ``ON DELETE CASCADE`` or ``ON DELETE SET NULL`` rewrites rows of another table
inside the database: no collection writes them, so nothing evicts them, nothing broadcasts them,
and their table's write generation does not move. Nothing in L1 ages, so a copy of such a row --
by key, or in a scan derived from its table -- would be served until the row is written again.

A collection declares each foreign key that points at it with an action
(:attr:`~threetears.core.collections.base.BaseCollection.delete_cascades`). Its
:meth:`~threetears.core.collections.base.BaseCollection.delete` reads the keys of the rows each
action will reach before the delete (:func:`read_delete_cascade`) and, once the delete has
committed, invalidates them through their own collections (:func:`announce_delete_cascade`), which
advances each table's generation for them. A table whose collection this registry does not hold is
advanced naming the rows by the primary key its collection class declares
(:func:`~threetears.core.collections.generation.primary_key_of_table`): its followers evict exactly
those rows. Nothing here records a change to the whole table. A delete written as raw SQL calls the
two itself, around its statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from threetears.core.backends.protocol import L3_RAIL_ROW_CAP
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import (
    advance_for_write,
    primary_key_of_table,
    tables_with_write_generation,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.observe import get_logger

__all__ = ["CascadedRows", "DeleteCascade", "announce_delete_cascade", "read_delete_cascade"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeleteCascade:
    """one foreign key from ``child_table.child_column`` to the declaring collection's ``parent_column``.

    :ivar child_table: the table whose rows the action reaches
    :ivar child_column: its column naming the parent row
    :ivar action: ``"CASCADE"`` deletes the child rows; ``"SET NULL"`` rewrites them
    :ivar parent_column: the declaring table's column the key references
    """

    child_table: str
    child_column: str
    action: Literal["CASCADE", "SET NULL"]
    parent_column: str = "id"


@dataclass(frozen=True, slots=True)
class CascadedRows:
    """the rows of one table a delete's foreign-key actions reach, by their collection's keys.

    :ivar table_name: the table
    :ivar keys: each row's pk value, or tuple of pk values in its collection's declared order; for a
        table with no collection here, by the key its collection class declares, and empty only when no
        class here declares it
    :ivar unheard: whether the table has no collection here, so it is advanced directly, naming
        :attr:`keys` (or, with none, as an epoch whose keys are unknown: its followers are behind)
    :ivar rows: each row's key columns and the columns its collection's messages carry
        (``invalidation_columns``), in ``keys`` order, so a cache derived from the table hears what
        each row was; empty when ``unheard``
    """

    table_name: str
    keys: tuple[Any, ...]
    unheard: bool = False
    rows: tuple[dict[str, Any], ...] = ()


async def read_delete_cascade(collection: Any, parent_keys: Sequence[Any]) -> list[CascadedRows]:
    """the rows each of ``collection``'s declared foreign-key actions will reach when ``parent_keys`` go.

    Read before the delete: afterwards a cascade has taken the rows and a ``SET NULL`` has cleared
    the column that found them. A cascaded table's own declared actions are followed too, since
    the database runs them as part of the same delete.

    :param collection: the collection whose rows are about to be deleted
    :ptype collection: BaseCollection
    :param parent_keys: their pk values, or tuples in declared order
    :ptype parent_keys: Sequence[Any]
    :return: the rows reached, one entry per (cascade, table); empty when nothing is declared, no
        key is given or there is no L3 pool to read
    :rtype: list[CascadedRows]
    """
    found: list[CascadedRows] = []
    cascades: tuple[DeleteCascade, ...] = getattr(type(collection), "delete_cascades", ())
    if not cascades or not parent_keys or collection.l3_pool is None:
        return found
    registry = collection.registry
    for cascade in cascades:
        values = await _parent_values(collection, parent_keys, cascade.parent_column)
        if not values:
            continue
        child = None if registry is None else registry.get_collection(cascade.child_table)
        if child is None:
            declared = primary_key_of_table(cascade.child_table)
            if declared is None:
                found.append(CascadedRows(cascade.child_table, (), unheard=True))
                continue
            # cache-bypass: the keys of the rows the action reaches, by the key the table's class declares
            rows = await _children(
                collection.l3_pool, cascade.child_table, cascade.child_column, declared, declared, values
            )
            keys = tuple(row[declared[0]] if len(declared) == 1 else tuple(row[c] for c in declared) for row in rows)
            if keys:
                found.append(CascadedRows(cascade.child_table, keys, unheard=True))
            continue
        columns = tuple(child.primary_key_columns)
        carried = tuple(dict.fromkeys((*columns, *getattr(child, "invalidation_columns", ()))))
        # cache-bypass: the rows a foreign-key action is about to reach, found by the column naming the parent, in
        # pages of the child's key under the L3 rail's row cap (a parent of more children than it is cut past it)
        rows = await _children(collection.l3_pool, cascade.child_table, cascade.child_column, carried, columns, values)
        keys = tuple(row[columns[0]] if len(columns) == 1 else tuple(row[c] for c in columns) for row in rows)
        if not keys:
            continue
        found.append(CascadedRows(cascade.child_table, keys, rows=tuple(dict(row) for row in rows)))
        if cascade.action == "CASCADE":
            found.extend(await read_delete_cascade(child, keys))
    return found


async def _parent_values(collection: Any, parent_keys: Sequence[Any], parent_column: str) -> list[Any]:
    """the values of ``parent_column`` on the rows ``parent_keys`` name.

    :param collection: the parent collection
    :ptype collection: BaseCollection
    :param parent_keys: the parent rows' keys
    :ptype parent_keys: Sequence[Any]
    :param parent_column: the referenced column
    :ptype parent_column: str
    :return: the values, in key order, rows not found left out
    :rtype: list[Any]
    """
    columns = tuple(collection.primary_key_columns)
    if parent_column in columns:
        index = columns.index(parent_column)
        return [collection.normalize_pk(key)[index] for key in parent_keys]
    values: list[Any] = []
    for key in parent_keys:
        row = await collection.fetch_from_store(key)
        if row is not None and row.get(parent_column) is not None:
            values.append(row[parent_column])
    return values


async def announce_delete_cascade(collection: Any, cascaded: Sequence[CascadedRows]) -> None:
    """invalidate every row :func:`read_delete_cascade` found, through its own collection, once the delete committed.

    Every table is attempted; the first failure to advance a generation is raised after the rest.

    :param collection: the collection whose delete reached the rows
    :ptype collection: BaseCollection
    :param cascaded: what :func:`read_delete_cascade` read before the delete
    :ptype cascaded: Sequence[CascadedRows]
    :return: nothing
    :rtype: None
    :raises GenerationUnavailableError: when a reached table's write generation could not be advanced
    """
    failure: GenerationUnavailableError | None = None
    registry = collection.registry
    unheard: list[CascadedRows] = []
    for rows in cascaded:
        child = None if rows.unheard or registry is None else registry.get_collection(rows.table_name)
        if child is None:
            unheard.append(rows)
            continue
        try:
            await child.invalidate_cache_many(list(rows.keys), rows=list(rows.rows) if rows.rows else None)
        except GenerationUnavailableError as exc:
            log.error(
                "a delete's foreign-key action was not announced: its table's generation did not advance",
                extra={"extra_data": {"table": rows.table_name, "rows": len(rows.keys), "error": str(exc)}},
            )
            failure = failure or exc
    # only a switched-on table has followers to tell, and only it may be advanced
    source = None if registry is None else registry.generation_source
    for rows in unheard:
        if rows.table_name not in tables_with_write_generation() or source is None:
            continue
        named = [
            ChangedRow(ids=[str(value) for value in (key if isinstance(key, tuple) else (key,))])  # convert at border
            for key in rows.keys
        ]
        try:
            # keyed when the keys are known; otherwise an epoch whose keys are unknown -- never the whole table
            token = await advance_for_write(
                source,
                rows.table_name,
                named or None,
                schema=None if registry is None else registry.l3_schema,
                keys_unknown=not named,
            )
            if not isinstance(token, str) or not token:
                raise GenerationUnavailableError(
                    f"the source advanced {rows.table_name} without returning a generation"
                )
        except GenerationUnavailableError as exc:
            log.error(
                "a delete's foreign-key action was not announced: its table's generation did not advance",
                extra={"extra_data": {"table": rows.table_name, "rows": len(rows.keys), "error": str(exc)}},
            )
            failure = failure or exc
    if failure is not None:
        raise failure


async def _children(
    pool: Any, table: str, column: str, carried: Sequence[str], key: Sequence[str], values: Sequence[Any]
) -> list[Any]:
    """the rows of ``table`` whose ``column`` is one of ``values``, read in pages of ``key`` under the L3 rail's row
    cap: ``WHERE column = ANY($1) [AND (key) > (...)] ORDER BY key LIMIT``.

    :param pool: the L3, an asyncpg-shaped ``fetch``
    :ptype pool: Any
    :param table: the child table, a declared identifier
    :ptype table: str
    :param column: the column naming the parent, a declared identifier
    :ptype column: str
    :param carried: the columns read, ``key`` among them
    :ptype carried: Sequence[str]
    :param key: the child's primary key, declared identifiers
    :ptype key: Sequence[str]
    :param values: the parents' values
    :ptype values: Sequence[Any]
    :return: the rows
    :rtype: list[Any]
    """
    page_size = L3_RAIL_ROW_CAP - 1
    order = ", ".join(key)
    head = f"SELECT {', '.join(carried)} FROM {table} WHERE {column} = ANY($1)"  # noqa: S608 -- declared identifiers
    found: list[Any] = []
    page = list(await pool.fetch(f"{head} ORDER BY {order} LIMIT {page_size}", values))
    found += page
    while len(page) == page_size:
        last = [page[-1][c] for c in key]
        after = ", ".join(f"${2 + i}" for i in range(len(key)))
        page = list(
            await pool.fetch(f"{head} AND ({order}) > ({after}) ORDER BY {order} LIMIT {page_size}", values, *last)
        )
        found += page
    return found
