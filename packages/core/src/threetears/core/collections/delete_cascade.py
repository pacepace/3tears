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
advances each table's generation for them. A table whose collection this registry does not hold
is advanced with no rows, so every follower drops it: the reach is unknown there. A delete written
as raw SQL calls the two itself, around its statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from threetears.core.collections.generation import announce_unheard_writes
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
    :ivar keys: each row's pk value, or tuple of pk values in its collection's declared order;
        empty when no collection for the table is registered (``unheard``)
    :ivar unheard: whether the table has no collection here, so its rows cannot be named and it is
        advanced with none
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
            found.append(CascadedRows(cascade.child_table, (), unheard=True))
            continue
        columns = tuple(child.primary_key_columns)
        carried = tuple(dict.fromkeys((*columns, *getattr(child, "invalidation_columns", ()))))
        # cache-bypass: the rows a foreign-key action is about to reach, found by the column naming the parent
        rows = await collection.l3_pool.fetch(
            f"SELECT {', '.join(carried)} FROM {cascade.child_table} WHERE {cascade.child_column} = ANY($1)",  # noqa: S608 -- declared identifiers
            values,
        )
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
    unheard: list[str] = []
    for rows in cascaded:
        if rows.unheard:
            unheard.append(rows.table_name)
            continue
        child = None if registry is None else registry.get_collection(rows.table_name)
        if child is None:
            unheard.append(rows.table_name)
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
    from threetears.core.collections.base import tables_with_write_generation  # noqa: PLC0415 -- base imports this module

    unheard = [table for table in unheard if table in tables_with_write_generation()]
    source = None if registry is None else registry.generation_source
    if unheard and source is not None:
        try:
            await announce_unheard_writes(source, unheard)
        except GenerationUnavailableError as exc:
            failure = failure or exc
    if failure is not None:
        raise failure
