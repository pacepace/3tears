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

**A stamped parent's delete runs no foreign-key action** (append-only L3, decision 2): its row is marked gone in place
(``ROW_STAMPED``), never deleted, so the database's ``ON DELETE CASCADE`` / ``SET NULL`` never fires, and the rows
it would have reached stay as they were (a deleted group's grants still granting). :func:`apply_delete_cascade` takes
each declared action itself, through the child's collection, once the parent is marked gone: a ``CASCADE`` marks the
child rows gone (a stamped child) or deletes them by key (any other), a ``SET NULL`` sets the column NULL by key, its
epoch risen for a stamped child. Each is a stamped write a restore hears, where the database's own action was a write
no restore could see.

**Nor does it run a ``RESTRICT`` / ``NO ACTION`` guard**: the database refuses a ``DELETE`` of a row its children still
name, and a gone mark is an ``UPDATE``, so a role with live grants would be marked gone and its grants left granting. A
collection declares each such foreign key (:attr:`~threetears.core.collections.base.BaseCollection.delete_restricts`,
:class:`DeleteRestrict`), and its delete of a stamped row first counts the live children each names, through the child
column's index (:func:`refuse_restricted_delete`), refusing while any is held.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from threetears.core.backends.protocol import L3_RAIL_ROW_CAP
from threetears.core.ddl.row_epoch import GONE_COLUMN
from threetears.core.collections.epoch_change import ChangedRow, key_text
from threetears.core.collections.generation import (
    advance_for_write,
    has_write_generation,
    primary_key_of_table,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.keyed import AnyOf, Fragment, Sort, counts_by, delete_keyed, keyset_rows, update_keyed
from threetears.interfaces.keyed.filters import quote_identifier
from threetears.interfaces.keyed.rows import KeyBatch, KeyColumns
from threetears.observe import get_logger

__all__ = [
    "CascadedRows",
    "DeleteCascade",
    "DeleteRestrict",
    "RestrictedDeleteError",
    "UnappliedCascadeError",
    "announce_delete_cascade",
    "apply_delete_cascade",
    "read_delete_cascade",
    "refuse_restricted_delete",
    "refuse_unappliable_cascade",
]

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
class DeleteRestrict:
    """one foreign key from ``child_table.child_column`` to the declaring collection's ``parent_column`` whose action
    refuses the parent's delete while a child names it (``ON DELETE RESTRICT`` / ``NO ACTION``).

    :ivar child_table: the table whose rows hold the parent
    :ivar child_column: its column naming the parent row, the leading column of an index of the child's
    :ivar action: the declared action, ``"RESTRICT"`` or ``"NO ACTION"`` (both refuse alike here)
    :ivar parent_column: the declaring table's column the key references
    """

    child_table: str
    child_column: str
    action: Literal["RESTRICT", "NO ACTION"]
    parent_column: str = "id"


class RestrictedDeleteError(RuntimeError):
    """a stamped parent's delete refused: a ``RESTRICT`` / ``NO ACTION`` foreign key's child rows still name it.

    :ivar table_name: the parent table
    :ivar child_table: the child table holding live rows
    :ivar count: how many live child rows name the parent
    """

    def __init__(self, table_name: str, child_table: str, child_column: str, count: int) -> None:
        """name the parent, the child table and how many of its live rows hold the parent.

        :param table_name: the parent table
        :ptype table_name: str
        :param child_table: the child table
        :ptype child_table: str
        :param child_column: its column naming the parent
        :ptype child_column: str
        :param count: its live rows naming the parent
        :ptype count: int
        :return: nothing
        :rtype: None
        """
        super().__init__(
            f"{table_name}: refusing the delete, {count} live row(s) of {child_table} still name it "
            f"({child_table}.{child_column}, ON DELETE RESTRICT); remove them first"
        )
        self.table_name = table_name
        self.child_table = child_table
        self.count = count


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
    :ivar action: the foreign-key action that reaches them (:attr:`DeleteCascade.action`)
    :ivar column: the child column naming the parent (:attr:`DeleteCascade.child_column`)
    """

    table_name: str
    keys: tuple[Any, ...]
    unheard: bool = False
    rows: tuple[dict[str, Any], ...] = ()
    action: Literal["CASCADE", "SET NULL"] = field(default="CASCADE", compare=False)
    column: str = field(default="", compare=False)


class UnappliedCascadeError(RuntimeError):
    """a stamped parent's delete reaches rows its foreign-key action cannot be taken on by 3tears: the child table has
    no collection in this registry, or one that cannot take it (a versioned child). Raised before the parent is marked
    gone, so nothing is left half done."""


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
                found.append(
                    CascadedRows(
                        cascade.child_table, (), unheard=True, action=cascade.action, column=cascade.child_column
                    )
                )
                continue
            # cache-bypass: the keys of the rows the action reaches, by the key the table's class declares
            rows = await _children(
                collection.l3_pool, cascade.child_table, cascade.child_column, declared, declared, values
            )
            keys = tuple(row[declared[0]] if len(declared) == 1 else tuple(row[c] for c in declared) for row in rows)
            if keys:
                found.append(
                    CascadedRows(
                        cascade.child_table, keys, unheard=True, action=cascade.action, column=cascade.child_column
                    )
                )
            continue
        columns = tuple(child.primary_key_columns)
        carried = tuple(dict.fromkeys((*columns, *getattr(child, "invalidation_columns", ()))))
        # cache-bypass: the rows a foreign-key action is about to reach, found by the column naming the parent, in
        # pages of the child's key under the L3 rail's row cap (a parent of more children than it is cut past it)
        rows = await _children(collection.l3_pool, cascade.child_table, cascade.child_column, carried, columns, values)
        keys = tuple(row[columns[0]] if len(columns) == 1 else tuple(row[c] for c in columns) for row in rows)
        if not keys:
            continue
        found.append(
            CascadedRows(
                cascade.child_table,
                keys,
                rows=tuple(dict(row) for row in rows),
                action=cascade.action,
                column=cascade.child_column,
            )
        )
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


async def refuse_restricted_delete(collection: Any, parent_keys: Sequence[Any]) -> None:
    """refuse, before a stamped parent is marked gone, a delete a declared ``RESTRICT`` / ``NO ACTION`` foreign key
    forbids: the database runs no such guard for a gone mark (an ``UPDATE``), so 3tears runs it. Each declared
    :class:`DeleteRestrict`'s live children are counted by the parents' values, one keyed ``GROUP BY`` through the
    child column's index (:func:`~threetears.core.keyed.counts_by`, a stamped child's gone rows left out by its live
    guard), never a scan.

    :param collection: the parent collection
    :ptype collection: BaseCollection
    :param parent_keys: the keys about to be marked gone
    :ptype parent_keys: Sequence[Any]
    :return: nothing
    :rtype: None
    :raises RestrictedDeleteError: when a child table holds live rows naming a parent
    """
    restricts: tuple[DeleteRestrict, ...] = getattr(type(collection), "delete_restricts", ())
    if not restricts or not parent_keys or collection.l3_pool is None:
        return
    for restrict in restricts:
        values = await _parent_values(collection, parent_keys, restrict.parent_column)
        if not values:
            continue
        # cache-bypass: whether a child still names the parent, read where the guard would have run (L3)
        counted = await counts_by(
            collection.l3_pool, restrict.child_table, column=restrict.child_column, values=AnyOf(values, cast=None)
        )
        held = sum(counted.values())
        if held:
            raise RestrictedDeleteError(collection.table_name, restrict.child_table, restrict.child_column, held)


def refuse_unappliable_cascade(collection: Any, cascaded: Sequence[CascadedRows]) -> None:
    """refuse, before a stamped parent is marked gone, a delete whose foreign-key actions 3tears cannot take itself
    (:func:`apply_delete_cascade`): rows of a table with no collection in this registry, or of a versioned one.

    :param collection: the parent collection
    :ptype collection: BaseCollection
    :param cascaded: what :func:`read_delete_cascade` read
    :ptype cascaded: Sequence[CascadedRows]
    :return: nothing
    :rtype: None
    :raises UnappliedCascadeError: when an action cannot be taken
    """
    registry = collection.registry
    for rows in cascaded:
        if not rows.keys:
            continue
        child = None if rows.unheard or registry is None else registry.get_collection(rows.table_name)
        schema = getattr(child, "schema", None)
        if child is None or schema is None or getattr(schema, "versioned", False):
            raise UnappliedCascadeError(
                f"{collection.table_name}: a delete marks its row gone (ROW_STAMPED), so the database's ON DELETE "
                f"{rows.action} of {rows.table_name}.{rows.column} never runs, and 3tears takes it only through the "
                f"child's own collection, which this registry does not hold (or holds versioned); register it"
            )


async def apply_delete_cascade(collection: Any, cascaded: Sequence[CascadedRows]) -> None:
    """take each foreign-key action a stamped parent's gone mark left undone, through the child's collection, in the
    order :func:`read_delete_cascade` read them (a child's own cascades after it): a ``CASCADE`` marks a stamped
    child's rows gone (the gone flag set, the epoch risen, a row already gone left as it is) or deletes any other
    child's by key; a ``SET NULL`` sets the column NULL by key, its epoch risen for a stamped child. Keyed ``UPDATE``
    / ``DELETE`` statements of a batch of keys each, never a whole-table write. Call
    :func:`refuse_unappliable_cascade` first; :func:`announce_delete_cascade` after, as for a real delete.

    :param collection: the parent collection, its row marked gone
    :ptype collection: BaseCollection
    :param cascaded: what :func:`read_delete_cascade` read before the gone mark
    :ptype cascaded: Sequence[CascadedRows]
    :return: nothing
    :rtype: None
    :raises UnappliedCascadeError: when an action cannot be taken
    """
    refuse_unappliable_cascade(collection, cascaded)
    registry = collection.registry
    for rows in cascaded:
        if not rows.keys:
            continue
        child = registry.get_collection(rows.table_name)
        schema = child.schema
        columns = tuple(schema.pk_columns)
        named = [key if isinstance(key, tuple) else (key,) for key in rows.keys]
        batch = KeyBatch(
            columns=columns,
            keys=[[key[place] for key in named] for place in range(len(columns))],
            casts=tuple(_sql_cast(schema, column) for column in columns),
        )
        stamped = bool(getattr(schema, "stamps_epoch", False))
        # the parent's L3, as the cascade's read of the children went (read_delete_cascade): one database
        pool = collection.required_l3_pool
        if rows.action == "SET NULL":
            await update_keyed(pool, schema.name, values=child.stamped_values({rows.column: None}), keys=batch)
        elif stamped:
            await update_keyed(
                pool,
                schema.name,
                values=child.stamped_values({GONE_COLUMN: True}),
                keys=batch,
                where=[Fragment(f"NOT {quote_identifier(GONE_COLUMN)}")],
            )
        else:
            await delete_keyed(pool, schema.name, keys=batch)
        log.info(
            "a stamped parent's foreign-key action taken by 3tears",
            extra={"extra_data": {"table": rows.table_name, "action": rows.action, "rows": len(rows.keys)}},
        )


def _sql_cast(schema: Any, column: str) -> str:
    """the SQL type a key column's batch binds as, by its declared type.

    :param schema: the child's schema
    :ptype schema: TableSchema
    :param column: the key column
    :ptype column: str
    :return: the type
    :rtype: str
    """
    return _CASTS.get(schema.column(column).column_type, "text")


#: the SQL type of each key column type tag a cascade's key batch binds (``schema_backed.sql_cast``'s, read here by
#: tag: this module is imported by the collections base, beneath ``schema_backed``)
_CASTS: Final[Mapping[str, str]] = {"uuid": "uuid", "bigint": "bigint", "int": "integer", "datetimetz": "timestamptz"}


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
        if source is None or not has_write_generation(rows.table_name):
            continue
        named = [
            ChangedRow(
                ids=[key_text(value) for value in (key if isinstance(key, tuple) else (key,))]
            )  # convert at border
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
    pool: Any, table: str, column: str, carried: Sequence[str], key: KeyColumns, values: Sequence[Any]
) -> list[Any]:
    """the rows of ``table`` whose ``column`` is one of ``values``, read in pages of ``key`` under the L3 rail's row
    cap (:func:`~threetears.core.keyed.keyset_rows`): ``WHERE column = ANY($1) [AND past the last row] ORDER BY key
    LIMIT``.

    The child is known here by its columns' names alone (a table only :func:`primary_key_of_table` declares), so the
    set and the key after the last row are bound untyped (``AnyOf(cast=None)``, ``Sort(cast=None)``): the database
    types them by the columns.

    :param pool: the L3, an asyncpg-shaped ``fetch``
    :ptype pool: Any
    :param table: the child table, a declared identifier
    :ptype table: str
    :param column: the column naming the parent, a declared identifier
    :ptype column: str
    :param carried: the columns read, ``key`` among them
    :ptype carried: Sequence[str]
    :param key: the child's primary key, declared identifiers
    :ptype key: KeyColumns
    :param values: the parents' values
    :ptype values: Sequence[Any]
    :return: the rows
    :rtype: list[Any]
    """
    # the table and column are one declared DeleteCascade's pair, every one of them literal, so the no-whole-reads
    # walk judges this read once per construction (its index on the child column), never as a run-time table
    order = tuple(Sort(name, cast=None) for name in key)
    # each statement asks one row under the cap, which the rail answers whole; a full answer says more follow
    return await keyset_rows(
        pool,
        table,
        columns=list(carried),
        by={column: AnyOf(list(values), cast=None)},
        order=order,
        page=L3_RAIL_ROW_CAP - 2,
    )
