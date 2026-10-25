"""a collection that writes its own SQL over a table it declares as a ``TableSchema``.

:class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` generates its statements
from its schema. A collection whose statements are its own -- a natural-key upsert, a partial update
-- still has a table every tier caches rows of, and an L1 table hand-copied from it drifts: a column
a migration adds is missing from L1 and its writes fail there. :class:`DeclaredTableCollection`
declares the table once, as the migrations leave it, and everything the tiers need is read off it:

- its L1 table, generated from the schema and created on whatever L1 its registry gives it, so a
  host that hands its registry an L1 declares nothing for it;
- its key (:attr:`~threetears.core.collections.base.BaseCollection.primary_key_column`), its
  instants (:attr:`~threetears.core.collections.base.BaseCollection.datetime_columns`) and every
  column's type (:attr:`DeclaredTableCollection.field_types`), which its L2 codec rehydrates rows by;
- the columns a write names and the columns an update rewrites
  (:attr:`DeclaredTableCollection.insert_columns`, :attr:`DeclaredTableCollection.update_columns`),
  and the upsert built from them (:meth:`DeclaredTableCollection.write_row`);
- the columns a write leaves to the database (:meth:`DeclaredTableCollection.columns_decided_by_store`),
  so the row every tier caches is the row L3 holds;
- the caches a natural-key upsert settles when it lands on a row stored under another key
  (:meth:`DeclaredTableCollection.settle_stored_key`).

None of these is written twice. A subclass may still state ``datetime_columns``, ``field_types``,
``insert_columns`` or ``update_columns``; each must then say what the schema says, or the class is
refused with a ``TypeError`` when it is defined.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from sqlalchemy import MetaData

from threetears.core.backends.protocol import parse_rowcount
from threetears.core.backends.schema_sql import normalize_write_value, render_param
from threetears.core.cache.base import UnsupportedL1ColumnError
from threetears.core.collections.base import BaseCollection, EntityT
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.schema_backed import (
    TSVECTOR_TYPE,
    Column,
    TableSchema,
    check_declared_row_types,
    schema_datetime_columns,
    schema_field_types,
)
from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

__all__ = ["DeclaredTableCollection"]

log = get_logger(__name__)


class RowCannotBeInsertedError(RuntimeError):
    """a row short of a ``NOT NULL`` column was written as an update, and no row holds its key.

    Postgres checks ``NOT NULL`` on the row an INSERT proposes before any conflict is resolved, so
    a row lacking such a column can only update a row already stored. When none is, the write
    stored nothing, and this names the columns an insert would have needed.
    """


#: what a subclass may state for itself; stating any of them has the class read off its schema again
_RESTATABLE: tuple[str, ...] = (
    "schema",
    "primary_key_column",
    "datetime_columns",
    "field_types",
    "conflict_columns",
    "insert_only_columns",
    "insert_columns",
    "update_columns",
)


def _write_columns(schema: TableSchema) -> tuple[str, ...]:
    """the columns a write may name: every declared column but a trigger-maintained ``tsvector``.

    :param schema: the table
    :ptype schema: TableSchema
    :return: their names, in declared order
    :rtype: tuple[str, ...]
    """
    return tuple(column.name for column in schema.columns if column.column_type != TSVECTOR_TYPE)


def _update_columns(schema: TableSchema, kept: set[str]) -> tuple[str, ...]:
    """the columns an update of a stored row rewrites.

    Every written column but an ``immutable`` one and those in ``kept``.

    :param schema: the table
    :ptype schema: TableSchema
    :param kept: the key, the conflict target and the collection's insert-only columns
    :ptype kept: set[str]
    :return: their names, in declared order
    :rtype: tuple[str, ...]
    """
    return tuple(
        column.name
        for column in schema.columns
        if column.column_type != TSVECTOR_TYPE and not column.immutable and column.name not in kept
    )


def _check_declared_list(cls: type, name: str, derived: tuple[str, ...]) -> None:
    """refuse a class whose own column list ``name`` differs from the one read off its schema.

    :param cls: the collection class being defined
    :ptype cls: type
    :param name: the attribute (``insert_columns`` or ``update_columns``)
    :ptype name: str
    :param derived: the list read off the schema
    :ptype derived: tuple[str, ...]
    :return: nothing
    :rtype: None
    :raises TypeError: when the class declares the list and it differs
    """
    declared = cls.__dict__.get(name)
    if declared is not None and tuple(declared) != derived:
        raise TypeError(
            f"{cls.__name__} declares {name} {tuple(declared)!r}, but its schema gives {derived!r}; "
            f"the schema is the declaration, so drop {name}"
        )


class DeclaredTableCollection(BaseCollection[EntityT]):
    """a collection writing its own SQL over the table its :attr:`schema` declares.

    Subclasses declare the table and write their statements::

        class WidgetCollection(DeclaredTableCollection[WidgetEntity]):
            schema = TableSchema(name="widgets", primary_key="id", columns=[...])

            async def save_to_store(self, data, original_timestamp=None, *, conn=None): ...

    Its L2 codec, and the upsert :meth:`write_row` builds, come from the schema too, so a
    subclass whose write is an upsert writes no column list::

            async def save_to_store(self, data, original_timestamp=None, *, conn=None):
                return await self.write_row(data, conn=conn)

    :cvar schema: the table as its migrations leave it -- every column, its type, nullability,
        server default, whether an update may rewrite it (``immutable``) and the primary key. Its
        L1 table is generated from it.
    """

    schema: ClassVar[TableSchema]
    #: every column's Python type, read off :attr:`schema`; :meth:`deserialize` rehydrates by it
    field_types: ClassVar[Mapping[str, Any]] = {}
    #: the upsert's conflict target; the schema's primary key unless a subclass names a natural key.
    #: When it is not the primary key, :meth:`write_row` settles the caches after a write that
    #: landed on a row stored under another key (:meth:`settle_stored_key`).
    conflict_columns: ClassVar[tuple[str, ...]] = ()
    #: columns this collection's writes insert but never update, though the columns are not
    #: ``immutable``: another writer owns their later values. A write naming one reads the row back
    #: (:meth:`columns_decided_by_store`), since an update keeps what that writer stored.
    insert_only_columns: ClassVar[tuple[str, ...]] = ()
    #: the columns a write may name, in declared order, read off :attr:`schema`
    insert_columns: ClassVar[tuple[str, ...]] = ()
    #: the columns an update of a stored row rewrites, read off :attr:`schema` and
    #: :attr:`conflict_columns`: not the key, the conflict target, an ``immutable`` column or one of
    #: :attr:`insert_only_columns`
    update_columns: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """read the key, the row types and the write columns off the schema, refusing a copy that disagrees.

        :param kwargs: forwarded to :meth:`BaseCollection.__init_subclass__`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises TypeError: when the class declares a ``primary_key_column``, ``datetime_columns``,
            ``field_types``, ``insert_columns`` or ``update_columns`` its schema does not, or
            ``conflict_columns`` or ``insert_only_columns`` naming a column its schema lacks
        """
        schema = getattr(cls, "schema", None)
        if isinstance(schema, TableSchema) and any(name in cls.__dict__ for name in _RESTATABLE):
            declared = cls.__dict__.get("primary_key_column")
            key = schema.primary_key
            if declared is not None and declared != key:
                raise TypeError(
                    f"{cls.__name__} declares primary_key_column {declared!r}, but its schema's primary key is {key!r}"
                )
            check_declared_row_types(cls, schema)
            # a class's own target, else the one it inherits, else the key; a new schema has a new key
            inherited = () if "schema" in cls.__dict__ else cls.conflict_columns
            conflict = tuple(cls.__dict__.get("conflict_columns") or inherited) or schema.pk_columns
            insert_only = tuple(cls.insert_only_columns)
            for name, columns in (("conflict_columns", conflict), ("insert_only_columns", insert_only)):
                unknown = [column for column in columns if schema.get_column(column) is None]
                if unknown:
                    raise TypeError(
                        f"{cls.__name__} declares {name} {columns!r}, but {unknown!r} are not columns of "
                        f"its schema {schema.name!r}"
                    )
            insert_columns = _write_columns(schema)
            update_columns = _update_columns(schema, {*schema.pk_columns, *conflict, *insert_only})
            _check_declared_list(cls, "insert_columns", insert_columns)
            _check_declared_list(cls, "update_columns", update_columns)
            cls.primary_key_column = key
            # derived before the base hook runs, which checks ``expires_at_column`` against this set
            cls.datetime_columns = schema_datetime_columns(schema)
            cls.field_types = schema_field_types(schema)
            cls.conflict_columns = conflict
            cls.insert_columns = insert_columns
            cls.update_columns = update_columns
        super().__init_subclass__(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """build the collection and create its L1 table on the L1 its registry gave it.

        The L1 backends' ``initialize`` is additive and leaves a table it already holds untouched,
        so a host that also declares the table, or builds the collection twice, loses nothing.

        :param args: positional arguments for :class:`BaseCollection`
        :ptype args: Any
        :param kwargs: keyword arguments for :class:`BaseCollection`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises UnsupportedL1ColumnError: when the L1 cannot hold a declared column exactly, naming
            the collection and the column
        """
        super().__init__(*args, **kwargs)
        #: the keys a write sent that landed on a row stored under another key, until its read back
        self._sent_to_another_key: set[Any] = set()
        l1 = self.l1_backend
        if l1 is not None:
            try:
                l1.initialize(self.schema.to_sqlalchemy_table(MetaData()).metadata)
            except UnsupportedL1ColumnError as exc:
                raise UnsupportedL1ColumnError(f"{type(self).__name__}: {exc}") from exc

    @property
    def table_name(self) -> str:
        """the declared table's name.

        :return: :attr:`TableSchema.name` of :attr:`schema`
        :rtype: str
        """
        return self.schema.name

    def serialize(self, data: dict[str, Any]) -> bytes:
        """encode a row for L2 as JSON.

        :param data: the row
        :ptype data: dict[str, Any]
        :return: UTF-8 JSON bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """decode a row :meth:`serialize` encoded, each column rehydrated to its :attr:`field_types` type.

        :param data: UTF-8 JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, dict(self.field_types))

    def carried_columns(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the :attr:`insert_columns` ``data`` carries, in declared order.

        A row carries a column when it names it, except that ``None`` for a ``NOT NULL`` column
        with a server default is not a value the column can hold, so it counts as not carried.

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :return: the carried column names
        :rtype: tuple[str, ...]
        """
        return tuple(
            name for name in self.insert_columns if name in data and not self._defaulted_none(name, data[name])
        )

    def _defaulted_none(self, name: str, value: Any) -> bool:
        """whether ``value`` is ``None`` for a ``NOT NULL`` column the database defaults.

        :param name: the column
        :ptype name: str
        :param value: the value a row names for it
        :ptype value: Any
        :return: ``True`` when no write can store it and the column's default stands instead
        :rtype: bool
        """
        column = self._column(name)
        return value is None and column.server_default is not None and not column.nullable

    def _column(self, name: str) -> Column:
        """the declared column ``name``.

        :param name: the column
        :ptype name: str
        :return: its declaration
        :rtype: Column
        :raises KeyError: when the schema declares no such column
        """
        column = self.schema.get_column(name)
        if column is None:
            raise KeyError(f"{self.schema.name}: no column {name!r}")
        return column

    def missing_required(self, data: dict[str, Any]) -> list[str]:
        """the ``NOT NULL`` columns with no default that ``data`` does not carry, in declared order.

        Empty when ``data`` can be inserted; otherwise :meth:`write_statement` writes it as an
        update of the stored row.

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :return: the missing column names
        :rtype: list[str]
        """
        carried = set(self.carried_columns(data))
        return [
            name
            for name in self.insert_columns
            if name not in carried and not self._column(name).nullable and self._column(name).server_default is None
        ]

    def write_statement(self, data: dict[str, Any], *, returning: bool = False) -> tuple[str, list[Any]]:
        """the statement writing exactly the columns ``data`` carries, with its parameters.

        One rule: **a column the row does not carry is never written.** An INSERT leaves it to the
        column's default; an update of a stored row keeps the stored value. A row carrying every
        ``NOT NULL`` column without a default is written as an upsert on :attr:`conflict_columns`
        that rewrites the carried :attr:`update_columns` (``DO NOTHING`` when it carries none). A
        row that does not is written as a plain ``UPDATE`` of the row holding its conflict key.
        Each value is bound as the column's type binds it (an aware instant, a UUID, ``jsonb``).

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :param returning: whether the statement returns the primary key of the row it wrote
        :ptype returning: bool
        :return: the SQL and its positional parameters
        :rtype: tuple[str, list[Any]]
        :raises KeyError: when ``data`` does not carry every conflict column, or is written as an
            update and carries no column one rewrites
        """
        table = self.schema.name
        conflict = self.conflict_columns
        carried = self.carried_columns(data)
        missing = [column for column in conflict if column not in carried]
        if missing:
            raise KeyError(f"{table}: a write must carry its conflict key {list(conflict)!r}, missing {missing!r}")
        updatable = set(self.update_columns)
        updated = [name for name in carried if name in updatable]
        if not self.missing_required(data):
            placeholders = ", ".join(render_param(self._column(name), index) for index, name in enumerate(carried, 1))
            if updated:
                assignments = ", ".join(f"{name} = EXCLUDED.{name}" for name in updated)
                action = f"DO UPDATE SET {assignments}"
            else:
                action = "DO NOTHING"
            sql = (
                f"INSERT INTO {table} ({', '.join(carried)}) VALUES ({placeholders}) "  # noqa: S608 - declared names
                f"ON CONFLICT ({', '.join(conflict)}) {action}"
            )
            bound = list(carried)
        else:
            if not updated:
                raise KeyError(f"{table}: a write of an existing row must carry a column to update")
            assignments = ", ".join(
                f"{name} = {render_param(self._column(name), index)}" for index, name in enumerate(updated, 1)
            )
            keys = " AND ".join(
                f"{name} = {render_param(self._column(name), index)}"
                for index, name in enumerate(conflict, start=len(updated) + 1)
            )
            sql = f"UPDATE {table} SET {assignments} WHERE {keys}"  # noqa: S608 - declared names
            bound = [*updated, *conflict]
        if returning:
            sql = f"{sql} RETURNING {', '.join(self.schema.pk_columns)}"
        params = [normalize_write_value(self._column(name), data[name]) for name in bound]
        return sql, params

    async def write_row(self, data: dict[str, Any], *, conn: Any = None) -> int:
        """write ``data`` with :meth:`write_statement` and return the rows affected.

        When :attr:`conflict_columns` is a natural key rather than the primary key, the statement
        returns the key of the row it wrote, and a write that landed on a row stored under another
        key settles the caches (:meth:`settle_stored_key`).

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected; 0 with no L3
        :rtype: int
        :raises KeyError: as :meth:`write_statement` does
        :raises RowCannotBeInsertedError: when a row short of a required column, written as an
            update, finds no row holding its key
        """
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        pk = self.schema.pk_columns
        if self.conflict_columns != pk:
            sql, params = self.write_statement(data, returning=True)
            row = await target.fetchrow(sql, *params)
            sent = data.get(pk[0]) if len(pk) == 1 else tuple(data.get(column) for column in pk)
            stored: Any = None
            if row is not None:
                stored = row[pk[0]] if len(pk) == 1 else tuple(row[column] for column in pk)
            await self.settle_stored_key(sent, stored, conn=conn)
            rows = 0 if row is None else 1
        else:
            sql, params = self.write_statement(data)
            rows = parse_rowcount(await target.execute(sql, *params))
        missing = self.missing_required(data)
        if rows == 0 and missing:
            key = {column: f"{data[column]}" for column in self.conflict_columns}
            log.error(
                "an update found no row, and the row lacks columns an insert needs",
                extra={"extra_data": {"table": self.table_name, "key": key, "missing_columns": missing}},
            )
            raise RowCannotBeInsertedError(
                f"{self.table_name}: no row holds {key!r}, and the row cannot be inserted: it lacks the NOT NULL "
                f"columns {missing!r}"
            )
        return rows

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the declared columns whose stored value writing ``data`` leaves to the database.

        See :meth:`BaseCollection.columns_decided_by_store`. Read off the schema, as
        :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` reads its own: a
        declared column ``data`` does not name (the database fills its default, or an update keeps
        the stored value), and a ``NOT NULL`` column with a server default that ``data`` names as
        ``None``, which no write can store. And every column, when the write landed on a row stored
        under another key (:meth:`settle_stored_key`): the key sent holds no row, so the read back
        finds none and caches nothing. And an :attr:`insert_only_columns` column ``data`` names:
        an update of a stored row keeps the stored value, which another writer may have changed.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the names of the columns the database decides, in declared order
        :rtype: tuple[str, ...]
        """
        key = tuple(data.get(column) for column in self.primary_key_columns)
        sent = key[0] if len(key) == 1 else key
        if sent in self._sent_to_another_key:
            self._sent_to_another_key.discard(sent)
            decided = tuple(column.name for column in self.schema.columns)
        else:
            decided = tuple(
                column.name
                for column in self.schema.columns
                if column.name not in data
                or column.name in self.insert_only_columns
                or (column.server_default is not None and not column.nullable and data[column.name] is None)
            )
        return decided

    async def settle_stored_key(self, sent: Any, stored: Any, *, conn: Any = None) -> None:
        """settle the caches after a natural-key upsert, given the key it sent and the key it landed on.

        Call it from :meth:`save_to_store` with the key the statement returned (``RETURNING``). An
        upsert whose natural key is already held by a row of another primary key updates that row:
        the row the caches hold under the stored key is replaced, and the key sent holds nothing.
        The stored key is evicted from every tier (on a caller's transaction, when it ends), and
        the write's own key is read back, found absent and cached nowhere. A write that landed on
        its own key, or stored nothing (``None``), needs nothing settled.

        :param sent: the primary key the write sent
        :ptype sent: Any
        :param stored: the primary key of the row it wrote, or ``None`` when it wrote none
        :ptype stored: Any
        :param conn: the caller's connection the write joined, or ``None`` for the pool
        :ptype conn: Any
        :return: nothing
        :rtype: None
        """
        if stored is not None and stored != sent:
            self._sent_to_another_key.add(sent)
            log.info(
                "a natural-key upsert landed on a row stored under another key; that row is evicted",
                extra={"extra_data": {"table": self.table_name, "sent": f"{sent}", "stored": f"{stored}"}},
            )
            if conn is None:
                await self.invalidate_cache(stored)
            else:
                CallerTransaction.join(conn, writer=f"{type(self).__name__}.save_to_store").enroll(self, stored)
