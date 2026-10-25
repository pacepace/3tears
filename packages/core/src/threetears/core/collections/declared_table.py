"""a collection that writes its own SQL over a table it declares as a ``TableSchema``.

:class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` generates its statements
from its schema. A collection whose statements are its own -- a natural-key upsert, a partial update
-- still has a table every tier caches rows of, and an L1 table hand-copied from it drifts: a column
a migration adds is missing from L1 and its writes fail there. :class:`DeclaredTableCollection`
declares the table once, as the migrations leave it, and everything the tiers need is read off it:

- its L1 table, generated from the schema and created on whatever L1 its registry gives it, so a
  host that hands its registry an L1 declares nothing for it;
- its key (:attr:`~threetears.core.collections.base.BaseCollection.primary_key_column`) and its
  instants (:attr:`~threetears.core.collections.base.BaseCollection.datetime_columns`);
- the columns a write leaves to the database (:meth:`DeclaredTableCollection.columns_decided_by_store`),
  so the row every tier caches is the row L3 holds;
- the caches a natural-key upsert settles when it lands on a row stored under another key
  (:meth:`DeclaredTableCollection.settle_stored_key`).
"""

from __future__ import annotations

from typing import Any, ClassVar

from sqlalchemy import MetaData

from threetears.core.collections.base import BaseCollection, EntityT
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.schema_backed import DATETIMETZ_TYPE, TableSchema
from threetears.observe import get_logger

__all__ = ["DeclaredTableCollection"]

log = get_logger(__name__)


class DeclaredTableCollection(BaseCollection[EntityT]):
    """a collection writing its own SQL over the table its :attr:`schema` declares.

    Subclasses declare the table and write their statements::

        class WidgetCollection(DeclaredTableCollection[WidgetEntity]):
            schema = TableSchema(name="widgets", primary_key="id", columns=[...])

            async def save_to_store(self, data, original_timestamp=None, *, conn=None): ...

    :cvar schema: the table as its migrations leave it -- every column, its type, nullability,
        server default and the primary key. Its L1 table is generated from it.
    """

    schema: ClassVar[TableSchema]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """key the collection by its schema's primary key and read its instants off the schema.

        :param kwargs: forwarded to :meth:`BaseCollection.__init_subclass__`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises TypeError: when the class declares a ``primary_key_column`` its schema does not
        """
        schema = cls.__dict__.get("schema")
        if isinstance(schema, TableSchema):
            declared = cls.__dict__.get("primary_key_column")
            key = schema.primary_key
            if declared is not None and declared != key:
                raise TypeError(
                    f"{cls.__name__} declares primary_key_column {declared!r}, but its schema's primary key is {key!r}"
                )
            cls.primary_key_column = key
            # derived before the base hook runs, which checks ``expires_at_column`` against this set
            cls.datetime_columns = cls.datetime_columns | frozenset(
                column.name for column in schema.columns if column.column_type == DATETIMETZ_TYPE
            )
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
        """
        super().__init__(*args, **kwargs)
        #: the keys a write sent that landed on a row stored under another key, until its read back
        self._sent_to_another_key: set[Any] = set()
        l1 = self.l1_backend
        if l1 is not None:
            l1.initialize(self.schema.to_sqlalchemy_table(MetaData()).metadata)

    @property
    def table_name(self) -> str:
        """the declared table's name.

        :return: :attr:`TableSchema.name` of :attr:`schema`
        :rtype: str
        """
        return self.schema.name

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the declared columns whose stored value writing ``data`` leaves to the database.

        See :meth:`BaseCollection.columns_decided_by_store`. Read off the schema, as
        :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection` reads its own: a
        declared column ``data`` does not name (the database fills its default, or an update keeps
        the stored value), and a ``NOT NULL`` column with a server default that ``data`` names as
        ``None``, which no write can store. And every column, when the write landed on a row stored
        under another key (:meth:`settle_stored_key`): the key sent holds no row, so the read back
        finds none and caches nothing.

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
