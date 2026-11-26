"""stamp a table that exists already: its epoch, its gone flag, its restore bucket, index and declaration.

Append-only L3 (``design-append-only-l3.md`` decision 3, Pace 2026-11-25): the tables are their own change record, so
the change-note ring can be retired. A table declared ``ROW_STAMPED`` (one row a key, updated in place) carries its
last write's epoch, a ``gone`` flag a delete sets in place, and a ``(epoch_bucket ASC, epoch)`` index a restore of a
lost epoch log reads a bucket at a time, its ``COMMENT ON INDEX`` naming how. ``TableSchema.create_statements``
renders all of it for a table created from its declaration; this module gives the same to a table some earlier
migration created without them, the one step every owning package's migration set takes (3tears' own tables; the
hub's v187 took the same steps for its own).

For each :class:`StampingTable`, in order (:func:`stamp_existing_table`):

1. ``epoch BIGINT``, added nullable (DDL).
2. ``gone BOOLEAN NOT NULL DEFAULT false`` (DDL), as ``TableSchema._declare_stamped_gone`` declares it: every row
   live. WHY before the backfill: a keyed read of a table declared stamped reads its live rows alone (``NOT gone``), and
   the backfill's pages are such a read, so the flag is there first.
3. **The backfill** (DML, its own statements): every row whose epoch is NULL stamped with one fresh epoch, one keyed
   ``UPDATE`` a page of keys (:func:`backfill_epochs`: ``keyset_pages`` then ``update_keyed``), never one ``UPDATE``
   of the whole table. WHY one fresh stamp: an existing row's last write is older than any restore's X, and a row
   stamped now is read by the first restore after the cutover, which re-evicts it at worst (harmless).
4. ``epoch`` set ``NOT NULL``, no default (DDL): a write that bypasses 3tears unstamped is refused by the database.
5. ``epoch_bucket SMALLINT GENERATED ALWAYS AS (<the key's hash> & 15) STORED NOT NULL`` (DDL): ``bucket_sql`` over
   each key column's ``key_hash_sql``, as ``TableSchema._bucket_expression`` renders it.
6. The ``(epoch_bucket ASC, epoch)`` index, ``SPLIT AT VALUES ((1), ..., (15))`` on YugabyteDB (a tablet a bucket;
   none on Postgres), then its ``COMMENT ON INDEX`` declaration (``"kind":"stamped"``).

Each step is the database's own (DDL and DML never share a statement: the YugabyteDB DDL/DML rule of
:mod:`~threetears.core.data.migrations.helpers`). **Idempotent**: each DDL ``IF NOT EXISTS``, the backfill touching only
rows whose epoch is NULL, the comment rewritten as it is. **The outage-window cutover** (Pace 2026-11-25: one cutover,
everything scaled to zero): nothing else writes while it runs; it never truncates and never drops a row.

A migration names its tables FROZEN, as they are at its version (:class:`StampingTable`): a later change of a table's
key is its own migration. :func:`stamping_table_of` derives one from a ``TableSchema``, so a test holds a migration's
frozen tables to what each declaration renders.

**Not imported by** ``threetears.core.data.migrations``: it reads through the keyed helpers, which import the
collections, and the collections import that package; a migration imports this module by its own path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    GONE_COLUMN,
    bucket_epoch_index_name,
    bucket_split_sql,
    bucket_sql,
    key_hash_sql,
    next_epoch,
    stamped_restore_declaration,
)
from threetears.core.keyed import update_keyed
from threetears.core.keyset import keyset_pages
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.filters import Fragment
from threetears.interfaces.keyed.rows import KeyBatch
from threetears.observe import get_logger

__all__ = [
    "BACKFILL_PAGE",
    "StampingTable",
    "backfill_epochs",
    "bucket_statements",
    "on_yugabyte",
    "stamp_existing_table",
    "stamp_existing_tables",
    "stamping_table_of",
]

log = get_logger(__name__)

#: the keys one backfill statement stamps. WHY 1,000 (as the hub's v187): a page the L3 rail's cap answers in one
#: statement, and an ``UPDATE`` of 1,000 keys a short one on every engine; 3tears' own tables are a few pages at most
BACKFILL_PAGE: Final[int] = 1000

#: the mark YugabyteDB's ``version()`` carries (``PostgreSQL 15.12-YB-2025.1...``)
_YUGABYTE_MARK: Final[str] = "-YB-"

#: the SQL type each key-hash kind (:func:`~threetears.core.ddl.row_epoch.key_hash_sql`) binds a key array as
_KEY_CASTS: Final[dict[str, str]] = {
    "text": "text",
    "uuid": "uuid",
    "int4": "integer",
    "int8": "bigint",
    "timestamptz": "timestamptz",
    "numeric": "numeric",
    "bool": "boolean",
    "bytea": "bytea",
    "jsonb": "jsonb",
}


@dataclass(frozen=True, slots=True)
class StampingTable:
    """one table a migration stamps, as its ``TableSchema`` declares it at that migration's version.

    :ivar name: the table, bare (the migration's ``search_path`` names its schema)
    :ivar keys: its primary key, ``(column, hash kind)`` in key order; the kind is the column's hash's
        (:func:`~threetears.core.ddl.row_epoch.key_hash_sql`: ``uuid``, ``text``, ...) and the SQL type its keys bind
        as
    :ivar instants: the key columns that are ``timestamptz``, as the restore declaration names them
    :ivar tenancy: the table's tenancy kind and its tenant column, as the restore declaration names them
    :ivar moveable: the columns a write may move its rows between tenants by (``TableSchema.moveable``), as the
        restore declaration names them; none for most tables
    :ivar columns: its declared invalidation columns (``TableSchema.invalidation_columns``), as the restore
        declaration names them; none for most tables
    :ivar schema: the schema holding it when not the migration's own (``search_path``): a table of a second schema of
        the same database (the hub's audit table, in ``<platform>_audit``). Its statements name it qualified; its index
        and declaration keep the bare name
    """

    name: str
    keys: tuple[tuple[str, str], ...]
    instants: tuple[str, ...] = ()
    tenancy: tuple[str, str | None] = ("undeclared", None)
    moveable: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    schema: str | None = None

    @property
    def qualified(self) -> str:
        """the table as its statements name it: quoted, under its own schema when it names one.

        :return: the name
        :rtype: str
        """
        quoted = quote_identifier(self.name)
        return quoted if self.schema is None else f"{quote_identifier(self.schema)}.{quoted}"

    @property
    def index(self) -> str:
        """the ``(epoch_bucket, epoch)`` index's name, as 3tears names it.

        :return: the name
        :rtype: str
        """
        return bucket_epoch_index_name(self.name)

    @property
    def key_columns(self) -> tuple[str, ...]:
        """the primary key's columns, in order.

        :return: them
        :rtype: tuple[str, ...]
        """
        return tuple(column for column, _ in self.keys)


def stamping_table_of(schema: Any) -> StampingTable:
    """the :class:`StampingTable` a ``TableSchema`` declared ``ROW_STAMPED`` renders, its key, instants and tenancy
    as its own restore comment names them: what a test holds a migration's frozen table to.

    :param schema: the table's ``TableSchema``
    :ptype schema: TableSchema
    :return: the table
    :rtype: StampingTable
    :raises ValueError: when the schema is not stamped, or a key column's type has no bucket hash
    """
    declared = schema.row_epoch
    if declared is None or declared.versions:
        raise ValueError(f"stamping_table_of: {schema.name} is not declared ROW_STAMPED")
    keys = tuple((column, schema.bucket_hash_kind(column)) for column in schema.pk_columns)
    instants = tuple(column for column, kind in keys if kind == "timestamptz")
    return StampingTable(
        schema.name,
        keys,
        instants,
        (schema.tenancy.kind, schema.tenancy.column),
        tuple(schema.moveable),
        tuple(schema.invalidation_columns),
    )


async def on_yugabyte(store: Any) -> bool:
    """whether the migration's database is YugabyteDB, whose restore index is split a tablet a bucket.

    :param store: the migration's store (``DataStore``), or anything with its ``query``
    :ptype store: Any
    :return: whether it is
    :rtype: bool
    """
    rows = await store.query("SELECT version() AS version")
    return _YUGABYTE_MARK in str(rows[0]["version"])


def bucket_statements(table: StampingTable, *, yugabyte: bool) -> tuple[str, ...]:
    """steps 5 and 6: the stored bucket, its ``(bucket, epoch)`` index (split a tablet a bucket on YugabyteDB), and the
    index's restore declaration, each as ``TableSchema.create_statements`` renders it.

    :param table: the table
    :ptype table: StampingTable
    :param yugabyte: whether the database is YugabyteDB
    :ptype yugabyte: bool
    :return: the statements
    :rtype: tuple[str, ...]
    """
    quoted = table.qualified
    expression = bucket_sql(tuple(key_hash_sql(quote_identifier(column), kind) for column, kind in table.keys))
    declaration = stamped_restore_declaration(
        table.key_columns, table.instants, table.tenancy, table.moveable, table.columns
    ).replace("'", "''")
    split = f" {bucket_split_sql()}" if yugabyte else ""
    # an index lives in its table's schema: created by its bare name, commented on under that schema
    index = quote_identifier(table.index)
    commented = index if table.schema is None else f"{quote_identifier(table.schema)}.{index}"
    return (
        f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {BUCKET_COLUMN} smallint "
        f"GENERATED ALWAYS AS ({expression}) STORED NOT NULL",
        "-- purpose: threetears.core.collections.table_restore::_bucket_rows\n"
        "-- reason: a lost change log is rebuilt from each bucket's rows stamped since the keeper's mark\n"
        f"CREATE INDEX IF NOT EXISTS {index} ON {quoted} ({BUCKET_COLUMN} ASC, {EPOCH_COLUMN})"
        f"{split}",
        f"COMMENT ON INDEX {commented} IS '{declaration}'",
    )


class _Rail:
    """the migration's store as the keyed helpers read and write through it: asyncpg underneath, so its arrays bind
    natively."""

    binds_native_arrays: Final = True

    def __init__(self, store: Any) -> None:
        """hold the store.

        :param store: the migration's store
        :ptype store: Any
        """
        self._store = store

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """run a read.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :return: its rows
        :rtype: list[dict[str, Any]]
        """
        rows: list[dict[str, Any]] = await self._store.query(sql, *params)
        return rows

    async def execute(self, sql: str, *params: Any) -> str:
        """run a write.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :return: its status
        :rtype: str
        """
        status: str = await self._store.execute(sql, *params)
        return status


async def backfill_epochs(store: Any, table: StampingTable, *, page_size: int = BACKFILL_PAGE) -> int:
    """step 2: stamp every row of ``table`` whose epoch is NULL with one fresh epoch, a page of keys a statement.

    The keys are read a page at a time (``keyset_pages``, on the primary key, walking a hashed lead on YugabyteDB),
    and each page's keys are stamped by one keyed ``UPDATE`` (``update_keyed``, the page's keys bound as one array a
    key column). Never one ``UPDATE`` of the whole table. The pages are handed over one at a time, so the migration
    holds one page of keys at once, function-local.

    :param store: the migration's store (``DataStore``), or anything with its ``query`` and ``execute``
    :ptype store: Any
    :param table: the table
    :ptype table: StampingTable
    :param page_size: the keys a statement stamps
    :ptype page_size: int
    :return: the rows stamped
    :rtype: int
    """
    rail = _Rail(store)
    # the keyed helpers quote a table themselves: it is named to them bare (a plain identifier, frozen), under its
    # own schema when it names one (a TRUSTED ``schema.table``, as they take it)
    name = table.name if table.schema is None else f"{table.schema}.{table.name}"
    stamp = next_epoch()
    columns = ", ".join(quote_identifier(column) for column in table.key_columns)
    casts = tuple(_KEY_CASTS.get(kind, "text") for _, kind in table.keys)
    lead_array = f"{casts[0]}[]" if len(table.keys) > 1 else None
    stamped = 0
    async for page in keyset_pages(
        rail,
        f"SELECT {columns} FROM {name}",  # noqa: S608 - a frozen table and its declared key
        table.key_columns,
        page_size=page_size,
        lead_array=lead_array,
        # named, so a table declared stamped is read as one: its live rows, every row before any delete marked one
        table=name,
    ):
        if not page:
            continue
        keys = KeyBatch(
            columns=table.key_columns,
            keys=[[row[column] for row in page] for column in table.key_columns],
            casts=casts,
        )
        written = await update_keyed(
            rail,
            name,
            values={EPOCH_COLUMN: stamp},
            keys=keys,
            where=[Fragment(f"{quote_identifier(EPOCH_COLUMN)} IS NULL")],
            batch=page_size,
        )
        stamped += written.count
    return stamped


async def stamp_existing_table(store: Any, table: StampingTable, *, yugabyte: bool) -> int:
    """the six steps for one table, in order: the epoch column, the gone flag, the epoch's backfill, required, the
    stored bucket, its index and declaration.

    :param store: the migration's store, bound to the table's schema via ``search_path``
    :ptype store: Any
    :param table: the table
    :ptype table: StampingTable
    :param yugabyte: whether the database is YugabyteDB (:func:`on_yugabyte`)
    :ptype yugabyte: bool
    :return: the rows the backfill stamped
    :rtype: int
    """
    quoted = table.qualified
    await store.execute(f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {EPOCH_COLUMN} bigint")
    await store.execute(f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {GONE_COLUMN} boolean NOT NULL DEFAULT false")
    stamped = await backfill_epochs(store, table)
    await store.execute(f"ALTER TABLE {quoted} ALTER COLUMN {EPOCH_COLUMN} SET NOT NULL")
    for statement in bucket_statements(table, yugabyte=yugabyte):
        await store.execute(statement)
    log.info(
        "a table carries its epoch and its restore index",
        extra={"extra_data": {"table": table.name, "rows_stamped": stamped}},
    )
    return stamped


async def stamp_existing_tables(store: Any, tables: tuple[StampingTable, ...]) -> int:
    """:func:`stamp_existing_table` for each of ``tables``, in order, the database asked once whether it is YugabyteDB.

    :param store: the migration's store, bound to the tables' schema via ``search_path``
    :ptype store: Any
    :param tables: the tables, frozen as the migration names them
    :ptype tables: tuple[StampingTable, ...]
    :return: the rows the backfills stamped, all told
    :rtype: int
    """
    yugabyte = await on_yugabyte(store)
    stamped = 0
    for table in tables:
        stamped += await stamp_existing_table(store, table, yugabyte=yugabyte)
    return stamped
