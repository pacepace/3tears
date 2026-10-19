"""the width of the embeddings a deployment stores, and changing it.

An embedding column is a plain pgvector ``vector(N)``: every row holds a vector of exactly N
dimensions, and the column's HNSW index is built for N. The width is a deployment setting, chosen
once for the embedding provider the host uses, and the packages that store embeddings take it as an
optional ``embedding_dim`` (default :data:`DEFAULT_EMBEDDING_DIM`) on their migration registration,
their collections and their table factories.

How wide a column may be is the backend's to say, because the column is useless without its index:

- PostgreSQL's pgvector HNSW index takes a ``vector`` of at most
  :data:`POSTGRES_MAX_INDEXED_EMBEDDING_DIM` (2,000) dimensions;
- YugabyteDB's ``ybhnsw`` takes up to :data:`MAX_EMBEDDING_DIM` (16,000), and has no ``halfvec``
  operator class, so the column stays a plain ``vector`` there too.

:func:`check_embedding_dim` refuses a width outside ``1..16,000`` wherever a width is configured;
:func:`require_indexable_embedding_dim` asks the database which backend it is, only for a width
above 2,000, when the DDL runs.

A migration runs once, so a width changed after a schema exists is not a migration.
:func:`verify_embedding_columns` reads each column's declared width from the catalogue and raises
:class:`EmbeddingWidthMismatch` when it is not the configured one; :func:`resize_embedding_columns`
is the explicit operator action that changes it. It clears every stored embedding of the column --
an embedding of one width is meaningless at another, so every row has to be embedded again -- and
rebuilds the column's indexes under their own names and options. Nothing calls it automatically. Run it with writes to the tables
stopped: on YugabyteDB the column change is a table rewrite that may not carry over concurrent writes
(yugabyte-db issue 19860).

The DDL follows the YugabyteDB-safe migration rules: one statement per ``execute`` (YugabyteDB
auto-commits DDL), ``IF EXISTS`` / ``IF NOT EXISTS`` on every drop and create, and catalogue lookups
limited to the current schema. The column type is changed with ``USING NULL``, so no ``UPDATE``
runs at all. A resize stopped part way is finished by running it again for the indexes the
collection declares: a column already at the width is not altered again, and a declared index that is
missing is created. An index the collection does not declare (one an operator added) that was dropped
before the stop is not recreated by a second run; its full definition is in the log line written before
anything is dropped, to recreate it by hand.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from threetears.core.collections.schema_backed import VECTOR_TYPE, IndexDef, TableSchema
from threetears.observe import get_logger

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "MAX_EMBEDDING_DIM",
    "POSTGRES_MAX_INDEXED_EMBEDDING_DIM",
    "EmbeddingColumn",
    "EmbeddingWidthMismatch",
    "EmbeddingWidthStore",
    "check_embedding_dim",
    "embedding_columns",
    "indexable_embedding_ceiling",
    "require_indexable_embedding_dim",
    "resize_embedding_columns",
    "schema_with_embedding_dim",
    "stored_embedding_dim",
    "verify_embedding_columns",
]

log = get_logger(__name__)

#: the width every embedding column has unless a deployment configures another: Voyage AI's
#: voyage-4, the provider the family's embedding tables were first built for.
DEFAULT_EMBEDDING_DIM = 1024
#: the widest vector any supported backend can index: YugabyteDB's ``ybhnsw``.
MAX_EMBEDDING_DIM = 16_000
#: the widest ``vector`` PostgreSQL's pgvector HNSW index takes.
POSTGRES_MAX_INDEXED_EMBEDDING_DIM = 2_000

#: ``SELECT version()`` on YugabyteDB reports e.g. ``PostgreSQL 15.12-YB-2025.2.0.0-b0``.
_YUGABYTE_VERSION_MARK = "-YB-"

_VECTOR_TYPE_PATTERN = re.compile(r"(?:[A-Za-z_][\w]*\.)?vector(?:\((\d+)\))?")

_CREATE_INDEX_PATTERN = re.compile(r"^CREATE (UNIQUE )?INDEX (?!IF NOT EXISTS )", re.IGNORECASE)

_WITH_CLAUSE_PATTERN = re.compile(r"\bWITH \(", re.IGNORECASE)

_CURRENT_SCHEMA_SQL = "SELECT current_schema() AS schema_name"

_VERSION_SQL = "SELECT version() AS version"

_COLUMN_TYPE_SQL = """
SELECT format_type(a.atttypid, a.atttypmod) AS column_type
  FROM pg_catalog.pg_attribute a
  JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
  JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname = current_schema()
   AND c.relname = $1
   AND a.attname = $2
   AND a.attnum > 0
   AND NOT a.attisdropped
"""

_COLUMN_INDEXES_SQL = """
SELECT ic.relname AS index_name, pg_get_indexdef(i.indexrelid) AS index_definition,
       ic.reloptions AS index_options
  FROM pg_catalog.pg_index i
  JOIN pg_catalog.pg_class ic ON ic.oid = i.indexrelid
  JOIN pg_catalog.pg_class tc ON tc.oid = i.indrelid
  JOIN pg_catalog.pg_namespace n ON n.oid = tc.relnamespace
  JOIN pg_catalog.pg_attribute a ON a.attrelid = tc.oid
 WHERE n.nspname = current_schema()
   AND tc.relname = $1
   AND a.attname = $2
   AND a.attnum = ANY(i.indkey)
 ORDER BY ic.relname
"""


@runtime_checkable
class EmbeddingWidthStore(Protocol):
    """the store surface reading and changing an embedding column's width needs.

    a :class:`~threetears.core.data.store.DataStore` has it. every statement must reach the same
    schema: a store bound to one session, or a pool whose connections share one search path.
    """

    async def execute(self, sql: str, *params: Any) -> str:
        """execute ``sql`` against the bound schema.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: status string from the underlying engine
        :rtype: str
        """
        ...

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """run ``sql`` against the bound schema and return its rows.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: one dict per row
        :rtype: list[dict[str, Any]]
        """
        ...


class EmbeddingWidthMismatch(Exception):
    """an embedding column in the database is not the width the deployment is configured for.

    raised by :func:`verify_embedding_columns`. a schema's width is fixed by the migration that
    created it, so the configured width cannot simply be changed: either configure the width the
    schema holds, or change the schema with ``resize_embedding_width``, which clears every stored
    embedding so each row is embedded again.

    :ivar table: the table holding the column
    :ivar column: the embedding column
    :ivar stored_width: the width the column is declared with, or ``None`` for an unsized ``vector``
    :ivar configured_width: the width the deployment is configured for
    """

    def __init__(self, table: str, column: str, stored_width: int | None, configured_width: int) -> None:
        """record the column and the two widths, and say what to do about them.

        :param table: the table holding the column
        :ptype table: str
        :param column: the embedding column
        :ptype column: str
        :param stored_width: the column's declared width, or ``None`` for an unsized ``vector``
        :ptype stored_width: int | None
        :param configured_width: the width the deployment is configured for
        :ptype configured_width: int
        :return: nothing
        :rtype: None
        """
        self.table = table
        self.column = column
        self.stored_width = stored_width
        self.configured_width = configured_width
        held = "an unsized vector" if stored_width is None else f"vector({stored_width})"
        super().__init__(
            f"{table}.{column} is {held} in the database, but the configured embedding width is "
            f"{configured_width}. Either configure embedding_dim={stored_width} to match the schema, or "
            f"change the schema with resize_embedding_width(store, {configured_width}): that clears every "
            f"stored embedding, so every row has to be re-embedded."
        )


@dataclass(frozen=True)
class EmbeddingColumn:
    """one embedding column, and the indexes its collection declares over it.

    :ivar table: the table holding the column
    :ivar column: the ``vector`` column
    :ivar indexes: the declared indexes covering the column; a resize creates any that are missing
    """

    table: str
    column: str
    indexes: tuple[IndexDef, ...] = ()


def check_embedding_dim(embedding_dim: int) -> int:
    """refuse an embedding width no supported backend can index.

    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: ``embedding_dim``, unchanged
    :rtype: int
    :raises ValueError: when the width is not an int, is below 1, or is above
        :data:`MAX_EMBEDDING_DIM`
    """
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
        raise ValueError(f"embedding_dim must be an int, got {embedding_dim!r}")
    if not 1 <= embedding_dim <= MAX_EMBEDDING_DIM:
        raise ValueError(
            f"embedding_dim must be between 1 and {MAX_EMBEDDING_DIM:,}, the widest vector YugabyteDB can "
            f"index (PostgreSQL indexes at most {POSTGRES_MAX_INDEXED_EMBEDDING_DIM:,}); got {embedding_dim}"
        )
    return embedding_dim


async def indexable_embedding_ceiling(store: EmbeddingWidthStore) -> int:
    """the widest ``vector`` the database ``store`` reaches can index.

    :param store: a store on the database
    :ptype store: EmbeddingWidthStore
    :return: :data:`MAX_EMBEDDING_DIM` on YugabyteDB, :data:`POSTGRES_MAX_INDEXED_EMBEDDING_DIM`
        otherwise
    :rtype: int
    """
    rows = await store.query(_VERSION_SQL)
    version = str(rows[0]["version"]) if rows else ""
    ceiling = MAX_EMBEDDING_DIM if _YUGABYTE_VERSION_MARK in version else POSTGRES_MAX_INDEXED_EMBEDDING_DIM
    return ceiling


async def require_indexable_embedding_dim(store: EmbeddingWidthStore, embedding_dim: int) -> None:
    """refuse a width the database ``store`` reaches cannot index, before any DDL runs.

    asks the database which backend it is only for a width above
    :data:`POSTGRES_MAX_INDEXED_EMBEDDING_DIM`, which every backend indexes; a width at or below it
    issues no statement.

    :param store: a store on the database
    :ptype store: EmbeddingWidthStore
    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: nothing
    :rtype: None
    :raises ValueError: when the width is out of range (:func:`check_embedding_dim`), or above
        PostgreSQL's HNSW ceiling on PostgreSQL
    """
    check_embedding_dim(embedding_dim)
    if embedding_dim <= POSTGRES_MAX_INDEXED_EMBEDDING_DIM:
        return
    ceiling = await indexable_embedding_ceiling(store)
    if embedding_dim > ceiling:
        raise ValueError(
            f"embedding_dim {embedding_dim} cannot be indexed on PostgreSQL: pgvector's HNSW index takes a "
            f"vector of at most {POSTGRES_MAX_INDEXED_EMBEDDING_DIM:,} dimensions. YugabyteDB indexes up to "
            f"{MAX_EMBEDDING_DIM:,}; on PostgreSQL configure {POSTGRES_MAX_INDEXED_EMBEDDING_DIM:,} or fewer."
        )


def schema_with_embedding_dim(schema: TableSchema, embedding_dim: int) -> TableSchema:
    """``schema`` with every ``vector`` column declared ``embedding_dim`` wide.

    the declaration a collection or table factory derives for a deployment's width; every other
    column, index and constraint is ``schema``'s own.

    :param schema: the declared schema
    :ptype schema: TableSchema
    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: ``schema`` itself when its vector columns already have the width, else a copy
    :rtype: TableSchema
    :raises ValueError: when the width is out of range, or ``schema`` has no vector column
    """
    check_embedding_dim(embedding_dim)
    if not any(col.column_type == VECTOR_TYPE for col in schema.columns):
        raise ValueError(f"table {schema.name!r} declares no vector column to give a width")
    if all(col.vector_dim == embedding_dim for col in schema.columns if col.column_type == VECTOR_TYPE):
        return schema
    columns = [
        dataclasses.replace(col, vector_dim=embedding_dim) if col.column_type == VECTOR_TYPE else col
        for col in schema.columns
    ]
    return dataclasses.replace(schema, columns=columns)


def embedding_columns(*schemas: TableSchema) -> tuple[EmbeddingColumn, ...]:
    """every ``vector`` column ``schemas`` declare, with the declared indexes covering it.

    :param schemas: the declared schemas
    :ptype schemas: TableSchema
    :return: one entry per vector column, in declaration order
    :rtype: tuple[EmbeddingColumn, ...]
    """
    found = tuple(
        EmbeddingColumn(
            table=schema.name,
            column=col.name,
            indexes=tuple(index for index in schema.indexes if col.name in index.columns),
        )
        for schema in schemas
        for col in schema.columns
        if col.column_type == VECTOR_TYPE
    )
    return found


async def stored_embedding_dim(store: EmbeddingWidthStore, table: str, column: str) -> int | None:
    """the width ``table.column`` is declared with in the current schema.

    :param store: a store bound to the schema
    :ptype store: EmbeddingWidthStore
    :param table: the table
    :ptype table: str
    :param column: the ``vector`` column
    :ptype column: str
    :return: the declared width, or ``None`` for an unsized ``vector``
    :rtype: int | None
    :raises LookupError: when the current schema has no such column (its migrations have not run)
    :raises ValueError: when the column is not a ``vector``
    """
    rows = await store.query(_COLUMN_TYPE_SQL, table, column)
    if not rows:
        raise LookupError(
            f"the current schema has no column {table}.{column}; run the package's migrations before "
            f"verifying or resizing its embedding width"
        )
    column_type = str(rows[0]["column_type"])
    match = _VECTOR_TYPE_PATTERN.fullmatch(column_type)
    if match is None:
        raise ValueError(f"{table}.{column} is {column_type}, not a vector column")
    width = None if match.group(1) is None else int(match.group(1))
    return width


async def verify_embedding_columns(
    store: EmbeddingWidthStore,
    columns: Sequence[EmbeddingColumn],
    embedding_dim: int,
) -> None:
    """refuse a schema whose embedding columns are not ``embedding_dim`` wide.

    for a host's start-up: the configured width is checked against the schema it will write to,
    so a width changed in configuration alone fails here, naming what to do, rather than on the
    first write.

    :param store: a store bound to the schema
    :ptype store: EmbeddingWidthStore
    :param columns: the embedding columns to check
    :ptype columns: Sequence[EmbeddingColumn]
    :param embedding_dim: the configured width
    :ptype embedding_dim: int
    :return: nothing
    :rtype: None
    :raises EmbeddingWidthMismatch: on the first column of another width
    :raises LookupError: when a column is missing (the migrations have not run)
    :raises ValueError: when the width is out of range
    """
    check_embedding_dim(embedding_dim)
    for col in columns:
        stored = await stored_embedding_dim(store, col.table, col.column)
        if stored != embedding_dim:
            raise EmbeddingWidthMismatch(col.table, col.column, stored, embedding_dim)


async def resize_embedding_columns(
    store: EmbeddingWidthStore,
    columns: Sequence[EmbeddingColumn],
    embedding_dim: int,
) -> int:
    """change ``columns`` to ``vector(embedding_dim)``, clearing every embedding they hold.

    an explicit operator action, never run automatically, and run with writes to the tables stopped:
    on YugabyteDB the ``ALTER`` is a table rewrite that may not carry over writes made while it runs
    ("Concurrent DMLs may not be reflected in the new table", yugabyte-db issue 19860). every row of
    a resized column loses its embedding and has to be embedded again with a model of the new width. per column, a column
    already at the width is left as it is; otherwise the indexes on it are dropped, the column is
    altered with ``USING NULL``, and each dropped index is created again from its own definition,
    under its own name and options (its ``reloptions`` added where the definition leaves them out, as
    YugabyteDB's does). then every index the column's collection declares is created
    if it is missing, which also completes a resize that was stopped between a drop and its
    create.

    a :class:`~threetears.core.data.store.DataStore` is used through
    :meth:`~threetears.core.data.store.DataStore.ddl_session`: one connection, holding the
    database-wide DDL lock, for the whole resize. any other store must already be one session on
    the schema.

    :param store: a store bound to the schema
    :ptype store: EmbeddingWidthStore
    :param columns: the embedding columns to resize
    :ptype columns: Sequence[EmbeddingColumn]
    :param embedding_dim: the new width
    :ptype embedding_dim: int
    :return: how many stored embeddings were cleared
    :rtype: int
    :raises ValueError: when the width is out of range, or above what the database can index
    :raises LookupError: when a column is missing (the migrations have not run)
    """
    from threetears.core.data.store import DataStore

    check_embedding_dim(embedding_dim)
    cleared = 0
    if isinstance(store, DataStore) and not store.holds_ddl_lock:
        async with store.ddl_session() as locked:
            cleared = await _resize_on_one_session(locked, columns, embedding_dim)
    else:
        cleared = await _resize_on_one_session(store, columns, embedding_dim)
    return cleared


async def _resize_on_one_session(
    store: EmbeddingWidthStore,
    columns: Sequence[EmbeddingColumn],
    embedding_dim: int,
) -> int:
    """resize every column through one session that already reaches the schema.

    :param store: a store on one session
    :ptype store: EmbeddingWidthStore
    :param columns: the embedding columns to resize
    :ptype columns: Sequence[EmbeddingColumn]
    :param embedding_dim: the new width
    :ptype embedding_dim: int
    :return: how many stored embeddings were cleared
    :rtype: int
    """
    await require_indexable_embedding_dim(store, embedding_dim)
    rows = await store.query(_CURRENT_SCHEMA_SQL)
    schema = str(rows[0]["schema_name"])
    cleared = 0
    for col in columns:
        cleared += await _resize_column(store, schema, col, embedding_dim)
    return cleared


async def _resize_column(store: EmbeddingWidthStore, schema: str, col: EmbeddingColumn, embedding_dim: int) -> int:
    """resize one column, rebuild its indexes, and return how many embeddings it held.

    :param store: a store on one session
    :ptype store: EmbeddingWidthStore
    :param schema: the current schema's name
    :ptype schema: str
    :param col: the column
    :ptype col: EmbeddingColumn
    :param embedding_dim: the new width
    :ptype embedding_dim: int
    :return: how many stored embeddings were cleared
    :rtype: int
    """
    table = f"{_quote(schema)}.{_quote(col.table)}"
    held = 0
    stored = await stored_embedding_dim(store, col.table, col.column)
    if stored != embedding_dim:
        indexes = await store.query(_COLUMN_INDEXES_SQL, col.table, col.column)
        counted = await store.query(f"SELECT count(*) AS held FROM {table} WHERE {_quote(col.column)} IS NOT NULL")
        held = int(counted[0]["held"])
        log.warning(
            "resizing an embedding column clears every embedding it holds",
            extra={
                "extra_data": {
                    "table": col.table,
                    "column": col.column,
                    "stored_width": stored,
                    "new_width": embedding_dim,
                    "embeddings_cleared": held,
                    "indexes_rebuilt": [str(row["index_name"]) for row in indexes],
                    # every definition, before anything is dropped: an index the collection does not
                    # declare cannot be rebuilt by a second run if this one stops after dropping it
                    "index_definitions": [
                        _recreate_index_sql(str(row["index_definition"]), row["index_options"]) for row in indexes
                    ],
                }
            },
        )
        for row in indexes:
            await store.execute(f"DROP INDEX IF EXISTS {_quote(schema)}.{_quote(str(row['index_name']))}")
        await store.execute(
            f"ALTER TABLE {table} ALTER COLUMN {_quote(col.column)} "
            f"TYPE public.vector({embedding_dim}) USING NULL::public.vector({embedding_dim})"
        )
        for row in indexes:
            await store.execute(_recreate_index_sql(str(row["index_definition"]), row["index_options"]))
    for index in col.indexes:
        await store.execute(_declared_index_sql(table, index))
    return held


def _recreate_index_sql(definition: str, options: Sequence[str] | None) -> str:
    """the statement creating an index again from its catalogue definition, options included.

    PostgreSQL's ``pg_get_indexdef`` carries the index's ``WITH (...)``; YugabyteDB's leaves it out
    (an ``ybhnsw`` index built ``WITH (m=16, ef_construction=64)`` reads back without it), though
    ``pg_class.reloptions`` still holds the options. a definition with no ``WITH (`` clause takes
    the stored options, before any partial-index ``WHERE``; one that has it is used as it is.

    :param definition: what ``pg_get_indexdef`` answers for the index
    :ptype definition: str
    :param options: the index's ``reloptions`` (``key=value`` strings), or ``None``
    :ptype options: Sequence[str] | None
    :return: a ``CREATE [UNIQUE] INDEX IF NOT EXISTS`` statement
    :rtype: str
    """
    sql = _CREATE_INDEX_PATTERN.sub(r"CREATE \1INDEX IF NOT EXISTS ", definition)
    if options and _WITH_CLAUSE_PATTERN.search(sql) is None:
        with_clause = f" WITH ({', '.join(options)})"
        head, where, predicate = sql.partition(" WHERE ")
        sql = f"{head}{with_clause}{where}{predicate}"
    return sql


def _declared_index_sql(table: str, index: IndexDef) -> str:
    """``CREATE INDEX IF NOT EXISTS`` for a declared index on ``table``.

    an operator class without a schema is the pgvector extension's, which the family installs in
    ``public``, the schema every agent migration names it in.

    :param table: the qualified table
    :ptype table: str
    :param index: the declared index
    :ptype index: IndexDef
    :return: the statement
    :rtype: str
    """
    ops = dict(index.ops or ())
    parts = []
    for name in index.columns:
        opclass = ops.get(name)
        if opclass is not None and "." not in opclass:
            opclass = f"public.{opclass}"
        parts.append(_quote(name) if opclass is None else f"{_quote(name)} {opclass}")
    unique = "UNIQUE " if index.unique else ""
    using = f" USING {index.using}" if index.using else ""
    with_clause = ""
    if index.pg_with:
        with_clause = " WITH (" + ", ".join(f"{key} = {value}" for key, value in index.pg_with) + ")"
    where = f" WHERE {index.where}" if index.where else ""
    sql = f"CREATE {unique}INDEX IF NOT EXISTS {_quote(index.name)} ON {table}{using} ({', '.join(parts)}){with_clause}{where}"
    return sql


def _quote(identifier: str) -> str:
    """``identifier`` as a quoted SQL identifier.

    :param identifier: a schema, table, column or index name
    :ptype identifier: str
    :return: the name in double quotes, any double quote in it doubled
    :rtype: str
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
