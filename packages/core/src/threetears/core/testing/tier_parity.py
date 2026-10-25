"""a gate holding every tier of a collection's table to one schema, top to bottom.

A collection that declares a :class:`~threetears.core.collections.schema_backed.TableSchema` keeps
its rows in up to four places: the migrated Postgres table (L3), the SQLite or DuckDB L1 generated
from the schema, and the L2 codec. A tier that changes a value's type or loses its precision serves
that changed value on every read it answers -- a ``NUMERIC`` read back as a ``float``, an instant
without its zone -- and nothing else notices. This gate writes one row of exacting values (a
``Decimal`` at its column's full precision, a negative one, an aware instant to the microsecond, a
UUID, nested JSON, a bool, raw bytes, an integer beyond a float's precision) through every tier and
reads it back, and names the collection, column and tier of every difference. Through L2 it also
sends each ``NUMERIC`` as a ``float``, as an older writer cached it, and expects its ``Decimal`` back.

Two halves, both for any host to run over its own collections:

- :func:`tier_parity_findings` needs nothing live: the schema against the SQLite L1, the DuckDB L1
  (when DuckDB is installed) and the collection's own L2 codec.
- :func:`live_tier_parity_findings` needs a Postgres connection holding the collections' tables,
  set up as the host's pool sets one up (:func:`~threetears.core.collections.init_connection`): the
  migrated table's columns, types, precision and scale, nullability, defaults and primary key
  against the schema, and the row written to a scratch copy of the table and read back.

Each returns its findings, one line each; an empty list is a pass.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Context, Decimal
from typing import Any

import asyncpg
from sqlalchemy import MetaData

from threetears.core.cache.base import UnsupportedL1ColumnError
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import NO_L2, BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    TableSchema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["live_tier_parity_findings", "tier_parity_findings"]

log = get_logger(__name__)

try:
    import duckdb

    from threetears.core.cache.duckdb import DuckDBBackend

    _DUCKDB: type[DuckDBBackend] | None = DuckDBBackend
    #: what a write or read the backends refuse raises
    _L1_ERRORS: tuple[type[Exception], ...] = (TypeError, ValueError, OverflowError, sqlite3.Error, duckdb.Error)
except ImportError:  # NOSILENT: DuckDB is an optional extra; without it there is no DuckDB L1 to hold
    _DUCKDB = None
    _L1_ERRORS = (TypeError, ValueError, OverflowError, sqlite3.Error)

#: builds a collection of a class for its L2 codec
CollectionBuilder = Callable[[type[BaseCollection[Any]]], BaseCollection[Any]]

#: the Postgres type each column type is migrated as (``information_schema.columns.data_type``)
_L3_DATA_TYPES: Mapping[str, frozenset[str]] = {
    UUID_TYPE: frozenset({"uuid"}),
    STRING_TYPE: frozenset({"text", "character varying"}),
    DATETIMETZ_TYPE: frozenset({"timestamp with time zone"}),
    JSONB_TYPE: frozenset({"jsonb"}),
    BYTES_TYPE: frozenset({"bytea"}),
    INT_TYPE: frozenset({"integer"}),
    BIGINT_TYPE: frozenset({"bigint"}),
    BOOL_TYPE: frozenset({"boolean"}),
    NUMERIC_TYPE: frozenset({"numeric"}),
    TSVECTOR_TYPE: frozenset({"tsvector"}),
    VECTOR_TYPE: frozenset({"USER-DEFINED"}),
    ENUM_TYPE: frozenset({"USER-DEFINED", "text", "character varying"}),
}


def _sample(column: Column) -> Any:
    """the exacting value the gate writes to ``column``, or ``None`` for a column no write sets.

    :param column: the column
    :ptype column: Column
    :return: the value
    :rtype: Any
    """
    kind = column.column_type
    value: Any = None
    if kind == NUMERIC_TYPE and column.precision is not None and column.scale is not None:
        # every digit the column holds, negative: a float or a narrower integer loses some of it.
        # the context carries the column's own precision: decimal's default 28 digits would round
        # a wider value up past the column
        value = Decimal(-(10**column.precision - 1)).scaleb(-column.scale, Context(prec=column.precision))
    elif kind == VECTOR_TYPE and column.vector_dim is not None:
        value = [0.5 if index % 2 else -0.25 for index in range(column.vector_dim)]
    elif kind == ENUM_TYPE and column.enum_type:
        value = column.enum_type[0]
    else:
        value = {
            UUID_TYPE: uuid.UUID("0190c0de-7a11-7e57-9b1a-5ca1ab1e0001"),
            STRING_TYPE: "tier parity ✓",
            DATETIMETZ_TYPE: datetime(2026, 10, 24, 12, 34, 56, 789012, tzinfo=UTC),
            JSONB_TYPE: {"nested": {"n": 1, "f": 1.5, "s": "x", "l": [1, "two", None], "b": True}},
            BYTES_TYPE: b"\x00\xfftier",
            INT_TYPE: -2147483647,
            BIGINT_TYPE: 9007199254740993,
            BOOL_TYPE: True,
        }.get(kind)
    return value


def _sample_row(schema: TableSchema) -> dict[str, Any]:
    """one row of every column a write sets (a trigger-maintained ``tsvector`` is not one).

    :param schema: the table
    :ptype schema: TableSchema
    :return: the row
    :rtype: dict[str, Any]
    """
    return {column.name: _sample(column) for column in schema.columns if column.column_type != TSVECTOR_TYPE}


def _difference(sent: Any, got: Any) -> str | None:
    """how ``got`` differs from ``sent`` in type or value, or ``None`` when it is the same.

    :param sent: the value written
    :ptype sent: Any
    :param got: the value read back
    :ptype got: Any
    :return: the difference, or ``None``
    :rtype: str | None
    """
    result: str | None = None
    if sent is None:
        result = None if got is None else f"read back {got!r} for NULL"
    elif not isinstance(got, type(sent)) or (isinstance(got, bool) and not isinstance(sent, bool)):
        result = f"read back {got!r} ({type(got).__name__}) for {sent!r} ({type(sent).__name__})"
    elif isinstance(sent, Decimal) and (got != sent or got.as_tuple().exponent != sent.as_tuple().exponent):
        result = f"read back {got!r} for {sent!r}"
    elif isinstance(sent, datetime) and got.tzinfo is None:
        result = f"read back {got!r}, an instant without its zone, for {sent!r}"
    elif got != sent:
        result = f"read back {got!r} for {sent!r}"
    return result


def _row_findings(name: str, tier: str, sent: Mapping[str, Any], got: Mapping[str, Any] | None) -> list[str]:
    """the findings of one tier's read back of the sample row.

    :param name: the collection's name
    :ptype name: str
    :param tier: the tier
    :ptype tier: str
    :param sent: the row written
    :ptype sent: Mapping[str, Any]
    :param got: the row read back, or ``None`` when none was
    :ptype got: Mapping[str, Any] | None
    :return: one line per column that came back different
    :rtype: list[str]
    """
    if got is None:
        return [f"{name}: {tier}: the row written was not read back"]
    findings = []
    for column, value in sent.items():
        difference = _difference(value, got.get(column)) if column in got else "the column is missing"
        if difference is not None:
            findings.append(f"{name}.{column}: {tier}: {difference}")
    return findings


def _l1_findings(name: str, tier: str, backend: Any, schema: TableSchema) -> list[str]:
    """create the schema's table on a fresh L1, write the sample row, and compare what it reads back.

    :param name: the collection's name
    :ptype name: str
    :param tier: the tier
    :ptype tier: str
    :param backend: a fresh, empty L1 backend
    :ptype backend: Any
    :param schema: the table
    :ptype schema: TableSchema
    :return: the findings
    :rtype: list[str]
    """
    findings: list[str] = []
    try:
        backend.initialize(schema.to_sqlalchemy_table(MetaData()).metadata)
    except UnsupportedL1ColumnError as exc:
        findings.append(f"{name}: {tier}: {exc}")
    else:
        row = _sample_row(schema)
        key = tuple(row[column] for column in schema.pk_columns)
        try:
            backend.upsert(schema.name, row, schema.pk_columns)
            got = backend.select_by_id(schema.name, key if len(key) > 1 else key[0], schema.pk_columns)
        except _L1_ERRORS as exc:
            findings.append(f"{name}: {tier}: the row could not be written and read: {type(exc).__name__}: {exc}")
        else:
            findings.extend(_row_findings(name, tier, row, got))
    finally:
        backend.reset()
    return findings


def _default_build(cls: type[BaseCollection[Any]]) -> BaseCollection[Any]:
    """a collection of ``cls`` on an empty registry with no L2, for its codec alone.

    :param cls: the collection class
    :ptype cls: type[BaseCollection[Any]]
    :return: the collection
    :rtype: BaseCollection[Any]
    """
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return cls(registry=CollectionRegistry(), config=config, nats_client=NO_L2)


def _schema_of(cls: type[BaseCollection[Any]]) -> TableSchema | None:
    """the ``TableSchema`` a collection class declares, or ``None``.

    :param cls: the collection class
    :ptype cls: type[BaseCollection[Any]]
    :return: the schema
    :rtype: TableSchema | None
    """
    schema = getattr(cls, "schema", None)
    return schema if isinstance(schema, TableSchema) else None


def tier_parity_findings(
    collections: Iterable[type[BaseCollection[Any]]], *, build: CollectionBuilder | None = None
) -> list[str]:
    """every difference between a collection's schema and what its SQLite L1, DuckDB L1 and L2 hand back.

    For each class that declares a ``TableSchema``: unless it keeps no L1 copy (``caches_in_l1 = False``),
    the schema's table is created on a fresh SQLite L1 and, when DuckDB is installed, a fresh DuckDB L1, each of which refuses a column it cannot
    hold exactly; the sample row is written to each and read back; and it is passed through the
    collection's own L2 codec (``serialize``, then ``decode_row``). A class with no schema is skipped.

    :param collections: the collection classes
    :ptype collections: Iterable[type[BaseCollection[Any]]]
    :param build: builds a collection of a class, for its L2 codec; the default builds one on an
        empty registry with no L2, which a class whose constructor takes more cannot use
    :ptype build: CollectionBuilder | None
    :return: one line per difference, naming the collection, the column and the tier; empty when
        every tier hands back what was written
    :rtype: list[str]
    """
    findings: list[str] = []
    builder = build or _default_build
    for cls in collections:
        schema = _schema_of(cls)
        if schema is None:
            continue
        name = cls.__qualname__
        if cls.caches_in_l1:
            findings.extend(_l1_findings(name, "SQLite L1", SQLiteBackend(f"tier_parity_{uuid.uuid7().hex}"), schema))
            if _DUCKDB is not None:
                findings.extend(_l1_findings(name, "DuckDB L1", _DUCKDB(), schema))
        row = _sample_row(schema)
        try:
            collection = builder(cls)
        except TypeError as exc:
            findings.append(f"{name}: L2: could not build the collection for its codec ({exc}); pass build=")
            continue
        findings.extend(_l2_findings(name, "L2", collection, row, row))
        floats = _float_numeric_row(schema)
        if floats is not None:
            sent, expected = floats
            findings.extend(_l2_findings(name, "L2 (a NUMERIC sent as a float)", collection, sent, expected))
    return findings


def _float_numeric_row(schema: TableSchema) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """the sample row with each ``NUMERIC`` column a ``float``, and the row the codec must hand back for it.

    A writer that passes a float, or a pod whose L1 held ``REAL``, puts a JSON number in L2; the codec
    must hand back the ``Decimal`` that float was written from, as L3 would.

    :param schema: the table
    :ptype schema: TableSchema
    :return: the row sent and the row expected back, or ``None`` when the table has no ``NUMERIC``
    :rtype: tuple[dict[str, Any], dict[str, Any]] | None
    """
    numeric = [
        column
        for column in schema.columns
        if column.column_type == NUMERIC_TYPE and column.precision is not None and column.scale is not None
    ]
    if not numeric:
        return None
    sent = _sample_row(schema)
    expected = dict(sent)
    for column in numeric:
        # fits every NUMERIC(p, s) with p > s, and 0.5 every one with a scale
        value = -0.5 if column.scale else -7.0
        sent[column.name] = value
        expected[column.name] = Decimal(repr(value))
    return sent, expected


def _l2_findings(
    name: str, tier: str, collection: BaseCollection[Any], sent: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    """pass ``sent`` through the collection's L2 codec and compare what it hands back with ``expected``.

    :param name: the collection's name
    :ptype name: str
    :param tier: the tier, as the findings name it
    :ptype tier: str
    :param collection: the collection, for its codec
    :ptype collection: BaseCollection[Any]
    :param sent: the row encoded
    :ptype sent: Mapping[str, Any]
    :param expected: the row the codec must hand back
    :ptype expected: Mapping[str, Any]
    :return: the findings
    :rtype: list[str]
    """
    try:
        decoded = collection.decode_row(collection.serialize(dict(sent)))
    except (TypeError, ValueError) as exc:
        return [f"{name}: {tier}: the row did not survive its codec: {type(exc).__name__}: {exc}"]
    return _row_findings(name, tier, expected, decoded)


def _has_default(expression: str | None) -> bool:
    """whether a default expression gives a column a value: ``NULL`` (or ``NULL::type``) is no default.

    A schema declares ``server_default="NULL"`` on a nullable column so its generated writes leave
    the column out when a row does not name it; Postgres records no default for it, and none is
    needed. Either way the column defaults to ``NULL``.

    :param expression: the declared ``server_default``, or ``information_schema``'s ``column_default``
    :ptype expression: str | None
    :return: whether it is a default other than ``NULL``
    :rtype: bool
    """
    return expression is not None and expression.strip().upper().split("::")[0] != "NULL"


def _l3_column_findings(
    name: str, schema: TableSchema, live: Sequence[Mapping[str, Any]], pk: tuple[str, ...], not_owned: frozenset[str]
) -> list[str]:
    """every difference between the schema and the migrated table's columns and key.

    :param name: the collection's name
    :ptype name: str
    :param schema: the table as declared
    :ptype schema: TableSchema
    :param live: ``information_schema.columns`` rows of the migrated table
    :ptype live: Sequence[Mapping[str, Any]]
    :param pk: the migrated table's primary key, in order
    :ptype pk: tuple[str, ...]
    :param not_owned: the migrated columns the collection deliberately neither reads nor writes
    :ptype not_owned: frozenset[str]
    :return: the findings
    :rtype: list[str]
    """
    findings: list[str] = []
    by_name = {str(row["column_name"]): row for row in live}
    declared = {column.name: column for column in schema.columns}
    findings.extend(
        f"{name}.{missing}: L3: the migrated table has the column, the TableSchema does not"
        for missing in sorted(set(by_name) - set(declared) - not_owned)
    )
    findings.extend(
        f"{name}.{column}: L3: listed as not owned, but the migrated table has no such column"
        if column not in by_name
        else f"{name}.{column}: L3: listed as not owned, but the TableSchema declares it"
        for column in sorted(not_owned)
        if column not in by_name or column in declared
    )
    for column in schema.columns:
        row = by_name.get(column.name)
        if row is None:
            findings.append(
                f"{name}.{column.name}: L3: the TableSchema declares the column, the migrated table does not"
            )
            continue
        data_type = str(row["data_type"])
        if data_type not in _L3_DATA_TYPES.get(column.column_type, frozenset()):
            findings.append(f"{name}.{column.name}: L3: migrated as {data_type}, declared {column.column_type}")
        if column.column_type == NUMERIC_TYPE and (row["numeric_precision"], row["numeric_scale"]) != (
            column.precision,
            column.scale,
        ):
            findings.append(
                f"{name}.{column.name}: L3: migrated as NUMERIC({row['numeric_precision']}, {row['numeric_scale']}), "
                f"declared NUMERIC({column.precision}, {column.scale})"
            )
        nullable = column.nullable and column.name not in schema.pk_columns
        if (row["is_nullable"] == "YES") != nullable:
            findings.append(
                f"{name}.{column.name}: L3: migrated {'NULL' if row['is_nullable'] == 'YES' else 'NOT NULL'}, "
                f"declared {'NULL' if nullable else 'NOT NULL'}"
            )
        if _has_default(row["column_default"]) != _has_default(column.server_default):
            findings.append(
                f"{name}.{column.name}: L3: migrated default {row['column_default']!r}, "
                f"declared server_default {column.server_default!r}"
            )
    if pk != schema.pk_columns:
        findings.append(f"{name}: L3: migrated primary key {pk!r}, declared {schema.pk_columns!r}")
    return findings


def _split_table(name: str) -> tuple[str | None, str]:
    """a table name's schema, when it names one (``platform_audit.audit_events``), and its table.

    :param name: the table's name as a ``TableSchema`` declares it
    :ptype name: str
    :return: ``(schema, table)``, ``schema`` ``None`` for the connection's own
    :rtype: tuple[str | None, str]
    """
    schema, _, table = name.rpartition(".")
    return (schema or None), table


def _qualified(name: str) -> str:
    """a table name as SQL, each part quoted: ``"platform_audit"."audit_events"``.

    :param name: the table's name as a ``TableSchema`` declares it
    :ptype name: str
    :return: the quoted name
    :rtype: str
    """
    schema, table = _split_table(name)
    return quote_identifier(table) if schema is None else f"{quote_identifier(schema)}.{quote_identifier(table)}"


def _fitted(value: Any, max_length: int | None) -> Any:
    """``value`` cut to a ``varchar(n)`` column's length, so the column can hold it.

    :param value: the sample value
    :ptype value: Any
    :param max_length: the column's ``character_maximum_length``, or ``None`` for no limit
    :ptype max_length: int | None
    :return: the value, a string no longer than the column holds
    :rtype: Any
    """
    return value[:max_length] if isinstance(value, str) and max_length is not None else value


async def _l3_row_findings(
    name: str, conn: Any, schema: TableSchema, max_lengths: Mapping[str, int | None]
) -> list[str]:
    """write each sample value to a scratch copy of the migrated table and compare what it reads back.

    The copy (``LIKE ... INCLUDING DEFAULTS``) has the table's columns, types and defaults and none
    of its foreign keys or checks; its ``NOT NULL`` constraints are dropped, so each column's value
    is written and read on a row of its own, each in its own savepoint, and a value the table refuses
    is that column's finding without stopping the rest. A string is cut to its column's
    ``varchar(n)``. The copy is a temporary table, dropped after.

    :param name: the collection's name
    :ptype name: str
    :param conn: the Postgres connection
    :ptype conn: Any
    :param schema: the table
    :ptype schema: TableSchema
    :param max_lengths: each column's ``character_maximum_length``
    :ptype max_lengths: Mapping[str, int | None]
    :return: the findings
    :rtype: list[str]
    """
    findings: list[str] = []
    scratch = quote_identifier(f"tier_parity_{uuid.uuid7().hex}")
    row = {
        column.name: _fitted(_sample(column), max_lengths.get(column.name))
        for column in schema.columns
        if column.column_type not in (TSVECTOR_TYPE, VECTOR_TYPE, ENUM_TYPE) and column.name in max_lengths
    }
    await conn.execute(f"CREATE TEMP TABLE {scratch} (LIKE {_qualified(schema.name)} INCLUDING DEFAULTS)")
    try:
        for column in row:
            await conn.execute(f"ALTER TABLE {scratch} ALTER COLUMN {quote_identifier(column)} DROP NOT NULL")
        for column, value in row.items():
            quoted = quote_identifier(column)
            try:
                async with conn.transaction():
                    got = await conn.fetchval(
                        f"INSERT INTO {scratch} ({quoted}) VALUES ($1) RETURNING {quoted}",  # noqa: S608
                        value,
                    )
            except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                findings.append(f"{name}.{column}: L3: the table refused {value!r}: {type(exc).__name__}: {exc}")
            else:
                findings.extend(_row_findings(name, "L3", {column: value}, {column: got}))
    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {scratch}")
    return findings


async def live_tier_parity_findings(
    conn: Any,
    collections: Iterable[type[BaseCollection[Any]]],
    *,
    not_owned: Mapping[type[BaseCollection[Any]], Iterable[str]] | None = None,
) -> list[str]:
    """every difference between a collection's schema and its migrated Postgres table, and what L3 hands back.

    For each class that declares a ``TableSchema``: the table of that name on ``conn``'s search
    path is read from ``information_schema`` and held to the schema -- every column, its type, its
    ``NUMERIC`` precision and scale, its nullability, whether it has a default -- and to its primary
    key; then the sample row is written to a scratch copy of the table and read back, so asyncpg's
    answer for each column is held to the value written. ``conn`` is set up as the host's pool sets
    a connection up (:func:`~threetears.core.collections.init_connection`), so JSON is bound and
    read as the collections do. A class with no schema is skipped. Run :func:`tier_parity_findings`
    too: this holds L3, that holds the L1s and L2.

    :param conn: an asyncpg connection (not a pool: the scratch table is the connection's own)
        holding the collections' tables
    :ptype conn: Any
    :param collections: the collection classes
    :ptype collections: Iterable[type[BaseCollection[Any]]]
    :param not_owned: per class, the migrated columns it deliberately neither reads nor writes --
        columns another writer owns, which the collection's generated reads never select, so no
        tier of it holds them. Each must exist in the table and be absent from the schema.
    :ptype not_owned: Mapping[type[BaseCollection[Any]], Iterable[str]] | None
    :return: one line per difference, naming the collection, the column and the tier; empty when
        the migrated tables are the schemas and L3 hands back what was written
    :rtype: list[str]
    """
    findings: list[str] = []
    for cls in collections:
        schema = _schema_of(cls)
        if schema is None:
            continue
        name = cls.__qualname__
        table_schema, table = _split_table(schema.name)
        live = [
            dict(row)
            for row in await conn.fetch(
                "SELECT column_name, data_type, numeric_precision, numeric_scale, is_nullable, column_default, "
                "character_maximum_length FROM information_schema.columns "
                "WHERE table_schema = COALESCE($2, current_schema()) AND table_name = $1",
                table,
                table_schema,
            )
        ]
        if not live:
            findings.append(f"{name}: L3: no table {schema.name!r} on the connection")
            continue
        pk_rows = await conn.fetch(
            "SELECT k.column_name FROM information_schema.table_constraints c "
            "JOIN information_schema.key_column_usage k ON k.constraint_name = c.constraint_name "
            "AND k.table_schema = c.table_schema AND k.table_name = c.table_name "
            "WHERE c.constraint_type = 'PRIMARY KEY' AND c.table_schema = COALESCE($2, current_schema()) "
            "AND c.table_name = $1 ORDER BY k.ordinal_position",
            table,
            table_schema,
        )
        pk = tuple(str(r["column_name"]) for r in pk_rows)
        others = frozenset((not_owned or {}).get(cls, ()))
        findings.extend(_l3_column_findings(name, schema, live, pk, others))
        max_lengths = {str(row["column_name"]): row["character_maximum_length"] for row in live}
        try:
            findings.extend(await _l3_row_findings(name, conn, schema, max_lengths))
        except asyncpg.PostgresError as exc:
            findings.append(f"{name}: L3: the scratch copy of {schema.name!r} failed: {type(exc).__name__}: {exc}")
    return findings
