"""schema-backed collection primitive.

collapses the CRUD boilerplate that every
:class:`~threetears.core.collections.base.BaseCollection` subclass used
to hand-roll (``fetch_from_store`` / ``save_to_store`` /
``delete_from_store`` / ``serialize`` / ``deserialize``) into one
schema-driven implementation. subclasses declare a
:class:`TableSchema` as a class attribute and stop writing SQL by hand
for the standard row lifecycle. domain-specific query methods
(``find_by_agent`` / ``hybrid_search`` / etc.) stay on subclasses
because query shape is legitimately per-collection; CRUD is not.

per CLAUDE.md "NO BACKWARDS-COMPATIBILITY SHIMS": a
subclass that adopts :class:`SchemaBackedCollection` deletes its
hand-rolled CRUD in the same commit. there is no dual-path
"override-wins" fallback -- schema is the declaration, the base class
is the implementation.

**pgvector:** ``VECTOR_TYPE`` is deliberately generic (not named
``EMBEDDING_TYPE``) because pgvector rows can carry non-embedding
vectors. write path emits the textual bracketed form plus ``::vector``
cast; read path coerces the textual form back to ``list[float]``.

**composite pk:** mirrors the
:class:`~threetears.core.collections.base.BaseCollection` composite-pk
contract added in namespace-task-01 phase 8.5l-1. declare
``TableSchema.primary_key = ("a", "b")`` and the generator emits
``WHERE a = $1 AND b = $2`` for fetch / delete, ``ON CONFLICT (a, b)``
for upsert, and ``WHERE a = $1 AND b = $2 AND date_updated = $N`` for
CAS.
"""

from __future__ import annotations

import re

import asyncio
import hashlib
import inspect
import json
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import wraps
from typing import Annotated, Any, ClassVar, Final, Generic, Literal, TypeVar, overload
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import BigInteger, Boolean, DateTime, Integer, Numeric, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, TSVECTOR
from sqlalchemy.engine import Dialect
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import TextClause, UnaryExpression

from threetears.interfaces.keyed.keys import (
    DeclaredIndex,
    TableKey,
    not_null_predicate,
    null_predicate,
    register_partial_predicates,
    register_table_indexes,
    register_table_keys,
    partial_predicate,
)
from threetears.interfaces.keyed.rows import DeclaredScope, KeyColumn, KeyColumns
from threetears.interfaces.keyed.filters import Fragment
from threetears.core.backends.arrow_copy import (
    arrow_python_columns,
    arrow_python_rows,
    with_constant_columns,
    with_typed_constants,
)
from threetears.core.loop_turns import STEP_ITEMS, LoopTurns, emptied, stepped
from threetears.core.backends.protocol import (
    ANALYZE_AFTER_ROWS,
    AnalyzingStore,
    ArrowWritingStore,
    BulkDeletingStore,
    ManyKeyReadingStore,
    ReturningDeletingStore,
    BulkDurableStore,
    CopyingStore,
    DurableStore,
    GoneMarkingStore,
    KeyLedReadingStore,
    LoadFile,
    LoadingStore,
    OrderedDurableStore,
    PrefixDeletingStore,
    TruncatingStore,
    copy_bounds,
    load_requests,
    side_by_side_requests,
)
from threetears.core.backends.schema_sql import (
    PGVECTOR_TYPES,
    build_select_column_list,
    bulk_update_columns,
    coerce_row as _coerce_row_fn,
    copy_writes,
    decode_l2_value as _decode_l2_value,
    decode_vector as _decode_vector,
    encode_jsonb as encode_jsonb,
    insert_columns_for_data,
    json_default as _json_default,
    upsert_set_columns,
)
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.l1_lru import L1_EVICTION_INFO, declare_l1_table
from threetears.core.eviction import Declared
from threetears.core.collections.base import BaseCollection, EntityT
from threetears.core.tenancy import TENANCY_UNDECLARED, Tenancy
from threetears.core.collections.generation import WRITE_GENERATION_UNDECLARED, WriteGenerationDeclaration
from threetears.core.collections.caller_transaction import CallerTransaction, save_delta_files
from threetears.core.backends.broker_generation import (
    collecting_advanced_generations,
    collecting_load_answers,
    committed_generations_record,
    named_delta_files,
    naming_delta_files,
    naming_unconfirmed_delta_files,
    reply_generations_since,
)
from threetears.core.collections.delta_files import (
    DeltaFileRefusedError,
    OWN_DELTA_SAVES,
    LoadDelta,
    TableDelta,
    check_delta_scope,
)
from threetears.core.collections.delta_keep import DeltaKeepRule
from threetears.core.collections.epoch_change import RESET_REASON, EpochChange, WholeReach, records_changes
from threetears.core.collections.flush import FlushStrategy
from threetears.core.collections.l2_order import L2_EPOCH_COLUMN, L2_ORDER_COLUMNS, L2_REVISION_COLUMN
from threetears.core.declared import whole_read
from threetears.core.ddl.row_epoch import (
    AppendOnlyWriteError,
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    GONE_COLUMN,
    RowEpoch,
    UnstampedWriteError,
    bucket_epoch_index_name,
    bucket_split_sql,
    bucket_sql,
    key_hash_sql,
    restore_declaration,
    scope_epoch_index_name,
    next_epoch,
    stamped_restore_declaration,
    versioned_primary_key_sql,
)
from threetears.core.keyset import keyset_pages
from threetears.core.sql_fragments import quote_identifier
from threetears.core.entities.base import BaseEntity
from threetears.core.serialization import resolve_base_type
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; rail_arrow_types refuses without it, saying so
    _PYARROW = False

__all__ = [
    "register_sa_table_indexes",
    "stamped_values",
    "rail_text",
    "sql_cast",
    "BIGINT_TYPE",
    "MAX_INDEX_COLUMNS",
    "BOOL_TYPE",
    "BYTES_TYPE",
    "Column",
    "DATETIMETZ_TYPE",
    "ENUM_TYPE",
    "encode_jsonb",
    "ForeignKey",
    "ForeignKeyDef",
    "INT_TYPE",
    "Index",
    "IndexDef",
    "JSONB_TYPE",
    "NUMERIC_TYPE",
    "OnConflict",
    "OnDelete",
    "PartitionEnforcementError",
    "SchemaBackedCollection",
    "STRING_TYPE",
    "TSVECTOR_TYPE",
    "TableSchema",
    "UUID_TYPE",
    "VECTOR_TYPE",
    "check_declared_row_types",
    "collection_for_schema",
    "l2_order_columns",
    "schema_datetime_columns",
    "schema_field_types",
    "rail_arrow_types",
    "spans_partitions",
]

# on_conflict enum values for :class:`TableSchema`.
#
# - ``"raise"``: emit plain INSERT with no ``ON CONFLICT`` clause; duplicate
#   primary-key inserts surface the asyncpg ``UniqueViolationError`` directly
#   to the caller. journal / append-only tables.
# - ``"ignore"``: emit ``INSERT ... ON CONFLICT (pk) DO NOTHING``; duplicate
#   primary-key inserts are silently dropped. dedup-on-redelivery tables
#   (audit envelopes, idempotent event logs).
# - ``"update"``: emit ``INSERT ... ON CONFLICT (pk) DO UPDATE SET <mutable>``
#   (default). standard upsert behaviour for editable rows.
OnConflict = Literal["raise", "ignore", "update"]

log = get_logger(__name__)

#: ``committed_before`` not given (:meth:`SchemaBackedCollection._drop_and_advance`): the change was named by no broker
#: reply (a reset, a failed load's settle, a key-prefix delete settled after its transaction)
_NO_REPLY: Final = object()

#: the restore read a versioned table's ``(scope, epoch)`` index serves, and the plan test proving it does
_RESTORE_READ: Final = "packages/core/src/threetears/core/collections/table_restore.py::_scope_versions"
_RESTORE_PLAN_TEST: Final = (
    "packages/core/tests/integration/test_table_restore_plans_live.py::"
    "test_a_restore_reads_a_scope_through_the_scope_epoch_index"
)

#: the restore read a stamped table's ``(bucket, epoch)`` index serves, and the plan test proving it does
_STAMPED_RESTORE_READ: Final = "packages/core/src/threetears/core/collections/table_restore.py::_bucket_rows"
_STAMPED_RESTORE_PLAN_TEST: Final = (
    "packages/core/tests/integration/test_table_restore_plans_live.py::"
    "test_a_restore_reads_a_stamped_table_a_bucket_at_a_time_on_its_own_tablet"
)


#: a page of more values than this (rows times the schema's columns) is read-coerced on a worker thread taking turns
#: with the loop (:func:`_coerced_page`), a smaller one on the loop. A refill's page at the production reply cap is
#: about 10,000 rows: of 49 columns, coerced in 55-59 ms and its rows as read freed in 3 ms more, in one stretch on
#: the loop (a 10-core Apple host, 2026-11-16). 100 rows of 49 columns (4,900 values) coerce in about half a
#: millisecond, under a thread's round trip (0.72 ms, the proxy's ``READ_ON_THREAD_BYTES``), so a page this small
#: stays on the loop
_COERCE_ON_THREAD_VALUES: Final = 4096


def _coerced_page(
    schema: TableSchema, page: list[dict[str, Any]], loop: asyncio.AbstractEventLoop | None
) -> list[dict[str, Any]]:
    """``page``'s rows each read-coerced (``coerce_row``), in order. On a worker thread (``loop`` named) the thread
    takes its turns with the loop between rows, then empties ``page`` a slice a turn (:func:`emptied`), so the rows as
    read are freed there and not where the page is next dropped; with no loop, the rows are coerced on the caller's
    thread and ``page`` is left as it was.

    The page is the reader's own (:func:`~threetears.core.keyset.keyset_pages` reads nothing of a page after handing it
    over), so emptying it frees what it alone holds.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param page: a page of rows as L3 answered them; emptied on a thread
    :ptype page: list[dict[str, Any]]
    :param loop: the caller's event loop, handed the thread's turns; None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the rows coerced
    :rtype: list[dict[str, Any]]
    """
    if loop is None:
        return [_coerce_row_fn(schema, row) for row in page]
    # a turn looked at every STEP_ITEMS values, not rows: a row of 49 values coerced in 5.5 us
    every = max(1, STEP_ITEMS // max(1, len(schema.columns)))
    rows = [_coerce_row_fn(schema, row) for row in stepped(page, LoopTurns(loop), every)]
    emptied([page], loop)
    return rows


def _load_requests_failed() -> Any:
    # fetched where it is moved, never held at import: a fresh import of the package would register it twice
    return counter(
        "threetears_load_requests_failed_total",
        description=(
            "load_files requests of a table that opts in to delta files that failed, by table and owner of their "
            "epoch record: 'hub' (the broker answered; it names the file unconfirmed when rows committed), 'hub_none' "
            "(it answered that none committed), 'process' (a direct pool: this process names the file unconfirmed), "
            "'unanswered' (no answer: a cancel or a lost reply; no record is written here)"
        ),
        label_names=("table", "owner"),
    )


def _deletes_settled_by_key() -> Any:
    # fetched where it is moved, never held at import: a fresh import of the package would register it twice
    return counter(
        "threetears_collection_deletes_settled_by_key_total",
        description=(
            "rows a bulk delete enrolled whose delete failed before returning them, so their invalidation names the "
            "key alone (a broader invalidation than their columns would give)"
        ),
        label_names=("table",),
    )


# sentinel strings for column type tags. chosen over real Python types
# because several logical types collapse to the same Python type
# (``list[float]`` vs ``list[dict]`` for vector vs jsonb arrays) and
# because the serialization layer already distinguishes types by tag
# rather than by runtime ``isinstance`` checks.
UUID_TYPE = "uuid"
STRING_TYPE = "string"
# ENUM_TYPE: column declared as a real Postgres enum via
# ``CREATE TYPE <name> AS ENUM (...)``. Verified via ``pg_type`` query
# against prod for ``memory_type``, ``conversation_status``,
# ``message_role``, ``user_role``, ``visibility_mode``,
# ``participation_role``, ``tool_invocation_status``. The
# :class:`Column` declaration carries ``enum_type`` (the allowed value
# tuple) and ``enum_name`` (the Postgres type name); the SQL generator
# / ``to_sqlalchemy_table`` shaper dispatch on this tag.
ENUM_TYPE = "enum"
# NUMERIC_TYPE: column declared as ``NUMERIC(precision, scale)`` --
# fixed-precision decimal. Required for ``media_content.cost`` which is
# ``Numeric(12, 8)`` in the v0.7.5 factory; no other existing tag covers
# this. The :class:`Column` declaration carries ``precision`` and
# ``scale``.
NUMERIC_TYPE = "numeric"
# TSVECTOR_TYPE: Postgres ``tsvector`` column (full-text search vector).
# Maintained server-side by a trigger; the DDL for the trigger stays in
# alembic on the consumer side. Declared with ``nullable=True,
# immutable=True`` so the Collection UPDATE generators exclude it from
# the ``SET`` clause -- otherwise an application UPDATE would overwrite
# the trigger-maintained value.
TSVECTOR_TYPE = "tsvector"
# DATETIMETZ_TYPE: column declared as ``TIMESTAMPTZ`` (timezone-aware
# instant) in the L3 DDL. This is the only datetime column type the
# platform supports; ``DATETIME_TYPE`` (TIMESTAMP / naive) was removed
# in collections-task-05 in favor of a single aware-UTC convention.
# asyncpg's TIMESTAMPTZ codec calls ``obj.astimezone(utc)`` on every
# value -- aware-UTC inputs round-trip cleanly; naive inputs would be
# silently shifted by the host's local-TZ offset, so the write
# coercion for this type ensures aware-UTC at the boundary regardless
# of input shape (naive inputs are wrapped with ``UTC``; aware inputs
# are normalized via ``astimezone(UTC)``) so the codec's
# ``obj.astimezone(utc)`` is a no-op and the wire value is the true
# UTC instant. read coercion ensures the returned value carries
# ``tzinfo=UTC`` so downstream callers (and CAS predicates that
# round-trip through this collection) see a stable shape.
DATETIMETZ_TYPE = "datetimetz"
JSONB_TYPE = "jsonb"
BYTES_TYPE = "bytes"
INT_TYPE = "int"
# BIGINT_TYPE: column declared as ``BIGINT`` (64-bit int8). INT_TYPE
# (int4) tops out at ~2.1e9, which overflows for byte-size columns of
# multi-GB artifacts (a pcap / db-dump ``size_bytes``); use this tag
# for any count that can exceed int4. Read/write coercion matches
# INT_TYPE (a plain Python ``int``); only the L3 DDL width differs.
BIGINT_TYPE = "bigint"
BOOL_TYPE = "bool"
VECTOR_TYPE = "vector"

#: the SQL type a column's value is cast to where a statement binds it in an array, by its type tag; ``text`` for
#: every tag not here
_SQL_CASTS: Final[Mapping[str, str]] = {
    UUID_TYPE: "uuid",
    INT_TYPE: "bigint",
    BIGINT_TYPE: "bigint",
    BOOL_TYPE: "boolean",
    NUMERIC_TYPE: "numeric",
    DATETIMETZ_TYPE: "timestamptz",
}

#: the kind of hash (:func:`~threetears.core.ddl.row_epoch.key_hash_sql`) a stamped table's key column of each type tag
#: feeds its restore bucket; a key column of any other type refuses the table
_BUCKET_HASH_KINDS: Final[Mapping[str, str]] = {
    STRING_TYPE: "text",
    UUID_TYPE: "uuid",
    INT_TYPE: "int4",
    BIGINT_TYPE: "int8",
    DATETIMETZ_TYPE: "timestamptz",
    NUMERIC_TYPE: "numeric",
    BOOL_TYPE: "bool",
    ENUM_TYPE: "enum",
    BYTES_TYPE: "bytea",
    JSONB_TYPE: "jsonb",
}


def stamped_values(schema: TableSchema, values: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any]:
    """``values`` for a keyed ``UPDATE`` of ``schema``'s table (``threetears.core.keyed.update_keyed``), stamped: a
    stamped table's (``ROW_STAMPED``) epoch set to ``GREATEST(epoch + 1, <a fresh stamp>)``, as its upsert sets it
    (``rising_epoch_assignment``), so the row a restore reads by its ``(bucket, epoch)`` index is never one an update
    left at its older stamp. The stamp is drawn from the caller's transaction when ``conn`` is in one
    (``CallerTransaction.in_place_epoch``, counted in its ``s_min``), else from the process clock (one statement, its
    own commit). Unchanged for a table that declares no row epoch.

    WHY a helper, not the keyed layer: ``update_keyed`` knows a table by its name alone, and only the schema says it is
    stamped; an update that left the epoch alone would succeed and be invisible to a restore.

    :param schema: the table's schema
    :ptype schema: TableSchema
    :param values: column -> its new value, as ``update_keyed`` takes them
    :ptype values: Mapping[str, Any]
    :param conn: the connection the update runs on, or None for the pool
    :ptype conn: Any
    :return: the values, the epoch's term added for a stamped table
    :rtype: dict[str, Any]
    :raises AppendOnlyWriteError: when the table is versioned (append-only: never updated in place)
    """
    declared = schema.row_epoch
    if declared is None:
        return dict(values)
    if declared.versions:
        raise AppendOnlyWriteError(f"stamped_values: {schema.name} is versioned (append-only), never updated in place")
    stamp = CallerTransaction.stamp_for(conn, schema.name)
    quoted = quote_identifier(EPOCH_COLUMN)
    return {**values, EPOCH_COLUMN: Fragment(f"GREATEST({quoted} + 1, $1::bigint)", (stamp,))}


def sql_cast(column_type: str) -> str:
    """the SQL type a value of a column of ``column_type`` is cast to where a statement binds it in an array.

    :param column_type: the column's type tag
    :ptype column_type: str
    :return: the SQL type (``uuid``, ``bigint``, ``timestamptz``; ``text`` for a string and every other tag)
    :rtype: str
    """
    return _SQL_CASTS.get(column_type, "text")


# the rail's Arrow types live here, beside the type tags, rather than in complete_copy (which re-exports them): a
# collection names its delta files' column types from them (SchemaBackedCollection.delta_file_types), and
# complete_copy imports this module, so it cannot be imported from here

#: the L3 column types the rail's JSON answers as text: ``fetch`` answers them as text (a ``NUMERIC`` arrives as
#: its digits, a timestamp outside ``date_*`` as its ISO text)
_RAIL_TEXT_TYPES: Final = frozenset({STRING_TYPE, UUID_TYPE, ENUM_TYPE, TSVECTOR_TYPE, NUMERIC_TYPE, DATETIMETZ_TYPE})


def rail_text(column_type: str) -> bool:
    """whether the L3 rail answers a column of ``column_type`` as text (:data:`_RAIL_TEXT_TYPES`): what a copy read
    off the rail holds it as, a timestamp outside ``date_*`` as its ISO text, a ``NUMERIC`` as its digits.

    :param column_type: the column's type tag
    :ptype column_type: str
    :return: True when the rail answers it as text
    :rtype: bool
    """
    return column_type in _RAIL_TEXT_TYPES


#: the prefix of the columns the rail's proxy answers as aware datetimes (``nats_proxy.deserialize_row``)
_DATED_PREFIX: Final = "date_"


def rail_arrow_types(columns: Sequence[Any]) -> dict[str, Any]:
    """each column's Arrow type as the L3 rail's reply carries it: what :meth:`NatsProxyL3Backend.fetch_arrow
    <threetears.core.backends.nats_proxy.NatsProxyL3Backend.fetch_arrow>` reads the rows as, so they equal ``fetch``'s.

    Text, a UUID, an enum, a ``NUMERIC`` and a timestamp are text (as ``fetch`` answers them); an integer is
    ``int64``; a boolean is ``bool``; a ``date_*`` timestamp is an instant in UTC (``fetch`` makes those aware
    datetimes). A column ``fetch`` answers otherwise than the reply's JSON (``jsonb``, bytes, a vector), or a
    ``date_*`` column that is not a timestamp (``fetch`` parses any ISO text there), has no such type: read it
    with ``fetch``.

    :param columns: the columns, each with ``name`` and ``column_type`` (a
        :class:`~threetears.core.collections.schema_backed.Column`), in the order they are selected
    :ptype columns: Sequence[Column]
    :return: column name -> its Arrow type, in order
    :rtype: dict[str, pyarrow.DataType]
    :raises ValueError: for a column the rail does not answer as one Arrow type exactly
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    types: dict[str, Any] = {}
    for column in columns:
        name, kind = column.name, column.column_type
        dated = name.startswith(_DATED_PREFIX)
        if dated and kind != DATETIMETZ_TYPE:
            raise ValueError(
                f"column {name!r}: fetch reads ISO text in a date_ column as a datetime; read it with fetch"
            )
        if dated:
            types[name] = pa.timestamp("us", tz="UTC")
        elif kind in _RAIL_TEXT_TYPES:
            types[name] = pa.string()
        elif kind in (INT_TYPE, BIGINT_TYPE):
            types[name] = pa.int64()
        elif kind == BOOL_TYPE:
            types[name] = pa.bool_()
        else:
            raise ValueError(
                f"column {name!r} ({kind}): the rail answers it otherwise than one Arrow type; read it with fetch"
            )
    return types


# what a delta file may hold of a column's database default (design-pod-delta-files.md, decision 20): the file is
# written before the commit, so it holds only what 3tears knows without the database's answer. A plain literal of the
# column's own type is a fixed value, written into the file's row as the rail answers it; anything else -- a function
# (now(), gen_random_uuid()), a sequence (nextval), an expression, a literal 3tears does not read exactly -- is the
# database's to decide, and refused where the table opts in rather than guessed

#: a quoted SQL literal, ``''`` its escaped quote
_QUOTED_LITERAL: Final = re.compile(r"'((?:[^']|'')*)'")
#: an unquoted SQL literal: a number, a boolean or NULL
_BARE_LITERAL: Final = re.compile(r"[+-]?\d+(?:\.\d+)?|true|false|null", re.IGNORECASE)
#: a trailing cast to a plain type name (``::text``, ``::character varying``)
_CAST: Final = re.compile(r"\s*::\s*([a-z_][a-z0-9_]*(?:\s+[a-z_][a-z0-9_]*)*)\s*", re.IGNORECASE)
#: the type names a cast may name for each column type and leave the literal's value as it is; a cast to anything
#: else (``varchar(2)`` cuts the text) is the database's to answer
_SAME_TYPE_CASTS: Final[dict[str, frozenset[str]]] = {
    STRING_TYPE: frozenset({"text", "character varying", "varchar"}),
    UUID_TYPE: frozenset({"uuid"}),
    INT_TYPE: frozenset({"integer", "int", "int4", "bigint", "int8"}),
    BIGINT_TYPE: frozenset({"integer", "int", "int4", "bigint", "int8"}),
    BOOL_TYPE: frozenset({"boolean", "bool"}),
    NUMERIC_TYPE: frozenset({"numeric"}),
}


class _NotFixed(ValueError):
    """a column's database default is not a value a delta file can hold before the commit."""


def _fixed_default(column: Column) -> Any:
    """the value ``column``'s ``server_default`` gives a new row, as the rail answers it: a plain literal of the
    column's own type (``'active'``, ``0``, ``false``, ``NULL``), optionally cast to that type.

    :param column: a column declaring a ``server_default``
    :ptype column: Column
    :return: the value: text for a text, UUID, enum or ``NUMERIC`` column (a ``NUMERIC`` at its declared scale, as the
        database stores it), an int, a bool, or None
    :rtype: Any
    :raises _NotFixed: when the default is not such a literal: the database computes it, or 3tears cannot read it as
        the database would
    """
    expression = (column.server_default or "").strip()
    quoted = _QUOTED_LITERAL.match(expression)
    bare = None if quoted else _BARE_LITERAL.match(expression)
    literal = quoted or bare
    if literal is None:
        raise _NotFixed("the database computes it")
    rest = expression[literal.end() :]
    cast = _CAST.fullmatch(rest) if rest else None
    if rest and cast is None:
        raise _NotFixed("the database computes it")
    if bare is not None and literal.group(0).lower() == "null":
        return None
    allowed = _SAME_TYPE_CASTS.get(column.column_type, frozenset())
    if column.column_type == ENUM_TYPE and column.enum_name:
        allowed = frozenset({column.enum_name.lower()})
    if cast is not None and " ".join(cast.group(1).lower().split()) not in allowed:
        raise _NotFixed(f"its cast to {cast.group(1)!r} is the database's to answer for a {column.column_type} column")
    text = quoted.group(1).replace("''", "'") if quoted else literal.group(0)
    kind = column.column_type
    try:
        if kind == STRING_TYPE and quoted:
            return text
        if kind == ENUM_TYPE and quoted and text in (column.enum_type or ()):
            return text
        if kind == UUID_TYPE and quoted:
            return str(UUID(text))
        if kind in (INT_TYPE, BIGINT_TYPE) and re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if kind == BOOL_TYPE and bare is not None and text.lower() in ("true", "false"):
            return text.lower() == "true"
        if kind == NUMERIC_TYPE and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text) and column.scale is not None:
            # the database stores a NUMERIC at its declared scale, and the rail answers its digits so ("0.50")
            return str(Decimal(text).quantize(Decimal(1).scaleb(-column.scale), rounding=ROUND_HALF_UP))
    except (ValueError, InvalidOperation) as exc:
        raise _NotFixed(f"3tears cannot read it as a {kind} value: {exc}") from exc
    raise _NotFixed(f"3tears does not read it as a fixed {kind} value exactly")


#: the most columns one index may hold on YugabyteDB, its ``INCLUDE`` columns and a primary key's columns counted:
#: 2026.1 refuses more ("cannot use more than 32 columns in an index", measured on 2026.1.0.0-b118). A schema naming
#: more could never be created there, so :class:`IndexDef` and :class:`TableSchema` refuse it when declared
MAX_INDEX_COLUMNS: Final = 32


# allowed values for :attr:`ForeignKeyDef.on_delete`. matches the
# Postgres referential-action clause used in production migrations; the
# rare ``on_update`` clause is not modeled (v0.8.0 locked decision).
OnDelete = Literal["CASCADE", "SET NULL", "RESTRICT", "NO ACTION"]

_ON_DELETE_VALUES: frozenset[str] = frozenset(
    {"CASCADE", "SET NULL", "RESTRICT", "NO ACTION"},
)


@dataclass(frozen=True)
class ForeignKeyDef:
    """foreign-key descriptor for :class:`TableSchema`.

    represents both single-column FKs (``local_cols`` and ``ref_cols``
    each a 1-tuple) and composite FKs (same-length N-tuples). single-
    column FKs may alternatively be declared via
    :attr:`Column.foreign_key` so simple cases stay terse; the composite
    case requires this dataclass on the :class:`TableSchema` because a
    single :class:`Column` declaration cannot reference multiple local
    columns.

    :cvar local_cols: tuple of column names on this table participating
        in the FK, in declared order
    :cvar ref_table: name of the referenced table
    :cvar ref_cols: tuple of column names on the referenced table, in
        the same order as :attr:`local_cols`
    :cvar on_delete: referential-action clause; defaults to
        ``"NO ACTION"`` (Postgres default semantics)
    """

    local_cols: tuple[str, ...]
    ref_table: str
    ref_cols: tuple[str, ...]
    on_delete: OnDelete = "NO ACTION"

    def __post_init__(self) -> None:
        """validate the FK shape after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``local_cols`` / ``ref_cols`` are
            empty, mismatched in length, when ``ref_table`` is empty,
            or when ``on_delete`` is not one of the allowed literals
        """
        if not self.local_cols:
            raise ValueError(
                "ForeignKeyDef: local_cols must be a non-empty tuple",
            )
        if not self.ref_cols:
            raise ValueError(
                "ForeignKeyDef: ref_cols must be a non-empty tuple",
            )
        if len(self.local_cols) != len(self.ref_cols):
            raise ValueError(
                f"ForeignKeyDef: local_cols (len={len(self.local_cols)}) and "
                f"ref_cols (len={len(self.ref_cols)}) must have the same length",
            )
        if not self.ref_table:
            raise ValueError(
                "ForeignKeyDef: ref_table must be non-empty",
            )
        if self.on_delete not in _ON_DELETE_VALUES:
            raise ValueError(
                f"ForeignKeyDef: on_delete must be one of {sorted(_ON_DELETE_VALUES)!r}; got {self.on_delete!r}",
            )


@dataclass(frozen=True)
class IndexDef:
    """index descriptor for :class:`TableSchema`.

    represents a named SQL index. supports plain composite indexes,
    UNIQUE indexes, and partial indexes via the ``where`` clause (a
    raw Postgres boolean expression -- the generator inlines it
    verbatim, no parameter binding).

    v0.8.1 added ``using`` / ``ops`` / ``pg_with`` so non-btree access
    methods (HNSW, GIN) and their per-column operator-class /
    parameter knobs can be modelled. These map 1:1 onto SQLAlchemy's
    ``postgresql_using`` / ``postgresql_ops`` / ``postgresql_with``
    dialect kwargs at :meth:`TableSchema.to_sqlalchemy_table` emission
    time. The ``ops`` and ``pg_with`` payloads are stored as tuples of
    ``(key, value)`` pairs (not ``dict``) because the dataclass is
    ``frozen=True`` and ``dict`` is unhashable; the
    :class:`Index` factory accepts the natural ``dict`` shape and
    coerces to tuples here.

    :cvar name: index name (matches the L3 DDL exactly; required, no
        auto-generation per v0.8.0 locked decision)
    :cvar columns: tuple of column names on this table covered by the
        index, in declared order
    :cvar unique: when ``True``, emit ``CREATE UNIQUE INDEX``
    :cvar where: optional partial-index predicate (raw SQL expression);
        when set, emit ``WHERE <where>`` after the column list
    :cvar using: optional access-method name (e.g. ``"hnsw"``, ``"gin"``,
        ``"btree"``). when ``None``, the default Postgres btree access
        method is used and SQLAlchemy emits no ``USING <method>``
        clause. v0.8.1+
    :cvar ops: optional per-column operator-class mapping, as a tuple
        of ``(column_name, opclass_name)`` pairs (e.g.
        ``(("embedding", "vector_cosine_ops"),)``). v0.8.1+
    :cvar pg_with: optional access-method parameter mapping, as a
        tuple of ``(parameter_name, value_str)`` pairs (e.g.
        ``(("m", "16"), ("ef_construction", "64"))``). values are
        strings to mirror the textual ``WITH (key = value, ...)`` DDL
        syntax verbatim. v0.8.1+
    :cvar desc: names of the covered columns that sort descending
        (``date_created DESC`` in the DDL); every other column sorts
        ascending. each name must also be in ``columns``. emitted as
        ``column.desc()`` so the declared index equals a migration
        that creates the column ``DESC``. a column cannot carry both
        ``desc`` and an operator class in ``ops``.
    :cvar asc: names of the covered columns declared ascending in so many words (``expires_at ASC``). Postgres
        sorts them the same either way; YugabyteDB range-shards an index whose first column is declared ``ASC`` (or
        ``DESC``) and hashes it otherwise, and a hashed first column seeks no range and answers no order: a read of a
        range (``expires_at < $1``) or in order (``ORDER BY expires_at LIMIT``) with nothing bound before that column
        reads every row. Such an index names its first column here. Each name must also be in ``columns``, and in
        neither ``desc`` nor ``ops``.
    :cvar collate: per-column collations, as ``(column_name, collation)`` pairs (``(("signal", "C"),)``): the
        column is indexed ``COLLATE "C"``, so a range by byte order (a prefix read, ``signal >= $1 AND signal < $2
        COLLATE "C"``) seeks it whatever the database's own collation; with ``asc`` it is range-sharded on
        YugabyteDB. Each name must be in ``columns`` and carry no operator class; each collation a plain name.
    :cvar nulls_not_distinct: when ``True`` (a ``unique`` index only), two
        rows whose covered columns are both NULL collide, as
        ``NULLS NOT DISTINCT`` says (Postgres 15+): a key with an optional
        column still holds one row per key while that column is unset.
    :cvar include: columns the index carries without ordering by them
        (``INCLUDE (...)`` in the DDL), so a read of them is answered from
        the index alone. On YugabyteDB a read through an index that does
        not carry a column visits the table for each row, several times the
        index read's cost. none may also be in ``columns``.
    :cvar purpose: ``<path>::<qualified function>`` of the keyed read
        that uses this index, or a tuple of them when it serves several; the no-whole-reads rule
        (:mod:`threetears.enforcement.no_whole_reads`) counts a read through
        it as keyed only in that function, and refuses an index with none.
        Not part of the DDL, so it never makes two indexes differ.
    :cvar plan_test: ``<test file>::<test function>`` proving that read's
        plan uses this index (it names the index and calls
        :func:`threetears.core.testing.plans.plan_uses_index`)
    :cvar reason: why the index carries ``include`` columns, or why its
        table carries more than three indexes; the rule asks for one in
        either case
    """

    name: str
    columns: tuple[str, ...]
    unique: bool = False
    where: str | None = None
    # ---new in v0.8.1:---
    using: str | None = None
    ops: tuple[tuple[str, str], ...] | None = None
    pg_with: tuple[tuple[str, str], ...] | None = None
    desc: tuple[str, ...] = ()
    nulls_not_distinct: bool = False
    include: tuple[str, ...] = ()
    asc: tuple[str, ...] = ()
    collate: tuple[tuple[str, str], ...] = ()
    purpose: str | tuple[str, ...] = field(default="", compare=False)
    plan_test: str = field(default="", compare=False)
    reason: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """validate the index shape after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty or ``columns`` is empty, or
            ``nulls_not_distinct`` is set on an index that is not unique
        """
        if not self.name:
            raise ValueError("IndexDef: name must be non-empty")
        if not self.columns:
            raise ValueError(
                f"IndexDef(name={self.name!r}): columns must be a non-empty tuple",
            )
        if self.nulls_not_distinct and not self.unique:
            raise ValueError(
                f"IndexDef(name={self.name!r}): nulls_not_distinct says how a unique index compares NULLs; "
                "this index is not unique",
            )
        stray = [col for col in self.desc if col not in self.columns]
        if stray:
            raise ValueError(
                f"IndexDef(name={self.name!r}): desc names {stray!r} which are not in columns {self.columns!r}",
            )
        if len(self.columns) + len(self.include) > MAX_INDEX_COLUMNS:
            raise ValueError(
                f"IndexDef(name={self.name!r}): {len(self.columns)} columns and {len(self.include)} included are "
                f"{len(self.columns) + len(self.include)}; YugabyteDB indexes at most {MAX_INDEX_COLUMNS}",
            )
        twice = [col for col in self.include if col in self.columns]
        if twice:
            raise ValueError(
                f"IndexDef(name={self.name!r}): include names {twice!r}, which are already its columns",
            )
        loose = [col for col in self.asc if col not in self.columns or col in self.desc or col in dict(self.ops or ())]
        if loose:
            raise ValueError(
                f"IndexDef(name={self.name!r}): asc names {loose!r}, each of which must be one of its columns "
                f"{self.columns!r} and neither desc nor carry an operator class",
            )
        both = [col for col in self.desc if col in dict(self.ops or ())]
        if both:
            raise ValueError(
                f"IndexDef(name={self.name!r}): {both!r} carry both an operator class and desc, which "
                "to_sqlalchemy_table cannot emit (SQLAlchemy drops the operator class of a descending column)",
            )
        collated = [
            col
            for col, collation in self.collate
            if col not in self.columns or col in dict(self.ops or ()) or not re.fullmatch(r"\w+", collation)
        ]
        if collated:
            raise ValueError(
                f"IndexDef(name={self.name!r}): collate names {collated!r}, each of which must be one of its columns "
                f"{self.columns!r}, carry no operator class, and name a plain collation",
            )


def ForeignKey(  # noqa: N802 -- factory function intentionally named like a class
    local_cols: tuple[str, ...] | str,
    ref_table: str,
    ref_cols: tuple[str, ...] | str,
    *,
    on_delete: OnDelete = "NO ACTION",
) -> ForeignKeyDef:
    """factory for :class:`ForeignKeyDef`.

    coerces bare-string ``local_cols`` / ``ref_cols`` arguments into
    1-tuples so the common single-column case stays terse::

        ForeignKey("user_id", "users", "user_id")
        # equivalent to:
        ForeignKey(("user_id",), "users", ("user_id",))

    composite FKs require tuples directly::

        ForeignKey(
            ("agent_id", "memory_id"),
            "memories",
            ("agent_id", "memory_id"),
            on_delete="CASCADE",
        )

    :param local_cols: column name(s) on this table; bare ``str``
        coerced to a 1-tuple
    :ptype local_cols: tuple[str, ...] | str
    :param ref_table: referenced table name
    :ptype ref_table: str
    :param ref_cols: column name(s) on the referenced table; bare
        ``str`` coerced to a 1-tuple
    :ptype ref_cols: tuple[str, ...] | str
    :param on_delete: referential-action clause; defaults to
        ``"NO ACTION"``
    :ptype on_delete: OnDelete
    :return: validated :class:`ForeignKeyDef` instance
    :rtype: ForeignKeyDef
    :raises ValueError: when validation in
        :meth:`ForeignKeyDef.__post_init__` fails
    """
    local_tup: tuple[str, ...] = (local_cols,) if isinstance(local_cols, str) else tuple(local_cols)
    ref_tup: tuple[str, ...] = (ref_cols,) if isinstance(ref_cols, str) else tuple(ref_cols)
    return ForeignKeyDef(
        local_cols=local_tup,
        ref_table=ref_table,
        ref_cols=ref_tup,
        on_delete=on_delete,
    )


def Index(  # noqa: N802 -- factory function intentionally named like a class
    name: str,
    *columns: str,
    unique: bool = False,
    where: str | None = None,
    using: str | None = None,
    ops: Mapping[str, str] | None = None,
    pg_with: Mapping[str, str] | None = None,
    desc: str | Sequence[str] = (),
    nulls_not_distinct: bool = False,
    include: Sequence[str] = (),
    purpose: str | Sequence[str] = "",
    plan_test: str = "",
    reason: str = "",
    asc: str | Sequence[str] = (),
    collate: Mapping[str, str] | Sequence[tuple[str, str]] = (),
) -> IndexDef:
    """factory for :class:`IndexDef`.

    varargs ``columns`` reads cleanly at call sites::

        Index("ix_memories_user_date", "user_id", "date_created")
        Index(
            "ix_memories_user_alias",
            "agent_id", "user_id", "alias",
            unique=True,
            where="alias IS NOT NULL",
        )
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            using="hnsw",
            ops={"embedding": "vector_cosine_ops"},
            pg_with={"m": "16", "ef_construction": "64"},
        )
        Index("idx_memories_tags", "tags", using="gin")
        Index("idx_history", "agent_id", "date_created", desc="date_created")
        Index("t_by_state", "state", "geo", *primary_key, include=("votes", "pad"))
        Index("t_by_expiry", "expires_at", asc="expires_at")  # a range read: range-sharded on YugabyteDB

    The ``ops`` and ``pg_with`` arguments accept any
    :class:`collections.abc.Mapping` (typically ``dict``) and are
    converted to tuples for storage on the frozen :class:`IndexDef`.

    :param name: index name (required; no auto-generation)
    :ptype name: str
    :param columns: column name(s) covered by the index, in declared
        order
    :ptype columns: str
    :param unique: emit ``CREATE UNIQUE INDEX`` when ``True``
    :ptype unique: bool
    :param where: optional partial-index predicate (raw Postgres SQL)
    :ptype where: str | None
    :param using: optional Postgres access-method name (e.g.
        ``"hnsw"``, ``"gin"``, ``"btree"``); maps to SQLAlchemy's
        ``postgresql_using`` dialect kwarg
    :ptype using: str | None
    :param ops: optional per-column operator-class mapping (e.g.
        ``{"embedding": "vector_cosine_ops"}``); maps to SQLAlchemy's
        ``postgresql_ops`` dialect kwarg
    :ptype ops: Mapping[str, str] | None
    :param pg_with: optional access-method parameter mapping (e.g.
        ``{"m": "16", "ef_construction": "64"}``); maps to
        SQLAlchemy's ``postgresql_with`` dialect kwarg. values are
        strings to mirror the textual ``WITH (key = value, ...)`` DDL
        syntax verbatim
    :ptype pg_with: Mapping[str, str] | None
    :param desc: the covered column(s) that sort descending, as the
        migration's ``<column> DESC`` says; a bare ``str`` names one
    :ptype desc: str | Sequence[str]
    :param nulls_not_distinct: a unique index whose NULLs collide
        (``NULLS NOT DISTINCT``); maps to SQLAlchemy's
        ``postgresql_nulls_not_distinct`` dialect kwarg
    :ptype nulls_not_distinct: bool
    :param include: columns carried without ordering (``INCLUDE (...)``),
        so a read of them never visits the table; maps to SQLAlchemy's
        ``postgresql_include`` dialect kwarg
    :ptype include: Sequence[str]
    :param purpose: ``<path>::<qualified function>`` of the keyed read
        that uses the index, or several (:attr:`IndexDef.purpose`)
    :ptype purpose: str | Sequence[str]
    :param plan_test: ``<test file>::<test function>`` proving that read
        uses it (:attr:`IndexDef.plan_test`)
    :ptype plan_test: str
    :param reason: why it is wide, or why its table carries more than
        three indexes (:attr:`IndexDef.reason`)
    :ptype reason: str
    :param asc: the covered column(s) declared ``ASC`` in so many words, so YugabyteDB range-shards an index led by
        one (:attr:`IndexDef.asc`); a bare ``str`` names one
    :ptype asc: str | Sequence[str]
    :param collate: per-column collations (``{"signal": "C"}``, :attr:`IndexDef.collate`)
    :ptype collate: Mapping[str, str] | Sequence[tuple[str, str]]
    :return: validated :class:`IndexDef` instance
    :rtype: IndexDef
    :raises ValueError: when validation in
        :meth:`IndexDef.__post_init__` fails
    """
    ops_tuple: tuple[tuple[str, str], ...] | None
    if ops is None:
        ops_tuple = None
    else:
        ops_tuple = tuple((k, v) for k, v in ops.items())
    pg_with_tuple: tuple[tuple[str, str], ...] | None
    if pg_with is None:
        pg_with_tuple = None
    else:
        pg_with_tuple = tuple((k, v) for k, v in pg_with.items())
    return IndexDef(
        name=name,
        columns=tuple(columns),
        unique=unique,
        where=where,
        using=using,
        ops=ops_tuple,
        pg_with=pg_with_tuple,
        desc=(desc,) if isinstance(desc, str) else tuple(desc),
        nulls_not_distinct=nulls_not_distinct,
        include=tuple(include),
        asc=(asc,) if isinstance(asc, str) else tuple(asc),
        collate=tuple(collate.items()) if isinstance(collate, Mapping) else tuple(tuple(pair) for pair in collate),  # type: ignore[misc]
        purpose=purpose if isinstance(purpose, str) else tuple(purpose),
        plan_test=plan_test,
        reason=reason,
    )


@dataclass(frozen=True)
class UniqueConstraintDef:
    """unique-constraint descriptor for :class:`TableSchema`.

    Represents a named ``UNIQUE`` table constraint declared via
    ``ALTER TABLE ... ADD CONSTRAINT <name> UNIQUE (<cols>)``. This is
    distinct from a UNIQUE index declared via
    :class:`IndexDef` with ``unique=True``: both shapes produce
    identical rows in ``pg_indexes`` so they are byte-for-byte
    indistinguishable at the storage level, BUT Alembic auto-gen reads
    ``information_schema.table_constraints`` and surfaces a
    ``drop_constraint`` + ``create_unique_constraint`` diff when a
    target metadata's ``UniqueConstraint`` does not match a prod
    deployment's ``CONSTRAINT ... UNIQUE``.

    Use :class:`UniqueConstraintDef` when the prod DDL is
    ``ADD CONSTRAINT <name> UNIQUE (...)``. Use
    :class:`IndexDef` with ``unique=True`` only when prod uses
    ``CREATE UNIQUE INDEX``.

    :cvar name: constraint name (matches the L3 DDL exactly; required,
        no auto-generation per the v0.8.0 locked decision)
    :cvar columns: tuple of column names covered by the constraint,
        in declared order
    """

    name: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """validate the constraint shape after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty or ``columns`` is empty
        """
        if not self.name:
            raise ValueError("UniqueConstraintDef: name must be non-empty")
        if not self.columns:
            raise ValueError(
                f"UniqueConstraintDef(name={self.name!r}): columns must be a non-empty tuple",
            )


def UniqueConstraint(  # noqa: N802 -- factory function intentionally named like a class
    name: str,
    *columns: str,
) -> UniqueConstraintDef:
    """factory for :class:`UniqueConstraintDef`.

    Varargs ``columns`` reads cleanly at call sites::

        UniqueConstraint("uq_memories_memory_id", "memory_id")
        UniqueConstraint(
            "uq_memories_agent_memory",
            "agent_id", "memory_id",
        )

    :param name: constraint name (required; no auto-generation)
    :ptype name: str
    :param columns: column name(s) covered by the constraint, in
        declared order
    :ptype columns: str
    :return: validated :class:`UniqueConstraintDef` instance
    :rtype: UniqueConstraintDef
    :raises ValueError: when validation in
        :meth:`UniqueConstraintDef.__post_init__` fails
    """
    return UniqueConstraintDef(name=name, columns=tuple(columns))


@dataclass(frozen=True)
class Column:
    """column descriptor for :class:`TableSchema`.

    :cvar name: column name (matches the L3 DDL and the row-dict key)
    :cvar column_type: one of the module-level type tags
        (``UUID_TYPE``, ``STRING_TYPE``, etc.); used by
        :func:`threetears.core.backends.schema_sql.normalize_write_value` /
        :func:`threetears.core.backends.schema_sql.normalize_read_value` to pick
        the right coercion rule and by the SQL generator to decide
        whether a parameter needs a ``::jsonb`` / ``::vector`` cast
    :cvar immutable: when ``True``, column is emitted on INSERT but
        excluded from ``DO UPDATE SET`` and from the CAS UPDATE path.
        primary-key columns are implicitly immutable (enforced by
        :class:`TableSchema`). typical uses: ``date_created``,
        ``agent_id``, ``customer_id``, ``conversation_id``
    :cvar nullable: when ``True``, missing values in the write dict
        default to ``None``; when ``False``, a missing key raises
        :class:`KeyError` at write time so an incomplete payload
        fails loudly instead of silently writing ``NULL``
    :cvar partition: when ``True``, column is the partition key for
        the table. partition columns must be part of
        :attr:`TableSchema.primary_key` so the schema enforces row
        uniqueness through the partition. partition columns are also
        :attr:`immutable` by definition (the value identifies WHICH
        partition the row belongs to and changing it would corrupt
        the partition map); the :class:`TableSchema` validator coerces
        ``immutable=True`` automatically when ``partition=True`` to
        keep declarations terse. on a Collection mixed with the
        :class:`SchemaBackedCollection` partition guard, every
        public read / write method must either accept the partition
        column as a required argument, be decorated
        ``@spans_partitions`` to opt into multi-partition reads, or
        be allowlisted via ``partition_exempt_methods`` with an
        explicit rationale -- protects against the cross-partition
        bleed class of bug.

        **A partition column whose value is DERIVED from another column
        can move, and no generated statement can move it.** ``namespaces
        .row_scope`` is the live example -- it is ``platform`` exactly
        when ``customer_id IS NULL`` -- so a row that gains a customer
        changes its own primary key. :func:`~threetears.core.backends
        .schema_sql.build_insert_sql` nominates the PK as its ``ON
        CONFLICT`` arbiter, which then does not name the stored row,
        while any SECONDARY unique index on the bare entity id still
        does: the write becomes an unretryable ``UniqueViolationError``
        rather than a conflict the upsert resolves. Such a move has to
        be an explicit, deliberate method on the owning Collection
        (:meth:`~threetears.agent.acl.NamespaceCollection.rescope`) --
        it is deliberately NOT handled here, because a generic
        "reconcile a conflicting id into whatever partition the writer
        asked for" is precisely the cross-partition bleed the composite
        key exists to prevent.
    :cvar nullable: when ``True``, missing values default to ``None``;
        ignored when ``partition=True`` (partition columns must be
        non-null by construction)

    :param name: column name
    :ptype name: str
    :param column_type: one of the module-level type tags
    :ptype column_type: str
    :param immutable: INSERT-only flag; defaults to ``False``
    :ptype immutable: bool
    :param nullable: defaultable flag; defaults to ``False``
    :ptype nullable: bool
    :param partition: partition-column flag; defaults to ``False``
    :ptype partition: bool
    """

    name: str
    column_type: str
    immutable: bool = False
    nullable: bool = False
    partition: bool = False
    # ---new in v0.8.0:---
    # single-column FK shape; composite FKs go on
    # :attr:`TableSchema.foreign_keys`. tuple shape: ``(ref_table, ref_col)``.
    foreign_key: tuple[str, str] | None = None
    # enum CHECK constraint values; only valid with ``column_type == ENUM_TYPE``
    enum_type: tuple[str, ...] | None = None
    # Postgres enum-type name (the ``CREATE TYPE <name> AS ENUM`` identifier);
    # required when ``enum_type`` is set
    enum_name: str | None = None
    # pgvector dimension; only valid with ``column_type == VECTOR_TYPE``
    vector_dim: int | None = None
    # raw Postgres expression for ``DEFAULT <expr>`` in the DDL; passed
    # through verbatim, no parameter binding
    server_default: str | None = None
    # NUMERIC(precision, scale); both required together with NUMERIC_TYPE
    precision: int | None = None
    scale: int | None = None

    def __post_init__(self) -> None:
        """validate cross-field constraints introduced in v0.8.0.

        Numbering matches the v0.8.0-task-01 shard spec
        §"Validation in `__post_init__`" items 1-10. Spec items 1
        (foreign_key has no cross-field restriction on column_type)
        and 7 (server_default has no cross-field restriction) are
        deliberate non-restrictions, no validator needed; the
        numbered comments below preserve the spec's enumeration so
        future maintainers can map code↔spec at a glance.

        All validators are wrapped in :class:`ValueError` so callers
        get a single, consistent exception class for declaration-time
        bugs.

        :return: nothing
        :rtype: None
        :raises ValueError: when any of the cross-field constraints
            are violated (see body for the full set of rules)
        """
        # 1. foreign_key × column_type: no restriction (foreign_key
        #    may be set on any column_type). No validator.
        # 2. enum_type only valid with ENUM_TYPE; requires enum_name;
        #    requires non-empty tuple.
        if self.enum_type is not None:
            if self.column_type != ENUM_TYPE:
                raise ValueError(
                    f"Column(name={self.name!r}): enum_type only valid with "
                    f"column_type=ENUM_TYPE; got column_type={self.column_type!r}",
                )
            if not self.enum_name:
                raise ValueError(
                    f"Column(name={self.name!r}): enum_type requires enum_name to be set",
                )
            if len(self.enum_type) == 0:
                raise ValueError(
                    f"Column(name={self.name!r}): enum_type must be a non-empty tuple",
                )
        # 3. ENUM_TYPE requires enum_type (inverse of #2).
        if self.column_type == ENUM_TYPE and self.enum_type is None:
            raise ValueError(
                f"Column(name={self.name!r}): column_type=ENUM_TYPE requires enum_type to be set",
            )
        # 4. vector_dim only valid with VECTOR_TYPE.
        if self.vector_dim is not None and self.column_type != VECTOR_TYPE:
            raise ValueError(
                f"Column(name={self.name!r}): vector_dim only valid with "
                f"column_type=VECTOR_TYPE; got column_type={self.column_type!r}",
            )
        # 5. VECTOR_TYPE requires vector_dim (inverse of #4).
        if self.column_type == VECTOR_TYPE and self.vector_dim is None:
            raise ValueError(
                f"Column(name={self.name!r}): column_type=VECTOR_TYPE requires vector_dim to be set",
            )
        # 6. foreign_key tuple shape -- must be ``(ref_table, ref_col)``,
        #    both non-empty strings.
        if self.foreign_key is not None:
            if (
                not isinstance(self.foreign_key, tuple)
                or len(self.foreign_key) != 2
                or not all(isinstance(part, str) and part for part in self.foreign_key)
            ):
                raise ValueError(
                    f"Column(name={self.name!r}): foreign_key must be a 2-tuple of "
                    f"non-empty strings (ref_table, ref_col); got {self.foreign_key!r}",
                )
        # 7. server_default: no cross-field restriction (any string
        #    accepted). No validator.
        # 8. precision/scale only valid with NUMERIC_TYPE.
        if (self.precision is not None or self.scale is not None) and self.column_type != NUMERIC_TYPE:
            raise ValueError(
                f"Column(name={self.name!r}): precision/scale only valid with "
                f"column_type=NUMERIC_TYPE; got column_type={self.column_type!r}",
            )
        # 9. NUMERIC_TYPE requires both precision and scale (inverse
        #    of #8).
        if self.column_type == NUMERIC_TYPE and (self.precision is None or self.scale is None):
            raise ValueError(
                f"Column(name={self.name!r}): column_type=NUMERIC_TYPE requires both precision and scale to be set",
            )
        # 10. TSVECTOR_TYPE requires immutable=True (trigger-maintained
        #     server-side; UPDATE generators must skip the column).
        if self.column_type == TSVECTOR_TYPE and not self.immutable:
            raise ValueError(
                f"Column(name={self.name!r}): column_type=TSVECTOR_TYPE columns are "
                f"trigger-maintained server-side and must be declared immutable=True "
                f"to exclude them from Collection UPDATE generators",
            )


def l2_order_columns() -> list[Column]:
    """the two columns a table needs so ``l2_cas_mutate`` can persist to it in order.

    Splice into a :class:`TableSchema`'s ``columns``. ``l2_epoch`` holds the creation time of the
    L2 stream a row's compare-and-swap won in, ``l2_revision`` the revision it won; together they
    are the order :meth:`BaseCollection.l2_cas_mutate` fences every L3 persist on (see
    :mod:`threetears.core.collections.l2_order`). Both nullable -- a row another path wrote carries
    no order and orders before every swap -- and mutable, since the fenced upsert rewrites them.

    :return: the ``l2_epoch`` (``TIMESTAMPTZ``) and ``l2_revision`` (``BIGINT``) columns
    :rtype: list[Column]
    """
    return [
        Column(L2_EPOCH_COLUMN, DATETIMETZ_TYPE, nullable=True),
        Column(L2_REVISION_COLUMN, BIGINT_TYPE, nullable=True),
    ]


def _servable_index(
    name: str, columns: Sequence[tuple[str, bool]], *, plain: bool, where: str | None, unique: bool = False
) -> DeclaredIndex | None:
    """a declared index as a keyed read's pins and order can be served by it, or None when none can be: one with an
    operator class or other access method (``plain`` False: it orders by something else), or a predicate of anything
    but ``column = 'text'`` and ``column IS NOT NULL`` conjuncts (serving only a read that pins each column to that
    text and states each guard, :func:`~threetears.interfaces.keyed.keys.partial_predicate`; a read cannot be judged to
    state any other). A collated column ends the columns a caller passes (:meth:`TableSchema._declared_indexes`: WHY
    there). One rule
    for a :class:`TableSchema`'s indexes and a SQLAlchemy ``Table``'s (:func:`register_sa_table_indexes`).

    :param name: the index's name
    :ptype name: str
    :param columns: its columns in order, each with whether it sorts descending
    :ptype columns: Sequence[tuple[str, bool]]
    :param plain: whether it is a B-tree with no operator class or collation
    :ptype plain: bool
    :param where: its predicate, or None for a whole index
    :ptype where: str | None
    :param unique: whether it is unique (a unique index or ``UNIQUE`` constraint)
    :ptype unique: bool
    :return: the index, or None
    :rtype: DeclaredIndex | None
    """
    if not plain:
        return None
    predicate = (frozenset(), frozenset()) if where is None else partial_predicate(where)
    if predicate is None:
        return None
    stated, guarded = predicate
    return DeclaredIndex(
        name,
        tuple(column for column, _ in columns),
        frozenset(column for column, descending in columns if descending),
        stated=stated,
        guarded=guarded,
        unique=unique,
    )


#: an index element a SQLAlchemy ``Table`` names as text: a plain column, optionally ``ASC`` or ``DESC``
_SA_TEXT_COLUMN: Final[re.Pattern[str]] = re.compile(r"\s*\"?(\w+)\"?(?:\s+(ASC|DESC))?\s*", re.IGNORECASE)


def _sa_index_columns(index: sa.Index) -> list[tuple[str, bool]] | None:
    """a SQLAlchemy index's elements as ``(column, descending)``, or None when any is other than a plain column
    (an expression, ``NULLS FIRST``, a collation)."""
    columns: list[tuple[str, bool]] = []
    for element in index.expressions:
        if isinstance(element, sa.Column):
            columns.append((element.name, False))
            continue
        if isinstance(element, TextClause):
            found = _SA_TEXT_COLUMN.fullmatch(element.text)
            if found is None:
                return None
            columns.append((found.group(1), (found.group(2) or "").upper() == "DESC"))
            continue
        if isinstance(element, UnaryExpression) and isinstance(element.element, sa.Column):
            if element.modifier is operators.desc_op:
                columns.append((element.element.name, True))
                continue
            if element.modifier is operators.asc_op:
                columns.append((element.element.name, False))
                continue
        return None
    return columns


def register_sa_table_indexes(table: sa.Table) -> None:
    """register the indexes of an L3 table a package declares as a SQLAlchemy ``Table`` (not a :class:`TableSchema`),
    as a :class:`TableSchema` registers its own: its primary key and each index :func:`_servable_index` admits, for the
    hint a keyed read names to YugabyteDB (:func:`~threetears.interfaces.keyed.keys.serving_index`), and its partial
    predicates' ``column = 'text'`` conjuncts, which a keyed read pinning them writes as literals
    (:func:`~threetears.interfaces.keyed.keys.register_partial_predicates`).

    WHY (3tears #100): agent-wake, agent-skills and scheduled-jobs declare their tables in SQLAlchemy, so none of their
    indexes was registered and YugabyteDB was never told of one: it counted a conversation's 10 active schedules
    through the primary key, reading its 400 rows, on 6 of 6 fresh ANALYZEs.

    :param table: the table, as declared (its schema, if any, names the relation)
    :ptype table: sa.Table
    :return: nothing
    :rtype: None
    """
    name = table.name if table.schema is None else f"{table.schema}.{table.name}"
    pk_cols = tuple(column.name for column in table.primary_key.columns)
    indexes = [DeclaredIndex("primary_key", pk_cols, primary=True)] if pk_cols else []
    predicates: list[str] = []
    for index in sorted(table.indexes, key=lambda one: str(one.name)):
        options = index.dialect_options["postgresql"]
        where = options["where"]
        predicate = None if where is None else str(where.text if isinstance(where, TextClause) else where)
        if predicate is not None:
            predicates.append(predicate)
        columns = _sa_index_columns(index)
        if columns is None or index.name is None:
            continue
        # SQLAlchemy reads an unset access method as False: the default B-tree
        using = options["using"]
        served = _servable_index(
            str(index.name),
            columns,
            plain=(not using or str(using).lower() == "btree") and not options["ops"],
            where=predicate,
        )
        if served is not None:
            indexes.append(served)
    register_table_indexes(name, indexes)
    register_partial_predicates(name, predicates)


@dataclass(frozen=True)
class TableSchema:
    """table descriptor for :class:`SchemaBackedCollection`.

    :cvar name: table name (matches the L3 DDL)
    :cvar primary_key: single column name (single-pk shape) or tuple
        of column names in declared order (composite-pk shape)
    :cvar columns: list of :class:`Column` in the order they should
        appear in generated INSERT statements. order is stable across
        invocations so asyncpg parameter positions (``$1``, ``$2``, ...)
        remain predictable for tests that assert on positional args
    :cvar cas_column: column used as the optimistic-concurrency fence
        on the UPDATE path. when set (typical value: ``"date_updated"``)
        and the caller passes a non-``None`` ``original_timestamp`` to
        :meth:`SchemaBackedCollection.save_to_store`, the generator
        emits a separate UPDATE fenced on ``cas_column = $N``. when
        ``None``, the ``original_timestamp`` argument is silently
        ignored
    :cvar on_conflict: behaviour of
        :meth:`SchemaBackedCollection.save_to_store` on primary-key
        conflict. ``"update"`` (default) emits ``ON CONFLICT (pk) DO
        UPDATE SET <mutable>`` -- the standard upsert path. ``"raise"``
        emits plain INSERT with no ``ON CONFLICT`` clause; duplicate
        primary keys raise :class:`asyncpg.exceptions.UniqueViolationError`
        from asyncpg naturally (journal / append-only tables).
        ``"ignore"`` emits ``ON CONFLICT (pk) DO NOTHING``; duplicate
        primary keys are silently dropped (dedup-on-redelivery tables
        like ``audit_events``, where a redelivered envelope repeats its
        ``id``)
    :cvar write_generation: ``WRITE_GENERATION`` switches the table on: every committed write
        advances its write generation, the creating migration gives it its change-note triggers, and
        the collection class reads its ``write_generation`` from here
    :cvar delta_files: ``True`` opts the table in to delta files: every commit that writes it saves the rows it
        wrote and the keys it deleted as one file on the shared volume, before the commit, and a write that cannot
        is refused (:mod:`~threetears.core.collections.delta_files`). Off by default: a table not opted in writes
        exactly as before. Every column must be one the rail answers as one Arrow type (``rail_arrow_types``), the
        type the file holds it as. A column's ``server_default`` must be a fixed value, which the file's row holds for
        a write that leaves the column out (:attr:`delta_file_defaults`); a default the database computes, a
        tsvector, a ``cas_column`` or the L2 order columns are refused where declared (:meth:`_refuse_undecided`)
    :cvar delta_keep: the rows of each state a full of the table keeps when the writer folds its delta files
        (:class:`~threetears.core.collections.delta_keep.DeltaKeepRule`, a subclass: it crosses to the fold's worker
        process as its name); None keeps every row. Read only for a table that opts in (:attr:`delta_files`): the
        same rule the table's copy keeps by, so a full holds what a copy holds (``design-pod-delta-files.md``,
        decision 13)
    :cvar delta_scope: for a table that opts in to delta files and whose key does not name a row's state (AP results,
        keyed by the race, ``office_key``, with the state in ``join_state_code``), the row column holding the state: a
        delete marker carries the key alone otherwise, and fulls are cut per state, so each of the table's deletes
        names the state of the keys it deletes (``delete_rows(..., delta_scope=...)``), the marker carries it in this
        column, and the fold files the delete under that state's part (``design-pod-delta-files.md``, decision 27). A
        delete naming no state is refused before it reaches L3. None: the key names the state, or the table is not cut
        per state
    :cvar delta_state_scope: for a table that opts in to delta files, the state each row is of as the writer's seqlock
        names it, ``(column, prefix)`` (the form of a copy's
        :attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`, ``("join_state_code", "state:")``):
        its column a key column, or :attr:`delta_scope` (so every delete marker carries it too). Declared, each commit
        reads the states it wrote off its delta file's rows and its record names every one of them, marked complete
        (:data:`~threetears.core.collections.epoch_change.EVERY_STATE_NAMED`), so a copy that finds the file gone
        reloads only those states from the full (``design-pod-delta-files.md``, decision 54 (c)). None: a record's
        states are never known complete, and a copy reloads the whole table
    :cvar cas_null_safe: opt in to a NULL-safe compare-and-swap fence
        that also covers the FIRST write of a row. default ``False``,
        which leaves every existing schema's emitted SQL untouched.

        the default fence cannot express "no row existed when i read".
        with ``cas_null_safe=False`` a save carrying
        ``original_timestamp=None`` is NOT fence-eligible at all: it
        falls through to an UNFENCED
        ``INSERT ... ON CONFLICT (pk) DO UPDATE SET ...``. that is
        correct for a random primary key (two writers cannot collide on
        one), and WRONG for a DERIVED primary key -- a
        ``uuid5``/hash-of-the-business-key id that two concurrent
        first-writers both compute to the same value. both of their
        statements report one row affected and the second silently
        overwrites the first, so a read-modify-write (a counter
        increment, a set-member append) loses the first writer's work
        with nothing raised anywhere.

        with ``cas_null_safe=True`` a save carrying
        ``original_timestamp=None`` -- a first write -- emits ONE
        statement (a save carrying a value read off an existing row takes
        the update-only CAS path every ``cas_column`` schema takes, so a
        row deleted since that read is never re-created)::

            INSERT INTO t (...) VALUES (...)
            ON CONFLICT (pk) DO UPDATE SET <mutable> = EXCLUDED....
            WHERE t.<cas_column> IS NOT DISTINCT FROM $N

        ``IS NOT DISTINCT FROM`` is the NULL-safe equality Postgres
        offers, so ``$N = NULL`` (a first write) matches a row whose
        fence column is NULL and matches NOTHING once a rival writer has
        stamped one. no conflicting row at all means the plain INSERT
        applies and the fence never runs, so the row IS created. the
        loser therefore affects 0 rows, which
        :meth:`BaseCollection.save_entity` turns into
        :class:`~threetears.core.exceptions.ConcurrentModificationError`
        for the caller's retry loop -- provided the entity is
        constructed ``is_new=False`` (``is_new=True`` maps 0 rows onto an
        unretryable :class:`RuntimeError` instead).

        requires ``cas_column`` set, ``on_conflict="update"``, and the
        ``cas_column`` to be a mutable non-pk column carrying no
        ``server_default`` -- otherwise the fence could never advance and
        would wave every later writer through. all four are validated in
        :meth:`__post_init__`.
    """

    name: str
    primary_key: KeyColumn | KeyColumns
    columns: list[Column]
    cas_column: str | None = None
    on_conflict: OnConflict = "update"
    # ---new in v0.8.0:---
    # composite or supplementary FK declarations. single-column FKs may
    # alternatively be declared via :attr:`Column.foreign_key` to keep
    # simple cases terse; both shapes coexist. tuple (not list) so the
    # frozen-dataclass hash + equality contract stays intact.
    foreign_keys: tuple[ForeignKeyDef, ...] = ()
    # named indexes on this table. ``UNIQUE`` + partial-index variants
    # supported via :class:`IndexDef`. tuple for the same reason.
    indexes: tuple[IndexDef, ...] = ()
    # named UNIQUE constraints (``ALTER TABLE ... ADD CONSTRAINT ...
    # UNIQUE (...)``). distinct from :class:`IndexDef` ``unique=True``
    # at the Alembic auto-gen level even though both produce identical
    # ``pg_indexes`` rows; see :class:`UniqueConstraintDef` docstring.
    # v0.8.1+
    unique_constraints: tuple[UniqueConstraintDef, ...] = ()
    # NULL-safe CAS fence opt-in. declared LAST among the init fields so
    # every existing positional construction keeps its argument
    # positions, and defaulted False so no existing schema changes the
    # SQL it emits. see the :cvar: entry above for the full contract.
    cas_null_safe: bool = False
    # every committed write to the table advances its write generation, and the table carries the
    # change-note triggers from the migration that creates it (threetears.core.collections.lost_writes).
    # the one declaration: the migration package that creates the table renders the trigger from it,
    # and the collection class reads its switch from it.
    write_generation: WriteGenerationDeclaration = WRITE_GENERATION_UNDECLARED
    # which tenant the table's change records belong to (threetears.core.tenancy); the
    # collection class reads it from here, as it reads write_generation
    tenancy: Tenancy = TENANCY_UNDECLARED
    # the table's eviction policy in a process's SQLite L1 (threetears.core.eviction: EVICTABLE, a cache of reads a
    # miss refills; KEEP, what a reader treats as complete or a write needs), with why. No default: the L1 refuses
    # a table that declares none (threetears.core.cache.l1_lru)
    l1_eviction: Declared | None = None
    # every commit that writes the table saves its delta file before it commits, enforced on every write path of the
    # collection (threetears.core.collections.delta_files); the one declaration, read by the collection class
    delta_files: bool = False
    # the rows of each state a full of the table keeps at a fold (threetears.core.collections.delta_fold); a class,
    # compared and hashed by its name like every other field, so the schema stays a value
    delta_keep: type[DeltaKeepRule] | None = None
    # the row column a delete marker carries the row's state in, for a table whose key does not name it (decision 27):
    # the table's declared state column (DeclaredScope), the one a local copy's state read is pinned by
    delta_scope: Annotated[str | None, DeclaredScope] = None
    # the state each row is of, (column, prefix): a commit's record names every state its file's rows are of (decision 54)
    delta_state_scope: tuple[str, str] | None = None
    # every row carries the epoch of the write that left it (threetears.core.ddl.row_epoch): row_versions(scope) keeps a
    # version a write, the epoch the last column of the stored key; ROW_STAMPED keeps one row a key, stamped. The
    # column is added when not declared; every write of the table stamps it, and a write without it is refused
    row_epoch: RowEpoch | None = None

    _pk_columns: KeyColumns = field(init=False, repr=False)
    # the key the table stores rows by: the primary key, then the epoch for a versioned table (row_epoch)
    _stored_key: KeyColumns = field(init=False, repr=False)
    _by_name: dict[str, Column] = field(init=False, repr=False)
    # each column's fixed database default, for a table that opts in to delta files (TableSchema.delta_file_defaults)
    _delta_defaults: dict[str, Any] = field(init=False, repr=False, compare=False)
    _partition_column: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """validate + index the schema after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when a pk column is not declared in
            :attr:`columns`, when :attr:`cas_column` is set but missing
            from :attr:`columns`, when more than one column is flagged
            ``partition=True`` (only one partition column per table),
            when a partition column is not part of
            :attr:`primary_key` (the schema must enforce row uniqueness
            through the partition), or when :attr:`cas_null_safe` is set
            without the four preconditions its fence needs (see
            :meth:`_validate_cas_null_safe`)
        """
        if self.l1_eviction is not None:
            # declared where the table is declared: every L1 and the cache metrics read it by the table's name
            declare_l1_table(self.name, self.l1_eviction)
        if self.row_epoch is not None:
            # before anything reads the columns or the indexes: the epoch column and the (scope, epoch) index are
            # the table's own, added where not declared
            self._declare_row_epoch()
        if self.delta_files:
            # refused where it is declared, not at the first write: a column no file can type would refuse every one
            rail_arrow_types(self.columns)
        object.__setattr__(self, "_delta_defaults", self._refuse_undecided() if self.delta_files else {})
        pk = self.primary_key
        pk_cols: tuple[str, ...] = (pk,) if isinstance(pk, str) else tuple(pk)
        if self.delta_scope is not None:
            self._check_delta_scope(pk_cols)
        if self.delta_state_scope is not None:
            self._check_delta_state_scope(pk_cols)
        # coerce partition columns to immutable so callers do not have
        # to declare both flags. partition columns identify the slot a
        # row belongs to; mutating the value would corrupt the partition
        # map.
        coerced_columns: list[Column] = []
        partition_cols: list[str] = []
        for col in self.columns:
            if col.partition:
                partition_cols.append(col.name)
                if not col.immutable:
                    col = Column(  # noqa: PLW2901 -- intentional coercion
                        name=col.name,
                        column_type=col.column_type,
                        immutable=True,
                        nullable=col.nullable,
                        partition=True,
                        # v0.8.0: pass through every new field. without
                        # this, a partition column declared with
                        # foreign_key= / enum_type= / vector_dim= /
                        # server_default= / precision= / scale= would
                        # silently drop those fields during coercion.
                        foreign_key=col.foreign_key,
                        enum_type=col.enum_type,
                        enum_name=col.enum_name,
                        vector_dim=col.vector_dim,
                        server_default=col.server_default,
                        precision=col.precision,
                        scale=col.scale,
                    )
            coerced_columns.append(col)
        if len(partition_cols) > 1:
            raise ValueError(
                f"TableSchema(name={self.name!r}): only one partition "
                f"column per table is supported, got {partition_cols!r}",
            )
        partition_column: str | None = partition_cols[0] if partition_cols else None
        if partition_column is not None and partition_column not in pk_cols:
            raise ValueError(
                f"TableSchema(name={self.name!r}): partition column "
                f"{partition_column!r} must be part of primary_key "
                f"(got primary_key={pk_cols!r}) so the schema enforces "
                f"row uniqueness through the partition",
            )
        by_name = {c.name: c for c in coerced_columns}
        stored_key = (*pk_cols, EPOCH_COLUMN) if self.row_epoch is not None and self.row_epoch.versions else pk_cols
        if len(stored_key) > MAX_INDEX_COLUMNS:
            raise ValueError(
                f"TableSchema(name={self.name!r}): primary_key lists {len(stored_key)} columns"
                f"{' with the epoch' if stored_key != pk_cols else ''}; YugabyteDB indexes at most {MAX_INDEX_COLUMNS}",
            )
        for pk_col in pk_cols:
            if pk_col not in by_name:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): primary_key column {pk_col!r} not found in columns",
                )
        if self.cas_column is not None and self.cas_column not in by_name:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_column {self.cas_column!r} not found in columns",
            )
        # dataclass is frozen, use object.__setattr__ to install derived
        # attributes
        object.__setattr__(self, "columns", coerced_columns)
        object.__setattr__(self, "_pk_columns", pk_cols)
        object.__setattr__(self, "_stored_key", stored_key)
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "_partition_column", partition_column)
        # v0.8.0 cross-field validators. placed AFTER ``_by_name`` is
        # installed because they reference the indexed column map.
        for fk in self.foreign_keys:
            for local_col in fk.local_cols:
                if local_col not in by_name:
                    raise ValueError(
                        f"TableSchema(name={self.name!r}): foreign_key references "
                        f"local_col={local_col!r} which is not declared in columns",
                    )
        for idx in self.indexes:
            for col_name in idx.columns:
                # a stamped table's bucket column is the L3 table's alone, declared with its restore index
                if col_name not in by_name and not (col_name == BUCKET_COLUMN and self._stamped()):
                    raise ValueError(
                        f"TableSchema(name={self.name!r}): index name={idx.name!r} "
                        f"references col={col_name!r} which is not declared in columns",
                    )
        index_names = [i.name for i in self.indexes]
        if len(index_names) != len(set(index_names)):
            raise ValueError(
                f"TableSchema(name={self.name!r}): duplicate index names in {index_names!r}",
            )
        # v0.8.1: validate the new ``unique_constraints`` field
        # symmetrically with ``indexes``.
        for uc in self.unique_constraints:
            for col_name in uc.columns:
                if col_name not in by_name:
                    raise ValueError(
                        f"TableSchema(name={self.name!r}): unique_constraint "
                        f"name={uc.name!r} references col={col_name!r} which "
                        f"is not declared in columns",
                    )
        uc_names = [u.name for u in self.unique_constraints]
        if len(uc_names) != len(set(uc_names)):
            raise ValueError(
                f"TableSchema(name={self.name!r}): duplicate unique_constraint names in {uc_names!r}",
            )
        # Cross-axis name collision: an index and a unique constraint
        # cannot share a name (Postgres allows it at separate
        # information-schema layers but the parity assertions would
        # produce ambiguous error messages).
        collision = set(index_names) & set(uc_names)
        if collision:
            raise ValueError(
                f"TableSchema(name={self.name!r}): name(s) {sorted(collision)!r} "
                f"used by both an IndexDef and a UniqueConstraintDef",
            )
        self._validate_cas_null_safe(pk_cols, by_name)
        self._validate_l2_order(pk_cols, by_name)
        # the keys a keyset read of this table must hold (its order and pins), for the pagers' run-time refusal. A
        # versioned table's key alone is not unique (each write keeps a version of it): the key it holds is the stored one
        register_table_keys(self.name, self._declared_keys(stored_key, partition_column))
        # the indexes a keyed read of it names to YugabyteDB, whose planner prefers the primary key (keys.serving_index)
        register_table_indexes(self.name, self._declared_indexes(stored_key))
        # the texts its partial indexes' predicates name, which a keyed read pinning them writes as literals, so a
        # generic plan can prove the predicate (keys.register_partial_predicates)
        register_partial_predicates(self.name, (index.where for index in self.indexes if index.where is not None))

    def _declare_row_epoch(self) -> None:
        """give a table that declares a row epoch its epoch column where not declared, and a versioned one its
        ``(scope, epoch)`` index; refuse what cannot hold beside it. The index is what a restore of a lost change log
        reads the table by, a scope at a time (:mod:`~threetears.core.collections.table_restore`).

        The column is ``BIGINT NOT NULL`` with no default: a write that bypasses 3tears and leaves it out is refused by
        the database. On a versioned table it is part of the stored key, so it is immutable (never in an upsert's
        ``SET``); on a stamped one each write sets it.

        A versioned table keeps a version of a key a write, so the key alone is unique no longer: a unique constraint
        or a unique index would refuse the second version, a compare-and-swap fence or the L2 order would fence one
        version against another. Each is refused where it is declared.

        :return: nothing
        :rtype: None
        :raises ValueError: when the epoch column is declared otherwise than as above, or named in the primary key; or,
            versioned, the scope column is not declared, or the table declares any of the above
        """
        declared = self.row_epoch
        assert declared is not None  # noqa: S101 -- called only when declared
        versions = declared.versions
        pk = self.primary_key
        if EPOCH_COLUMN in ((pk,) if isinstance(pk, str) else tuple(pk)):
            raise ValueError(
                f"TableSchema(name={self.name!r}): {EPOCH_COLUMN!r} is not named in primary_key: a versioned table "
                "stores rows by the key and then the epoch on its own, a stamped one by the key alone"
            )
        expected = Column(EPOCH_COLUMN, BIGINT_TYPE, immutable=versions, nullable=False)
        held = next((column for column in self.columns if column.name == EPOCH_COLUMN), None)
        if held is None:
            object.__setattr__(self, "columns", [*self.columns, expected])
        elif held != expected:
            raise ValueError(
                f"TableSchema(name={self.name!r}): its {EPOCH_COLUMN!r} column must be {expected!r} (or left out, "
                f"and it is added so): a BIGINT NOT NULL with no default, so a write that leaves it out is refused, "
                f"immutable on a versioned table"
            )
        if not versions:
            self._declare_bucket_index()
            return
        # the "gone" marker's flag: a delete appends the key's version at its epoch with it set (append-only L3, chunk 2)
        gone = Column(GONE_COLUMN, BOOL_TYPE, immutable=True, nullable=False, server_default="false")
        held_gone = next((column for column in self.columns if column.name == GONE_COLUMN), None)
        if held_gone is None:
            object.__setattr__(self, "columns", [*self.columns, gone])
        elif held_gone != gone:
            raise ValueError(
                f"TableSchema(name={self.name!r}): its {GONE_COLUMN!r} column must be {gone!r} (or left out, and it "
                f"is added so): a delete appends the key's version with it set, and a write that leaves it out writes "
                f"a live version"
            )
        scope = declared.scope
        if scope is None or not any(column.name == scope for column in self.columns):
            raise ValueError(f"TableSchema(name={self.name!r}): row_versions scope {scope!r} is not one of its columns")
        refused = [
            (self.cas_column is not None, f"cas_column={self.cas_column!r}"),
            (self.cas_null_safe, "cas_null_safe=True"),
            (any(column.name in L2_ORDER_COLUMNS for column in self.columns), "the L2 order columns"),
            (bool(self.unique_constraints), "a unique constraint"),
            (any(index.unique for index in self.indexes), "a unique index"),
        ]
        for refuse, what in refused:
            if refuse:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): row_versions with {what}: each write keeps its own version of "
                    "a key, so the key alone is unique no longer and one version cannot be fenced against another"
                )
        name = scope_epoch_index_name(self.name)
        if not any(index.name == name for index in self.indexes):
            # WHY the scope leads, hashed on YugabyteDB (its default for a first column): the epochs rise, so an index
            # led by them sends every insert to the last tablet; led by the scope, the rising end spreads across scopes,
            # and a restore reads one scope's versions since X as one range (design-append-only-l3.md, decision 3)
            restore = Index(
                name,
                scope,
                EPOCH_COLUMN,
                purpose=_RESTORE_READ,
                plan_test=_RESTORE_PLAN_TEST,
                reason="a lost change log is rebuilt from each scope's versions since the keeper's mark",
            )
            object.__setattr__(self, "indexes", (*self.indexes, restore))

    def _declare_bucket_index(self) -> None:
        """give a stamped table its ``(bucket, epoch)`` index, the read a restore of a lost change log makes of it a
        bucket at a time (:mod:`~threetears.core.collections.table_restore`). The bucket column is the L3 table's alone
        (:data:`~threetears.core.ddl.row_epoch.BUCKET_COLUMN`, rendered by :meth:`_sqlalchemy_table` when stored): no
        cache, entity or write names it.

        WHY a stored generated column, not an expression index: the keyed helpers, the declared indexes and the
        no-whole-reads rule all know an index by its columns, so a read pinning ``epoch_bucket = $1`` is one range of a
        declared index, judged keyed like any other; an expression index serves only a read that repeats its
        expression word for word, which none of them can check. Both engines store a generated column (Postgres 12+,
        YugabyteDB 2.25+).

        :return: nothing
        :rtype: None
        :raises ValueError: when a key column's type has no hash a bucket can be made of
        """
        self._bucket_expression()  # refused where declared: a key no bucket can hash refuses the table
        name = bucket_epoch_index_name(self.name)
        if any(index.name == name for index in self.indexes):
            return
        # WHY the bucket leads, range-sharded on YugabyteDB (``asc``) and split a tablet a bucket (create_statements):
        # the epochs rise, so an index led by them sends every write to the last tablet; led by the bucket, the rising
        # end is spread over RESTORE_BUCKETS tablets, and a restore reads one bucket's rows since X as one range
        restore = Index(
            name,
            BUCKET_COLUMN,
            EPOCH_COLUMN,
            asc=BUCKET_COLUMN,
            purpose=_STAMPED_RESTORE_READ,
            plan_test=_STAMPED_RESTORE_PLAN_TEST,
            reason="a lost change log is rebuilt from each bucket's rows stamped since the keeper's mark",
        )
        object.__setattr__(self, "indexes", (*self.indexes, restore))

    def _stamped(self) -> bool:
        """whether this is a stamped table (``ROW_STAMPED``: one row a key, updated in place).

        :return: whether it is
        :rtype: bool
        """
        return self.row_epoch is not None and not self.row_epoch.versions

    def _bucket_expression(self) -> str:
        """a stamped table's :data:`~threetears.core.ddl.row_epoch.BUCKET_COLUMN` expression, from its key columns.

        :return: the expression
        :rtype: str
        :raises ValueError: when a key column's type has no hash a bucket can be made of
        """
        pk = self.primary_key
        keys = (pk,) if isinstance(pk, str) else tuple(pk)
        by_name = {column.name: column for column in self.columns}
        hashes = []
        for key in keys:
            column = by_name.get(key)
            kind = None if column is None else _BUCKET_HASH_KINDS.get(column.column_type)
            if kind is None:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): ROW_STAMPED key column {key!r} is of no type a restore bucket "
                    f"can hash ({sorted(_BUCKET_HASH_KINDS)})"
                )
            hashes.append(key_hash_sql(quote_identifier(key), kind))
        return bucket_sql(tuple(hashes))

    def _declared_indexes(self, pk_cols: tuple[str, ...]) -> list[DeclaredIndex]:
        """the B-tree indexes a keyed read's pins and order can be served by: the primary key, and each declared index
        :func:`_servable_index` admits.

        :param pk_cols: the primary key's columns
        :ptype pk_cols: tuple[str, ...]
        :return: the indexes
        :rtype: list[DeclaredIndex]
        """
        indexes = [DeclaredIndex("primary_key", pk_cols, primary=True)] if pk_cols else []
        for index in self.indexes:
            # a column held in an explicit collation (``conversation_ref COLLATE "C"``) serves only comparisons made in
            # that collation, which no pin of a keyed read is: the index serves a read by the plain columns before it.
            # WHY: skipped whole, a collated index was never named to YugabyteDB, so a read pinning its leading columns
            # was hinted to a wider index of the same lead (find_by_ref_prefix to idx_conv_agent_ref_created, every ref
            # of the channel read and filtered, D7 2026-11-15)
            collated = {column for column, _ in index.collate}
            columns = tuple(index.columns)
            if collated:
                columns = columns[: next(at for at, column in enumerate(columns) if column in collated)]
            served = (
                _servable_index(
                    index.name,
                    tuple((column, column in index.desc) for column in columns),
                    plain=index.using is None and not index.ops,
                    where=index.where,
                    # a unique index cut short at a collated column is unique no longer
                    unique=index.unique and columns == tuple(index.columns),
                )
                if columns
                else None
            )
            if served is not None:
                indexes.append(served)
        # each UNIQUE constraint, a unique B-tree of its columns: a read binding them all (a set of values allowed)
        indexes.extend(
            DeclaredIndex(constraint.name, tuple(constraint.columns), unique=True)
            for constraint in self.unique_constraints
        )
        return indexes

    def _declared_keys(self, pk_cols: tuple[str, ...], partition_column: str | None) -> list[TableKey]:
        """the keys this declaration says the table holds: its primary key, each ``UNIQUE`` constraint, each unique
        index (a partial one by its ``IS NOT NULL`` or ``IS NULL`` predicate's column, or as a key outright when its
        predicate is any other: the run-time refusal never refuses what it cannot prove). The partition column places a row only.

        :param pk_cols: the primary key's columns
        :ptype pk_cols: tuple[str, ...]
        :param partition_column: the partition column, or None
        :ptype partition_column: str | None
        :return: the keys
        :rtype: list[TableKey]
        """
        placement = (partition_column,) if partition_column is not None else ()

        def key(columns: tuple[str, ...], not_null: tuple[str, ...] = (), null: tuple[str, ...] = ()) -> TableKey:
            return TableKey(columns, not_null, tuple(c for c in placement if c in columns), null)

        keys = [key(pk_cols), *(key(tuple(unique.columns)) for unique in self.unique_constraints)]
        for index in self.indexes:
            if not index.unique:
                continue
            valued = not_null_predicate(index.where) if index.where else None
            nulled = null_predicate(index.where) if index.where else None
            keys.append(
                key(
                    tuple(index.columns),
                    (valued,) if valued is not None else (),
                    (nulled,) if nulled is not None else (),
                )
            )
        return keys

    def _validate_l2_order(self, pk_cols: tuple[str, ...], by_name: dict[str, Column]) -> None:
        """reject order columns a compare-and-swap persist could not fence on.

        The two columns are declared together or not at all: one without the other leaves a
        table that looks fenced and is not. Each must be nullable (a row some other path wrote
        carries no order), mutable and outside the key (the fenced ``DO UPDATE`` must rewrite
        them), and of the one type the fence compares as. The fence lives on the ``ON CONFLICT
        DO UPDATE`` branch, and it replaces the ``cas_null_safe`` fence rather than composing
        with it.

        :param pk_cols: primary-key column names in declared order
        :ptype pk_cols: tuple[str, ...]
        :param by_name: declared columns indexed by name
        :ptype by_name: dict[str, Column]
        :return: nothing
        :rtype: None
        :raises ValueError: when only one order column is declared, when either is of the wrong
            type, non-nullable, immutable, part of the key or carries a ``server_default``, or
            when a table declaring them has ``on_conflict`` other than ``"update"`` or sets
            ``cas_null_safe``
        """
        declared = [name for name in L2_ORDER_COLUMNS if name in by_name]
        if not declared:
            return
        if len(declared) != len(L2_ORDER_COLUMNS):
            raise ValueError(
                f"TableSchema(name={self.name!r}): declares {declared!r} but not both of "
                f"{L2_ORDER_COLUMNS!r}; the compare-and-swap order is the pair, and half of it "
                f"cannot fence anything. Declare both with l2_order_columns()",
            )
        expected = {L2_EPOCH_COLUMN: DATETIMETZ_TYPE, L2_REVISION_COLUMN: BIGINT_TYPE}
        for name, column_type in expected.items():
            col = by_name[name]
            unsound = (
                col.column_type != column_type
                or not col.nullable
                or col.immutable
                or name in pk_cols
                or col.server_default is not None
            )
            if unsound:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): order column {name!r} must be a nullable, "
                    f"mutable, non-key {column_type!r} column with no server_default -- declare it "
                    f"with l2_order_columns()",
                )
        if self.on_conflict != "update" or self.cas_null_safe:
            raise ValueError(
                f"TableSchema(name={self.name!r}): a table carrying the compare-and-swap order "
                f"needs on_conflict='update' and no cas_null_safe fence -- its persist is the "
                f"ordered ON CONFLICT DO UPDATE, fenced on the order and nothing else",
            )

    def _check_delta_scope(self, pk_cols: tuple[str, ...]) -> None:
        """refuse a :attr:`delta_scope` no delete marker could carry, or none needs.

        :param pk_cols: the key columns
        :ptype pk_cols: tuple[str, ...]
        :return: nothing
        :rtype: None
        :raises ValueError: when the table does not opt in to delta files, the column is not one of its columns, or
            its key names the column already
        """
        name = self.delta_scope
        if not self.delta_files:
            raise ValueError(
                f"TableSchema(name={self.name!r}): delta_scope={name!r} names the state a delete marker carries; only "
                f"a table that opts in to delta files writes one (delta_files=True)"
            )
        if name not in {column.name for column in self.columns}:
            raise ValueError(f"TableSchema(name={self.name!r}): delta_scope={name!r} is not a column of the table")
        if name in pk_cols:
            raise ValueError(
                f"TableSchema(name={self.name!r}): delta_scope={name!r} is a key column: the key names it already, "
                f"so every delete marker carries it"
            )

    def _check_delta_state_scope(self, pk_cols: tuple[str, ...]) -> None:
        """refuse a :attr:`delta_state_scope` whose state a delete marker would not carry: then a file's markers could
        not say their states, and a record would name its states complete without them.

        :param pk_cols: the key columns
        :ptype pk_cols: tuple[str, ...]
        :return: nothing
        :rtype: None
        :raises ValueError: when the table does not opt in to delta files, the prefix is empty, or the column is
            neither a key column nor :attr:`delta_scope`
        """
        assert self.delta_state_scope is not None  # noqa: S101 -- the caller's condition
        column, prefix = self.delta_state_scope
        if not self.delta_files:
            raise ValueError(
                f"TableSchema(name={self.name!r}): delta_state_scope={self.delta_state_scope!r} names the states a "
                f"delta file's rows are of; only a table that opts in to delta files writes one (delta_files=True)"
            )
        if not prefix:
            raise ValueError(f'TableSchema(name={self.name!r}): delta_state_scope needs a scope prefix ("state:")')
        if column not in pk_cols and column != self.delta_scope:
            raise ValueError(
                f"TableSchema(name={self.name!r}): delta_state_scope's column {column!r} is neither a key column nor "
                f"delta_scope, so a delete marker would not carry its state"
            )

    def _refuse_undecided(self) -> dict[str, Any]:
        """refuse a table opting in to delta files when the database decides part of what a write leaves: a column
        whose default it computes, a column a trigger maintains, or a guarded write it may skip. The file is written
        before the commit and cannot know the database's answer (design-pod-delta-files.md, decision 20), so such a
        table is refused where it is declared rather than given a file that guesses.

        :return: each column's fixed default, as the rail answers it (:attr:`delta_file_defaults`)
        :rtype: dict[str, Any]
        :raises ValueError: naming the table, the column or feature, and why
        """
        why = "a delta file is written before the commit and cannot know the database's answer"
        guards = [
            (self.cas_column is not None, f"cas_column={self.cas_column!r}", "a compare-and-swap fenced write"),
            (
                any(c.name in L2_ORDER_COLUMNS for c in self.columns),
                f"the L2 order columns {L2_ORDER_COLUMNS!r}",
                "an ordered upsert fenced on the L2 order",
            ),
        ]
        for guarded, feature, kind in guards:
            if guarded:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): delta_files=True with {feature}: {kind} applies only if its "
                    f"condition holds, which the database may skip; {why}"
                )
        defaults: dict[str, Any] = {}
        for column in self.columns:
            if column.column_type == TSVECTOR_TYPE:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): delta_files=True with column {column.name!r}: a tsvector is "
                    f"maintained by a database trigger, so the database computes its value; {why}"
                )
            if column.server_default is None:
                continue
            try:
                defaults[column.name] = _fixed_default(column)
            except _NotFixed as exc:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): delta_files=True with column {column.name!r} whose "
                    f"server_default {column.server_default!r} is not a fixed value ({exc}); {why}"
                ) from exc
        return defaults

    @property
    def delta_file_defaults(self) -> Mapping[str, Any]:
        """each column whose database default is a fixed value, and that value as the rail answers it, for a table
        that opts in to delta files: a delta file's row holds it for a column its write left out. Empty for a table
        that does not opt in.

        :return: column -> its fixed default
        :rtype: Mapping[str, Any]
        """
        return self._delta_defaults

    @property
    def declares_l2_order(self) -> bool:
        """whether this table carries the compare-and-swap order columns.

        :return: whether both ``l2_epoch`` and ``l2_revision`` are declared (validated as a pair)
        :rtype: bool
        """
        return L2_EPOCH_COLUMN in self._by_name

    def _validate_cas_null_safe(self, pk_cols: tuple[str, ...], by_name: dict[str, Column]) -> None:
        """reject a :attr:`cas_null_safe` opt-in whose fence could never work.

        the NULL-safe fence is only sound when the generated statement can
        (a) exist at all and (b) ADVANCE the fence column on every write, so
        the next writer's expected value stops matching. four preconditions
        carry that, and each one silently degrades to a lost write when
        violated -- hence a hard failure at construction rather than a
        surprise at 2am.

        :param pk_cols: primary-key column names in declared order
        :ptype pk_cols: tuple[str, ...]
        :param by_name: declared columns indexed by name
        :ptype by_name: dict[str, Column]
        :return: nothing
        :rtype: None
        :raises ValueError: when :attr:`cas_null_safe` is set and
            :attr:`cas_column` is unset, :attr:`on_conflict` is not
            ``"update"``, the ``cas_column`` is part of the primary key or
            declared ``immutable=True`` (the ``DO UPDATE SET`` clause would
            omit it, so the fence never advances), or the ``cas_column``
            declares a ``server_default`` (the INSERT generator drops an
            omitted server-default column, so the fence would silently
            vanish from the statement)
        """
        if not self.cas_null_safe:
            return
        if self.cas_column is None:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_null_safe=True requires cas_column to be set",
            )
        if self.on_conflict != "update":
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_null_safe=True requires "
                f"on_conflict='update', got {self.on_conflict!r} -- the fence lives "
                f"on the ON CONFLICT DO UPDATE branch, which the other modes do not emit",
            )
        cas_col = by_name[self.cas_column]
        if self.cas_column in pk_cols or cas_col.immutable:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_null_safe=True requires cas_column "
                f"{self.cas_column!r} to be a mutable non-pk column -- an immutable or pk "
                f"fence column is excluded from DO UPDATE SET, so the fence would never "
                f"advance and every later writer would match the same expected value",
            )
        if cas_col.server_default is not None:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_null_safe=True requires cas_column "
                f"{self.cas_column!r} to declare no server_default -- the INSERT generator "
                f"drops a server-default column the payload omits, which would drop the "
                f"fence out of the statement without saying so",
            )

    @property
    def pk_columns(self) -> KeyColumns:
        """return primary-key column names as a tuple (always tuple-shape).

        :return: pk column names in declared order
        :rtype: tuple[str, ...]
        """
        return self._pk_columns

    @property
    def stored_key(self) -> KeyColumns:
        """the columns the table stores rows by, its database primary key: :attr:`pk_columns`, then the epoch for a
        versioned table (:attr:`row_epoch`), whose key keeps a version a write. What an upsert's ``ON CONFLICT``
        names; every read and every cache keys rows by :attr:`pk_columns`.

        :return: the columns, in order
        :rtype: tuple[str, ...]
        """
        return self._stored_key

    @property
    def versioned(self) -> bool:
        """whether the table keeps a version a write, append-only (:attr:`row_epoch` is ``row_versions``): a write
        appends the key's new version and a delete appends a "gone" marker, never an upsert or a ``DELETE``.

        :return: whether it does
        :rtype: bool
        """
        return self.row_epoch is not None and self.row_epoch.versions

    @property
    def gone_marker_columns(self) -> tuple[str, ...]:
        """the columns a versioned table's "gone" marker carries: the key, the scope (when the key does not name it),
        the epoch and the gone flag. Empty for any other table.

        :return: the columns, in that order
        :rtype: tuple[str, ...]
        """
        if not self.versioned:
            return ()
        assert self.row_epoch is not None and self.row_epoch.scope is not None  # noqa: S101 -- narrowed by versioned
        scope = () if self.row_epoch.scope in self._pk_columns else (self.row_epoch.scope,)
        return (*self._pk_columns, *scope, EPOCH_COLUMN, GONE_COLUMN)

    @property
    def live_only_columns(self) -> tuple[str, ...]:
        """a versioned table's required columns a "gone" marker leaves out: every ``NOT NULL`` column but the key, the
        epoch, the gone flag and the scope. Nullable in the stored table, where one ``CHECK`` requires them of a live
        version (:meth:`create_statements`); empty for any other table.

        :return: the columns, in declared order
        :rtype: tuple[str, ...]
        """
        if not self.versioned:
            return ()
        assert self.row_epoch is not None  # noqa: S101 -- narrowed by versioned
        kept = {*self._pk_columns, EPOCH_COLUMN, GONE_COLUMN, self.row_epoch.scope}
        return tuple(column.name for column in self.columns if not column.nullable and column.name not in kept)

    @property
    def stamps_epoch(self) -> bool:
        """whether every write of the table stamps each row with its epoch (:attr:`row_epoch` declared).

        :return: whether it does
        :rtype: bool
        """
        return self.row_epoch is not None

    def create_statements(self, *, yugabyte: bool) -> tuple[str, ...]:
        """the DDL that creates the table in L3 and its indexes, as :meth:`to_sqlalchemy_table` declares them but stored
        by :attr:`stored_key`, for Postgres or YugabyteDB.

        A versioned table (:attr:`row_epoch`) is stored by its key and then the epoch, newest first: on YugabyteDB its
        primary key is written out as :func:`~threetears.core.ddl.row_epoch.versioned_primary_key_sql` writes it (the
        first key column hashed, ``epoch DESC`` the last range column), which SQLAlchemy's primary key cannot say.
        Every other table, and every table on Postgres, is SQLAlchemy's DDL as it renders it.

        :param yugabyte: whether the database is YugabyteDB
        :ptype yugabyte: bool
        :return: the ``CREATE TABLE``, then each ``CREATE INDEX``
        :rtype: tuple[str, ...]
        """
        # typed through Dialect: SQLAlchemy leaves the dialect classes' constructors unannotated
        dialect_class: type[Dialect] = postgresql.dialect
        dialect = dialect_class()
        table = self._sqlalchemy_table(sa.MetaData(), stored=True)
        create = str(CreateTable(table).compile(dialect=dialect))
        if yugabyte and self.row_epoch is not None and self.row_epoch.versions:
            quote = dialect.identifier_preparer.quote
            rendered = f"PRIMARY KEY ({', '.join(quote(name) for name in self._stored_key)})"
            if create.count(rendered) != 1:
                raise ValueError(f"TableSchema(name={self.name!r}): no one {rendered!r} in its DDL to order: {create}")
            ordered = versioned_primary_key_sql(
                tuple(quote(name) for name in self._pk_columns), quote(EPOCH_COLUMN), yugabyte=True
            )
            create = create.replace(rendered, ordered)
        indexes = tuple(self._index_statement(index, dialect, yugabyte) for index in table.indexes)
        return (create, *indexes, *self._restore_comment())

    def _index_statement(self, index: sa.Index, dialect: Dialect, yugabyte: bool) -> str:
        """an index's ``CREATE INDEX``; a stamped table's restore index split a tablet a bucket on YugabyteDB
        (:func:`~threetears.core.ddl.row_epoch.bucket_split_sql`: WHY there).

        :param index: the index
        :ptype index: sa.Index
        :param dialect: the dialect it is rendered in
        :ptype dialect: Dialect
        :param yugabyte: whether the database is YugabyteDB
        :ptype yugabyte: bool
        :return: the statement
        :rtype: str
        """
        statement = str(CreateIndex(index).compile(dialect=dialect))
        if yugabyte and self._stamped() and index.name == bucket_epoch_index_name(self.name):
            statement = f"{statement} {bucket_split_sql()}"
        return statement

    def _restore_comment(self) -> tuple[str, ...]:
        """a versioned or stamped table's ``COMMENT ON INDEX`` naming how a restore reads it (:func:`~threetears.core.ddl.
        row_epoch.restore_declaration`, :func:`~threetears.core.ddl.row_epoch.stamped_restore_declaration`); nothing
        for any other table.

        WHY in the catalog: the keeper that restores a lost change log runs in the hub, which imports no consumer's
        table classes; the comment rides with the index the read goes through, created and dropped with it.

        :return: the statement, or nothing
        :rtype: tuple[str, ...]
        """
        declared = self.row_epoch
        if declared is None:
            return ()
        instants = tuple(name for name in self._pk_columns if self._by_name[name].column_type == DATETIMETZ_TYPE)
        tenancy = (self.tenancy.kind, self.tenancy.column)
        if declared.versions:
            text = restore_declaration(declared, self._pk_columns, instants, tenancy)
            index = scope_epoch_index_name(self.name)
        else:
            text = stamped_restore_declaration(self._pk_columns, instants, tenancy)
            index = bucket_epoch_index_name(self.name)
        quoted = text.replace("'", "''")
        return (f"COMMENT ON INDEX {quote_identifier(index)} IS '{quoted}'",)

    def _live_check(self) -> Any:
        """a versioned table's ``CHECK`` that a live version holds every required column a "gone" marker leaves out
        (:attr:`live_only_columns`): ``gone OR (c IS NOT NULL AND ...)``, so the database still refuses a live version
        missing one, as the column's ``NOT NULL`` did.

        WHY a CHECK and not a marker carrying every column: a delete names the key (and the scope), never the row it
        removes; filling the rest would mean reading the old version back a key at a time, the read the append was
        made to avoid. A separate marker table would make every newest-per-key read a second probe; in the table, the
        marker is the key's first entry newest-first, which the read stops at.

        :return: the constraint
        :rtype: sqlalchemy.CheckConstraint
        """
        required = " AND ".join(f'"{name}" IS NOT NULL' for name in self.live_only_columns)
        name = f"{self.name}_live"
        if len(name) > 63:
            # a constraint name is bounded (NAMEDATALEN): a digest keeps it one per table
            name = f"live_{hashlib.sha256(self.name.encode()).hexdigest()[:24]}"
        return sa.CheckConstraint(f'"{GONE_COLUMN}" OR ({required})', name=name)

    def keyset_order(self, pinned: Iterable[str] = ()) -> tuple[str, ...]:
        """the columns a keyset read of the rows ``pinned``'s equality filters name pages on, so an index answers
        each page in that order.

        A keyset read pages ``ORDER BY`` the columns it is given (:mod:`threetears.core.keyset`). Ordered by the
        primary key, a read filtered on other columns is found through another index, and every page reads and
        sorts every row the filters leave. This picks the declared index (or the primary key) that the filters
        lead furthest: its leading columns all in ``pinned``, and every primary-key column in it, so its columns
        after that lead are unique among the read's rows. The answer is those columns, in the index's order. An
        index led by the filters but without the key in it cannot answer a page in order: declare the primary
        key's columns as its tail (``Index("t_by_state", "state", "geo", *primary_key)``).

        Only a plain ascending btree index is chosen: not a partial one (it holds only its ``where``'s rows), one
        of another access method, one with a descending column or an operator class. When no index is led by
        the filters (none pinned, or none such), the primary key. On a tie the primary key wins, then the index
        declared first.

        A set filter (``= ANY``) pins nothing: its column is read in order, as the index orders it.

        :param pinned: the columns the read's equality filters fix
        :ptype pinned: Iterable[str]
        :return: the columns to page on, unique among the rows the filters name
        :rtype: tuple[str, ...]
        """
        fixed = set(pinned)
        candidates = [self._pk_columns]
        candidates += [
            index.columns
            for index in self.indexes
            if index.where is None
            and index.using in (None, "btree")
            and not index.desc
            and not index.ops
            and set(self._pk_columns) <= set(index.columns)
        ]
        best, order = 0, self._pk_columns
        for columns in candidates:
            led = next((place for place, column in enumerate(columns) if column not in fixed), len(columns))
            # a lead that pins every column leaves nothing to page on: one row, read by the key
            if best < led < len(columns):
                best, order = led, tuple(columns[led:])
        return order

    def hashed_lead_array(self, order: Sequence[str]) -> str | None:
        """the SQL array type of ``order``'s first column when a keyset read on ``order`` pages a hashed lead: when
        ``order`` is the primary key (what :meth:`keyset_order` answers for a read no filter leads), whose first
        column YugabyteDB hashes (no declaration here names a range-sharded key). None for an order a filter leads,
        which pages inside the bucket of the pinned lead (:mod:`threetears.core.keyset`, "A hashed lead").

        :param order: the columns the read pages on
        :ptype order: Sequence[str]
        :return: the lead's array type (``text[]``), or None
        :rtype: str | None
        """
        if len(order) < 2 or tuple(order) != tuple(self._pk_columns):
            return None
        return f"{sql_cast(self.column(order[0]).column_type)}[]"

    @property
    def partition_column(self) -> str | None:
        """return the partition-column name, or ``None`` for non-partitioned tables.

        :return: partition column name when one column is flagged
            ``partition=True``, ``None`` otherwise
        :rtype: str | None
        """
        return self._partition_column

    def column(self, name: str) -> Column:
        """resolve a column descriptor by name.

        :param name: column name
        :ptype name: str
        :return: matching :class:`Column`
        :rtype: Column
        :raises KeyError: when no column with the given name exists
        """
        return self._by_name[name]

    def get_column(self, name: str) -> Column | None:
        """resolve a column descriptor by name, returning ``None`` on miss.

        non-raising variant of :meth:`column`. used by the read /
        deserialize paths when a row carries extra columns not
        declared in the schema (ad-hoc joined projections,
        migration-in-flight rows with stray fields) -- those pass
        through untouched rather than raising.

        :param name: column name
        :ptype name: str
        :return: matching :class:`Column` or ``None`` when not declared
        :rtype: Column | None
        """
        return self._by_name.get(name)

    def mutable_columns(self) -> list[Column]:
        """return non-pk, non-immutable columns in declared order.

        used by the upsert generator (``DO UPDATE SET``) and the CAS
        UPDATE generator.

        :return: list of columns eligible for update
        :rtype: list[Column]
        """
        pk_set = set(self._pk_columns)
        return [c for c in self.columns if not c.immutable and c.name not in pk_set]

    def to_sqlalchemy_table(self, metadata: Any) -> Any:
        """register this schema's canonical SQLAlchemy ``Table`` on ``metadata``.

        idempotent: if a table with :attr:`name` already exists on the
        passed metadata, returns the existing Table without
        re-registering.

        Its primary key is :attr:`pk_columns`, the key every cache holds one row of: a process's L1 is made from it.
        A versioned table's L3 table is stored by :attr:`stored_key` (its key, then the epoch), which
        :meth:`create_statements` renders.

        SQLAlchemy + pgvector imports are deferred inside the method
        body so non-Alembic / non-SQLAlchemy consumers of
        :mod:`threetears.core.collections.schema_backed` (e.g. the
        per-pod async-CRUD execution path) do not pay the import cost.

        :param metadata: SQLAlchemy ``MetaData`` instance to attach the
            table to
        :ptype metadata: sqlalchemy.MetaData
        :return: the registered :class:`sqlalchemy.Table`
        :rtype: sqlalchemy.Table
        :raises KeyError: when any column carries a ``column_type`` tag
            that is not registered in the type mapping (catches
            unknown-tag drift loudly rather than silently dropping the
            column)
        :raises ImportError: when the schema contains a VECTOR_TYPE
            column and ``pgvector`` is not installed (mirrors the
            v0.7.5 factory behaviour -- the failure is legible at
            registration time, not at first INSERT)
        """
        return self._sqlalchemy_table(metadata, stored=False)

    def _sqlalchemy_table(self, metadata: Any, *, stored: bool) -> Any:
        """:meth:`to_sqlalchemy_table`, its primary key :attr:`stored_key` when ``stored`` (the L3 table's) and
        :attr:`pk_columns` otherwise (a cache's).

        :param metadata: SQLAlchemy ``MetaData`` instance to attach the table to
        :ptype metadata: sqlalchemy.MetaData
        :param stored: whether the primary key is the stored key
        :ptype stored: bool
        :return: the registered :class:`sqlalchemy.Table`
        :rtype: sqlalchemy.Table
        """
        if self.name in metadata.tables:
            return metadata.tables[self.name]

        # tag → SQLAlchemy type constructor. each lambda receives the
        # :class:`Column` so vector-dim / numeric-precision / enum-value
        # parameters thread through cleanly without a second lookup.
        type_mapping: dict[str, Callable[[Column], Any]] = {
            UUID_TYPE: lambda col: PgUUID(as_uuid=True),
            STRING_TYPE: lambda col: Text(),
            DATETIMETZ_TYPE: lambda col: DateTime(timezone=True),
            JSONB_TYPE: lambda col: JSONB(),
            BYTES_TYPE: lambda col: BYTEA(),
            INT_TYPE: lambda col: Integer(),
            BIGINT_TYPE: lambda col: BigInteger(),
            BOOL_TYPE: lambda col: Boolean(),
            VECTOR_TYPE: lambda col: require_pgvector()(col.vector_dim),
            NUMERIC_TYPE: lambda col: Numeric(col.precision, col.scale),
            TSVECTOR_TYPE: lambda col: TSVECTOR(),
            ENUM_TYPE: lambda col: SAEnum(
                *(col.enum_type or ()),
                name=col.enum_name,
                create_constraint=True,
            ),
        }

        # the stored key: a versioned table's epoch is the last column of it, its constraint given in the key's own
        # order (below) so the epoch is last whatever the columns' order
        versioned = stored and self.row_epoch is not None and self.row_epoch.versions
        pk_set = set(self._stored_key if stored else self._pk_columns)
        live_only = frozenset(self.live_only_columns) if stored else frozenset()
        sa_columns: list[Any] = []
        for col in self.columns:
            if col.column_type not in type_mapping:
                raise KeyError(
                    f"TableSchema(name={self.name!r}): column "
                    f"{col.name!r} has unknown column_type "
                    f"{col.column_type!r}; no entry in to_sqlalchemy_table "
                    f"type mapping",
                )
            sa_type = type_mapping[col.column_type](col)
            is_pk = col.name in pk_set
            # PK columns are NOT NULL by definition; for non-PK columns
            # carry through the schema's :attr:`Column.nullable` flag.
            nullable = False if is_pk else col.nullable
            if col.name in live_only:
                # a "gone" marker leaves it out: required of a live version by the table's CHECK below
                nullable = True
            kwargs: dict[str, Any] = {
                "primary_key": is_pk and not versioned,
                "nullable": nullable,
            }
            if col.server_default is not None:
                # a raw Postgres expression (``now()``, ``'[]'::jsonb``): a plain string would be quoted as a literal
                kwargs["server_default"] = sa.text(col.server_default)
            if col.foreign_key is not None:
                ref_table, ref_col = col.foreign_key
                sa_columns.append(
                    sa.Column(
                        col.name,
                        sa_type,
                        sa.ForeignKey(f"{ref_table}.{ref_col}"),
                        **kwargs,
                    ),
                )
            else:
                sa_columns.append(sa.Column(col.name, sa_type, **kwargs))

        # composite (or supplementary single-column) FKs declared at
        # table level via :attr:`foreign_keys`. ``ondelete`` is omitted
        # when the action is "NO ACTION" so the emitted DDL matches the
        # hand-written factories byte-for-byte (SQLAlchemy treats
        # ``ondelete=None`` as "no clause"; passing the literal string
        # would force an explicit ``ON DELETE NO ACTION`` that the
        # factories don't emit).
        sa_fk_constraints: list[Any] = []
        for fk in self.foreign_keys:
            ondelete: str | None = fk.on_delete if fk.on_delete != "NO ACTION" else None
            sa_fk_constraints.append(
                sa.ForeignKeyConstraint(
                    list(fk.local_cols),
                    [f"{fk.ref_table}.{ref_col}" for ref_col in fk.ref_cols],
                    ondelete=ondelete,
                ),
            )

        # v0.8.1: emit ``sa.UniqueConstraint`` (NOT ``sa.Index(unique=True)``)
        # so Alembic auto-gen reads them out of
        # ``information_schema.table_constraints`` and matches the prod
        # ``ADD CONSTRAINT ... UNIQUE`` DDL byte-for-byte. See
        # :class:`UniqueConstraintDef` docstring for the distinction.
        sa_unique_constraints: list[Any] = [
            sa.UniqueConstraint(*uc.columns, name=uc.name) for uc in self.unique_constraints
        ]

        bucketed = stored and self._stamped()
        if bucketed:
            # the stamped table's restore bucket, derived by the database on every write (_declare_bucket_index: WHY)
            sa_columns.append(
                sa.Column(
                    BUCKET_COLUMN,
                    sa.SmallInteger(),
                    sa.Computed(self._bucket_expression(), persisted=True),
                    nullable=False,
                )
            )
        table = sa.Table(
            self.name,
            metadata,
            *sa_columns,
            *((sa.PrimaryKeyConstraint(*self._stored_key),) if versioned else ()),
            *(() if not live_only else (self._live_check(),)),
            *sa_fk_constraints,
            *sa_unique_constraints,
            info={} if self.l1_eviction is None else {L1_EVICTION_INFO: self.l1_eviction},
        )

        # indexes are built against the table's own columns so a ``desc``
        # column can be emitted as ``column.desc()``; an Index over bound
        # columns attaches itself to their table.
        for idx in self.indexes:
            if not bucketed and BUCKET_COLUMN in idx.columns and self._stamped():
                # the restore index of a stamped table is the L3 table's alone: a cache holds no bucket column
                continue
            idx_kwargs: dict[str, Any] = {
                "unique": idx.unique,
            }
            if idx.where is not None:
                idx_kwargs["postgresql_where"] = sa.text(idx.where)
            if idx.using is not None:
                idx_kwargs["postgresql_using"] = idx.using
            if idx.ops is not None:
                # convert the frozen tuple-of-pairs storage shape back
                # to the natural dict shape SA's dialect_kwargs expects.
                idx_kwargs["postgresql_ops"] = dict(idx.ops)
            if idx.pg_with is not None:
                idx_kwargs["postgresql_with"] = dict(idx.pg_with)
            if idx.nulls_not_distinct:
                idx_kwargs["postgresql_nulls_not_distinct"] = True
            if idx.include:
                idx_kwargs["postgresql_include"] = list(idx.include)
            collations = dict(idx.collate)

            def _item(col: str) -> Any:
                item = table.c[col].collate(collations[col]) if col in collations else table.c[col]
                return item.desc() if col in idx.desc else item.asc() if col in idx.asc else item

            sa.Index(idx.name, *(_item(col) for col in idx.columns), **idx_kwargs)

        return table


def require_pgvector() -> Any:
    """return pgvector's ``Vector`` column class.

    pgvector is imported with this module when the pgvector extra is
    installed; without it this raises, so the failure is legible at
    table-registration / collection-init, not deep inside an asyncpg INSERT.

    :return: the :class:`pgvector.sqlalchemy.Vector` class
    :rtype: type
    :raises ImportError: when pgvector is not installed
    """
    if not PGVECTOR_TYPES:
        raise ImportError(
            "pgvector is required to materialise VECTOR_TYPE columns; "
            "install ``pgvector`` or remove the column from the "
            "TableSchema",
        )
    return PGVECTOR_TYPES[0]


class PartitionEnforcementError(TypeError):
    """raised at class-definition time when a partitioned collection
    declares a public method that neither accepts the partition column
    nor opts into the cross-partition path via :func:`spans_partitions`.

    catches the cross-partition bleed class of bug at import time --
    long before a stray ``find_by_user`` lacking ``conversation_id``
    can ship to production and silently surface another conversation's
    rows.

    resolution: add the partition column to the method signature, mark
    the method with ``@spans_partitions``, or list it in
    ``cls.partition_exempt_methods`` with a ``# rationale: ...``
    inline comment. blanket exemptions ("internal helper", "tests need
    this") are rejected by code review.
    """


F = TypeVar("F", bound=Callable[..., Any])


@overload
def spans_partitions(method: F) -> F: ...


@overload
def spans_partitions(
    method: None = None,
    *,
    marker_only: bool = False,
) -> Callable[[F], F]: ...


def spans_partitions(
    method: F | None = None,
    *,
    marker_only: bool = False,
) -> F | Callable[[F], F]:
    """mark a Collection method as deliberately cross-partition.

    decorated methods are exempt from the
    :class:`SchemaBackedCollection` partition-column guard; they may
    span multiple partitions intentionally (e.g. an authorized
    cross-agent retrieval where the caller has resolved the authorized
    set of agent IDs upstream and passes them in as a tuple).

    contract: at call time, the decorator validates that the caller
    passed a ``tuple`` (not a list, not a single value, not a
    generator) for the partition-spanning argument. tuples are an
    explicit "I deliberately fan out across these partitions" signal;
    anything else is treated as a programming error and raises
    :class:`TypeError`. the wrapped callable inspects the method
    signature to find the first parameter whose name ends in ``_ids``,
    a plural form of the partition column. callers that need a
    different sentinel should override the convention by passing the
    ``tuple`` argument under any name ending in ``_ids``.

    when no ``_ids``-suffix parameter is found, decoration FAILS at
    class-definition time with :class:`PartitionEnforcementError`
    unless the caller explicitly passed ``marker_only=True``. the
    marker-only opt-in is a deliberate noise -- adding it to a method
    requires the author to justify why the method is structurally
    cross-partition by other means (e.g. an opaque pre-built SQL
    string where ACL is enforced upstream). this closes the silent-
    degrade failure mode flagged by review-task-01 finding D-2 in
    partition-hardening-task-01 sub-task 2.

    two calling forms are supported:

    - ``@spans_partitions`` (bare) -- method must declare an
      ``_ids``-suffix parameter; runtime guard validates tuple shape.
    - ``@spans_partitions(marker_only=True)`` -- method has no
      ``_ids`` parameter; runtime guard is skipped entirely; the
      decorator becomes a pure compile-time exemption marker against
      :meth:`SchemaBackedCollection.__init_subclass__`'s partition
      discipline.

    :param method: collection method to decorate (bare-decorator form)
    :ptype method: Callable[..., Any] | None
    :param marker_only: when ``True``, skip the ``_ids`` parameter
        scan + runtime guard; method becomes a pure compile-time
        exemption marker
    :ptype marker_only: bool
    :return: wrapped method, or a decorator factory when invoked with
        keyword arguments
    :rtype: Callable[..., Any] | Callable[[F], F]
    :raises PartitionEnforcementError: at decoration time when the
        target method has no ``_ids``-suffix parameter and
        ``marker_only=True`` was not passed
    """

    def _decorate(target: F) -> F:
        sig = inspect.signature(target)
        plural_param: str | None = None
        for name in sig.parameters:
            if name.endswith("_ids"):
                plural_param = name
                break
        if plural_param is None and not marker_only:
            raise PartitionEnforcementError(
                f"{target.__qualname__}: @spans_partitions found no "
                f"parameter ending in '_ids'. either rename the "
                f"partition-spanning parameter to end in '_ids' so the "
                f"runtime tuple-shape guard can validate the fan-out, "
                f"OR opt into the marker-only path with "
                f"@spans_partitions(marker_only=True) -- the latter "
                f"declares 'this method is structurally cross-partition "
                f"by other means' (e.g. opaque pre-built SQL with ACL "
                f"enforced upstream) and skips the runtime guard.",
            )

        @wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """call-site guard for ``@spans_partitions``-marked methods.

            :raises TypeError: when the cross-partition argument is not
                a ``tuple`` or has length zero
            """
            if plural_param is not None and not marker_only:
                bound = sig.bind(*args, **kwargs)
                value = bound.arguments.get(plural_param)
                if value is not None:
                    if not isinstance(value, tuple):
                        raise TypeError(
                            f"{target.__qualname__}: cross-partition "
                            f"argument {plural_param!r} must be a tuple "
                            f"(deliberate fan-out signal); got "
                            f"{type(value).__name__}",
                        )
                    if len(value) == 0:
                        raise TypeError(
                            f"{target.__qualname__}: cross-partition "
                            f"argument {plural_param!r} is an empty "
                            f"tuple; refusing to emit a query that "
                            f"spans zero partitions",
                        )
            return target(*args, **kwargs)

        setattr(wrapper, "_spans_partitions", True)
        return wrapper  # type: ignore[return-value]

    if method is not None:
        return _decorate(method)
    return _decorate


def _is_partition_exempt(method: Any) -> bool:
    """true iff ``method`` is decorated with :func:`spans_partitions`.

    :param method: function or method object
    :ptype method: Any
    :return: presence of the marker attribute
    :rtype: bool
    """
    return getattr(method, "_spans_partitions", False) is True


def _method_accepts_partition(method: Any, partition_column: str) -> bool:
    """true iff ``method`` declares a parameter named ``partition_column``.

    used by the :class:`SchemaBackedCollection` ``__init_subclass__``
    hook to verify every public method on a partitioned Collection
    accepts the partition column. matches positional and keyword-only
    parameters.

    :param method: function or method object
    :ptype method: Any
    :param partition_column: name of the table's partition column
    :ptype partition_column: str
    :return: presence of a parameter named ``partition_column``
    :rtype: bool
    """
    try:
        sig = inspect.signature(method)
    except TypeError, ValueError:
        return False
    return partition_column in sig.parameters


#: the Python type a row holds for each column type: what a codec rehydrates a column to, and what
#: a hand-written field-type map has to name for the column to agree with its schema
_COLUMN_PYTHON_TYPES: Mapping[str, Any] = {
    UUID_TYPE: UUID,
    STRING_TYPE: str,
    ENUM_TYPE: str,
    TSVECTOR_TYPE: str,
    NUMERIC_TYPE: Decimal,
    DATETIMETZ_TYPE: datetime,
    # any JSON value: an object, an array or a scalar, passed through as decoded
    JSONB_TYPE: object,
    BYTES_TYPE: bytes,
    INT_TYPE: int,
    BIGINT_TYPE: int,
    BOOL_TYPE: bool,
    VECTOR_TYPE: list,
}

#: the types a hand-written field-type map may name for a column type besides
#: :data:`_COLUMN_PYTHON_TYPES`'s: a ``jsonb`` column holds whichever JSON shape its rows carry
_ALSO_AGREEING: Mapping[str, frozenset[Any]] = {
    JSONB_TYPE: frozenset({dict, list, Any}),
}


def schema_field_types(schema: TableSchema) -> dict[str, Any]:
    """every column of ``schema`` with the Python type its rows hold, in declared order.

    :param schema: the table
    :ptype schema: TableSchema
    :return: column name to Python type (:data:`_COLUMN_PYTHON_TYPES`)
    :rtype: dict[str, Any]
    """
    return {column.name: _COLUMN_PYTHON_TYPES[column.column_type] for column in schema.columns}


def schema_datetime_columns(schema: TableSchema) -> frozenset[str]:
    """the columns of ``schema`` holding instants (``DATETIMETZ_TYPE``).

    :param schema: the table
    :ptype schema: TableSchema
    :return: their names
    :rtype: frozenset[str]
    """
    return frozenset(column.name for column in schema.columns if column.column_type == DATETIMETZ_TYPE)


def _field_type_disagreements(schema: TableSchema, declared: Mapping[str, Any]) -> list[str]:
    """how a hand-written field-type map differs from ``schema``, one line per difference.

    :param schema: the table
    :ptype schema: TableSchema
    :param declared: the hand-written map, column name to type hint (``UUID``, ``datetime | None``)
    :ptype declared: Mapping[str, Any]
    :return: the differences; empty when the map agrees
    :rtype: list[str]
    """
    found: list[str] = []
    for name in declared:
        if schema.get_column(name) is None:
            found.append(f"{name!r} is not a column of {schema.name!r}")
    for column in schema.columns:
        if column.name not in declared:
            found.append(f"{column.name!r} is missing")
            continue
        hint = declared[column.name]
        base = resolve_base_type(hint)
        agreeing = {_COLUMN_PYTHON_TYPES[column.column_type], *_ALSO_AGREEING.get(column.column_type, ())}
        if base not in agreeing:
            found.append(
                f"{column.name!r} is declared {hint!r}, but the schema's {column.column_type!r} column holds "
                f"{_COLUMN_PYTHON_TYPES[column.column_type].__name__}"
            )
    return found


def _restates_row_types(cls: type) -> bool:
    """whether ``cls`` itself declares its schema or a copy of the row types read off one.

    :param cls: the collection class being defined
    :ptype cls: type
    :return: ``True`` when its own namespace holds ``schema``, ``datetime_columns`` or ``field_types``
    :rtype: bool
    """
    return any(name in cls.__dict__ for name in ("schema", "datetime_columns", "field_types"))


def check_declared_row_types(cls: type, schema: TableSchema) -> None:
    """refuse a class whose hand-written row types disagree with the schema it declares.

    The schema says each column's type, so :attr:`~threetears.core.collections.base.BaseCollection.datetime_columns`
    and a ``field_types`` map are read off it. A class may still state either, and then each must
    say what the schema says: a copy kept beside the schema that is not checked is how a column
    added by a migration goes missing from it.

    :param cls: the collection class being defined
    :ptype cls: type
    :param schema: the table it declares
    :ptype schema: TableSchema
    :return: nothing
    :rtype: None
    :raises TypeError: when the class's own ``datetime_columns`` or ``field_types`` differ from the
        schema's, naming every difference
    """
    instants = schema_datetime_columns(schema)
    declared_instants = cls.__dict__.get("datetime_columns")
    if declared_instants is not None and frozenset(declared_instants) != instants:
        raise TypeError(
            f"{cls.__name__} declares datetime_columns {sorted(declared_instants)!r}, but its schema's instants "
            f"are {sorted(instants)!r}; the schema is the declaration, so drop datetime_columns"
        )
    declared_types = cls.__dict__.get("field_types")
    if declared_types is not None:
        found = _field_type_disagreements(schema, declared_types)
        if found:
            raise TypeError(
                f"{cls.__name__} declares field_types that disagree with its schema {schema.name!r}: "
                f"{'; '.join(found)}. the schema is the declaration, so drop field_types"
            )


class SchemaBackedCollection(BaseCollection[EntityT], Generic[EntityT]):
    """base collection that generates CRUD SQL from a :class:`TableSchema`.

    subclasses declare::

        class MyCollection(SchemaBackedCollection[MyEntity]):
            schema = TableSchema(
                name="my_table",
                primary_key="id",
                columns=[
                    Column("id", UUID_TYPE),
                    Column("name", STRING_TYPE),
                    Column("date_created", DATETIMETZ_TYPE, immutable=True),
                    Column("date_updated", DATETIMETZ_TYPE),
                ],
                cas_column="date_updated",
            )
            entity_class = MyEntity
            primary_key_column = "id"

    and inherit the full CRUD implementation:
    :meth:`fetch_from_store`, :meth:`save_to_store`,
    :meth:`delete_from_store`, :meth:`serialize`, :meth:`deserialize`.
    domain-specific query methods (e.g. ``find_by_agent``) stay on the
    subclass because query shape is per-collection.

    the L3 pool is read from :attr:`l3_pool` (the public extension
    point on :class:`BaseCollection`). subclasses that accept a
    ``postgres_pool`` constructor kwarg should assign it onto
    ``self.l3_pool`` in their ``__init__`` after calling ``super()``
    so the generic path finds the pool uniformly.

    **partition-column guard.** when :attr:`schema` declares a
    partition column (one :class:`Column` flagged ``partition=True``),
    every public read / write method on the subclass must satisfy at
    least one of:

    1. accept the partition column as a parameter (positional or
       keyword) -- the SQL generator will then enforce the
       partition predicate naturally.
    2. be decorated with :func:`spans_partitions` -- the caller is
       deliberately fanning out across an authorized set of
       partition values supplied as a tuple.
    3. appear in :attr:`partition_exempt_methods` -- a narrow,
       audited allowlist for methods that legitimately span every
       partition (e.g. ``count_total`` on a per-pod operational
       summary).

    a subclass that violates the guard fails at class-definition
    time with :class:`PartitionEnforcementError`. resolution priority:
    (1) add the partition column to the method signature; (2) decorate
    with :func:`spans_partitions`; (3) extend
    :attr:`partition_exempt_methods` with a documented rationale.

    :cvar partition_exempt_methods: public subclass extension point --
        names of public methods on this collection that legitimately
        span every partition and are therefore excused from the
        partition-column guard. subclasses extend it by declaring
        their own frozenset (union the base's value to inherit its
        entries), each addition carrying a specific rationale
        comment; blanket rationales are rejected by code review
    """

    #: the table this collection reads and writes. declared on the subclass; an instance may hold
    #: its own copy, set before ``__init__`` registers it, to vary what a deployment configures --
    #: an embedding column's width (:func:`threetears.core.data.embedding_width.schema_with_embedding_dim`).
    schema: TableSchema
    partition_exempt_methods: ClassVar[frozenset[str]] = frozenset()
    #: whether every L3 write this collection makes is a statement generated from :attr:`schema`,
    #: so :meth:`complete_written_row` may name, as ``None``, the nullable columns a write that
    #: reaches L2 before L3 leaves out: those statements write the ``None`` sent as ``NULL``.
    #: ``None`` infers it: ``True`` unless the subclass overrides :meth:`save_to_store`. A
    #: subclass whose override only wraps the generated write (calls ``super().save_to_store``
    #: for every row it writes) declares ``True``; one that writes its own SQL leaves it inferred.
    stores_through_generated_sql: ClassVar[bool | None] = None
    #: every column's Python type (:data:`_COLUMN_PYTHON_TYPES`), read off :attr:`schema` when the
    #: class is defined. A subclass may state it; it must then agree with the schema, or the class
    #: is refused (:func:`check_declared_row_types`).
    field_types: ClassVar[Mapping[str, Any]] = {}
    # lazily-built (underlying_pool, SqlL3Backend) cache for the case where a raw
    # transport is assigned directly to ``l3_pool`` (registry not in play); keeps the
    # on-demand wrapper stable across CRUD calls. ``None`` until first wrap.
    _durable_store_cache: tuple[Any, DurableStore] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """validate the partition-column contract at class-definition time.

        executed once per subclass at import time. when
        :attr:`schema` declares a partition column, every public
        async method that is not inherited from the framework must
        either accept the partition column as a parameter, be
        decorated with :func:`spans_partitions`, or be listed in
        :attr:`partition_exempt_methods`.

        framework-level methods inherited from
        :class:`BaseCollection` / :class:`SchemaBackedCollection`
        are exempt by construction (they thread the pk through
        :meth:`BaseCollection.normalize_pk` which already includes
        the partition column when the table has one). only methods
        defined directly on the subclass are checked.

        :param kwargs: PEP-487 keyword arguments forwarded to base
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises TypeError: when the class's own ``datetime_columns`` or ``field_types`` disagree
            with its schema (:func:`check_declared_row_types`)
        :raises PartitionEnforcementError: when a public subclass
            method violates the partition contract
        """
        schema = cls.__dict__.get("schema")
        effective = getattr(cls, "schema", None)
        if isinstance(effective, TableSchema) and _restates_row_types(cls):
            # the schema already says which columns hold instants; restating them as
            # ``datetime_columns`` is how one gets left out, so a restatement must agree. derived
            # before the base hook runs, which checks ``expires_at_column`` against this set.
            check_declared_row_types(cls, effective)
            cls.datetime_columns = schema_datetime_columns(effective)
            cls.field_types = schema_field_types(effective)
        super().__init_subclass__(**kwargs)
        if schema is None or not isinstance(schema, TableSchema):
            return None
        partition_column = schema.partition_column
        if partition_column is None:
            return None
        exempt = cls.partition_exempt_methods
        violations: list[str] = []
        for name, member in cls.__dict__.items():
            if not callable(member):
                continue
            if name.startswith("_"):
                continue
            if isinstance(member, (classmethod, staticmethod, property)):
                continue
            if name in exempt:
                continue
            if _is_partition_exempt(member):
                continue
            if _method_accepts_partition(member, partition_column):
                continue
            violations.append(name)
        if violations:
            raise PartitionEnforcementError(
                f"{cls.__name__}: schema declares partition column "
                f"{partition_column!r}; method(s) {violations!r} neither "
                f"accept it nor opt into @spans_partitions, and are not "
                f"in partition_exempt_methods. add the partition "
                f"column to the signature, decorate the method with "
                f"@spans_partitions, or extend partition_exempt_methods "
                f"with a documented rationale.",
            )
        return None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """construct the collection and register its schema with the L3 backend.

        :class:`BaseCollection.__init__` resolves :attr:`l3_pool` from the
        registry. When that backend is a
        :class:`~threetears.core.backends.sql.SqlL3Backend` (the default — the
        registry wraps a raw asyncpg pool / ``NatsProxyL3Backend`` in one), the
        schema is registered so the backend's structured
        :class:`~threetears.core.backends.protocol.DurableStore` ops emit
        schema-aware SQL byte-identical to the hand-rolled path. A backend that is
        not a ``SqlL3Backend`` (e.g. a git-backed ``DurableStore``, or no backend
        at all) needs no registration.

        :param args: positional args forwarded to :class:`BaseCollection`
        :ptype args: Any
        :param kwargs: keyword args forwarded to :class:`BaseCollection`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        self._reject_deferred_flush_on_cas_null_safe()
        self._register_schema()

    @property
    def writes_delta_files(self) -> bool:
        """whether every commit of this table saves a delta file before it commits: its schema opts in
        (:attr:`TableSchema.delta_files`).

        :return: whether it does
        :rtype: bool
        """
        return self.schema.delta_files

    def table_delta(self) -> TableDelta:
        """an empty delta of this table, its columns typed as the rail answers them, for a transaction to record its
        writes in.

        :return: the delta
        :rtype: TableDelta
        """
        return TableDelta(
            name=self.table_name,
            types=rail_arrow_types(self.schema.columns),
            key=self.schema.pk_columns,
            defaults=self.schema.delta_file_defaults,
            scope_column=self.schema.delta_scope,
            state_scope=self.schema.delta_state_scope,
        )

    def _refuse_unfiled(self, writer: str) -> None:
        """refuse a write to a table that opts in to delta files on a path that cannot record its rows for the file.

        :param writer: the refusing call
        :ptype writer: str
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table opts in
        """
        if self.writes_delta_files:
            raise DeltaFileRefusedError(
                f"{type(self).__name__}.{writer}: {self.table_name} opts in to delta files, and this write cannot "
                f"record its rows for one; write it by key on a CallerTransaction (save_rows, save_arrow, "
                f"delete_rows, delete_arrow, save_entity(conn=...))"
            )

    def _epoch_stamp(self, conn: Any, writer: str) -> dict[str, int]:
        """the epoch column one write statement stamps its rows with, for a table that declares a row epoch: a fresh
        epoch of the write the caller's transaction took its turn at (:meth:`CallerTransaction.row_epoch`). Empty for
        any other table.

        **A stamped table** (``ROW_STAMPED``) is stamped whether or not its write took a turn: on the caller's
        transaction from it (:meth:`CallerTransaction.in_place_epoch`, so its ``s_min`` and age refusal count the
        stamp), and a write that joins none (one statement, its own commit) from the process clock. WHY no turn: its
        epoch rises on every write (``GREATEST``), so newest-wins needs no fence (``in_place_epoch``: WHY); the hub's
        own update-in-place tables are written outside any seqlock writer, and every one of their writes is stamped.

        :param conn: the connection the write joins, or None
        :ptype conn: Any
        :param writer: the call writing, for the refusal
        :ptype writer: str
        :return: ``{"epoch": ...}``, or nothing
        :rtype: dict[str, int]
        :raises UnstampedWriteError: when the table is versioned and the write joins no transaction that took its
            write's turn
        """
        if not self.schema.stamps_epoch:
            return {}
        transaction = None if conn is None else CallerTransaction.enclosing(conn)
        if not self.schema.versioned:
            # stamped: in place, its epoch risen on every write, so no write's turn is needed (in_place_epoch: WHY)
            return {EPOCH_COLUMN: CallerTransaction.stamp_for(conn, self.table_name)}
        if transaction is None:
            raise UnstampedWriteError(
                f"{type(self).__name__}.{writer}: {self.table_name} declares a row epoch, so every write carries its "
                f"write's epoch, and this one joins no transaction to take it from: write it on a CallerTransaction "
                f"that took its write's turn (ScopeEpochs.touch)"
            )
        return {EPOCH_COLUMN: transaction.row_epoch(self.table_name)}

    def stamped_values(self, values: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any]:
        """``values`` for a keyed ``UPDATE`` of this table, stamped (:func:`stamped_values`).

        :param values: column -> its new value, as ``update_keyed`` takes them
        :ptype values: Mapping[str, Any]
        :param conn: the connection the update runs on, or None for the pool
        :ptype conn: Any
        :return: the values, the epoch's term added for a stamped table
        :rtype: dict[str, Any]
        :raises AppendOnlyWriteError: when the table is versioned (append-only: never updated in place)
        """
        return stamped_values(self.schema, values, conn=conn)

    def _record_one(self, data: dict[str, Any], conn: Any, writer: str, *, fenced: bool = False) -> None:
        """record one row a store write sends, and the columns its statement sets in a row already held
        (:func:`~threetears.core.backends.schema_sql.upsert_set_columns`), for the delta file of a table that opts
        in: on the caller's transaction, or refused when there is none to save the file before.

        :param data: the row as sent
        :ptype data: dict[str, Any]
        :param conn: the connection the write joins, or None
        :ptype conn: Any
        :param writer: the call writing it
        :ptype writer: str
        :param fenced: the store writes the fenced ``UPDATE`` of a row the caller read
        :ptype fenced: bool
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table opts in and the write joins no CallerTransaction
        """
        if not self.writes_delta_files:
            return
        transaction = None if conn is None else CallerTransaction.enclosing(conn)
        if transaction is None:
            self._refuse_unfiled(writer)
        else:
            transaction.record_rows(self, [data], upsert_set_columns(self.schema, data, fenced=fenced))

    def _bulk_written(
        self, rows: Sequence[Mapping[str, Any]], store: Any, *, copy: bool
    ) -> list[str] | list[list[str]]:
        """the columns :meth:`save_rows`' statements set in a row already held, for a table that opts in to delta
        files: a bulk statement's one ``SET`` list (every mutable column its first row's columns emit,
        :func:`~threetears.core.backends.schema_sql.bulk_update_columns`), none for a copy (it inserts and never
        updates), or each row's own when the store upserts a row at a time.

        :param rows: the rows, stamped
        :ptype rows: Sequence[Mapping[str, Any]]
        :param store: the durable store writing them
        :ptype store: Any
        :param copy: the rows are copied, never upserted
        :ptype copy: bool
        :return: one list for every row, or one a row
        :rtype: list[str] | list[list[str]]
        """
        if not self.writes_delta_files or not rows or copy:
            return []
        if isinstance(store, BulkDurableStore):
            return bulk_update_columns(self.schema, insert_columns_for_data(self.schema, dict(rows[0]))) or []
        return [upsert_set_columns(self.schema, row) for row in rows]

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        """save ``entity`` through the three-tier write path, refusing a vector its column cannot hold.

        see :meth:`BaseCollection.save_entity`. a ``vector(N)`` column holds only N-dimensional
        vectors, so a row carrying one of another width is refused here, before any tier takes it
        or any statement runs (:func:`_refuse_misfit_vectors`); the database's own refusal would
        come only once the L3 write ran, and a write-behind collection's L2 would hold the row by
        then.

        :param entity: entity instance to persist
        :ptype entity: BaseEntity
        :param conn: see :meth:`BaseCollection.save_entity`
        :ptype conn: Any
        :return: nothing
        :rtype: None
        :raises ValueError: when a vector is not its column's declared width, and as
            :meth:`BaseCollection.save_entity` raises it
        """
        if any(col.column_type == VECTOR_TYPE for col in self.schema.columns):
            _refuse_misfit_vectors(self.schema, entity.to_dict())
        await super().save_entity(entity, conn=conn)

    @property
    def emits_cas_fence(self) -> bool:
        """whether every generated L3 write carries a CAS fence (``cas_null_safe``).

        see :attr:`BaseCollection.emits_cas_fence` for what the framework does
        with the answer.

        :return: :attr:`TableSchema.cas_null_safe`
        :rtype: bool
        """
        return bool(self.schema.cas_null_safe)

    @property
    def persists_l2_order(self) -> bool:
        """whether this table carries the order columns and its L3 backend can write fenced on them.

        see :attr:`BaseCollection.persists_l2_order`. ``True`` when the schema declares
        :func:`l2_order_columns` and the durable store implements
        :class:`~threetears.core.backends.protocol.OrderedDurableStore` (the SQL backend does).
        A collection with no L3 backend answers from its schema alone: nothing is persisted there.

        :return: whether ``l2_cas_mutate`` can persist to this collection in order
        :rtype: bool
        """
        store = self._durable_store()
        return self.schema.declares_l2_order and (store is None or isinstance(store, OrderedDurableStore))

    def complete_written_row(self, data: dict[str, Any]) -> dict[str, Any]:
        """``data`` with each nullable column a write ahead of L3 leaves out named as ``None``.

        see :meth:`BaseCollection.complete_written_row`: consulted only by a write that reaches L1
        and L2 before L3, which sends L3 the completed row. a nullable, mutable, non-key column with
        no ``server_default`` that ``data`` leaves out is named ``None``, and every statement this
        collection generates writes a named ``None`` as ``NULL`` -- the INSERT, its ``DO UPDATE
        SET`` and the fenced UPDATE alike -- so L3 stores the row the tiers cache. (a synchronous
        write is never completed: its update rewrites only the columns the row names, and a column
        it leaves out keeps its stored value, read back for the caches.)

        only while this collection's L3 writes are the generated statements
        (:attr:`stores_through_generated_sql`): a subclass that writes its own SQL may not write a
        ``None`` it is sent, so nothing is completed for it, and such a write is refused.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: ``data`` with those columns present as ``None``; ``data`` itself when none is missing
            or this collection writes its own SQL
        :rtype: dict[str, Any]
        """
        generated = self.stores_through_generated_sql
        if generated is None:
            generated = type(self).save_to_store is SchemaBackedCollection.save_to_store
        known_nulls = [
            col.name
            for col in self.schema.columns
            if generated
            and col.name not in data
            and col.nullable
            and not col.immutable
            and col.server_default is None
            and col.name not in self.schema.pk_columns
        ]
        completed = {**data, **dict.fromkeys(known_nulls)} if known_nulls else data
        return completed

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the declared columns whose stored value writing ``data`` leaves to the database.

        see :meth:`BaseCollection.columns_decided_by_store` for what the framework does with the
        answer. two shapes, both read off the schema:

        * a declared column ``data`` does not name. the generated INSERT drops it when it has a
          ``server_default`` and the database fills the default in; any update of an existing row
          keeps its stored value (the generated statements rewrite only the columns the row
          names); a statement this collection writes itself may keep any of them. none of those
          is knowable from ``data``. (a write ahead of L3 asks after
          :meth:`complete_written_row`, so a nullable column it sends as ``None`` is present.)
        * a ``NOT NULL`` ``server_default`` column ``data`` names as ``None``. the fenced UPDATE
          skips it and keeps the stored value (:func:`~threetears.core.backends.schema_sql
          .cas_mutable_columns_for_data`).

        conservative by construction: a column it names costs one read of the row after the
        commit, while a column it missed would be served wrong from every tier.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the names of the columns the database decides, in declared order
        :rtype: tuple[str, ...]
        """
        decided: list[str] = []
        for col in self.schema.columns:
            if col.name not in data:
                decided.append(col.name)
            elif col.server_default is not None and not col.nullable and data[col.name] is None:
                decided.append(col.name)
        return tuple(decided)

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        """persist a compare-and-swap row fenced on its order, via the ordered durable store.

        see :meth:`BaseCollection.save_ordered_to_store` for the contract. the SQL backend emits
        ``INSERT ... ON CONFLICT (pk) DO UPDATE ... WHERE`` the stored order is strictly older
        than the incoming one, ``NULL`` counting as older than every order.

        :param data: row payload, order columns included
        :ptype data: dict[str, Any]
        :param conn: optional asyncpg-compatible connection the write joins
        :ptype conn: Any
        :return: rows affected: 1 when written, 0 when the stored order is newer or equal
        :rtype: int
        :raises TypeError: when the schema carries no order columns, or the durable store cannot
            write fenced on them
        """
        store = self._durable_store()
        if not self.schema.declares_l2_order or not isinstance(store, OrderedDurableStore):
            raise TypeError(
                f"{type(self).__name__}: table {self.schema.name!r} cannot take an ordered write -- it needs "
                f"l2_order_columns() in its schema and a durable store implementing OrderedDurableStore, "
                f"got store {type(store).__name__}"
            )
        data = {**data, **self._epoch_stamp(conn, "save_ordered_to_store")}
        self._record_one(data, conn, "save_ordered_to_store")
        return await store.upsert_ordered(self.schema.name, data, conn=conn)

    def _reject_deferred_flush_on_cas_null_safe(self) -> None:
        """fail construction when a fenced table is also configured for deferred flush.

        the two are mutually exclusive by construction, and the failure mode is
        silent. a deferred write never reaches ``save_to_store`` from
        :meth:`BaseCollection.save_entity` at all -- it lands in the write
        buffer and is replayed later through ``persist_to_store``, which
        supplies NO ``original_timestamp``. on a ``cas_null_safe`` table that
        replay fences against ``NULL``, so it matches only a row nobody has
        written since; against any row that DOES exist it affects 0 rows, which
        the flush loop reports and evicts rather than retries. every deferred
        write to an existing row would evaporate with nothing raised.

        cheap to detect at construction, so detect it there.

        :return: nothing
        :rtype: None
        :raises ValueError: when the schema declares ``cas_null_safe=True`` and
            this collection's table is configured for deferred flush
        """
        if not self.schema.cas_null_safe:
            return
        deferred = (
            self._flush_strategy != FlushStrategy.ALWAYS
            and self.table_name in self._flush_tables
            and self._write_buffer is not None
        )
        if deferred:
            raise ValueError(
                f"{type(self).__name__}: table {self.table_name!r} declares "
                f"cas_null_safe=True but is also listed in collection_flush_tables "
                f"with collection_flush={self._flush_strategy.value!r}. a deferred "
                f"write is replayed without the CAS fence value it was decided "
                f"under, so every deferred write to an existing row would be "
                f"silently dropped. remove the table from collection_flush_tables, "
                f"or drop cas_null_safe and fence in hand-written SQL.",
            )

    def _register_schema(self) -> None:
        """register :attr:`schema` with the resolved ``SqlL3Backend``, if any.

        a non-``SqlL3Backend`` backend (or ``None``) is left untouched.

        :return: nothing
        :rtype: None
        """
        backend = self.l3_pool
        if isinstance(backend, SqlL3Backend):
            backend.register_schema(self.schema.name, self.schema)

    def _durable_store(self) -> DurableStore | None:
        """return the L3 backend as a :class:`DurableStore`, or ``None`` when absent.

        The structured CRUD lifecycle routes through this seam. The resolved
        :attr:`l3_pool` is normally a :class:`~threetears.core.backends.sql.SqlL3Backend`
        (the registry wraps a raw pool) or another ``DurableStore`` (e.g. scriob's
        ``GitL3Backend``) — returned as-is. A raw transport assigned directly to
        ``l3_pool`` (bypassing the registry — e.g. a test harness) is wrapped in a
        schema-registered ``SqlL3Backend`` on demand, cached so the wrapper is stable
        and re-used until the underlying pool changes.

        :return: the backend typed as :class:`DurableStore`, or ``None``
        :rtype: DurableStore | None
        """
        backend = self.l3_pool
        if backend is None:
            return None
        if isinstance(backend, DurableStore):
            return backend
        # raw transport assigned directly (registry not in play): wrap + register
        # the schema once, cache by the underlying pool's identity so the wrapper
        # is stable across calls and rebuilds only if the pool is reassigned.
        cached = self._durable_store_cache
        if cached is not None and cached[0] is backend:
            return cached[1]
        wrapper = SqlL3Backend(backend)
        wrapper.register_schema(self.schema.name, self.schema)
        self._durable_store_cache = (backend, wrapper)
        return wrapper

    # --- BaseCollection contract implementations ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one row from L3 by primary key via the :class:`DurableStore` seam.

        accepts scalar ``entity_id`` for single-pk tables or a tuple of
        pk values for composite-pk tables; shape is validated by
        :meth:`BaseCollection.normalize_pk`. the schema-aware backend emits the
        declared-columns-only SELECT (``VECTOR``/``TSVECTOR`` cast ``::text``) and
        read-coerces the row.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        store = self._durable_store()
        if store is None:
            return None
        pk_values = self.normalize_pk(entity_id)
        pk = dict(zip(self.schema.pk_columns, pk_values, strict=True))
        return await store.fetch_one(self.schema.name, pk)

    def _l3_pages(self, *, page_size: int) -> AsyncIterator[list[dict[str, Any]]]:
        """every row of the table, in key order, a page per statement, read-coerced as :meth:`fetch_from_store` reads one.

        :param page_size: rows per page; ``page_size + 1`` must not pass the transport's cap
        :ptype page_size: int
        :return: the pages
        :rtype: AsyncIterator[list[dict[str, Any]]]
        :raises NotImplementedError: when the collection has no L3 pool
        """
        pool = self.l3_pool
        if pool is None:
            raise NotImplementedError(f"{type(self).__name__} has no L3 to read its rows from")
        schema = self.schema
        pages = whole_read(
            keyset_pages(
                pool,
                f"SELECT {build_select_column_list(schema)} FROM {schema.name}",  # noqa: S608 - the declared schema
                list(schema.pk_columns),
                page_size=page_size,
                lead_array=schema.hashed_lead_array(schema.pk_columns),
            ),
            kind="cold_start",
            reason="L2 holds nothing for this table (its first fill, or NATS lost the memory bucket and the complete mark with it)",
            ruling="Pace 2026-11-12: the one full load is a true no-data cold start",
            proof="packages/epoch/tests/unit/test_l2_first.py::test_the_refill_reads_l3_only_while_l2_holds_no_data",
            speed="packages/epoch/tests/integration/test_l2_refill_speed_live.py::test_refill_from_l3_speed_at_real_size",
        )

        async def coerced() -> AsyncIterator[list[dict[str, Any]]]:
            async for page in pages:
                if len(page) * len(schema.columns) > _COERCE_ON_THREAD_VALUES:
                    yield await asyncio.to_thread(_coerced_page, schema, page, asyncio.get_running_loop())
                else:
                    yield _coerced_page(schema, page, None)

        return coerced()

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch one row on a caller's open transaction, through the :class:`DurableStore` seam.

        See :meth:`BaseCollection.fetch_from_store_in_transaction`. The same declared-columns
        SELECT and read coercion as :meth:`fetch_from_store`, bound to ``conn`` so it sees the
        transaction's uncommitted writes. A subclass that overrides :meth:`fetch_from_store` reads
        its own way, which this cannot know, so it reads nothing here unless it overrides this too.

        :param entity_id: pk value (single-pk) or tuple of pk values (composite-pk)
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit; ``None`` on miss, with no durable store, or when a subclass reads
            its own way
        :rtype: dict[str, Any] | None
        """
        store = self._durable_store()
        if store is None or type(self).fetch_from_store is not SchemaBackedCollection.fetch_from_store:
            return None
        pk_values = self.normalize_pk(entity_id)
        pk = dict(zip(self.schema.pk_columns, pk_values, strict=True))
        return await store.fetch_one(self.schema.name, pk, conn=conn)

    #: the most rows one bulk statement writes: well inside a broker's statement timeout for a
    #: row of a few dozen columns
    BULK_MAX_ROWS: ClassVar[int] = 1000

    #: the most bytes one bulk statement's whole request comes to -- its SQL text, its parameters
    #: and the envelope around them, measured by the transport that sends it
    #: (:func:`~threetears.core.backends.schema_sql.bulk_insert_batches`): under a NATS
    #: deployment's default 1 MiB message, with room to spare
    BULK_MAX_BYTES: ClassVar[int] = 768 * 1024

    #: the most leading-key values one key-led read names: a value usually holds a few rows, so a
    #: batch's answer stays under the L3 rail's row cap
    #: (:data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP`) and is rarely read again in halves
    LED_READ_MAX_VALUES: ClassVar[int] = 250

    #: the most rows one transaction of a load from files (:meth:`load_files`) commits: a load of
    #: millions of rows commits as it goes, so no one transaction holds them all
    LOAD_ROWS_PER_TRANSACTION: ClassVar[int] = 50_000

    #: the most time one transaction of a load from files may take, in milliseconds. A binary
    #: ``COPY`` of :attr:`LOAD_ROWS_PER_TRANSACTION` rows takes seconds, so a minute is generous
    #: headroom, and it lets one ``l3.load_files`` request carry 59 transactions inside the pod's
    #: wait (:data:`~threetears.core.backends.protocol.LOAD_FILES_MAX_WAIT_SECONDS`)
    LOAD_TIMEOUT_MS: ClassVar[int] = 60_000

    async def save_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        conn: Any,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        copy: bool = False,
    ) -> int:
        """upsert many rows on a caller's transaction, through the collection's durable store.

        For a load, where :meth:`save_entity` would make a round trip per row. The rows go to the
        store's :meth:`~threetears.core.backends.protocol.BulkDurableStore.upsert_many` when it has
        one -- a SQL store writes one ``INSERT ... VALUES (...), (...) ON CONFLICT`` per batch, each
        batch within ``max_rows`` rows, the bind-parameter limit and a request of ``max_bytes`` --
        and a row at a time through ``upsert`` when it has not.

        **On the caller's transaction only, like** ``save_entity(conn=...)``: the rows are final
        when the caller's :class:`CallerTransaction` commits, and a write that fails fails the
        transaction, so every batch rolls back with it. Each row's key leaves this process's L1 now
        and is enrolled; when the transaction ends the keys are settled together, one call for the
        collection (:meth:`~threetears.core.collections.base.BaseCollection.invalidate_cache_many`):
        a collection with no bus does no L2 or broadcast work at all, and one with a bus settles its
        keys side by side rather than one after another. The cost that remains with a bus is two
        messages a key (its L2 delete and its broadcast), because the broadcast's envelope names one
        entity and a peer of an older release reads no other shape.

        Each row is stamped ``date_created`` and ``date_updated`` (when the schema has them) with
        the write's time; an upsert of a row already held keeps its ``date_created``, an immutable
        column the conflict clause does not set. No CAS fence applies: a bulk write replaces.

        **With** ``copy=True`` **the write inserts and never updates**, for a load into a table just
        emptied (:meth:`truncate`): the rows go to the store's
        :meth:`~threetears.core.backends.protocol.CopyingStore.copy_many` -- ``COPY``, the rows ordered
        by key, through ``l3.tx.copy`` on the L3 rail -- and a row at a time through ``upsert`` with
        ``on_conflict="raise"`` to a store without it. A key already held fails the write, and so the
        caller's transaction. Everything else -- the checks, the stamps, the keys leaving L1 and
        settling with the transaction -- is as for an upsert.

        :param rows: the rows, each keyed by column; a nullable column may be absent
        :ptype rows: list[dict[str, Any]]
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :param max_rows: the most rows one statement writes; :attr:`BULK_MAX_ROWS` when None
        :ptype max_rows: int | None
        :param max_bytes: the most bytes one statement's whole request comes to on the L3 rail;
            :attr:`BULK_MAX_BYTES` when None
        :ptype max_bytes: int | None
        :param copy: insert with ``COPY`` rather than upsert; a key already held fails the write
        :ptype copy: bool
        :return: how many rows were submitted: every row given, whether the conflict clause updated
            a row already held, left it as it was (``on_conflict="ignore"``) or raised
        :rtype: int
        :raises ValueError: when no :class:`CallerTransaction` is open on ``conn``; when the
            collection has no durable store, caches absences, defers its L3 writes or fences with a
            null-safe CAS; when two rows share a key (one statement cannot upsert a key twice); when
            the rows disagree on which server-default columns they supply
        :raises KeyError: when a row lacks a key column, naming the table and the column
        """
        transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.save_rows")
        store = self._durable_store()
        if store is None:
            raise ValueError(f"{type(self).__name__}.save_rows: {self.table_name} has no durable store to write to")
        if self._negative_cache_writes_advance or self._defers_l3_writes or self.schema.cas_null_safe:
            raise ValueError(
                f"{type(self).__name__}.save_rows: a bulk upsert cannot keep this collection's write "
                f"contract (it caches absences, defers its L3 writes, or fences with a null-safe CAS)"
            )
        now = datetime.now(UTC)
        stamps: dict[str, Any] = {
            name: now for name in ("date_created", "date_updated") if self.schema.get_column(name) is not None
        }
        # one epoch for the statement's rows: one call writes a key once (refused below otherwise)
        stamps.update(self._epoch_stamp(conn, "save_rows"))
        # each row checked, stamped and keyed, and its enrollment made, on a worker thread taking turns with the loop,
        # from the count where a thread costs less than the work: on the loop, 33,786 rows of 179 columns held it 87-103 ms with the
        # collector held (the WD2 bench, 2026-11-16)
        stamped, keys, enrollments = (
            await asyncio.to_thread(self._stamped_rows, rows, stamps, asyncio.get_running_loop())
            if len(rows) >= self.L1_EVICT_OFF_LOOP_KEYS
            else self._stamped_rows(rows, stamps)
        )
        # enrolled before the write, so a write whose outcome is unknown is settled with the rest
        transaction.enroll_all(enrollments)
        transaction.record_rows(self, stamped, self._bulk_written(stamped, store, copy=copy))
        # every key leaves L1 in one write before the store's, not one write a key
        await self._evict_l1_many_off_loop(keys)
        written = 0
        if copy:
            written = await self._copy_rows(store, stamped, conn=conn, max_rows=max_rows, max_bytes=max_bytes)
        elif stamped and self._appends_by_copy(store, stamped[0]):
            # a versioned table's rewrite is an append of the keys' new versions: a COPY, never an upsert
            # (TableSchema.versioned). WHY: an upsert into a table with a row trigger reads and flushes a row at a time
            # on YugabyteDB, 1.58 ms a row against 0.09 for a copy (append-only L3, scratch copies of
            # latest_results_candidates_by_id, 2026-11-25)
            written = await self._copy_rows(
                store,
                stamped,
                conn=conn,
                max_rows=max_rows or self.BULK_MAX_ROWS,
                max_bytes=max_bytes or self.BULK_MAX_BYTES,
                append=True,
            )
        elif stamped and isinstance(store, BulkDurableStore):
            written = await store.upsert_many(
                self.table_name,
                stamped,
                max_rows=max_rows or self.BULK_MAX_ROWS,
                max_bytes=max_bytes or self.BULK_MAX_BYTES,
                conn=conn,
            )
        elif stamped:
            # a store that saves one row at a time: still one transaction, a write per row
            for data in stamped:
                await store.upsert(
                    self.table_name, data, pk=self.schema.pk_columns, on_conflict=self.schema.on_conflict, conn=conn
                )
                written += 1
        log.debug(
            "bulk copy written on the caller's transaction"
            if copy
            else "bulk upsert written on the caller's transaction",
            extra={"extra_data": {"table": self.table_name, "rows": written}},
        )
        return written

    def _stamped_rows(
        self,
        rows: Sequence[dict[str, Any]],
        stamps: Mapping[str, Any],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any], list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]]:
        """:meth:`save_rows`' rows checked and stamped, each one's key, and their enrollments; pure, for a thread.

        :param rows: the rows, each keyed by column
        :ptype rows: Sequence[dict[str, Any]]
        :param stamps: the write's stamp columns and their value: its time, and its epoch for a table that declares one
        :ptype stamps: Mapping[str, Any]
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
            or off any loop (a few rows: no turn taken)
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the stamped rows, their keys (a value, or a tuple for a key of many columns) and their enrollments
            (:meth:`CallerTransaction.enrollments`), in row order
        :rtype: tuple[list[dict[str, Any]], list[Any], list[tuple[BaseCollection, Any, dict[str, Any] | None]]]
        :raises KeyError: when a row lacks a key column, naming the table and the column
        :raises ValueError: when a row's vector is not its column's width, or two rows share a key
        """
        key_columns = self.schema.pk_columns
        # the vector check reads every column of every row; a table with no vector column has nothing to check
        vectors = any(col.column_type == VECTOR_TYPE and col.vector_dim is not None for col in self.schema.columns)
        turns = LoopTurns(loop) if loop is not None else None
        stamped: list[dict[str, Any]] = []
        keys: list[Any] = []
        for index, row in enumerate(rows):
            if turns is not None and index % STEP_ITEMS == 0:
                turns.take()
            missing = [name for name in key_columns if name not in row]
            if missing:
                raise KeyError(f"{self.table_name}.save_rows: row {index} has no key column {missing[0]!r}")
            if vectors:
                _refuse_misfit_vectors(self.schema, row)
            data = {**row, **stamps}
            stamped.append(data)
            key = tuple(data[name] for name in key_columns)
            keys.append(key if len(key) > 1 else key[0])
        if _shares_a_key(keys, turns):
            raise ValueError(f"{type(self).__name__}.save_rows: two rows share a key; one statement upserts a key once")
        return stamped, keys, CallerTransaction.enrollments(self, keys, stamped, turns)

    async def save_arrow(self, rows: Any, *, conn: Any, copy: bool = False, max_rows: int | None = None) -> int:
        """:meth:`save_rows` for rows already in an Arrow table: upserted (or, with ``copy``, inserted) as they are,
        through the store's Arrow write door, never a Python value a row.

        The same write as ``save_rows(rows.to_pylist(), conn=conn, copy=copy)``: the same checks, the same stamps
        (``date_created`` and ``date_updated``, the write's time, added as constant Arrow columns), the same columns
        and conflict clause, every key leaving L1 now and enrolled to settle with the caller's transaction, a key
        twice refused. Only the key columns (and the collection's :attr:`invalidation_columns`) become Python, for
        the duplicate check, the enrolment and the L1 eviction. The rows go to the hub as Arrow, cut by Arrow slices
        against the request budget (:meth:`~threetears.core.backends.sql.SqlL3Backend.write_arrow`); the hub stages
        them and merges them in one statement a request, so row triggers and change notes fire as for the bulk
        ``INSERT ... VALUES``. An upsert's request carries at most :attr:`BULK_MAX_ROWS` rows, as :meth:`save_rows`'s
        statement does; a copy's is cut by its bytes alone, as :meth:`save_rows` cuts a copy on the rail
        (:func:`~threetears.core.backends.protocol.copy_bounds`); ``max_rows`` bounds either.

        WHY an upsert is bounded by rows (2026-11-21): the hub merges a request as ONE statement under the
        transaction's statement timeout (the rail's five seconds), and a staged upsert costs ~1.16 ms a row on
        YugabyteDB at 179 columns (the hub's ``_stage_rows`` table: 300 rows 344.7 ms, 64 rows 70.3 ms), so a
        5,000-row page cut by bytes alone was one ~5.8 s statement, over the timeout on an idle database; the
        election replay's page retried as upserts was cancelled so (``QueryCanceledError``). :attr:`BULK_MAX_ROWS`
        keeps it ~1.2 s.

        **Falls back to** :meth:`save_rows` (the rows read into Python a slice at a time on a thread, today's cost)
        when the store has no Arrow write door or declines these rows
        (:meth:`~threetears.core.backends.sql.SqlL3Backend.arrow_write_plan`): a direct connection, a hub that
        announced no ``write_arrow``, a jsonb or vector column, a column type the door does not carry, a table whose
        conflict clause is ``raise``.

        :param rows: the rows, a column each; a nullable column may be absent
        :ptype rows: pyarrow.Table
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :param copy: insert with ``COPY`` rather than upsert; a key already held fails the write
        :ptype copy: bool
        :param max_rows: the most rows one request writes, at least one; when None, :attr:`BULK_MAX_ROWS` for an
            upsert and the request's bytes alone for a copy
        :ptype max_rows: int | None
        :return: how many rows were submitted
        :rtype: int
        :raises ValueError: as :meth:`save_rows`: no :class:`CallerTransaction` open on ``conn``; no durable store, a
            collection that caches absences, defers its L3 writes or fences with a null-safe CAS; two rows sharing a
            key; ``max_rows`` under one
        :raises KeyError: when the rows lack a key column, or a required column
        """
        transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.save_arrow")
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"{self.table_name}.save_arrow: max_rows must be at least 1, got {max_rows}")
        store = self._durable_store()
        if store is None:
            raise ValueError(f"{type(self).__name__}.save_arrow: {self.table_name} has no durable store to write to")
        if self._negative_cache_writes_advance or self._defers_l3_writes or self.schema.cas_null_safe:
            raise ValueError(
                f"{type(self).__name__}.save_arrow: a bulk upsert cannot keep this collection's write "
                f"contract (it caches absences, defers its L3 writes, or fences with a null-safe CAS)"
            )
        missing = [name for name in self.schema.pk_columns if name not in rows.column_names]
        if missing:
            raise KeyError(f"{self.table_name}.save_arrow: the rows have no key column {missing[0]!r}")
        if not rows.num_rows:
            return 0
        now = datetime.now(UTC)
        stamps: dict[str, datetime | int] = {
            name: now for name in ("date_created", "date_updated") if self.schema.get_column(name) is not None
        }
        # one epoch for the statement's rows, a constant Arrow column as the times are
        stamps.update(self._epoch_stamp(conn, "save_arrow"))
        arrow_store = store if isinstance(store, ArrowWritingStore) else None
        plan = None
        stamped = rows
        if arrow_store is not None:
            stamped = await asyncio.to_thread(with_constant_columns, rows, stamps)
            plan = arrow_store.arrow_write_plan(
                self.table_name,
                stamped.column_names,
                stamped.schema.types,
                kind="copy" if copy else "upsert",
                conn=conn,
            )
        if arrow_store is None or plan is None:
            # no door for these rows: today's write, the rows read into Python off the loop
            return await self.save_rows(
                await asyncio.to_thread(arrow_python_rows, rows), conn=conn, copy=copy, max_rows=max_rows
            )
        # the keys read, checked and enrolled from a worker thread taking turns with the loop: on the loop, 33,786 of
        # them held it 6-7 ms with the collector held and 50-55 ms with it running (the WD2 bench, 179 columns,
        # 2026-11-16)
        keys, enrollments = await asyncio.to_thread(self._arrow_enrollments, stamped, asyncio.get_running_loop())
        # enrolled before the write, so a write whose outcome is unknown is settled with the rest
        transaction.enroll_all(enrollments)
        # an append (a versioned table's new version) replaces the key's row whole, as the upsert it stands for set every
        # mutable column: its delta file records that SET list, never the append's empty one
        written = (
            bulk_update_columns(self.schema, insert_columns_for_data(self.schema, dict.fromkeys(stamped.column_names)))
            or []
            if plan.kind == "append"
            else plan.update_columns
        )
        transaction.record_arrow(self, stamped, written)
        # every key leaves L1 in one write before the store's, not one write a key
        await self._evict_l1_many_off_loop(keys)
        bound = max_rows if max_rows is not None or copy else self.BULK_MAX_ROWS
        await arrow_store.write_arrow(self.table_name, stamped, plan, conn=conn, max_rows=bound)
        log.debug(
            "Arrow bulk copy written on the caller's transaction"
            if copy
            else "Arrow bulk upsert written on the caller's transaction",
            extra={"extra_data": {"table": self.table_name, "rows": int(stamped.num_rows)}},
        )
        return int(stamped.num_rows)

    def _arrow_enrollments(
        self, rows: Any, loop: asyncio.AbstractEventLoop | None = None
    ) -> tuple[list[Any], list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]]:
        """each row's key, as :meth:`save_rows` enrols it, and its enrollment with the row its settling message reads
        its :attr:`invalidation_columns` from (None for a collection that declares none): the only columns of an Arrow
        write that become Python, read a slice at a time. Blocking: run it on a thread.

        :param rows: the rows
        :ptype rows: pyarrow.Table
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None off any loop
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the keys (a value, or a tuple for a key of many columns), in row order, and their enrollments
            (:meth:`CallerTransaction.enrollments`)
        :rtype: tuple[list[Any], list[tuple[BaseCollection, Any, dict[str, Any] | None]]]
        :raises ValueError: when two rows share a key
        """
        turns = LoopTurns(loop)
        key_columns = list(self.schema.pk_columns)
        named = [name for name in self.invalidation_columns if name in rows.column_names]
        read = [*key_columns, *(name for name in named if name not in key_columns)]
        columns = dict(zip(read, arrow_python_columns(rows, read, turns), strict=True))
        values = [columns[name] for name in key_columns]
        keys: list[Any] = list(stepped(zip(*values, strict=True), turns)) if len(values) > 1 else values[0]
        if _shares_a_key(keys, turns):
            raise ValueError(
                f"{type(self).__name__}.save_arrow: two rows share a key; one statement upserts a key once"
            )
        seen: list[dict[str, Any] | None] = (
            [
                dict(zip(named, each, strict=True))
                for each in stepped(zip(*(columns[name] for name in named), strict=True), turns)
            ]
            if named
            else [None] * len(keys)
        )
        return keys, CallerTransaction.enrollments(self, keys, seen, turns)

    def _appends_by_copy(self, store: DurableStore, first: Mapping[str, Any]) -> bool:
        """whether :meth:`save_rows`' upsert of a versioned table is written as a copy of the keys' new versions: the
        table is versioned, its store copies, and a copy writes every column the rows send
        (:func:`~threetears.core.backends.schema_sql.copy_writes`: not a jsonb or a vector). Otherwise the store's bulk
        statement writes them, an ``INSERT`` with no conflict clause for a versioned table.

        :param store: the collection's durable store
        :ptype store: DurableStore
        :param first: the first row, whose columns every row supplies
        :ptype first: Mapping[str, Any]
        :return: whether
        :rtype: bool
        """
        return (
            self.schema.versioned
            and isinstance(store, CopyingStore)
            and all(copy_writes(column) for column in insert_columns_for_data(self.schema, dict(first)))
        )

    async def _copy_rows(
        self,
        store: DurableStore,
        rows: list[dict[str, Any]],
        *,
        conn: Any,
        max_rows: int | None,
        max_bytes: int | None,
        append: bool = False,
    ) -> int:
        """insert ``rows``, already stamped and checked, on the caller's transaction: :meth:`save_rows` with ``copy``.

        :param store: the collection's durable store
        :ptype store: DurableStore
        :param rows: the rows
        :ptype rows: list[dict[str, Any]]
        :param conn: the caller's connection
        :ptype conn: Any
        :param max_rows: the most rows one copy carries; when neither bound is given, :func:`copy_bounds`
        :ptype max_rows: int | None
        :param max_bytes: the most bytes one copy's request comes to; when neither bound is given, :func:`copy_bounds`
        :ptype max_bytes: int | None
        :param append: the rows are a versioned table's new versions, noted a row each (``CopyingStore.copy_many``)
        :ptype append: bool
        :return: rows written
        :rtype: int
        """
        if not rows:
            return 0
        if isinstance(store, CopyingStore):
            if max_rows is None and max_bytes is None:
                max_rows, max_bytes = copy_bounds(conn, len(rows))
            appending: dict[str, bool] = {"append": True} if append else {}
            return await store.copy_many(
                self.table_name,
                rows,
                max_rows=max_rows or self.BULK_MAX_ROWS,
                max_bytes=max_bytes or self.BULK_MAX_BYTES,
                conn=conn,
                **appending,
            )
        # a store that cannot copy: still insert-only, a write per row
        for data in rows:
            await store.upsert(self.table_name, data, pk=self.schema.pk_columns, on_conflict="raise", conn=conn)
        return len(rows)

    @property
    def _records_epochs(self) -> bool:
        """whether this table's writes advance a write generation whose epochs are recorded: a whole-table change of
        it would make every follower read the table again.

        :return: whether it does
        :rtype: bool
        """
        return self.advances_write_generation or self._negative_cache_writes_advance

    async def reset(self) -> None:
        """an operator's reset: empty the table in L3 with ``TRUNCATE``, then drop what every cache holds of it. The one
        whole-table change a writer makes; its epoch names ``reason="reset"``, and a follower reads the table again as
        on a first start. Never from a load.

        For a table reloaded whole from a source that changed: deleting every row leaves a
        tombstone per row behind, and on YugabyteDB one transaction deleting millions of them runs
        for minutes and blocks every other writer. ``TRUNCATE`` leaves none. It is not
        transactional and runs alone, so it takes no connection: the store runs it outside any
        transaction (:meth:`~threetears.core.backends.protocol.TruncatingStore.truncate`; on the L3
        rail the ``l3.truncate`` request, which the broker runs under the database's DDL lock). Load
        the table again after with :meth:`load_files` from the source's export files, or with
        :meth:`save_rows` and ``copy=True`` for rows the caller holds.

        Then this process drops the table from its caches (:meth:`CollectionRegistry.drop_table
        <threetears.core.collections.registry.CollectionRegistry.drop_table>`): every row leaves L1,
        cached scans and derived caches are dropped, and its L2 entries are read past until each is
        read through from L3 again (:meth:`drop_cached_table`), since a pod cannot list L2 to delete
        them. Other processes learn of it from the table's write generation: ``TRUNCATE`` fires no
        row trigger (:mod:`~threetears.core.collections.lost_writes`),
        so this advances the generation itself, as a commit does, with no row broadcast to cover
        the advance; a follower's catch-up pass finds an advance it heard no row of and drops the
        table too. Safe to call again: an empty table truncates, and the caches drop, alike. A
        ``TRUNCATE`` that fails drops and advances too, since a reply lost after the broker ran it
        leaves the table emptied; the store's failure still reaches the caller.

        :return: nothing
        :rtype: None
        :raises ValueError: when the collection has no durable store, its store cannot truncate, or
            it defers its L3 writes (a buffered row would land after the table was emptied)
        :raises GenerationUnavailableError: when the table's write generation had to advance and
            could not; the table is emptied and this process's caches dropped
        """
        await self._truncate_l3()

    async def _truncate_l3(self) -> None:
        """``TRUNCATE`` the table in L3 for an operator's reset, then drop and advance (:meth:`reset`, its one caller).

        :return: nothing
        :rtype: None
        """
        store = self._durable_store()
        if store is None:
            raise ValueError(f"{type(self).__name__}.truncate: {self.table_name} has no durable store to empty")
        if not isinstance(store, TruncatingStore):
            raise ValueError(f"{type(self).__name__}.truncate: the store of {self.table_name} cannot truncate")
        if self._defers_l3_writes:
            raise ValueError(
                f"{type(self).__name__}.truncate: {self.table_name} defers its L3 writes, and a buffered row "
                f"would land after the table was emptied"
            )
        reach = WholeReach(reason=RESET_REASON)
        try:
            await whole_read(
                store.truncate(self.table_name, reason=RESET_REASON),
                kind="operator_reset",
                reason="an operator's reset empties the table in L3 with TRUNCATE; its epoch names the reset",
                ruling="Pace 2026-11-12: operator_reset declared in code; coordinator 2026-11-12: plain truncate() retired, reset() the one caller",
                proof="packages/core/tests/unit/collections/test_truncate.py::test_the_store_s_truncate_is_reached_only_from_an_operator_s_reset",
            )
        except (Exception, asyncio.CancelledError) as exc:
            # a reply lost or timed out may still have emptied the table: drop as if it had
            await self._drop_and_advance_after_failure("reset failed", exc, reach=reach)
            raise
        await self._drop_and_advance("reset", reach=reach)

    async def load_files(
        self,
        files: Sequence[LoadFile],
        *,
        columns: Sequence[str],
        constants: Mapping[str, Any] | None = None,
        rows_per_transaction: int | None = None,
        timeout_ms: int | None = None,
        connections: int = 1,
        prefixes: Sequence[Mapping[str, Any]] | None = None,
        reload: bool = False,
        delta: LoadDelta | None = None,
        on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None] | None = None,
        write_version: int | None = None,
    ) -> int:
        """load the table in L3 from parquet files, then drop what every cache holds of it.

        For a table reloaded whole from a warehouse export, after :meth:`truncate`: the files are
        the export's own (:func:`~threetears.datasources.partitioned_export.presigned_counted_partition` presigns
        them), and the store fetches and copies them itself
        (:meth:`~threetears.core.backends.protocol.LoadingStore.load_files`; on the L3 rail the
        ``l3.load_files`` request, so the broker streams the files and the rows never cross the
        bus). It runs outside any transaction of the caller's and commits every
        ``rows_per_transaction`` rows. Insert-only: a key already held fails the load. A column no
        file holds -- a part's own key, a load's ``date_created`` -- takes its value from
        ``constants``; nothing is stamped for the caller.

        Then, exactly as :meth:`truncate`: this process drops the table from its caches and advances
        the table's write generation once, with no row broadcast. ``COPY`` through the store leaves
        no row-level write mark either, so the advance is how a following pod learns of the load.
        A load that fails drops and advances too, before its failure reaches the caller: the
        transactions before the failure stay committed, and a reply lost or timed out can follow a
        load the broker finished.

        A load too long for one ``l3.load_files`` reply is sent as several requests, one after
        another in the files' order, each of whole files and each within the pod's wait
        (:func:`~threetears.core.backends.protocol.load_requests`); the caller sends one call
        however many files it has. The drop and the advance happen once, after the last request
        (or after the one that failed). Only a single file too long for the wait on its own is
        refused, before anything is sent.

        **Over several connections** (``connections`` above one): the broker copies a request's files one after
        another on one connection (about 16,000 rows a second on the local stack), so the requests are cut further,
        into runs of whole files of about the load's rows over ``connections``
        (:func:`~threetears.core.backends.protocol.side_by_side_requests`), and up to ``connections`` of them are
        sent at once, each its own connection and transactions. The rows are the same, insert-only as before; a
        request that fails stops the others (cancelled, then awaited, so none is left running), and the drop and the
        advance happen once, before the failure reaches the caller.

        **A reload is analysed** (``reload=True``, said only by a caller refilling a table it emptied with
        :meth:`reset`, or a new one): after its last request, a load of :data:`ANALYZE_AFTER_ROWS` rows or more has
        its store refresh the table's planner statistics (:class:`~threetears.core.backends.protocol.AnalyzingStore`),
        so its first reads plan by its indexes. Any other load, however large, is never analysed here: the database's
        own auto-analyze keeps up with it.

        :param files: the files, in order
        :ptype files: Sequence[LoadFile]
        :param columns: the table's columns each file supplies
        :ptype columns: Sequence[str]
        :param constants: column -> the value every row takes for it; none when None
        :ptype constants: Mapping[str, Any] | None
        :param rows_per_transaction: the most rows one committed transaction writes;
            :attr:`LOAD_ROWS_PER_TRANSACTION` when None
        :ptype rows_per_transaction: int | None
        :param timeout_ms: the most time one transaction may take, in milliseconds;
            :attr:`LOAD_TIMEOUT_MS` when None
        :ptype timeout_ms: int | None
        :param connections: the most requests sent at once; one sends them one after another
        :ptype connections: int
        :param prefixes: the key prefixes the load's rows fall in, each the leading key columns and their values (an
            AP load of a state names each office's key): its epoch names them, so a follower reads each as a key range.
            None takes the one prefix the constants lead the key with. A load naming neither is refused: its epoch
            could name only the whole table
        :ptype prefixes: Sequence[Mapping[str, Any]] | None
        :param reload: whether this load refills a table emptied (:meth:`reset`) or new: only such a load is analysed
        :ptype reload: bool
        :param delta: the rows the files hold, as the caller read them from the same files, and the key prefixes the
            load replaces: what a table that opts in to delta files is loaded with (see **A table that opts in**);
            None for any other table
        :ptype delta: LoadDelta | None
        :param on_commit: with a ``delta``, called on the event loop once for each commit the load made, as it lands,
            with that commit's delta files and the generation tokens its advance wrote, by table: the shapes a
            :class:`CallerTransaction` names them (``delta_files``, ``committed_generations``), for the writer's
            :meth:`~threetears.core.collections.delta_fold.DeltaFolds.committed`. The delete of the prefixes the load
            replaces comes first, then each request that committed, in the order each landed. A request that failed
            is never handed (its file may hold rows L3 never took, and its record names it unconfirmed: the fold hears
            it once the writer has confirmed it from L3, ``DeltaFolds.confirmed``); the requests that committed before
            it were. A callback, not the return value: a load that fails after some requests
            committed raises, and the fold must still hear the ones that did. It must not raise
            (``DeltaFolds.committed`` never does)
        :ptype on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None] | None
        :param write_version: the write's version (``ScopeEpochs.begin``) the load is part of. For a table that declares a
            row epoch (``TableSchema.row_epoch``) it is required, and every row loaded is stamped with one fresh epoch.
            With a ``delta``, its delta files are named by it (:func:`~threetears.core.collections.delta_files.versioned_write_id`, decision 53), so a
            full's manifest covers them exactly. None names them in the older, unversioned form, which a recovery always
            reads (decision 53's expand step; its contract makes it required)
        :ptype write_version: int | None
        :return: the rows written
        :rtype: int
        :raises ValueError: when the collection has no durable store, its store cannot load files,
            it defers its L3 writes (a buffered row would land among the loaded ones), a bound is
            under one, ``connections`` is under one, or one file alone needs a longer wait than the cap; when
            ``on_commit`` is given with no ``delta`` (a load that saves no file has none to hand)
        :raises GenerationUnavailableError: when the table's write generation had to advance and
            could not; the rows are loaded and this process's caches dropped
        :raises DeltaFileRefusedError: when the table opts in to delta files and no ``delta`` is given: the store
            copies the files itself, so no row passes through this process to be saved in one
        """
        if delta is None and on_commit is not None:
            raise ValueError(
                f"{type(self).__name__}.load_files: on_commit is given with no delta; a load that saves no delta "
                f"file has none to hand"
            )
        if delta is None:
            self._refuse_unfiled("load_files")
        elif not self.writes_delta_files:
            raise ValueError(
                f"{type(self).__name__}.load_files: {self.table_name} does not opt in to delta files; a load of it "
                f"takes no delta"
            )
        elif len(delta.rows) != len(files):
            raise ValueError(
                f"{type(self).__name__}.load_files: a load's delta gives the rows of each of its files, one entry a "
                f"file: {len(delta.rows)} given for {len(files)} files"
            )
        store = self._durable_store()
        if store is None:
            raise ValueError(f"{type(self).__name__}.load_files: {self.table_name} has no durable store to load")
        if not isinstance(store, LoadingStore):
            raise ValueError(f"{type(self).__name__}.load_files: the store of {self.table_name} cannot load files")
        if self._defers_l3_writes:
            raise ValueError(
                f"{type(self).__name__}.load_files: {self.table_name} defers its L3 writes, and a buffered row "
                f"would land among the loaded ones"
            )
        if delta is not None and delta.replaces:
            await self._delete_replaced(store, delta.replaces, on_commit, write_version=write_version)
        if not files:
            return 0
        if self.schema.stamps_epoch:
            if write_version is None:
                raise UnstampedWriteError(
                    f"{type(self).__name__}.load_files: {self.table_name} declares a row epoch, so every row loaded "
                    f"carries its write's epoch: name the write's version (write_version)"
                )
            # one epoch for the load's rows, the process clock's next stamp: a load inserts only, so it writes a key once.
            # The write's version stays required: its begin is what moved this process's clock past the previous
            # writer's last_epoch (ScopeEpochs.begin), so the load's rows sort above that writer's
            constants = {**(constants or {}), EPOCH_COLUMN: next_epoch()}
        reach = WholeReach(prefixes=tuple(self._load_reach(constants or {}, prefixes)))
        per_transaction = self.LOAD_ROWS_PER_TRANSACTION if rows_per_transaction is None else rows_per_transaction
        bound = self.LOAD_TIMEOUT_MS if timeout_ms is None else timeout_ms
        # refused here, before anything is sent: only a single file too long for one reply
        requests = load_requests(files, rows_per_transaction=per_transaction, timeout_ms=bound)
        if connections > 1:
            requests = side_by_side_requests(requests, connections)
        elif connections < 1:
            raise ValueError(f"a load runs over at least one connection, not {connections} connections")
        written = 0
        slots = asyncio.Semaphore(connections)
        # each file's rows the caller gave, by the file: a request's delta file holds its files' rows alone
        given = {id(file): rows for file, rows in zip(files, delta.rows, strict=True)} if delta is not None else {}

        async def one(request: Sequence[LoadFile]) -> int:
            async with slots:
                named: dict[str, str] = {}
                if delta is not None:
                    named = await self._save_load_delta(request, given, constants or {}, write_version)
                # the generations the request's commit wrote, learned as a CallerTransaction learns its own: from the
                # broker's reply on a pod, from this process's advance on a direct pool
                before = committed_generations_record()
                advanced: dict[str, str] = {}
                # the broker's answer, success or failure: the rows it committed (function-local, this request's)
                answers: list[int] = []
                # the request names its file: the broker's epoch record of it does, and so does this process's own
                # advance on a direct pool (decision 11: saved before the request's commits)
                landed = False
                try:
                    with (
                        naming_delta_files(named),
                        collecting_advanced_generations(advanced),
                        collecting_load_answers(answers),
                    ):
                        try:
                            result = await store.load_files(
                                self.table_name,
                                request,
                                columns=columns,
                                constants=dict(constants or {}),
                                rows_per_transaction=per_transaction,
                                timeout_ms=bound,
                                key_prefixes=[dict(prefix) for prefix in reach.prefixes],
                            )
                        except (Exception, asyncio.CancelledError) as exc:
                            if named:
                                # kept (named by an unconfirmed record), or an orphan for the sweep
                                landed = await self._after_failed_request(named, reach, answers, exc)
                            raise
                        if delta is not None:
                            # one epoch a request, each naming its own file: a file holding a later request's rows
                            # would hand a copy rows L3 does not hold yet, or never will once that request fails
                            await self._drop_and_advance(
                                "loaded from files",
                                reach=reach,
                                committed_before=before,
                                rows=result.row_count,
                                requests=1,
                            )
                    landed = True
                finally:
                    if named:
                        # a request that failed is kept when a record names its file unconfirmed (the writer confirms
                        # it from L3), else an orphan for the sweep
                        OWN_DELTA_SAVES.settled(list(named.values()), landed=landed)
                if on_commit is not None and named:
                    # only here, after the request and its advance landed: a request that raised hands nothing
                    _hand_commit(on_commit, named, {**reply_generations_since(before), **advanced})
            return result.row_count

        try:
            if connections == 1:
                for request in requests:
                    written += await one(request)
            else:
                written = sum(await _all_or_first_failure([one(request) for request in requests]))
        except (Exception, asyncio.CancelledError) as exc:
            # the load commits as it goes: the requests and transactions before a failure stay
            # committed, and a reply lost or timed out may follow a load the broker finished. A table that opts in to
            # delta files had each failed request's record settled by its owner (_after_failed_request): never a record
            # of this process's naming no file, which every copy would read back from L3 (ruling C of 2026-11-23)
            if delta is None:
                await self._drop_and_advance_after_failure("load from files failed", exc, reach=reach)
            raise
        if reload and written >= ANALYZE_AFTER_ROWS and isinstance(store, AnalyzingStore):
            # Only a reload is analysed here, once, after its last request (coordinator 2026-11-15, within Pace's
            # 11-12 cold-start ruling). Why a reload at all: a table emptied and filled again keeps the
            # statistics of what it held before (or none), so YugabyteDB planned its first reads blind -- a state's
            # rows of some geographies took 157 ms on a primary-key scan and a sort, 0.4 ms once analysed
            # (packages/core/tests/integration/test_reload_analyzed_live.py). Why only a reload: YugabyteDB's ANALYZE
            # reads about every row (10k rows 44 ms, 200k 132 ms, 1M 540 ms, best of 3 on a local YugabyteDB 2026.1
            # with block-based sampling, 2026-11-14), so analysing after every large load paid for the whole table on each
            # incremental load too. What covers the other loads: the database's own auto-analyze -- YugabyteDB
            # 2026.1's ysql_enable_auto_analyze (on in the local test YugabyteDB: threshold 50 rows + 10% of the
            # table, a 10 s per-table cooldown) and Postgres's autovacuum (on: the same threshold and scale factor).
            # A table already analysed keeps statistics close enough for an incremental load until it catches up.
            try:
                await store.analyze(self.table_name)
            # prawduct:allow prawduct/broad-except -- the rows are loaded; stale statistics cost plans, not data
            except Exception as exc:
                contain(
                    exc, log=log, what="could not analyse a table after loading it", extra={"table": self.table_name}
                )
        if delta is None:
            await self._drop_and_advance(
                "loaded from files",
                reach=reach,
                rows=written,
                requests=len(requests),
            )
        return written

    async def _after_failed_request(
        self, named: Mapping[str, str], reach: WholeReach, answers: Sequence[int], failure: BaseException
    ) -> bool:
        """settle the epoch record of a load request of a table that opts in to delta files that failed (ruling C of
        2026-11-23). The request commits a transaction per ``rows_per_transaction`` rows, so a leading part of its file's
        rows may be in L3; its file holds every row it sent.

        - A direct pool (this process writes its records): this process advances once, its record naming the file
          unconfirmed, and applies it here as a follower would; the writer confirms the file from L3 before any copy
          applies it.
        - A pod whose broker answered: the broker owns the record (one naming the file unconfirmed when rows committed,
          none when none did); this process writes none.
        - A pod with no answer (a cancel, a lost reply): no record is written here; the broker records the request if
          it goes on to commit (decision 26). Counted and logged.

        Never raises: the request's own failure is the one the caller gets.

        :param named: the request's delta file, by table
        :ptype named: Mapping[str, str]
        :param reach: the load's key prefixes
        :ptype reach: WholeReach
        :param answers: the rows the broker answered it committed, one an answer (none: no answer)
        :ptype answers: Sequence[int]
        :param failure: the request's failure
        :ptype failure: BaseException
        :return: whether a record names the file (unconfirmed), so it is kept rather than swept as an orphan
        :rtype: bool
        """
        file = named.get(self.table_name)
        registry = self.registry
        source = None if registry is None else registry.generation_source
        if file is None:
            return False
        if registry is not None and source is not None and records_changes(source):
            _load_requests_failed().labels(table=self.table_name, owner="process").inc()
            try:
                await self._advance_unconfirmed(registry, reach, file, failure=type(failure).__name__)
            # prawduct:allow prawduct/broad-except -- the request's own failure is raised; this one is recorded
            except Exception as exc:
                contain(
                    exc,
                    log=log,
                    what="a failed load request's epoch naming its file unconfirmed was not advanced",
                    extra={"table": self.table_name, "file": file, "failure": type(failure).__name__},
                    level="error",
                )
            return True
        if answers:
            committed = answers[-1]
            _load_requests_failed().labels(table=self.table_name, owner="hub" if committed else "hub_none").inc()
            log.warning(
                "a load request failed and its broker answered: the broker's record names its file unconfirmed when "
                "rows committed (the writer confirms it from L3), none when none did; this process writes none",
                extra={
                    "extra_data": {
                        "table": self.table_name,
                        "file": file,
                        "committed": committed,
                        "failure": type(failure).__name__,
                    }
                },
            )
            return committed > 0
        _load_requests_failed().labels(table=self.table_name, owner="unanswered").inc()
        log.error(
            "a load request failed with no answer from its broker (a cancel or a lost reply): this process writes no "
            "record; the broker records the request if it commits",
            extra={"extra_data": {"table": self.table_name, "file": file, "failure": type(failure).__name__}},
        )
        return False

    async def _advance_unconfirmed(self, registry: Any, reach: WholeReach, file: str, **detail: Any) -> None:
        """advance once for a failed load request on a direct pool, the record naming its file unconfirmed, and apply
        the epoch here as a follower hears it: a copy here waits for the writer to confirm the file, reading nothing
        from L3 (:meth:`CollectionRegistry.apply_epoch_change`).

        :param registry: the collection's registry
        :ptype registry: CollectionRegistry
        :param reach: the load's key prefixes
        :ptype reach: WholeReach
        :param file: the request's delta file
        :ptype file: str
        :param detail: more to log
        :ptype detail: Any
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the generation had to advance and could not
        """
        # the request's own naming of its file (confirmed) ends here: only the unconfirmed name is recorded
        with naming_delta_files({}), naming_unconfirmed_delta_files({self.table_name: file}):
            bump = await self._bump_after_commit(None, reach=reach)
        if bump.token is not None:
            await registry.apply_epoch_change(
                EpochChange(
                    table=self.table_name,
                    token=bump.token,
                    whole_table=True,
                    prefixes=[dict(prefix) for prefix in reach.prefixes],
                    delta_file=file,
                    delta_file_unconfirmed=True,
                )
            )
        log.warning(
            "a load request failed on a direct pool: its epoch names its file unconfirmed, for the writer to confirm "
            "from L3 before any copy applies it",
            extra={"extra_data": {"table": self.table_name, "file": file, "generation": bump.token, **detail}},
        )
        if bump.failure is not None:
            raise bump.failure

    async def _delete_replaced(
        self,
        store: Any,
        replaces: Sequence[Mapping[str, Any]],
        on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None] | None = None,
        *,
        write_version: int | None,
    ) -> None:
        """delete every row of the key prefixes a load replaces, in a transaction of its own committed before the
        load: a load is insert-only, so the rows it replaces leave L3 first. Its delta file names the prefixes
        (:meth:`delete_prefixes`), so a copy and the fold clear their own rows under them, with no L3 read
        (decision 23).

        :param store: the collection's durable store
        :ptype store: Any
        :param replaces: the prefixes, each the leading key columns and their values
        :ptype replaces: Sequence[Mapping[str, Any]]
        :param on_commit: told the delete's delta file and generation once it committed (:meth:`load_files`)
        :ptype on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None] | None
        :param write_version: the load's write version, naming the delete's delta file (:meth:`load_files`)
        :ptype write_version: int | None
        :return: nothing
        :rtype: None
        :raises ValueError: when the store opens no connection of its own, or as :meth:`delete_prefixes`
        """
        acquire = getattr(store, "acquire", None)
        if acquire is None:
            raise ValueError(
                f"{type(self).__name__}.load_files: the store of {self.table_name} opens no connection to delete the "
                f"prefixes the load replaces on"
            )
        async with acquire() as conn, CallerTransaction(conn) as transaction:
            if write_version is not None:
                transaction.note_write_version(write_version)
            await self.delete_prefixes(replaces, conn=conn)
        if on_commit is not None and transaction.delta_files:
            _hand_commit(on_commit, transaction.delta_files, transaction.committed_generations)

    async def _save_load_delta(
        self,
        request: Sequence[LoadFile],
        given: Mapping[int, Any],
        constants: Mapping[str, Any],
        write_version: int | None,
    ) -> dict[str, str]:
        """save one request's delta file before the request is sent: its files' rows the caller gave, with the load's
        constants, written as a copy writes them (inserted, never merged onto a row held).

        :param request: the request's files
        :ptype request: Sequence[LoadFile]
        :param given: each file's rows the caller gave, by the file's id
        :ptype given: Mapping[int, Any]
        :param constants: the load's constants
        :ptype constants: Mapping[str, Any]
        :param write_version: the load's write version, naming the file (:meth:`load_files`)
        :ptype write_version: int | None
        :return: the table -> its file's name
        :rtype: dict[str, str]
        :raises ValueError: when a file's rows given are not as many as it holds
        :raises DeltaFileRefusedError: when no delta files directory is configured
        """
        table_delta = self.table_delta()
        # joined, typed, checked against the files' counts and saved in the worker process the save runs in
        table_delta.load([given[id(file)] for file in request], [file.rows for file in request], constants)
        return await save_delta_files({self.table_name: table_delta}, version=write_version)

    def _key_columns(self) -> tuple[str, ...]:
        key = self.schema.primary_key
        return tuple(key) if isinstance(key, tuple) else (key,)

    def _load_reach(
        self, constants: Mapping[str, Any], prefixes: Sequence[Mapping[str, Any]] | None
    ) -> list[dict[str, str]]:
        """a load's or a clear's key prefixes as text, each checked to name the key's leading columns.

        :param constants: the load's constants
        :ptype constants: Mapping[str, Any]
        :param prefixes: the prefixes named, or None for the constants' one
        :ptype prefixes: Sequence[Mapping[str, Any]] | None
        :return: the prefixes, each once
        :rtype: list[dict[str, str]]
        :raises ValueError: when none is named, or one does not lead the key
        """
        if prefixes is None:
            prefix = self.key_prefix_of(constants)
            if not prefix:
                raise ValueError(
                    f"{type(self).__name__}: {self.table_name}'s load names no key prefix and its constants "
                    f"({sorted(constants)}) do not lead the key {list(self._key_columns())}: its epoch could name only "
                    f"the whole table. Name the key prefixes its rows fall in (prefixes=)"
                )
            return [prefix]
        named: list[dict[str, str]] = []
        for given in prefixes:
            prefix = self.key_prefix_of(given)
            if not prefix or set(prefix) != set(given):
                raise ValueError(
                    f"{type(self).__name__}: {self.table_name}: {dict(given)} is not a key prefix (the leading columns "
                    f"of {list(self._key_columns())}, each with a value)"
                )
            if prefix not in named:
                named.append(prefix)
        if not named:
            raise ValueError(f"{type(self).__name__}: {self.table_name}: prefixes= names none")
        return named

    def key_prefix_of(self, constants: Mapping[str, Any]) -> dict[str, str]:
        """the longest run of the key's leading columns ``constants`` names, as text: a load's or a clear's reach as an
        index reads it exactly (a key range). Empty when the first key column is not among them.

        :param constants: columns and the value every row holds
        :ptype constants: Mapping[str, Any]
        :return: the prefix
        :rtype: dict[str, str]
        """
        prefix: dict[str, str] = {}
        for column in self._key_columns():
            if constants.get(column) is None:
                break
            prefix[column] = str(constants[column])
        return prefix

    async def _drop_and_advance_after_failure(
        self, reason: str, failure: BaseException, *, reach: WholeReach | None = None
    ) -> None:
        """drop and advance as :meth:`_drop_and_advance` does, after a store call that failed.

        A store call that raised may still have changed L3 (a load commits as it goes; a lost reply
        can follow a change the broker made), and a drop where nothing changed costs only a
        read-through. The store's failure is the one the caller gets, so a failure here is recorded
        rather than raised over it.

        :param reason: what was being done to the table, for the drop and the log
        :ptype reason: str
        :param failure: the store call's failure, which the caller re-raises
        :ptype failure: BaseException
        :return: nothing
        :rtype: None
        """
        try:
            await self._drop_and_advance(reason, reach=reach, failure=type(failure).__name__)
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="caches not dropped or generation not advanced after a failed table change",
                extra={"table": self.table_name, "change": reason, "failure": type(failure).__name__},
                level="error",
            )

    async def _drop_and_advance(
        self,
        reason: str,
        *,
        reach: WholeReach | None = None,
        committed_before: object | None = _NO_REPLY,
        **detail: Any,
    ) -> None:
        """drop the table from this process's caches, then advance its write generation once.

        After a change to the table that left no row-level write mark (:meth:`truncate`,
        :meth:`load_files`): the registry drops it (L1 rows, cached scans, derived caches; L2
        entries read past), and the generation advances with no row broadcast, so a following pod
        drops it at its next catch-up.

        :param reason: what changed the table, for the drop and the log
        :ptype reason: str
        :param reach: a load's or a clear's key prefixes, or an operator's reset: a derived copy here reads those
            prefixes again, or the table as on a first start (:meth:`CollectionRegistry.drop_partitions`), and the
            epoch's record names the same, recorded here on a direct pool and by the broker that advances for a pod
        :ptype reach: WholeReach | None
        :param committed_before: this task's record of its last commit's generations as it stood before the request
            (``committed_generations_record``), for a request a broker answered: the epoch its reply names is the one
            this process applies from the file when the collection advances none of its own
            (:meth:`_advance_then_apply_file`); not given for a change no broker reply names
        :ptype committed_before: object | None
        :param detail: more to log
        :ptype detail: Any
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the generation had to advance and could not
        """
        registry = self.registry
        delta_file = named_delta_files().get(self.table_name)
        from_file = registry is not None and reach is not None and bool(reach.prefixes) and delta_file is not None
        # with a delta file the registry applies the epoch from it once the generation advances
        # (_advance_then_apply_file); without one it drops what the change reached here
        if registry is not None and not from_file:
            if reach is not None and reach.prefixes:
                await registry.drop_partitions(self.table_name, reach.prefixes, reason=reason)
            else:
                why = "reset" if reach is not None and reach.reason == RESET_REASON else "local"
                registry.drop_table(self.table_name, reason=reason, why=why)
        if registry is None or registry.get_collection(self.table_name) is not self:
            # an instance the registry does not hold hears no epoch it applies, so it drops its own L1 here, after the
            # commit (a read from now on is of L3 as the change left it): one site for both paths, the whole-table
            # change the record names (announced by the advance below)
            self.drop_cached_table()
        if from_file:
            assert registry is not None and reach is not None and delta_file is not None  # noqa: S101 -- from_file
            await self._advance_then_apply_file(
                registry, reason, reach, delta_file, committed_before=committed_before, **detail
            )
            return
        bump = await self._bump_after_commit(None, reach=reach)
        log.info(
            "table changed with no row write mark; caches dropped and generation advanced",
            extra={"extra_data": {"table": self.table_name, "change": reason, "generation": bump.token, **detail}},
        )
        if bump.failure is not None:
            raise bump.failure

    async def _advance_then_apply_file(
        self,
        registry: Any,
        reason: str,
        reach: WholeReach,
        delta_file: str,
        *,
        committed_before: object | None = _NO_REPLY,
        **detail: Any,
    ) -> None:
        """:meth:`_drop_and_advance` for a change whose delta file was saved (a load's request, a key-prefix delete of
        a table that opts in): the generation advances first, then this process applies the epoch as a follower hears
        it (:meth:`CollectionRegistry.apply_epoch_change`), so a copy here takes the change from the file, as every
        other pod's does, never by reading the prefixes from L3. An advance that names no epoch (a collection that
        keeps no write generation) drops the prefixes as before.

        :param registry: the collection's registry
        :ptype registry: CollectionRegistry
        :param reason: what changed the table, for the drop and the log
        :ptype reason: str
        :param reach: the change's key prefixes
        :ptype reach: WholeReach
        :param delta_file: the change's delta file, as its record names it
        :ptype delta_file: str
        :param committed_before: as :meth:`_drop_and_advance` takes it
        :ptype committed_before: object | None
        :param detail: more to log
        :ptype detail: Any
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the generation had to advance and could not
        """
        bump = await self._bump_after_commit(None, reach=reach)
        token = bump.token
        if token is None and committed_before is not _NO_REPLY:
            # a collection that advances no generation of its own (a pod's table built without L2, NO_L2): its broker
            # advanced the table after the request and named the epoch in its reply, the record naming this file.
            # This process applies that epoch from the file, as every other pod does, and accounts it as its own so its
            # catch-up does not apply it again. Dropping the prefixes instead told this process's own copy to read them
            # from L3, which a copy of a table that opts in refuses: the writer's copy refused every load of its own
            # and waited for the next full (2026-11-24, reports' LSF loads in test_refresh_live's rollout tests)
            token = reply_generations_since(committed_before).get(self.table_name)
            if token is not None:
                registry.account_generation(self.table_name, token)
        if token is not None:
            await registry.apply_epoch_change(
                EpochChange(
                    table=self.table_name,
                    token=token,
                    whole_table=True,
                    prefixes=[dict(prefix) for prefix in reach.prefixes],
                    delta_file=delta_file,
                )
            )
        else:
            await registry.drop_partitions(self.table_name, reach.prefixes, reason=reason)
        log.info(
            "table changed by key prefixes with its delta file; generation advanced, the file applied here",
            extra={
                "extra_data": {
                    "table": self.table_name,
                    "change": reason,
                    "generation": token,
                    "file": delta_file,
                    **detail,
                }
            },
        )
        if bump.failure is not None:
            raise bump.failure

    async def clear_partition(self, prefixes: Sequence[Mapping[str, Any]], *, conn: Any) -> int:
        """delete every row of some key prefixes on a caller's transaction: what a load that replaces a part does
        before it writes the part again, in place of emptying the table.

        Each prefix names the key's leading columns and their values (an office's key, a run's). The keys the
        prefixes hold are read on the caller's transaction, every statement led by the key (:meth:`read_rows_led_by`),
        then deleted by key (:meth:`delete_rows`): the epoch names exactly the rows that left, so a follower reads
        only those keys, never the table.

        :param prefixes: the key prefixes to clear
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :return: the rows deleted
        :rtype: int
        :raises ValueError: when a prefix does not lead the key, or as :meth:`read_rows_led_by` and :meth:`delete_rows`
        """
        reach = self._load_reach({}, prefixes)
        lead = self._key_columns()[0]
        key = list(self.schema.pk_columns)
        by_lead: dict[str, list[dict[str, str]]] = {}
        values: dict[str, Any] = {}
        for text in reach:
            by_lead.setdefault(text[lead], []).append(text)
        for given in prefixes:
            values.setdefault(str(given[lead]), given[lead])
        rows = await self.read_rows_led_by(list(values.values()), conn=conn)
        doomed = [
            tuple(row[column] for column in key)
            for row in rows
            if any(all(str(row[c]) == v for c, v in prefix.items()) for prefix in by_lead.get(str(row[lead]), []))
        ]
        if not doomed:
            return 0
        return await self.delete_rows(doomed, conn=conn)

    async def delete_prefixes(self, prefixes: Sequence[Mapping[str, Any]], *, conn: Any) -> int:
        """delete every row of some key prefixes on a caller's transaction, in statements whose count and cost do not
        grow with the rows already held: what a load does when whole runs, or whole parts, are gone from its source.

        Each prefix names the key's leading columns and their values (a state's run: ``{"state": "PA", "run_id":
        7}``). The store deletes them without reading a row back
        (:meth:`~threetears.core.backends.protocol.PrefixDeletingStore.delete_prefixes`): a key-led ``DELETE`` a
        group of prefixes that differ only in their last column. Where :meth:`clear_partition` reads the prefixes'
        keys and deletes them by key, so its epoch names each row, this one's epoch names the prefixes: the change is
        recorded as a whole-table change narrowed to them, as a load from files is (:meth:`load_files`), never key by
        key.

        **On the caller's transaction only**, as :meth:`delete_rows` is: the rows are gone when the caller's
        :class:`CallerTransaction` commits, and a failing delete fails the transaction. When it ends, whichever way,
        the prefixes are settled (:meth:`settle_prefixes`): this process drops the table from its L1 and cached scans,
        each derived cache that takes a partition (a :class:`~threetears.core.collections.copy_follower.CopyFollower`)
        reads each prefix again as a key range, and the table's write generation advances once, its epoch naming the
        prefixes, so every following process does the same at its catch-up. The change note the delete leaves is one
        for the table, naming the prefixes too, so a log restored from the notes says what the epoch said.

        **One kind of write a table a transaction**: a transaction that deletes a table's prefixes writes no row of
        that table by key (:meth:`CallerTransaction.enroll_prefixes`); its new rows go in a transaction of their own.

        **A table that opts in to delta files** (:attr:`TableSchema.delta_files`): the commit's delta file names the
        prefixes and no rows (:meth:`CallerTransaction.record_prefixes`), saved before the commit and named in its
        epoch record; a copy and the fold delete their own rows under each, with no L3 read (decision 23).

        :param prefixes: the key prefixes to delete, each the key's leading columns and their values
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :return: the rows deleted
        :rtype: int
        :raises ValueError: when no :class:`CallerTransaction` is open on ``conn``; when the collection has no durable
            store, its store cannot delete by prefix, or it defers its L3 writes; when a prefix does not lead the key;
            when the transaction wrote rows of the table by key
        """
        transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.delete_prefixes")
        store = self._durable_store()
        if store is None:
            raise ValueError(
                f"{type(self).__name__}.delete_prefixes: {self.table_name} has no durable store to delete from"
            )
        if not isinstance(store, PrefixDeletingStore):
            raise ValueError(
                f"{type(self).__name__}.delete_prefixes: the store of {self.table_name} cannot delete by key prefix"
            )
        if self._defers_l3_writes:
            raise ValueError(
                f"{type(self).__name__}.delete_prefixes: {self.table_name} defers its L3 writes, and a buffered row "
                f"of a prefix would land after the prefix was deleted"
            )
        if not prefixes:
            return 0
        reach = self._load_reach({}, prefixes)
        # enrolled before the delete, so a delete whose outcome is unknown is settled with the rest
        transaction.enroll_prefixes(self, reach)
        # a table that opts in to delta files: the commit's file names the prefixes, which a copy and the fold clear
        # from their own rows (decision 23); no row is listed, so L3 is never read for them
        transaction.record_prefixes(self, prefixes)
        # a versioned table's prefixes are run markers at the delete's epoch, ONE a prefix, never a DELETE
        marking = {"epoch": self._epoch_stamp(conn, "delete_prefixes")[EPOCH_COLUMN]} if self.schema.versioned else {}
        deleted = await store.delete_prefixes(
            self.table_name, [dict(prefix) for prefix in prefixes], conn=conn, **marking
        )
        log.debug(
            "key-prefix delete written on the caller's transaction",
            extra={"extra_data": {"table": self.table_name, "prefixes": len(reach), "rows": deleted}},
        )
        return deleted

    async def settle_prefixes(self, prefixes: Sequence[Mapping[str, str]]) -> None:
        """settle the caches of the key prefixes a caller's transaction deleted (:meth:`delete_prefixes`), once it has
        ended: as :meth:`load_files` settles a load's prefixes (:meth:`_drop_and_advance`).

        :param prefixes: each prefix, the key's leading columns and their values as text
        :ptype prefixes: Sequence[Mapping[str, str]]
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the table's write generation had to advance and could not; this
            process's caches were dropped
        """
        await self._drop_and_advance(
            "key prefixes deleted", reach=WholeReach(prefixes=tuple(dict(prefix) for prefix in prefixes))
        )

    async def delete_rows(
        self,
        keys: Sequence[Sequence[Any]],
        *,
        conn: Any,
        max_rows: int | None = None,
        delta_scope: Any = None,
        scope: Any = None,
    ) -> int:
        """delete many rows by key on a caller's transaction, through the collection's durable store.

        **A versioned table** (:attr:`TableSchema.versioned`) is never deleted from: each key is a "gone" marker
        appended at the delete's epoch (the store's ``mark_gone``), carrying the key, the table's scope and the epoch.

        The other half of :meth:`save_rows`, for a load that makes a table match a source: the rows
        the source no longer holds leave in a few multi-row ``DELETE`` statements, not one round trip
        each. The store's :meth:`~threetears.core.backends.protocol.BulkDeletingStore.delete_many`
        deletes them when it has one, and a key at a time through ``delete`` when it has not.

        **On the caller's transaction only**, as :meth:`save_rows` is: the rows are gone when the
        caller's :class:`CallerTransaction` commits, a failing delete fails the transaction, and every
        key leaves this process's L1 now and is settled with the rest when the transaction ends.

        A collection that declares :attr:`invalidation_columns` reads each row before deleting it,
        on the same transaction, so each row's message carries those columns of the row removed.

        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :param max_rows: the most keys one statement names, at least one; :attr:`BULK_MAX_ROWS` when None
        :ptype max_rows: int | None
        :param delta_scope: for a table whose key does not name a row's state
            (:attr:`TableSchema.delta_scope`), the state of every key deleted: each delete marker of the table's delta
            file carries it, and the fold files the delete under that state's part (decision 27). A writer deletes
            within one state's load, so it knows it. None for any other table
        :ptype delta_scope: Any
        :param scope: for a versioned table whose key does not name its scope column (``row_versions(scope)``), the
            scope of every key deleted, which each gone marker carries; when None, ``delta_scope`` where the table's
            delta scope is that column. None for any other table
        :ptype scope: Any
        :return: how many keys were named (a key no row holds is not an error)
        :rtype: int
        :raises ValueError: when no :class:`CallerTransaction` is open on ``conn``; when the collection
            has no durable store, caches absences or defers its L3 writes; when ``max_rows`` is under
            one; when a key is not as wide as the table's key; when ``delta_scope`` is given for a table that declares
            none; when a versioned table's markers' scope cannot be named
        :raises DeltaFileRefusedError: when the table declares a :attr:`TableSchema.delta_scope` and no
            ``delta_scope`` is given, before any statement
        """
        transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.delete_rows")
        self._check_delta_scope(delta_scope)
        store = self._durable_store()
        if store is None:
            raise ValueError(
                f"{type(self).__name__}.delete_rows: {self.table_name} has no durable store to delete from"
            )
        if self._negative_cache_writes_advance or self._defers_l3_writes:
            raise ValueError(
                f"{type(self).__name__}.delete_rows: a bulk delete cannot keep this collection's write "
                f"contract (it caches absences or defers its L3 writes)"
            )
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"{self.table_name}.delete_rows: max_rows must be at least 1, got {max_rows}")
        width = len(self.schema.pk_columns)
        # each key checked, and later keyed and enrolled, on a worker thread taking turns with the loop, from the
        # count where a thread costs less than the work, as save_rows' rows are (_stamped_rows)
        loop = asyncio.get_running_loop() if len(keys) >= self.L1_EVICT_OFF_LOOP_KEYS else None
        named = (
            await asyncio.to_thread(self._delete_keys_named, keys, loop)
            if loop is not None
            else self._delete_keys_named(keys)
        )
        # each key named is a delete marker in the table's delta file, whichever way it is enrolled below
        transaction.record_deletes(self, named, scope=delta_scope)
        # a versioned table is never deleted from (its "gone" markers are appended below), so never DELETE ... RETURNING
        versioned = self.schema.versioned
        marker_scope = self._marker_scope(delta_scope, scope) if versioned else None
        if self.invalidation_columns and named and isinstance(store, ReturningDeletingStore) and not versioned:
            return await self._delete_rows_returning(store, named, transaction, conn=conn, max_rows=max_rows)
        # a collection whose row messages carry columns reads each row before it goes, on the caller's
        # transaction, so the message can say whose it was: a statement a batch of keys where the store reads
        # many (one a key was a round trip each, on the rail a NATS request each), a read a key where it
        # cannot, none for a collection that declares no columns
        held: dict[tuple[Any, ...], dict[str, Any]] | None = None
        if self.invalidation_columns and named and isinstance(store, ManyKeyReadingStore):
            held = await store.fetch_many(self.table_name, named, max_rows=max_rows or self.BULK_MAX_ROWS, conn=conn)
        if not self.invalidation_columns or held is not None:
            entity_keys, enrollments = (
                await asyncio.to_thread(self._delete_enrollments, named, held, loop)
                if loop is not None
                else self._delete_enrollments(named, held)
            )
            # enrolled before the delete, so a delete whose outcome is unknown is settled with the rest
            transaction.enroll_all(enrollments)
        else:
            entity_keys = []
            for key in named:
                entity_key = key if width > 1 else key[0]
                seen = await store.fetch_one(
                    self.table_name, dict(zip(self.schema.pk_columns, key, strict=True)), conn=conn
                )
                # enrolled before the delete, so a delete whose outcome is unknown is settled with the rest
                transaction.enroll(self, entity_key, row=seen)
                entity_keys.append(entity_key)
        # every key leaves L1 in one write before the store's delete, not one write a key
        await self._evict_l1_many_off_loop(entity_keys)
        deleted = 0
        if versioned and named:
            # append-only: a "gone" marker a key at the delete's epoch, never a DELETE (TableSchema.versioned)
            if not isinstance(store, GoneMarkingStore):
                raise ValueError(
                    f"{type(self).__name__}.delete_rows: {self.table_name} is versioned, and its store cannot append "
                    f"gone markers"
                )
            deleted = await store.mark_gone(
                self.table_name,
                named,
                epoch=self._epoch_stamp(conn, "delete_rows")[EPOCH_COLUMN],
                scope=marker_scope,
                max_rows=max_rows or self.BULK_MAX_ROWS,
                conn=conn,
            )
        elif named and isinstance(store, BulkDeletingStore):
            deleted = await store.delete_many(
                self.table_name, named, max_rows=max_rows or self.BULK_MAX_ROWS, conn=conn
            )
        else:
            for key in named:
                await store.delete(self.table_name, dict(zip(self.schema.pk_columns, key, strict=True)), conn=conn)
                deleted += 1
        log.debug(
            "bulk delete written on the caller's transaction",
            extra={"extra_data": {"table": self.table_name, "rows": deleted}},
        )
        return deleted

    def _check_delta_scope(self, delta_scope: Any) -> None:
        """refuse a bulk delete whose delete markers could not name their state, before anything is sent
        (:func:`~threetears.core.collections.delta_files.check_delta_scope`): the table declares a
        :attr:`TableSchema.delta_scope` and the delete names no state, or it declares none and the delete names one.

        :param delta_scope: the state the delete named, or None
        :ptype delta_scope: Any
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table's markers carry a state and none is named
        :raises ValueError: when a state is named for a table whose markers carry none
        """
        check_delta_scope(self.table_name, self.schema.delta_scope, delta_scope)

    def _marker_scope(self, delta_scope: Any, scope: Any) -> Any:
        """the scope value every "gone" marker of a versioned table's delete carries when its key does not name the
        scope column: the delete's ``scope``, else its ``delta_scope`` (the state of every key it deletes) when the
        table's delta scope is that column. None when the key names it.

        :param delta_scope: the delete's named state, or None
        :ptype delta_scope: Any
        :param scope: the delete's named scope, or None
        :ptype scope: Any
        :return: the scope value, or None
        :rtype: Any
        :raises ValueError: when the key names the scope and a scope is given anyway, or does not and none can be named
        """
        epoch = self.schema.row_epoch
        assert epoch is not None and epoch.scope is not None  # noqa: S101 -- a versioned table's
        if epoch.scope in self.schema.pk_columns:
            if scope is not None:
                raise ValueError(f"{self.table_name}: its key names its scope {epoch.scope!r}; give no scope")
            return None
        if scope is not None:
            return scope
        if delta_scope is None or self.schema.delta_scope != epoch.scope:
            raise ValueError(
                f"{type(self).__name__}: {self.table_name}'s key does not name its scope {epoch.scope!r}, and each gone "
                f"marker carries it (a restore finds the marker by it): name the scope of every key deleted (scope)"
            )
        return delta_scope

    def _delete_keys_named(
        self, keys: Sequence[Sequence[Any]], loop: asyncio.AbstractEventLoop | None = None
    ) -> list[tuple[Any, ...]]:
        """:meth:`delete_rows`' keys as tuples, each checked for the table's key width; pure, for a thread.

        :param keys: each row's key values, in the schema's key order
        :ptype keys: Sequence[Sequence[Any]]
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
            or off any loop (a few keys: no turn taken)
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the keys, in order
        :rtype: list[tuple[Any, ...]]
        :raises ValueError: when a key is not as wide as the table's key, naming the first such key
        """
        width = len(self.schema.pk_columns)
        named = [tuple(key) for key in (keys if loop is None else stepped(keys, LoopTurns(loop)))]
        wrong = next((key for key in named if len(key) != width), None)
        if wrong is not None:
            raise ValueError(
                f"{self.table_name}.delete_rows: key {wrong!r} has {len(wrong)} values where the key has {width}"
            )
        return named

    def _delete_enrollments(
        self,
        named: Sequence[tuple[Any, ...]],
        held: Mapping[tuple[Any, ...], dict[str, Any]] | None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> tuple[list[Any], list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]]:
        """:meth:`delete_rows`' keys as each is enrolled (a value, or a tuple for a key of many columns) and their
        enrollments, each with the row read before it went (``held``) or none; pure, for a thread.

        :param named: the keys as tuples, in the schema's key order
        :ptype named: Sequence[tuple[Any, ...]]
        :param held: the rows read before the delete, by key; None for a collection that declares no
            :attr:`invalidation_columns`
        :ptype held: Mapping[tuple[Any, ...], dict[str, Any]] | None
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
            or off any loop (a few keys: no turn taken)
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the entity keys and their enrollments (:meth:`CallerTransaction.enrollments`), in key order
        :rtype: tuple[list[Any], list[tuple[BaseCollection, Any, dict[str, Any] | None]]]
        """
        turns = LoopTurns(loop) if loop is not None else None
        keys: Iterable[tuple[Any, ...]] = named if turns is None else stepped(named, turns)
        many = len(self.schema.pk_columns) > 1
        entity_keys = [key if many else key[0] for key in keys]
        seen: list[dict[str, Any] | None] = (
            [None] * len(named)
            if held is None
            else [held.get(key) for key in (named if turns is None else stepped(named, turns))]
        )
        return entity_keys, CallerTransaction.enrollments(self, entity_keys, seen, turns)

    async def delete_arrow(
        self, keys: Any, *, conn: Any, max_rows: int | None = None, delta_scope: Any = None, scope: Any = None
    ) -> int:
        """:meth:`delete_rows` for keys already in an Arrow table: deleted through the store's Arrow write door, the
        keys sent as Arrow and merged by one ``DELETE ... USING`` a request, at most ``max_rows`` keys a request
        (:attr:`BULK_MAX_ROWS` when None), as :meth:`delete_rows` names them a statement: the request is one
        statement under the transaction's statement timeout (:meth:`save_arrow` has the measure).

        As :meth:`delete_rows`: on the caller's transaction only, every key leaving L1 now and settled with the rest
        when the transaction ends. The keys become Python only for the enrolment and the eviction. **Falls back to**
        :meth:`delete_rows` when the store has no Arrow door or declines the keys
        (:meth:`~threetears.core.backends.sql.SqlL3Backend.arrow_write_plan`), and for a collection that declares
        :attr:`invalidation_columns`, whose messages need each row read as it goes.

        :param keys: the keys, a column for each of the table's key columns (other columns are not sent)
        :ptype keys: pyarrow.Table
        :param conn: the caller's connection, its transaction opened by :class:`CallerTransaction`
        :ptype conn: Any
        :param max_rows: the most keys one request names, at least one; :attr:`BULK_MAX_ROWS` when None
        :ptype max_rows: int | None
        :param delta_scope: the state of every key deleted, for a table whose key does not name it (as
            :meth:`delete_rows`)
        :ptype delta_scope: Any
        :param scope: a versioned table's scope of every key, when neither its key nor ``keys`` carry the scope column
            (as :meth:`delete_rows`)
        :ptype scope: Any
        :return: how many keys were named (a key no row holds is not an error)
        :rtype: int
        :raises ValueError: as :meth:`delete_rows`: no :class:`CallerTransaction` open on ``conn``; no durable store,
            a collection that caches absences or defers its L3 writes; when the keys lack a key column; when
            ``max_rows`` is under one; a ``delta_scope`` for a table that declares none
        :raises DeltaFileRefusedError: as :meth:`delete_rows`, when the table's markers carry a state and none is given
        """
        transaction = CallerTransaction.join(conn, writer=f"{type(self).__name__}.delete_arrow")
        self._check_delta_scope(delta_scope)
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"{self.table_name}.delete_arrow: max_rows must be at least 1, got {max_rows}")
        store = self._durable_store()
        if store is None:
            raise ValueError(
                f"{type(self).__name__}.delete_arrow: {self.table_name} has no durable store to delete from"
            )
        if self._negative_cache_writes_advance or self._defers_l3_writes:
            raise ValueError(
                f"{type(self).__name__}.delete_arrow: a bulk delete cannot keep this collection's write "
                f"contract (it caches absences or defers its L3 writes)"
            )
        key_columns = list(self.schema.pk_columns)
        missing = [name for name in key_columns if name not in keys.column_names]
        if missing:
            raise ValueError(f"{self.table_name}.delete_arrow: the keys have no key column {missing[0]!r}")
        if not keys.num_rows:
            return 0
        # the keys read and their enrollments made on a worker thread, as save_arrow's (_arrow_enrollments)
        named, entity_keys, enrollments = await asyncio.to_thread(
            self._arrow_delete_enrollments, keys, asyncio.get_running_loop()
        )
        arrow_store = store if isinstance(store, ArrowWritingStore) and not self.invalidation_columns else None
        sent = keys
        if arrow_store is not None and self.schema.versioned:
            # append-only: the keys become "gone" markers at the delete's epoch, appended (TableSchema.versioned)
            sent = await self._gone_markers(keys, delta_scope, scope, conn)
        plan = (
            arrow_store.arrow_write_plan(
                self.table_name,
                sent.column_names,
                sent.schema.types,
                kind="gone" if self.schema.versioned else "delete",
                conn=conn,
            )
            if arrow_store is not None
            else None
        )
        if arrow_store is None or plan is None:
            # no door for these keys, or messages that carry each row's columns: today's delete
            return await self.delete_rows(named, conn=conn, max_rows=max_rows, delta_scope=delta_scope, scope=scope)
        # enrolled before the delete, so a delete whose outcome is unknown is settled with the rest
        transaction.enroll_all(enrollments)
        transaction.record_deletes(self, named, scope=delta_scope)
        # every key leaves L1 in one write before the store's delete, not one write a key
        await self._evict_l1_many_off_loop(entity_keys)
        await arrow_store.write_arrow(
            self.table_name, sent, plan, conn=conn, max_rows=max_rows if max_rows is not None else self.BULK_MAX_ROWS
        )
        log.debug(
            "Arrow bulk delete written on the caller's transaction",
            extra={"extra_data": {"table": self.table_name, "rows": len(named)}},
        )
        return len(named)

    async def _gone_markers(self, keys: Any, delta_scope: Any, scope: Any, conn: Any) -> Any:
        """a versioned table's "gone" markers of ``keys``, as Arrow: the key's columns, the scope (the keys' own
        column when they carry it, else the delete's ``delta_scope``), the delete's epoch and the flag, each constant
        laid out by Arrow with its declared type (:attr:`TableSchema.gone_marker_columns`).

        :param keys: the keys, a column for each key column
        :ptype keys: pyarrow.Table
        :param delta_scope: the delete's named state, or None
        :ptype delta_scope: Any
        :param scope: the delete's named scope, or None
        :ptype scope: Any
        :param conn: the caller's connection, its transaction's write turn taken
        :ptype conn: Any
        :return: the markers
        :rtype: pyarrow.Table
        :raises ValueError: when the markers' scope cannot be named (:meth:`_marker_scope`)
        :raises UnstampedWriteError: when the transaction took no write's turn
        """
        epoch = self.schema.row_epoch
        assert epoch is not None and epoch.scope is not None  # noqa: S101 -- a versioned table's
        carried = epoch.scope in keys.column_names
        selected = [
            *self.schema.pk_columns,
            *((epoch.scope,) if carried and epoch.scope not in self.schema.pk_columns else ()),
        ]
        constants: dict[str, tuple[Any, Any]] = {
            EPOCH_COLUMN: (self._epoch_stamp(conn, "delete_arrow")[EPOCH_COLUMN], pa.int64()),
            GONE_COLUMN: (True, pa.bool_()),
        }
        if not carried:
            named = self._marker_scope(delta_scope, scope)
            if named is not None:
                column = self.schema.column(epoch.scope)
                constants[epoch.scope] = (named, rail_arrow_types([column])[epoch.scope])
        return await asyncio.to_thread(with_typed_constants, keys.select(selected), constants)

    def _arrow_delete_enrollments(
        self, keys: Any, loop: asyncio.AbstractEventLoop | None = None
    ) -> tuple[list[tuple[Any, ...]], list[Any], list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]]:
        """:meth:`delete_arrow`'s keys as Python, a slice at a time: each a tuple in key order, each as a key is
        enrolled (a value, or a tuple for a key of many columns), and their enrollments with no row. Blocking: run it on
        a thread.

        :param keys: the keys, a column for each of the table's key columns
        :ptype keys: pyarrow.Table
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None off any loop
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: the keys as tuples, as entity keys, and their enrollments (:meth:`CallerTransaction.enrollments`)
        :rtype: tuple[list[tuple[Any, ...]], list[Any], list[tuple[BaseCollection, Any, dict[str, Any] | None]]]
        """
        turns = LoopTurns(loop)
        named = list(stepped(zip(*arrow_python_columns(keys, list(self.schema.pk_columns), turns), strict=True), turns))
        entity_keys = [key if len(key) > 1 else key[0] for key in stepped(named, turns)]
        return named, entity_keys, CallerTransaction.enrollments(self, entity_keys, [None] * len(entity_keys), turns)

    async def _delete_rows_returning(
        self,
        store: ReturningDeletingStore,
        named: list[tuple[Any, ...]],
        transaction: CallerTransaction,
        *,
        conn: Any,
        max_rows: int | None,
    ) -> int:
        """:meth:`delete_rows` for a collection whose messages carry columns, on a store that returns what it deletes.

        Each key is enrolled before the delete with no row, so a delete whose outcome is unknown is still
        settled (its message names the key alone: a broader invalidation, never a missed one), then restated
        with the row the delete returned. One ``DELETE ... RETURNING`` a batch, where a read then a delete took
        two statements (2.7 times one keyed delete on YugabyteDB). A failed delete is counted
        (``threetears_collection_deletes_settled_by_key_total``) and logged, so a message that went out without
        its columns is seen.

        :param store: the collection's durable store
        :ptype store: ReturningDeletingStore
        :param named: the keys, each a tuple in key order
        :ptype named: list[tuple[Any, ...]]
        :param transaction: the caller's transaction
        :ptype transaction: CallerTransaction
        :param conn: the caller's connection
        :ptype conn: Any
        :param max_rows: the most keys one statement names; :attr:`BULK_MAX_ROWS` when None
        :ptype max_rows: int | None
        :return: the keys named
        :rtype: int
        """
        # each key enrolled, and later restated with its row, from a worker thread taking turns with the loop, from
        # the count where a thread costs less than the work: 33,786 keys enrolled in 1.3 ms and restated in 3.6 ms, each
        # in one stretch on the loop (2026-11-16)
        loop = asyncio.get_running_loop() if len(named) >= self.L1_EVICT_OFF_LOOP_KEYS else None
        entity_keys, enrollments = (
            await asyncio.to_thread(self._delete_enrollments, named, None, loop)
            if loop is not None
            else self._delete_enrollments(named, None)
        )
        transaction.enroll_all(enrollments)
        await self._evict_l1_many_off_loop(entity_keys)
        try:
            deleted = await store.delete_many_returning(
                self.table_name, named, max_rows=max_rows or self.BULK_MAX_ROWS, conn=conn
            )
        except BaseException:  # prawduct:allow prawduct/broad-except -- counted and logged, then re-raised
            _deletes_settled_by_key().labels(table=self.table_name).inc(len(named))
            log.warning(
                "a bulk delete failed before it returned its rows: its keys are settled without their columns, "
                "a broader invalidation",
                extra={"extra_data": {"table": self.table_name, "keys": len(named)}},
            )
            raise
        # the rows keyed off the loop for many keys; restated on the loop, where the transaction's state is replaced
        restated = (
            await asyncio.to_thread(self._restated_rows, deleted, loop)
            if loop is not None
            else self._restated_rows(deleted)
        )
        transaction.restate_many(self, restated)
        log.debug(
            "bulk delete written on the caller's transaction, its rows returned",
            extra={"extra_data": {"table": self.table_name, "rows": len(deleted)}},
        )
        return len(named)

    def _restated_rows(
        self,
        deleted: Mapping[tuple[Any, ...], dict[str, Any]],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> dict[Any, dict[str, Any]]:
        """each key :meth:`_delete_rows_returning` enrolled, as enrolled, onto the row its delete returned, for
        :meth:`CallerTransaction.restate_many`; pure, so a thread may make them.

        :param deleted: each deleted row, under its key as a tuple in key order
        :ptype deleted: Mapping[tuple[Any, ...], dict[str, Any]]
        :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
            or off any loop (a few keys: no turn taken)
        :ptype loop: asyncio.AbstractEventLoop | None
        :return: key -> its row, a dict of this call's own
        :rtype: dict[Any, dict[str, Any]]
        """
        turns = LoopTurns(loop) if loop is not None else None
        many = len(self.schema.pk_columns) > 1
        items = deleted.items()
        return {key if many else key[0]: row for key, row in (items if turns is None else stepped(items, turns))}

    async def read_rows_led_by(
        self,
        values: Sequence[Any],
        *,
        columns: Sequence[str] | None = None,
        max_values: int | None = None,
        conn: Any = None,
    ) -> list[dict[str, Any]]:
        """the rows whose leading key column is one of ``values``, read from L3 with every statement led by the key.

        For a table whose key leads with a column many rows share (a feature and its generations, a
        race and its geographies): which keys are held for these values, or a few columns of those
        rows. On YugabyteDB the leading key column is the hashed one, so a statement it does not
        lead reads every row of the table -- ``SELECT DISTINCT``, ``ORDER BY`` the key, a filter on
        a later key column -- and runs into the statement timeout on a big table. Here each
        statement names a batch of values, ``WHERE <lead> = ANY($1)``; an answer that reaches the
        transport's row cap (on the L3 rail
        :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP`), which cuts without saying so,
        is read again in halves; and a value that alone holds that many rows is paged by the rest
        of its key.

        Reads L3 only, around the caches: nothing is read from or written to L1 or L2. The
        store's :meth:`~threetears.core.backends.protocol.KeyLedReadingStore.fetch_led_by` reads
        them when it has one. A store without it is scanned a value at a time through
        ``DurableStore.scan``, which takes no connection, so there a ``conn`` is refused rather
        than read around.

        :param values: the leading key column's values; each is read once, however often named
        :ptype values: Sequence[Any]
        :param columns: columns to read beside the key; the key alone when None. Every row carries
            the key columns either way
        :ptype columns: Sequence[str] | None
        :param max_values: the most values one statement names, at least one;
            :attr:`LED_READ_MAX_VALUES` when None
        :ptype max_values: int | None
        :param conn: a connection to read on (the caller's transaction); the store's own when None.
            Only a :class:`~threetears.core.backends.protocol.KeyLedReadingStore` can read on one
        :ptype conn: Any
        :return: the rows, each keyed by column, in no promised order
        :rtype: list[dict[str, Any]]
        :raises ValueError: when the collection has no durable store, a column is not the table's,
            ``max_values`` is under one, or ``conn`` is given to a store that cannot read on it
        """
        store = self._durable_store()
        if store is None:
            raise ValueError(f"{type(self).__name__}.read_rows_led_by: {self.table_name} has no durable store to read")
        key = list(self.schema.pk_columns)
        wanted = key + [name for name in dict.fromkeys(columns or ()) if name not in key]
        unknown = [name for name in wanted if self.schema.get_column(name) is None]
        if unknown:
            raise ValueError(f"{self.table_name}.read_rows_led_by: the table has no column {unknown[0]!r}")
        per_statement = self.LED_READ_MAX_VALUES if max_values is None else max_values
        if per_statement < 1:
            raise ValueError(f"{self.table_name}.read_rows_led_by: max_values must be at least 1, got {max_values}")
        reads_on_conn = isinstance(store, KeyLedReadingStore)
        if conn is not None and not reads_on_conn:
            raise ValueError(
                f"{self.table_name}.read_rows_led_by: its store has no key-led read and scans without a "
                f"connection, so it cannot read on the conn given; call it without conn"
            )
        unique = list(dict.fromkeys(values))
        if not unique:
            return []
        if isinstance(store, KeyLedReadingStore):
            rows = await store.fetch_led_by(
                self.table_name, unique, columns=wanted, max_values=per_statement, conn=conn
            )
        else:
            rows = []
            for value in unique:
                held = await store.scan(self.table_name, {key[0]: value})
                rows += [{name: row[name] for name in wanted} for row in held]
        log.debug(
            "key-led read",
            extra={"extra_data": {"table": self.table_name, "values": len(unique), "rows": len(rows)}},
        )
        return rows

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist one row to L3 via the :class:`DurableStore` seam.

        the schema-aware backend selects INSERT / CAS-UPDATE from the schema's
        ``on_conflict`` + ``cas_column`` and the ``cas`` (``original_timestamp``)
        argument, drops omitted server-default columns, and casts JSONB / vector
        params — byte-identical to the prior hand-rolled SQL.

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value; ignored
            when :attr:`TableSchema.cas_column` is ``None`` or when
            :attr:`TableSchema.on_conflict` is not ``"update"``
        :ptype original_timestamp: datetime | None
        :param conn: optional asyncpg-compatible connection that overrides
            :attr:`l3_pool` for this single write so it commits atomically with
            the caller's transaction. ``None`` defers to the collection's pool.
        :ptype conn: Any
        :return: rows affected reported by the backend (0 on CAS failure or
            ON CONFLICT DO NOTHING miss, 1 on success)
        :rtype: int
        """
        store = self._durable_store()
        if store is None:
            return 0
        fenced = (
            self.schema.cas_column is not None
            and self.schema.on_conflict == "update"
            and original_timestamp is not None
        )
        data = {**data, **self._epoch_stamp(conn, "save_to_store")}
        self._record_one(data, conn, "save_to_store", fenced=fenced)
        pk = list(self.schema.pk_columns)
        return await store.upsert(
            self.schema.name,
            data,
            pk=pk,
            on_conflict=self.schema.on_conflict,
            cas=original_timestamp,
            conn=conn,
        )

    @property
    def saves_many_to_store(self) -> bool:
        """whether :meth:`save_many_to_store` saves rows as :meth:`save_to_store` would: a table upserted on conflict,
        on a store that saves many rows in a few statements.

        :return: whether it does
        :rtype: bool
        """
        return self.schema.on_conflict == "update" and isinstance(self._durable_store(), BulkDurableStore)

    async def save_many_to_store(self, rows: list[dict[str, Any]]) -> int:
        """save many rows to L3 in a few statements, as :meth:`save_to_store` saves each with no fence.

        A store hook of the write path, as :meth:`save_to_store` is: the L2-first saver writes a backlog of rows
        already committed and announced at L2 through it (:mod:`threetears.core.collections.l2_first`), and only
        where :attr:`saves_many_to_store`.

        :param rows: the rows, each keyed by column
        :ptype rows: list[dict[str, Any]]
        :return: rows written
        :rtype: int
        :raises ValueError: when the collection does not save many rows at once (:attr:`saves_many_to_store`)
        """
        self._refuse_unfiled("save_many_to_store")
        store = self._durable_store()
        if self.schema.on_conflict != "update" or not isinstance(store, BulkDurableStore):
            raise ValueError(f"{self.table_name}: save_many_to_store needs a bulk store and an upsert on conflict")
        written: int = await store.upsert_many(
            self.schema.name, rows, max_rows=self.BULK_MAX_ROWS, max_bytes=self.BULK_MAX_BYTES
        )
        return written

    async def delete_from_store(self, entity_id: Any) -> None:
        """delete one row from L3 by primary key via the :class:`DurableStore` seam.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        self._refuse_unfiled("delete_from_store")
        store = self._durable_store()
        if store is None:
            return
        pk_values = self.normalize_pk(entity_id)
        pk = dict(zip(self.schema.pk_columns, pk_values, strict=True))
        await store.delete(self.schema.name, pk)

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize row dict to JSON bytes for L2 storage.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: UTF-8 JSON bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_default).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 back to a row dict.

        :param data: UTF-8 JSON bytes produced by :meth:`serialize`
        :ptype data: bytes
        :return: row dict with typed values
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        result: dict[str, Any] = {}
        for key, value in raw.items():
            col = self.schema.get_column(key)
            if col is None:
                result[key] = value
            else:
                result[key] = _decode_l2_value(col, value)
        return result

    # --- row helpers ---

    def _coerce_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """apply read-boundary coercion to every known column in a row.

        retained as an instance method because domain query methods on subclasses
        (custom ``find_by_*`` / search SQL) call ``self._coerce_row(dict(row))`` on
        their own asyncpg rows. delegates to the relocated pure
        :func:`threetears.core.backends.schema_sql.coerce_row` so the coercion logic
        lives in exactly one place.

        :param row: row dict as returned by asyncpg
        :ptype row: dict[str, Any]
        :return: same-shape dict with coerced values
        :rtype: dict[str, Any]
        """
        return _coerce_row_fn(self.schema, row)


async def _all_or_first_failure(calls: Sequence[Coroutine[Any, Any, int]]) -> list[int]:
    """every call's answer, run side by side; the first failure stops the rest (cancelled, then awaited) and is
    raised as itself, not inside a group.

    :param calls: the calls, not started
    :ptype calls: Sequence[Coroutine[Any, Any, int]]
    :return: each call's answer, in order
    :rtype: list[int]
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(call) for call in calls]
    except* BaseException as failed:
        raise failed.exceptions[0] from None
    return [task.result() for task in tasks]


def _hand_commit(
    on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None],
    files: Mapping[str, str],
    generations: Mapping[str, str],
) -> None:
    """tell a load's caller one commit it made: its delta files, and the generations its advance wrote for those tables.

    :param on_commit: the caller's callback (:meth:`SchemaBackedCollection.load_files`)
    :ptype on_commit: Callable[[Mapping[str, str], Mapping[str, str]], None]
    :param files: table -> the commit's delta file
    :ptype files: Mapping[str, str]
    :param generations: table -> the token written, any table; only the files' tables are handed
    :ptype generations: Mapping[str, str]
    :return: nothing
    :rtype: None
    """
    on_commit(dict(files), {table: token for table, token in generations.items() if table in files})


def _text_vector_width(text: str) -> int | None:
    """the width of pgvector's bracketed text, or ``None`` for text that is not one (the database judges it)."""
    try:
        decoded = _decode_vector(text)
    except ValueError:  # json.JSONDecodeError is a ValueError: text that is not a vector
        decoded = None
    return None if decoded is None else len(decoded)


def _shares_a_key(keys: Sequence[Any], turns: LoopTurns | None) -> bool:
    """whether two of ``keys`` are equal, read with the thread's turns between slices (a set of 33,786 keys of seven
    texts made in one call held the lock 3-5 ms).

    :param keys: the keys
    :ptype keys: Sequence[Any]
    :param turns: the thread's turns with the caller's loop; None to read them in one call
    :ptype turns: LoopTurns | None
    :return: whether a key repeats
    :rtype: bool
    """
    if turns is None:
        return len(set(keys)) != len(keys)
    held: set[Any] = set()
    for start in range(0, len(keys), STEP_ITEMS):
        held.update(keys[start : start + STEP_ITEMS])
        turns.take()
    return len(held) != len(keys)


def _refuse_misfit_vectors(schema: TableSchema, row: Mapping[str, Any]) -> None:
    """refuse a row carrying a vector that is not its column's declared width.

    a ``vector(N)`` column cannot hold any other width. a value is measured as a sequence, or as
    pgvector's bracketed text; ``None`` and a value of neither shape pass, for the database to
    judge.

    :param schema: the schema the row is written to
    :ptype schema: TableSchema
    :param row: the row, keyed by column
    :ptype row: Mapping[str, Any]
    :return: nothing
    :rtype: None
    :raises ValueError: naming the table, the column, its declared width and the vector's width
    """
    for col in schema.columns:
        if col.column_type != VECTOR_TYPE or col.vector_dim is None:
            continue
        value = row.get(col.name)
        width: int | None = None
        if isinstance(value, str):
            width = _text_vector_width(value)
        elif value is not None and hasattr(value, "__len__"):
            width = len(value)
        if width is not None and width != col.vector_dim:
            raise ValueError(
                f"{schema.name}.{col.name} holds {col.vector_dim}-dimensional embeddings, and this row's "
                f"has {width} dimensions (expected {col.vector_dim}, got {width}); nothing was written. "
                f"Embed with a model of the configured width, or configure embedding_dim={width} (an "
                f"existing schema then needs resize_embedding_width)."
            )


def collection_for_schema(
    schema: TableSchema, *, entity_class: type[BaseEntity] | None = None, module: str | None = None
) -> type[SchemaBackedCollection[Any]]:
    """a collection class over ``schema``, for a table declared as data rather than as a class.

    A pod whose tables come from a catalogue (one per map layer, one per warehouse table it
    mirrors) builds each one's collection from its schema instead of writing a class per
    table. Build each class once and keep it: every call makes a new class.

    :param schema: the table
    :ptype schema: TableSchema
    :param entity_class: the entities the collection builds; for a single-column key, a plain
        :class:`~threetears.core.entities.base.BaseEntity` keyed on that column when None. A
        composite key needs one: an entity's ``id`` names its bare row id, not the column the key
        leads with, and which column that is only the caller knows
    :ptype entity_class: type[BaseEntity] | None
    :param module: the module the built classes say they are defined in (``__module__``), so a census of a
        consumer's modules counts them as the consumer's; ``None`` (the default) is the calling module
    :ptype module: str | None
    :return: a :class:`SchemaBackedCollection` subclass over ``schema``, carrying the schema's
        ``write_generation`` declaration (switched on, kept in L2, or opted out with its reason)
    :rtype: type[SchemaBackedCollection[Any]]
    :raises ValueError: for a composite key with no ``entity_class``
    """
    if module is None:
        caller = inspect.currentframe()
        module = str(caller.f_back.f_globals.get("__name__", __name__)) if caller and caller.f_back else __name__
    defined_in: str = module
    described = schema
    entity = entity_class
    if entity is None and len(schema.pk_columns) > 1:
        raise ValueError(
            f"{schema.name}: a composite key {schema.pk_columns} needs an entity_class naming the row's "
            f"own id as its primary_key_field; the first key column is not guessed"
        )
    if entity is None:
        entity = type(
            f"{schema.name}_row",
            (BaseEntity,),
            {
                "__doc__": f"one row of {schema.name}.",
                "primary_key_field": schema.pk_columns[0],
                "__module__": defined_in,
            },
        )
    entity_type: type[BaseEntity] = entity

    class _SchemaCollection(SchemaBackedCollection[Any]):
        """one schema's rows."""

        primary_key_column: str | tuple[str, ...] = described.primary_key
        # the schema, set in the body so the class's definition reads its write_generation declaration from
        # it, whichever it is (BaseCollection.__init_subclass__), registration included
        schema = described
        __module__ = defined_in

        @property
        def table_name(self) -> str:
            """the schema's table.

            :return: the table name
            :rtype: str
            """
            return schema.name

        @property
        def entity_class(self) -> type[Any]:
            """the entity class.

            :return: the entity class
            :rtype: type[Any]
            """
            return entity_type

    _SchemaCollection.__name__ = f"SchemaCollection[{schema.name}]"
    _SchemaCollection.__qualname__ = _SchemaCollection.__name__
    return _SchemaCollection
