"""Schema-aware SQL generation + per-column codec for :class:`SqlL3Backend`.

These are the SQL builders and value-coercion functions that used to live on
:class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`. They are
relocated here as **pure functions** so the durable SQL backend
(:class:`~threetears.core.backends.sql.SqlL3Backend`) can route the standard CRUD
lifecycle through the :class:`~threetears.core.backends.protocol.DurableStore` seam while
emitting **byte-identical** SQL to the hand-rolled collection path.

``TableSchema`` / ``Column`` are imported only under ``TYPE_CHECKING``: every function
here reads the schema/column through its duck-typed surface
(``schema.columns`` / ``schema.name`` / ``schema.pk_columns`` / ``schema.cas_column`` /
``schema.on_conflict`` / ``schema.mutable_columns()`` / ``schema.column(name)`` /
``column.column_type`` / ``column.name`` / ``column.nullable`` / ``column.server_default``),
so there is **no runtime ``backends → collections`` import** (which would be a cycle —
``collections`` imports ``backends``).
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import Column, TableSchema

__all__ = [
    "build_cas_params",
    "build_cas_update_sql",
    "build_delete_sql",
    "build_fetch_sql",
    "build_insert_params",
    "build_insert_sql",
    "build_select_column_list",
    "build_where_pk",
    "cas_mutable_columns_for_data",
    "coerce_datetime_for_read_tz",
    "coerce_datetime_for_write_tz",
    "coerce_row",
    "coerce_uuid",
    "decode_jsonb",
    "decode_l2_value",
    "decode_vector",
    "encode_jsonb",
    "encode_vector",
    "insert_columns_for_data",
    "json_default",
    "normalize_read_value",
    "normalize_write_value",
    "pull_value",
    "render_param",
]

# column type tags (string-duplicated here so this module stays import-cycle-free;
# the canonical definitions live in ``schema_backed`` and these mirror them verbatim).
_UUID_TYPE = "uuid"
_DATETIMETZ_TYPE = "datetimetz"
_JSONB_TYPE = "jsonb"
_BYTES_TYPE = "bytes"
_INT_TYPE = "int"
_BIGINT_TYPE = "bigint"
_BOOL_TYPE = "bool"
_NUMERIC_TYPE = "numeric"
_VECTOR_TYPE = "vector"
_TSVECTOR_TYPE = "tsvector"


# ── codec (verbatim from schema_backed) ──────────────────────────────────────────


def coerce_uuid(value: Any) -> UUID | None:
    """coerce a value to a stdlib :class:`UUID`.

    asyncpg returns ``asyncpg.pgproto.pgproto.UUID`` for UUID columns,
    which is a subclass of stdlib UUID. SQLite's parameter binder
    rejects the subclass (it special-cases stdlib UUID only), so every
    read coerces to stdlib UUID unconditionally.

    :param value: UUID-shaped input (stdlib UUID, pgproto UUID, or str)
    :ptype value: Any
    :return: stdlib UUID or ``None`` pass-through
    :rtype: UUID | None
    """
    if value is None:
        result: UUID | None = None
    elif type(value) is UUID:  # noqa: E721 -- exact type match, see docstring
        result = value
    elif isinstance(value, UUID):  # pgproto subclass
        result = UUID(str(value))
    else:
        result = UUID(str(value))
    return result


def coerce_datetime_for_write_tz(value: Any) -> datetime | None:
    """coerce a datetime to aware-UTC form for TIMESTAMPTZ column writes.

    asyncpg's TIMESTAMPTZ codec runs ``obj.astimezone(utc)`` on every
    bound parameter. for a naive input the call interprets the wall-
    clock value as the **client's local timezone**, so a naive value
    that semantically represents UTC gets shifted by the host's local
    offset before it reaches the database. this coercion eliminates the
    ambiguity: naive inputs are interpreted as UTC; aware inputs are
    normalized via ``astimezone(UTC)``. ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: aware-UTC datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)
    else:
        result = value.astimezone(UTC)
    return result


def coerce_datetime_for_read_tz(value: Any) -> datetime | None:
    """coerce a TIMESTAMPTZ read value to aware-UTC form.

    asyncpg's TIMESTAMPTZ decoder already returns aware-UTC datetimes,
    so the typical path is a passthrough. naive values (from a
    non-asyncpg source) are wrapped with ``UTC``; aware values are
    normalized to UTC tzinfo so equality comparisons are byte-stable.
    ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: aware-UTC datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)
    else:
        result = value.astimezone(UTC)
    return result


def encode_vector(value: Any) -> str | None:
    """encode a vector for pgvector's ``::vector`` cast.

    asyncpg has no native pgvector codec; values must be passed as the
    literal bracketed textual representation (``"[1.0, 2.0, ...]"``).
    list inputs get JSON-encoded at the WRITE boundary. already-encoded
    strings pass through. ``None`` pass-through.

    :param value: list of floats, pre-encoded string, or ``None``
    :ptype value: Any
    :return: textual vector, passthrough string, or ``None``
    :rtype: str | None
    """
    if value is None:
        result: str | None = None
    elif isinstance(value, str):
        result = value
    elif isinstance(value, list):
        result = json.dumps(value)
    else:
        result = str(value)
    return result


def decode_vector(value: Any) -> list[float] | None:
    """decode a pgvector textual vector back to ``list[float]``.

    :param value: textual vector (e.g. ``"[1.0,2.0]"``), list passthrough,
        or ``None``
    :ptype value: Any
    :return: list of floats or ``None``
    :rtype: list[float] | None
    """
    if value is None:
        result: list[float] | None = None
    elif isinstance(value, list):
        result = [float(v) for v in value]
    elif isinstance(value, str):
        parsed = json.loads(value)
        result = [float(v) for v in parsed] if isinstance(parsed, list) else None
    else:
        result = None
    return result


def encode_jsonb(value: Any) -> Any:
    """validate / pass through a JSONB column value for asyncpg.

    public canonical encoder: every collection that writes a JSONB column --
    including hand-rolled custom-SQL upserts in sibling packages -- binds the
    value through this function as a NATIVE python object (``$N``, no
    ``::text::jsonb`` cast) and lets the registered codec apply the single
    ``json.dumps``. this is the one jsonb-encode step in the platform; a
    second ``json.dumps`` (or a ``::text::jsonb`` cast over a pre-encoded
    string) is the double-encode bug this centralization removes.

    so this function is a typed pass-through: dict/list/None flow through to
    the codec; pre-encoded strings are decoded back into structures so the
    codec receives a single Python value to encode. anything else fails fast.

    :param value: dict, list, pre-encoded JSON string, or ``None``
    :ptype value: Any
    :return: dict / list / ``None`` ready for asyncpg's jsonb codec
    :rtype: Any
    :raises TypeError: when ``value`` is not dict / list / str / None
    """
    if value is None:
        result: Any = None
    elif isinstance(value, (dict, list)):
        result = value
    elif isinstance(value, str):
        # legacy callers (e.g. test fixtures) handing a pre-encoded
        # JSON string; decode back to a structure so the codec applies
        # exactly one encoding step. malformed input surfaces as a
        # clean ValueError from json.loads rather than as silent
        # double-encoding deep in the write path.
        result = json.loads(value)
    else:
        raise TypeError(
            f"JSONB column requires dict / list / str / None; got {type(value).__name__}",
        )
    return result


def decode_jsonb(value: Any) -> Any:
    """decode a JSONB column value into a Python dict / list.

    asyncpg returns JSONB as a string unless a codec is registered;
    callers downstream expect the decoded structure. malformed payload
    surfaces as the raw string so the corruption is visible rather than
    silently swallowed.

    :param value: JSON string, decoded object, or ``None``
    :ptype value: Any
    :return: decoded structure (typically ``dict``), passthrough, or
        ``None``
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif isinstance(value, str) and value:
        try:
            result = json.loads(value)
        except ValueError, TypeError:
            result = value
    else:
        result = value
    return result


def json_default(obj: object) -> Any:
    """default handler for :func:`json.dumps` covering platform types.

    :param obj: value that :func:`json.dumps` cannot encode natively
    :ptype obj: object
    :return: JSON-compatible representation
    :rtype: Any
    :raises TypeError: when ``obj`` type is not covered
    """
    if isinstance(obj, UUID):
        result: Any = str(obj)
    elif isinstance(obj, datetime):
        result = obj.isoformat()
    elif isinstance(obj, bytes):
        result = base64.b64encode(obj).decode("ascii")
    elif isinstance(obj, Decimal):
        # NUMERIC columns carry Decimal; emit the exact decimal string so
        # ``decode_l2_value`` rehydrates it losslessly (str avoids the
        # float-rounding a JSON number would introduce).
        result = str(obj)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return result


def decode_l2_value(column: Column, value: Any) -> Any:
    """decode a single field from the L2 JSON payload back to native type.

    :param column: column descriptor controlling the coercion rule
    :ptype column: Column
    :param value: JSON-decoded raw value
    :ptype value: Any
    :return: value coerced to the column's native Python type
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif column.column_type == _UUID_TYPE and isinstance(value, str):
        result = UUID(value)
    elif column.column_type == _DATETIMETZ_TYPE and isinstance(value, str):
        # isoformat strings carrying ``+00:00`` round-trip to aware UTC
        # naturally; strings without an offset (legacy L2 payloads from
        # before this column type existed) are interpreted as UTC.
        parsed = datetime.fromisoformat(value)
        result = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    elif column.column_type == _BYTES_TYPE and isinstance(value, str):
        result = base64.b64decode(value.encode("ascii"))
    elif column.column_type == _BOOL_TYPE and isinstance(value, (bool, int)):
        result = bool(value)
    elif column.column_type in (_INT_TYPE, _BIGINT_TYPE) and isinstance(value, (int, float)):
        result = int(value)
    elif column.column_type == _NUMERIC_TYPE and isinstance(value, (str, int, float)):
        # NUMERIC round-trips as the exact decimal string json_default emits;
        # tolerate int/float too (legacy L2 payloads, hand-built rows).
        result = Decimal(str(value))
    elif column.column_type == _VECTOR_TYPE:
        result = decode_vector(value)
    elif column.column_type == _JSONB_TYPE:
        # JSONB round-trips through JSON natively; value is already the
        # decoded structure
        result = value
    else:
        result = value
    return result


# ── value normalization ──────────────────────────────────────────────────────────


def normalize_write_value(column: Column, value: Any) -> Any:
    """apply the column's write-boundary coercion.

    :param column: column descriptor
    :ptype column: Column
    :param value: raw value from the caller's data dict
    :ptype value: Any
    :return: coerced value ready to pass to asyncpg
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif column.column_type == _UUID_TYPE:
        result = coerce_uuid(value)
    elif column.column_type == _DATETIMETZ_TYPE:
        result = coerce_datetime_for_write_tz(value)
    elif column.column_type == _JSONB_TYPE:
        result = encode_jsonb(value)
    elif column.column_type == _VECTOR_TYPE:
        result = encode_vector(value)
    else:
        result = value
    return result


def normalize_read_value(column: Column, value: Any) -> Any:
    """apply the column's read-boundary coercion.

    :param column: column descriptor
    :ptype column: Column
    :param value: raw value from asyncpg
    :ptype value: Any
    :return: coerced value ready to hand back to callers
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif column.column_type == _UUID_TYPE:
        # always coerce to stdlib UUID -- pgproto UUID is a subclass
        # that SQLite's binder rejects
        result = coerce_uuid(value)
    elif column.column_type == _DATETIMETZ_TYPE:
        result = coerce_datetime_for_read_tz(value)
    elif column.column_type == _JSONB_TYPE:
        result = decode_jsonb(value)
    elif column.column_type == _VECTOR_TYPE:
        result = decode_vector(value)
    else:
        result = value
    return result


def coerce_row(schema: TableSchema, row: dict[str, Any]) -> dict[str, Any]:
    """apply read-boundary coercion to every known column in a row.

    columns not declared in the schema pass through untouched so
    ad-hoc joined projections still round-trip cleanly.

    :param schema: table schema describing the declared columns
    :ptype schema: TableSchema
    :param row: row dict as returned by asyncpg
    :ptype row: dict[str, Any]
    :return: same-shape dict with coerced values
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {}
    for key, value in row.items():
        col = schema.get_column(key)
        if col is None:
            result[key] = value
        else:
            result[key] = normalize_read_value(col, value)
    return result


def pull_value(schema: TableSchema, data: dict[str, Any], column: Column) -> Any:
    """read a column's value from the caller's data dict.

    :param schema: table schema (used for the error message only)
    :ptype schema: TableSchema
    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :param column: column descriptor
    :ptype column: Column
    :return: raw value from the dict
    :rtype: Any
    :raises KeyError: when the column is required (``nullable=False``)
        and the key is missing from ``data``
    """
    if column.name in data:
        result: Any = data[column.name]
    elif column.nullable:
        result = None
    else:
        raise KeyError(
            f"{schema.name}.save_to_store: required column {column.name!r} missing from data",
        )
    return result


# ── SQL generation ───────────────────────────────────────────────────────────────


def build_where_pk(schema: TableSchema, start: int = 1) -> tuple[str, tuple[str, ...]]:
    """build ``WHERE pk...`` fragment for fetch / delete / CAS.

    :param schema: table schema
    :ptype schema: TableSchema
    :param start: first positional-parameter index
    :ptype start: int
    :return: (where clause fragment, tuple of pk column names in
        declared order)
    :rtype: tuple[str, tuple[str, ...]]
    """
    pk_cols = schema.pk_columns
    parts = [f"{col} = ${start + i}" for i, col in enumerate(pk_cols)]
    return " AND ".join(parts), pk_cols


def render_param(column: Column, idx: int) -> str:
    """render ``$N`` with optional ``::jsonb`` / ``::text::public.vector`` cast.

    :param column: column descriptor
    :ptype column: Column
    :param idx: positional-parameter index (1-based)
    :ptype idx: int
    :return: placeholder string ready for SQL interpolation
    :rtype: str
    """
    if column.column_type == _JSONB_TYPE:
        result = f"${idx}::jsonb"
    elif column.column_type == _VECTOR_TYPE:
        # schema-qualify ``public.vector`` (not bare ``vector``): a Collection
        # accessed through the agent L3 proxy runs under a search_path scoped
        # to the per-agent schema (``agent_<hex>``), which does not include the
        # ``public`` schema where the pgvector type lives -- bare ``::vector``
        # fails to parse with ``type "vector" does not exist``. the ``::text``
        # pin encodes the bracketed-text value as text so a pool without the
        # pgvector codec registered can bind it and let Postgres cast
        # text -> vector server-side (the documented YB write pattern).
        result = f"${idx}::text::public.vector"
    else:
        result = f"${idx}"
    return result


def insert_columns_for_data(schema: TableSchema, data: dict[str, Any]) -> list[Column]:
    """select the column list to emit in INSERT SQL + params.

    columns with a declared ``server_default`` are OMITTED from the
    INSERT when the caller did not supply a value, so the server-side
    default applies.

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: ordered list of columns to emit; subset of
        ``schema.columns`` preserving declared order
    :rtype: list[Column]
    """
    included: list[Column] = []
    for col in schema.columns:
        if col.server_default is not None and col.name not in data:
            # let Postgres apply the server-side default
            continue
        included.append(col)
    return included


def build_insert_sql(schema: TableSchema, data: dict[str, Any] | None = None) -> str:
    """build INSERT SQL with the conflict clause selected by schema.

    column order matches ``schema.columns`` exactly, FILTERED to the
    subset selected by :func:`insert_columns_for_data`: columns with a
    declared ``server_default`` that the caller omitted are dropped from
    the column list AND the parameter list so the server-side default
    applies.

    branches on ``schema.on_conflict``:

    * ``"raise"``: plain INSERT with no ON CONFLICT clause
    * ``"ignore"``: ``INSERT ... ON CONFLICT (pk) DO NOTHING``
    * ``"update"`` (default): ``INSERT ... ON CONFLICT (pk) DO UPDATE SET
      <mutable>``; falls back to ``DO NOTHING`` when every non-pk column
      is immutable so existing rows are not clobbered.

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict to drive column selection. ``None`` is treated
        as an empty dict (every server-default column omitted).
    :ptype data: dict[str, Any] | None
    :return: parameterized INSERT SQL
    :rtype: str
    """
    cols = insert_columns_for_data(schema, data or {})
    col_names = ", ".join(c.name for c in cols)
    placeholders = ", ".join(render_param(c, i + 1) for i, c in enumerate(cols))
    sql = f"INSERT INTO {schema.name} ({col_names}) VALUES ({placeholders})"
    pk_cols = ", ".join(schema.pk_columns)
    if schema.on_conflict == "raise":
        result = sql
    elif schema.on_conflict == "ignore":
        result = f"{sql} ON CONFLICT ({pk_cols}) DO NOTHING"
    else:
        # DO UPDATE SET references EXCLUDED.<col> for each mutable
        # column. Intersect with the columns ACTUALLY emitted in
        # the INSERT -- an EXCLUDED reference to a column that
        # was not in the column list would be a SQL error.
        emitted = {c.name for c in cols}
        mutable_emitted = [c for c in schema.mutable_columns() if c.name in emitted]
        if mutable_emitted:
            set_clause = ", ".join(f"{c.name} = EXCLUDED.{c.name}" for c in mutable_emitted)
            result = f"{sql} ON CONFLICT ({pk_cols}) DO UPDATE SET {set_clause}"
        else:
            # pk-only table or every non-pk column is immutable or
            # omitted-via-server-default: emit DO NOTHING so
            # existing rows are not clobbered.
            result = f"{sql} ON CONFLICT ({pk_cols}) DO NOTHING"
    return result


def cas_mutable_columns_for_data(schema: TableSchema, data: dict[str, Any]) -> list[Column]:
    """select the mutable column list to emit in CAS UPDATE SQL + params.

    symmetric counterpart to :func:`insert_columns_for_data` for the CAS
    UPDATE path: a mutable column declared with ``server_default`` is
    optional in the caller's payload and dropped when omitted, and a
    ``None`` value on a NOT NULL server-default column is treated as
    "use existing DB value".

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: ordered list of mutable columns to emit
    :rtype: list[Column]
    """
    included: list[Column] = []
    for col in schema.mutable_columns():
        if col.server_default is not None and col.name not in data:
            continue
        if col.server_default is not None and not col.nullable and col.name in data and data[col.name] is None:
            # L1 round-trip surfaced a None on a NOT NULL
            # server-default column. Preserve existing DB value.
            continue
        included.append(col)
    return included


def build_cas_update_sql(schema: TableSchema, data: dict[str, Any] | None = None) -> str:
    """build UPDATE SQL for the CAS (optimistic-concurrency) path.

    parameter ordering: pk values ($1..$Npk), then mutable columns
    (filtered by :func:`cas_mutable_columns_for_data`) in declared order,
    then the CAS fence value as the last parameter.

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict to drive column selection. ``None`` is treated
        as an empty dict.
    :ptype data: dict[str, Any] | None
    :return: parameterized UPDATE SQL
    :rtype: str
    :raises RuntimeError: when the schema has no ``cas_column``
    """
    if schema.cas_column is None:
        raise RuntimeError(
            f"TableSchema(name={schema.name!r}): CAS UPDATE requires cas_column to be set",
        )
    pk_cols = schema.pk_columns
    mutable = cas_mutable_columns_for_data(schema, data or {})
    # $1..$Npk are pk values; $Npk+1..$Npk+Nmut are mutable values
    pk_where = " AND ".join(f"{col} = ${i + 1}" for i, col in enumerate(pk_cols))
    set_parts: list[str] = []
    next_idx = len(pk_cols) + 1
    for col in mutable:
        set_parts.append(f"{col.name} = {render_param(col, next_idx)}")
        next_idx += 1
    # final $ is the CAS fence value
    cas_idx = next_idx
    set_clause = ", ".join(set_parts)
    result = f"UPDATE {schema.name} SET {set_clause} WHERE {pk_where} AND {schema.cas_column} = ${cas_idx}"
    return result


def build_select_column_list(schema: TableSchema) -> str:
    """render the by-pk SELECT projection from declared schema columns.

    projects ONLY the schema's declared columns -- never ``SELECT *`` --
    so a real table column the schema does not declare is never read. a
    declared codec-less column (``VECTOR_TYPE`` / ``TSVECTOR_TYPE``) is
    cast to ``::text`` so asyncpg returns the string form instead of
    raising on the missing codec.

    :param schema: table schema
    :ptype schema: TableSchema
    :return: comma-joined projection list in declared column order
    :rtype: str
    """
    text_cast_types = (_VECTOR_TYPE, _TSVECTOR_TYPE)
    parts: list[str] = []
    for col in schema.columns:
        if col.column_type in text_cast_types:
            parts.append(f"{col.name}::text AS {col.name}")
        else:
            parts.append(col.name)
    return ", ".join(parts)


def build_fetch_sql(schema: TableSchema) -> str:
    """build SELECT SQL for pk-based fetch.

    :param schema: table schema
    :ptype schema: TableSchema
    :return: parameterized SELECT SQL
    :rtype: str
    """
    where, _ = build_where_pk(schema, start=1)
    columns = build_select_column_list(schema)
    return f"SELECT {columns} FROM {schema.name} WHERE {where}"


def build_delete_sql(schema: TableSchema) -> str:
    """build DELETE SQL for pk-based delete.

    :param schema: table schema
    :ptype schema: TableSchema
    :return: parameterized DELETE SQL
    :rtype: str
    """
    where, _ = build_where_pk(schema, start=1)
    return f"DELETE FROM {schema.name} WHERE {where}"


def build_insert_params(schema: TableSchema, data: dict[str, Any]) -> list[Any]:
    """build the parameter list for the INSERT path in declared order.

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: parameter list matching the SQL column order
    :rtype: list[Any]
    """
    return [normalize_write_value(c, pull_value(schema, data, c)) for c in insert_columns_for_data(schema, data)]


def build_cas_params(
    schema: TableSchema,
    data: dict[str, Any],
    original_timestamp: datetime,
) -> list[Any]:
    """build the parameter list for the CAS UPDATE path.

    layout: pk values in declared order, then mutable columns in
    declared order, then the CAS fence value last.

    :param schema: table schema
    :ptype schema: TableSchema
    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :param original_timestamp: pre-mutation CAS fence value
    :ptype original_timestamp: datetime
    :return: parameter list matching the CAS UPDATE placeholder layout
    :rtype: list[Any]
    """
    params: list[Any] = []
    for pk_name in schema.pk_columns:
        pk_col = schema.column(pk_name)
        params.append(normalize_write_value(pk_col, pull_value(schema, data, pk_col)))
    # filter mutable columns to those the caller supplied (or that
    # have no server_default) so the params list aligns positionally
    # with the SET clause produced by build_cas_update_sql(data).
    for col in cas_mutable_columns_for_data(schema, data):
        params.append(normalize_write_value(col, pull_value(schema, data, col)))
    if schema.cas_column is None:
        # unreachable -- build_cas_update_sql enforces the guard
        raise RuntimeError(
            f"TableSchema(name={schema.name!r}): CAS path reached without cas_column",
        )
    cas_col = schema.column(schema.cas_column)
    params.append(normalize_write_value(cas_col, original_timestamp))
    return params
