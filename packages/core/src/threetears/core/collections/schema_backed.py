"""schema-backed collection primitive.

collapses the CRUD boilerplate that every
:class:`~threetears.core.collections.base.BaseCollection` subclass used
to hand-roll (``fetch_from_postgres`` / ``save_to_postgres`` /
``delete_from_postgres`` / ``serialize`` / ``deserialize``) into one
schema-driven implementation. subclasses declare a
:class:`TableSchema` as a class attribute and stop writing SQL by hand
for the standard row lifecycle. domain-specific query methods
(``find_by_agent`` / ``hybrid_search`` / etc.) stay on subclasses
because query shape is legitimately per-collection; CRUD is not.

per CLAUDE.md (14-eng-ai-bot) "NO BACKWARDS-COMPATIBILITY SHIMS": a
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

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic
from uuid import UUID

from threetears.core.collections.base import BaseCollection, EntityT
from threetears.observe import get_logger

__all__ = [
    "BOOL_TYPE",
    "BYTES_TYPE",
    "Column",
    "DATETIME_TYPE",
    "INT_TYPE",
    "JSONB_TYPE",
    "SchemaBackedCollection",
    "STRING_TYPE",
    "TableSchema",
    "UUID_TYPE",
    "VECTOR_TYPE",
]

log = get_logger(__name__)


# sentinel strings for column type tags. chosen over real Python types
# because several logical types collapse to the same Python type
# (``list[float]`` vs ``list[dict]`` for vector vs jsonb arrays) and
# because the serialization layer already distinguishes types by tag
# rather than by runtime ``isinstance`` checks.
UUID_TYPE = "uuid"
STRING_TYPE = "string"
DATETIME_TYPE = "datetime"
JSONB_TYPE = "jsonb"
BYTES_TYPE = "bytes"
INT_TYPE = "int"
BOOL_TYPE = "bool"
VECTOR_TYPE = "vector"


@dataclass(frozen=True)
class Column:
    """column descriptor for :class:`TableSchema`.

    :cvar name: column name (matches the L3 DDL and the row-dict key)
    :cvar column_type: one of the module-level type tags
        (``UUID_TYPE``, ``STRING_TYPE``, etc.); used by
        :meth:`SchemaBackedCollection._normalize_write_value` /
        :meth:`SchemaBackedCollection._normalize_read_value` to pick
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

    :param name: column name
    :ptype name: str
    :param column_type: one of the module-level type tags
    :ptype column_type: str
    :param immutable: INSERT-only flag; defaults to ``False``
    :ptype immutable: bool
    :param nullable: defaultable flag; defaults to ``False``
    :ptype nullable: bool
    """

    name: str
    column_type: str
    immutable: bool = False
    nullable: bool = False


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
        :meth:`SchemaBackedCollection.save_to_postgres`, the generator
        emits a separate UPDATE fenced on ``cas_column = $N``. when
        ``None``, the ``original_timestamp`` argument is silently
        ignored
    :cvar append_only: when ``True``, the generator emits plain INSERT
        with no ``ON CONFLICT`` clause. duplicate-key conflicts raise
        from asyncpg naturally. intended for journal tables
    """

    name: str
    primary_key: str | tuple[str, ...]
    columns: list[Column]
    cas_column: str | None = None
    append_only: bool = False

    _pk_columns: tuple[str, ...] = field(init=False, repr=False)
    _by_name: dict[str, Column] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """validate + index the schema after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when a pk column is not declared in
            :attr:`columns`, when a declared pk column is not marked
            immutable-compatible, or when :attr:`cas_column` is set but
            missing from :attr:`columns`
        """
        pk = self.primary_key
        pk_cols: tuple[str, ...] = (pk,) if isinstance(pk, str) else tuple(pk)
        by_name = {c.name: c for c in self.columns}
        for pk_col in pk_cols:
            if pk_col not in by_name:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): primary_key column "
                    f"{pk_col!r} not found in columns",
                )
        if self.cas_column is not None and self.cas_column not in by_name:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_column "
                f"{self.cas_column!r} not found in columns",
            )
        # dataclass is frozen, use object.__setattr__ to install derived
        # attributes
        object.__setattr__(self, "_pk_columns", pk_cols)
        object.__setattr__(self, "_by_name", by_name)

    @property
    def pk_columns(self) -> tuple[str, ...]:
        """return primary-key column names as a tuple (always tuple-shape).

        :return: pk column names in declared order
        :rtype: tuple[str, ...]
        """
        return self._pk_columns

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


def _coerce_uuid(value: Any) -> UUID | None:
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


def _coerce_datetime_for_write(value: Any) -> datetime | None:
    """coerce a datetime to naive-UTC form for TIMESTAMP column writes.

    belt-and-suspenders alongside
    :meth:`BaseCollection.save_entity`: aware datetimes become naive
    (UTC-projected), naive pass through. ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: naive datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value
    else:
        result = value.astimezone(UTC).replace(tzinfo=None)
    return result


def _encode_vector(value: Any) -> str | None:
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


def _decode_vector(value: Any) -> list[float] | None:
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


def _encode_jsonb(value: Any) -> str | None:
    """encode a JSONB column value for the ``::jsonb`` cast.

    dict/list values become JSON strings; pre-encoded strings pass
    through; ``None`` pass-through.

    :param value: dict, list, pre-encoded string, or ``None``
    :ptype value: Any
    :return: JSON string or ``None``
    :rtype: str | None
    """
    if value is None:
        result: str | None = None
    elif isinstance(value, str):
        result = value
    elif isinstance(value, (dict, list)):
        result = json.dumps(value, default=_json_default)
    else:
        result = json.dumps(value, default=_json_default)
    return result


def _decode_jsonb(value: Any) -> Any:
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
        except (ValueError, TypeError):
            result = value
    else:
        result = value
    return result


def _json_default(obj: object) -> Any:
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
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return result


def _decode_l2_value(column: Column, value: Any) -> Any:
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
    elif column.column_type == UUID_TYPE and isinstance(value, str):
        result = UUID(value)
    elif column.column_type == DATETIME_TYPE and isinstance(value, str):
        result = datetime.fromisoformat(value)
    elif column.column_type == BYTES_TYPE and isinstance(value, str):
        result = base64.b64decode(value.encode("ascii"))
    elif column.column_type == BOOL_TYPE and isinstance(value, (bool, int)):
        result = bool(value)
    elif column.column_type == INT_TYPE and isinstance(value, (int, float)):
        result = int(value)
    elif column.column_type == VECTOR_TYPE:
        result = _decode_vector(value)
    elif column.column_type == JSONB_TYPE:
        # JSONB round-trips through JSON natively; value is already the
        # decoded structure
        result = value
    else:
        result = value
    return result


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
                    Column("date_created", DATETIME_TYPE, immutable=True),
                    Column("date_updated", DATETIME_TYPE),
                ],
                cas_column="date_updated",
            )
            entity_class = MyEntity
            primary_key_column = "id"

    and inherit the full CRUD implementation:
    :meth:`fetch_from_postgres`, :meth:`save_to_postgres`,
    :meth:`delete_from_postgres`, :meth:`serialize`, :meth:`deserialize`.
    domain-specific query methods (e.g. ``find_by_agent``) stay on the
    subclass because query shape is per-collection.

    the L3 pool is read from :attr:`l3_pool` (the public extension
    point on :class:`BaseCollection`). subclasses that accept a
    ``postgres_pool`` constructor kwarg should assign it onto
    ``self.l3_pool`` in their ``__init__`` after calling ``super()``
    so the generic path finds the pool uniformly.
    """

    schema: ClassVar[TableSchema]

    # --- SQL generation ---

    def _build_where_pk(self, start: int = 1) -> tuple[str, tuple[str, ...]]:
        """build ``WHERE pk...`` fragment for fetch / delete / CAS.

        :param start: first positional-parameter index
        :ptype start: int
        :return: (where clause fragment, tuple of pk column names in
            declared order)
        :rtype: tuple[str, tuple[str, ...]]
        """
        pk_cols = self.schema.pk_columns
        parts = [f"{col} = ${start + i}" for i, col in enumerate(pk_cols)]
        return " AND ".join(parts), pk_cols

    def _render_param(self, column: Column, idx: int) -> str:
        """render ``$N`` with optional ``::jsonb`` / ``::vector`` cast.

        :param column: column descriptor
        :ptype column: Column
        :param idx: positional-parameter index (1-based)
        :ptype idx: int
        :return: placeholder string ready for SQL interpolation
        :rtype: str
        """
        if column.column_type == JSONB_TYPE:
            result = f"${idx}::jsonb"
        elif column.column_type == VECTOR_TYPE:
            result = f"${idx}::vector"
        else:
            result = f"${idx}"
        return result

    def _build_insert_sql(self) -> str:
        """build INSERT SQL (with or without ON CONFLICT per schema).

        column order matches :attr:`TableSchema.columns` exactly; tests
        that assert on positional asyncpg parameters can rely on that.

        :return: parameterized INSERT SQL
        :rtype: str
        """
        schema = self.schema
        cols = schema.columns
        col_names = ", ".join(c.name for c in cols)
        placeholders = ", ".join(self._render_param(c, i + 1) for i, c in enumerate(cols))
        sql = f"INSERT INTO {schema.name} ({col_names}) VALUES ({placeholders})"
        if schema.append_only:
            result = sql
        else:
            pk_cols = ", ".join(schema.pk_columns)
            mutable = schema.mutable_columns()
            if mutable:
                set_clause = ", ".join(f"{c.name} = EXCLUDED.{c.name}" for c in mutable)
                result = f"{sql} ON CONFLICT ({pk_cols}) DO UPDATE SET {set_clause}"
            else:
                # pk-only table or every non-pk column is immutable:
                # emit DO NOTHING so existing rows are not clobbered.
                result = f"{sql} ON CONFLICT ({pk_cols}) DO NOTHING"
        return result

    def _build_cas_update_sql(self) -> str:
        """build UPDATE SQL for the CAS (optimistic-concurrency) path.

        parameter ordering: pk values ($1..$Npk), then mutable columns
        in declared order, then the CAS fence value as the last
        parameter. callers match this layout in :meth:`save_to_postgres`.

        :return: parameterized UPDATE SQL
        :rtype: str
        :raises RuntimeError: when the schema has no :attr:`cas_column`
            (caller misuse -- CAS path is unreachable without a fence)
        """
        schema = self.schema
        if schema.cas_column is None:
            raise RuntimeError(
                f"TableSchema(name={schema.name!r}): CAS UPDATE requires "
                f"cas_column to be set",
            )
        pk_cols = schema.pk_columns
        mutable = schema.mutable_columns()
        # $1..$Npk are pk values; $Npk+1..$Npk+Nmut are mutable values
        pk_where = " AND ".join(f"{col} = ${i + 1}" for i, col in enumerate(pk_cols))
        set_parts: list[str] = []
        next_idx = len(pk_cols) + 1
        for col in mutable:
            set_parts.append(f"{col.name} = {self._render_param(col, next_idx)}")
            next_idx += 1
        # final $ is the CAS fence value
        cas_idx = next_idx
        set_clause = ", ".join(set_parts)
        result = (
            f"UPDATE {schema.name} SET {set_clause} "
            f"WHERE {pk_where} AND {schema.cas_column} = ${cas_idx}"
        )
        return result

    def _build_fetch_sql(self) -> str:
        """build SELECT SQL for pk-based fetch.

        :return: parameterized SELECT SQL
        :rtype: str
        """
        where, _ = self._build_where_pk(start=1)
        return f"SELECT * FROM {self.schema.name} WHERE {where}"

    def _build_delete_sql(self) -> str:
        """build DELETE SQL for pk-based delete.

        :return: parameterized DELETE SQL
        :rtype: str
        """
        where, _ = self._build_where_pk(start=1)
        return f"DELETE FROM {self.schema.name} WHERE {where}"

    # --- value normalization ---

    def _normalize_write_value(self, column: Column, value: Any) -> Any:
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
        elif column.column_type == UUID_TYPE:
            result = _coerce_uuid(value)
        elif column.column_type == DATETIME_TYPE:
            result = _coerce_datetime_for_write(value)
        elif column.column_type == JSONB_TYPE:
            result = _encode_jsonb(value)
        elif column.column_type == VECTOR_TYPE:
            result = _encode_vector(value)
        else:
            result = value
        return result

    def _normalize_read_value(self, column: Column, value: Any) -> Any:
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
        elif column.column_type == UUID_TYPE:
            # always coerce to stdlib UUID -- pgproto UUID is a subclass
            # that SQLite's binder rejects
            result = _coerce_uuid(value)
        elif column.column_type == JSONB_TYPE:
            result = _decode_jsonb(value)
        elif column.column_type == VECTOR_TYPE:
            result = _decode_vector(value)
        else:
            result = value
        return result

    def _pull_value(self, data: dict[str, Any], column: Column) -> Any:
        """read a column's value from the caller's data dict.

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
                f"{self.schema.name}.save_to_postgres: required column "
                f"{column.name!r} missing from data",
            )
        return result

    def _build_insert_params(self, data: dict[str, Any]) -> list[Any]:
        """build the parameter list for the INSERT path in declared order.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: parameter list matching :attr:`TableSchema.columns` order
        :rtype: list[Any]
        """
        return [self._normalize_write_value(c, self._pull_value(data, c)) for c in self.schema.columns]

    def _build_cas_params(
        self, data: dict[str, Any], original_timestamp: datetime,
    ) -> list[Any]:
        """build the parameter list for the CAS UPDATE path.

        layout: pk values in declared order, then mutable columns in
        declared order, then the CAS fence value last. matches the
        placeholder layout produced by :meth:`_build_cas_update_sql`.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value
        :ptype original_timestamp: datetime
        :return: parameter list matching the CAS UPDATE placeholder
            layout
        :rtype: list[Any]
        """
        schema = self.schema
        params: list[Any] = []
        for pk_name in schema.pk_columns:
            pk_col = schema.column(pk_name)
            params.append(self._normalize_write_value(pk_col, self._pull_value(data, pk_col)))
        for col in schema.mutable_columns():
            params.append(self._normalize_write_value(col, self._pull_value(data, col)))
        # CAS fence value: CAS column is typically date_updated; caller
        # supplies the pre-mutation value directly
        if schema.cas_column is None:
            # unreachable -- _build_cas_update_sql enforces the guard
            raise RuntimeError(
                f"TableSchema(name={schema.name!r}): CAS path reached without cas_column",
            )
        cas_col = schema.column(schema.cas_column)
        params.append(self._normalize_write_value(cas_col, original_timestamp))
        return params

    # --- BaseCollection contract implementations ---

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one row from L3 by primary key.

        accepts scalar ``entity_id`` for single-pk tables or a tuple of
        pk values for composite-pk tables; shape is validated by
        :meth:`BaseCollection.normalize_pk`.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        result: dict[str, Any] | None
        if self.l3_pool is None:
            result = None
        else:
            pk_values = self.normalize_pk(entity_id)
            # coerce pk values uniformly (str -> UUID when column is
            # declared UUID, etc.)
            coerced: list[Any] = []
            for pk_name, raw in zip(self.schema.pk_columns, pk_values, strict=True):
                coerced.append(self._normalize_write_value(self.schema.column(pk_name), raw))
            row = await self.l3_pool.fetchrow(self._build_fetch_sql(), *coerced)
            if row is None:
                result = None
            else:
                result = self._coerce_row(dict(row))
        return result

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """persist one row to L3.

        behaviour branches on schema configuration:

        * ``append_only=True``: INSERT only, no ``ON CONFLICT``;
          duplicate-key conflicts raise from asyncpg naturally
        * ``cas_column`` set AND ``original_timestamp is not None``:
          emit UPDATE fenced on ``cas_column``; returns 0 on fence
          mismatch (:class:`~threetears.core.exceptions.ConcurrentModificationError`
          surfaces from the caller)
        * otherwise: upsert via INSERT ... ON CONFLICT (pk) DO UPDATE

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value; ignored
            when :attr:`TableSchema.cas_column` is ``None``
        :ptype original_timestamp: datetime | None
        :return: rows affected reported by asyncpg (0 on CAS failure, 1
            on success)
        :rtype: int
        """
        result: int
        if self.l3_pool is None:
            result = 0
        else:
            schema = self.schema
            if schema.cas_column is not None and original_timestamp is not None and not schema.append_only:
                sql = self._build_cas_update_sql()
                params = self._build_cas_params(data, original_timestamp)
            else:
                sql = self._build_insert_sql()
                params = self._build_insert_params(data)
            status = await self.l3_pool.execute(sql, *params)
            result = self._parse_rowcount(status)
        return result

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """delete one row from L3 by primary key.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is not None:
            pk_values = self.normalize_pk(entity_id)
            coerced: list[Any] = []
            for pk_name, raw in zip(self.schema.pk_columns, pk_values, strict=True):
                coerced.append(self._normalize_write_value(self.schema.column(pk_name), raw))
            await self.l3_pool.execute(self._build_delete_sql(), *coerced)

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

    # --- row / rowcount helpers ---

    def _coerce_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """apply read-boundary coercion to every known column in a row.

        columns not declared in the schema pass through untouched so
        ad-hoc joined projections still round-trip cleanly.

        :param row: row dict as returned by asyncpg
        :ptype row: dict[str, Any]
        :return: same-shape dict with coerced values
        :rtype: dict[str, Any]
        """
        result: dict[str, Any] = {}
        for key, value in row.items():
            col = self.schema.get_column(key)
            if col is None:
                result[key] = value
            else:
                result[key] = self._normalize_read_value(col, value)
        return result

    @staticmethod
    def _parse_rowcount(status: Any) -> int:
        """parse asyncpg's command-tag string into a row count.

        asyncpg returns a string like ``"INSERT 0 1"`` /
        ``"UPDATE 1"`` / ``"DELETE 1"``. empty / falsy values
        (mock pools sometimes return ``None`` or ``""``) resolve to 0.

        :param status: asyncpg status tag
        :ptype status: Any
        :return: rows-affected count
        :rtype: int
        """
        result: int
        if not status:
            result = 0
        elif isinstance(status, str):
            parts = status.split()
            if parts:
                try:
                    result = int(parts[-1])
                except ValueError:
                    result = 0
            else:
                result = 0
        else:
            result = 0
        return result
