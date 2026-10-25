"""L1 cache backend protocol and what every backend shares.

- :class:`L1Backend`, the protocol every L1 backend implements.
- :data:`MISSING`, the cache-miss sentinel (distinct from a cached ``None``).
- :func:`build_select_clause` and :func:`bulk_columns`, shared SQL construction.
- :func:`exact_decimal_type`, :func:`stored_decimal` and :class:`UnsupportedL1ColumnError`, how
  an exact ``NUMERIC`` column keeps its ``Decimal``, or is refused.

Nothing cached in L1 ages: a row stays until a write that reaches it is heard
(the collection broadcasts and their tables' write generations).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import Float, Numeric

from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "L1Backend",
    "MISSING",
    "build_select_clause",
    "bulk_columns",
    "UnsupportedL1ColumnError",
    "exact_decimal_type",
    "stored_decimal",
    # released here (v0.66.0) before it moved to threetears.core.sql_fragments; still importable
    "quote_identifier",
]

MISSING = object()
"""Sentinel for cache miss. Distinct from None (which is a valid cached value)."""


def build_select_clause(
    schema: dict[str, str] | None,
    table: str,
    columns: Sequence[str] | None,
) -> str:
    """Build a validated SELECT column list, ``*`` when unprojected.

    Shared by every backend so projection validation cannot drift
    between them.

    :param schema: the table's registered column-to-type mapping, or
        ``None``/empty when the table is not registered; validation is
        skipped then and the engine reports unknown columns itself
    :ptype schema: dict[str, str] | None
    :param table: target table name, used in error messages
    :ptype table: str
    :param columns: requested projection, or ``None`` for all columns;
        duplicates collapse, first occurrence wins
    :ptype columns: Sequence[str] | None
    :return: the SELECT clause column list
    :rtype: str
    :raises ValueError: if ``columns`` is empty, or names a column the
        registered schema does not have
    """
    if columns is None:
        return "*"
    deduped = list(dict.fromkeys(columns))
    if not deduped:
        raise ValueError("columns must be None or a non-empty sequence")
    if schema:
        unknown = [c for c in deduped if c not in schema]
        if unknown:
            raise ValueError(f"unknown columns for table {table}: {unknown}")
    return ", ".join(quote_identifier(c) for c in deduped)


def bulk_columns(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, str]) -> list[str]:
    """the columns a bulk write writes: those the rows name, in the table's order, all rows alike.

    :param rows: the rows to write
    :ptype rows: Sequence[Mapping[str, Any]]
    :param schema: the table's declared columns (empty when unknown: every named column is written)
    :ptype schema: Mapping[str, str]
    :return: the columns, filtered to the table's as ``upsert`` filters them
    :rtype: list[str]
    :raises ValueError: when the rows do not all name the same columns
    """
    named = set(rows[0]) if rows else set()
    ragged = next((i for i, row in enumerate(rows) if set(row) != named), None)
    if ragged is not None:
        raise ValueError(f"row {ragged} names different columns from row 0; a bulk write needs every row alike")
    return [c for c in schema if c in named] if schema else sorted(named)


#: wide enough for any value a Postgres ``NUMERIC`` holds, so quantizing one to its scale never
#: runs out of digits (the default context's 28 would refuse a ``NUMERIC(38, 10)`` value)
_NUMERIC_CONTEXT = Context(prec=1000, rounding=ROUND_HALF_UP)


class UnsupportedL1ColumnError(TypeError):
    """a column an L1 backend cannot hold exactly, refused when its table is created.

    An L1 that kept such a column anyway would hand back a different type or a rounded value from
    the one L3 holds, on every read it serves. Its message names the table and the column.
    """


def exact_decimal_type(
    sa_type: Any, *, table: str, column: str, max_precision: int, backend: str
) -> tuple[int, int] | None:
    """the declared ``(precision, scale)`` of an exact ``NUMERIC`` column an L1 can hold exactly.

    ``Float`` and its subclasses are ``Numeric`` to SQLAlchemy too, but Postgres stores them as binary
    floats and asyncpg answers a ``float``; they are not exact and answer ``None`` here, as every
    non-numeric type does. An exact ``NUMERIC`` is held only at a declared ``NUMERIC(p, s)`` with
    ``0 <= s <= p <= max_precision``: anything else is refused, never kept approximately.

    :param sa_type: the column's SQLAlchemy type
    :ptype sa_type: Any
    :param table: the table, for the refusal
    :ptype table: str
    :param column: the column, for the refusal
    :ptype column: str
    :param max_precision: the widest precision the backend holds exactly
    :ptype max_precision: int
    :param backend: the backend's name, for the refusal
    :ptype backend: str
    :return: ``(precision, scale)``, or ``None`` when the column is not an exact ``NUMERIC``
    :rtype: tuple[int, int] | None
    :raises UnsupportedL1ColumnError: when the column declares no precision and scale, or one the
        backend cannot hold exactly
    """
    result: tuple[int, int] | None = None
    if isinstance(sa_type, Numeric) and not isinstance(sa_type, Float):
        precision, scale = sa_type.precision, sa_type.scale
        if precision is None or scale is None:
            raise UnsupportedL1ColumnError(
                f"{table}.{column}: NUMERIC with no declared precision and scale cannot be cached in the "
                f"{backend} L1, which holds an exact NUMERIC only as a declared NUMERIC(p, s); declare both"
            )
        if not 0 <= scale <= precision <= max_precision:
            raise UnsupportedL1ColumnError(
                f"{table}.{column}: NUMERIC({precision}, {scale}) cannot be cached in the {backend} L1, which "
                f"holds an exact NUMERIC(p, s) only with 0 <= s <= p <= {max_precision}"
            )
        result = (precision, scale)
    return result


def stored_decimal(value: Any, scale: int) -> Decimal:
    """``value`` as the ``Decimal`` a ``NUMERIC`` column of ``scale`` stores and L3 answers.

    Postgres rounds a value to its column's scale half away from zero, and asyncpg hands it back
    carrying exactly that many places: ``1.5`` written to ``NUMERIC(12, 4)`` reads back as
    ``Decimal("1.5000")``. A float is taken by its shortest repr, which is the decimal it was
    written from for any value of at most 15 significant digits.

    :param value: the value written: a ``Decimal``, ``int``, ``float`` or decimal string
    :ptype value: Any
    :param scale: the column's declared scale
    :ptype scale: int
    :return: the value as stored
    :rtype: Decimal
    :raises ValueError: when ``value`` is ``NaN`` or infinite, which no L1 holds at a scale
    :raises decimal.InvalidOperation: when ``value`` is not a number
    """
    number = value if isinstance(value, Decimal) else Decimal(repr(value) if isinstance(value, float) else str(value))
    if not number.is_finite():
        raise ValueError(f"{number} has no value at scale {scale}; an L1 holds only finite NUMERIC values")
    return number.quantize(Decimal(1).scaleb(-scale), context=_NUMERIC_CONTEXT)


@runtime_checkable
class L1Backend(Protocol):
    """Protocol defining the interface for L1 cache backends.

    All methods are synchronous — L1 cache is local in-memory,
    so async adds overhead for no benefit.
    """

    def initialize(self, sa_metadata: Any) -> None:
        """Initialize the backend with schema derived from SQLAlchemy metadata."""
        ...

    def get_connection(self) -> Any:
        """Return a connection (or connection proxy) for the current thread."""
        ...

    def upsert(self, table: str, data: dict[str, Any], primary_key: str | tuple[str, ...] = "id") -> None:
        """insert or update row atomically.

        :param table: destination table name
        :ptype table: str
        :param data: row data keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK). all pk
            columns named here MUST be present in ``data``.
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        ...

    def upsert_many(
        self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...] = "id"
    ) -> int:
        """insert or update many rows in one statement, as ``upsert`` would one by one.

        every row must name the same columns; a bulk write with ragged rows has
        no single meaning for the columns some rows leave out.

        :param table: destination table name
        :ptype table: str
        :param rows: the rows, each keyed by column name, every pk column present
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: how many rows were written
        :rtype: int
        :raises ValueError: when the rows do not all name the same columns
        """
        ...

    def column_types(self, table: str) -> Mapping[str, str]:
        """the type codes this backend reads and writes a table's columns by, by column name.

        the backend's serialization codes (``TEXT_UUID``, ``VARCHAR_JSON``), not SQL
        types; a backend's own bookkeeping columns are not included.

        :param table: the table
        :ptype table: str
        :return: each column's type code (empty for a table it does not know)
        :rtype: Mapping[str, str]
        """
        ...

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        """select single row by primary key, returning None on miss.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK). length of tuple MUST
            equal length of ``primary_key`` tuple.
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added. Projection skips
            deserialization of every unselected column, which is the
            point: a wide row with one large JSON column costs its full
            parse on every unprojected read.
        :ptype columns: Sequence[str] | None
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        ...

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """select multiple rows by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_ids: list of pk values (single-PK) or list of
            tuples of pk values (composite-PK). every tuple MUST match
            the length of ``primary_key``.
        :ptype entity_ids: list[Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: list of matching row dicts; empty list when ``entity_ids`` is empty
        :rtype: list[dict[str, Any]]
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        ...

    def delete_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
    ) -> None:
        """delete single row by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK)
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        ...

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a generic SELECT query, returning list of row dicts."""
        ...

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """Serialize a Python value for storage based on column type hint."""
        ...

    def deserialize_field(self, value: Any, col_type: str) -> Any:
        """Deserialize a stored value back to the correct Python type."""
        ...

    def reset(self) -> None:
        """Close all connections and clear state."""
        ...

    def is_initialized(self) -> bool:
        """Return True if the backend has been initialized."""
        ...

    def has_table(self, table: str) -> bool:
        """Return True if ``table`` was registered via ``initialize()``.

        A pod's L1 backend is only ever initialized with the tables its OWN
        collections were created for (``collection_factory.create_dynamic_collection``
        calls ``initialize()`` per-table, lazily, the first time a Collection for
        that table is instantiated) -- a pod that never touches a given table's
        Collection locally never has it in its L1 cache at all, which is expected,
        not an error: a cross-pod cache-invalidation broadcast (``threetears.
        cache.invalidate``) is heard by EVERY pod regardless of which tables each
        one actually caches. Callers use this to skip a table their L1 backend was
        never told about, the same "unknown receipts are expected" treatment
        already given to an unrecognized ``Collection`` entirely.
        """
        ...
